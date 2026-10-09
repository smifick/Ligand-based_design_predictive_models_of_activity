"""
SmilesToFingerprint — sklearn-совместимый трансформер для генерации
молекулярных фингерпринтов из SMILES-строк.

Поддерживает: Morgan (ECFP), MACCS, RDKit, AtomPair, Torsion, Avalon.
Совместим с sklearn Pipeline, GridSearchCV, MLflow, Optuna.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import MACCSkeys, rdMolDescriptors
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Отключаем спам RDKit в stdout (можно включить обратно через logging)
RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)

def _smiles_to_fp(
    smiles: str,
    fp_type: str,
    radius: int,
    n_bits: int,
    use_features: bool,
) -> Optional[np.ndarray]:
    """Возвращает numpy-вектор фингерпринта или None, если SMILES невалиден."""
    if not isinstance(smiles, str) or not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if fp_type == "morgan":
        # В новых RDKit useFeatures заменён на atomInvariantsGenerator
        if use_features:
            gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=radius,
                fpSize=n_bits,
                atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
            )
        else:
            gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=radius,
                fpSize=n_bits,
            )
        fp = gen.GetFingerprint(mol)

    elif fp_type == "rdkit":
        gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits)
        fp = gen.GetFingerprint(mol)

    elif fp_type == "atompair":
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
        fp = gen.GetFingerprint(mol)

    elif fp_type == "torsion":
        gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
        fp = gen.GetFingerprint(mol)

    elif fp_type == "maccs":
        # MACCS всегда 167 бит — n_bits игнорируется
        fp = MACCSkeys.GenMACCSKeys(mol)

    elif fp_type == "avalon":
        from rdkit.Avalon import pyAvalonTools
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)

    else:
        raise ValueError(f"Unknown fp_type: {fp_type!r}")

    arr = np.zeros((fp.GetNumBits(),), dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# ---------------------------------------------------------------------------
# sklearn-совместимый трансформер
# ---------------------------------------------------------------------------
class SmilesToFingerprint(BaseEstimator, TransformerMixin):
    """
    Генерирует молекулярные фингерпринты из SMILES.

    Параметры
    ----------
    smiles_col : str, default='std_smiles'
        Имя колонки с SMILES. Используется только если на вход подан DataFrame.
        Если подан Series/ndarray — колонка игнорируется.
    fp_type : {'morgan','rdkit','atompair','torsion','maccs','avalon'}
        Тип фингерпринта.
    radius : int, default=2
        Радиус для Morgan (2 == ECFP4).
    n_bits : int, default=2048
        Длина вектора (игнорируется для MACCS).
    use_features : bool, default=False
        Для Morgan: использовать фармакофорные фичи вместо атомных (FCFP).
    n_jobs : int, default=-1
        Число процессов joblib. -1 = все ядра.
    on_error : {'zero','raise','drop'}
        Что делать с невалидными SMILES:
          - 'zero'  — вернуть вектор из нулей (безопасно для пайплайна);
          - 'raise' — бросить исключение;
          - 'drop'  — пометить как NaN (для последующей фильтрации).
    verbose : int, default=0
        Уровень логирования.

    Атрибуты
    --------
    n_features_ : int
        Размер выходного вектора.
    n_invalid_ : int
        Сколько SMILES не удалось распарсить во время transform.
    """

    def __init__(
        self,
        smiles_col: str = "std_smiles",
        fp_type: str = "morgan",
        radius: int = 2,
        n_bits: int = 2048,
        use_features: bool = False,
        n_jobs: int = -1,
        on_error: str = "zero",
        verbose: int = 0,
    ):
        self.smiles_col = smiles_col
        self.fp_type = fp_type
        self.radius = radius
        self.n_bits = n_bits
        self.use_features = use_features
        self.n_jobs = n_jobs
        self.on_error = on_error
        self.verbose = verbose

    # ---- helpers ----------------------------------------------------------
    def _extract_smiles(self, X) -> Sequence[str]:
        """Достаёт список SMILES из DataFrame/Series/list/ndarray."""
        # pandas DataFrame
        if hasattr(X, "columns") and self.smiles_col in getattr(X, "columns", []):
            return X[self.smiles_col].tolist()
        # pandas Series / ndarray / list
        if hasattr(X, "tolist"):
            return X.tolist()
        return list(X)

    def _resolve_n_features(self) -> int:
        if self.fp_type == "maccs":
            return 167
        return self.n_bits

    # ---- sklearn API ------------------------------------------------------
    def fit(self, X, y=None):
        """Ничего не обучает, только фиксирует размерность. Возвращает self."""
        # Валидируем параметры «на сухую»: генерируем один фейковый FP
        probe = _smiles_to_fp("C", self.fp_type, self.radius, self.n_bits, self.use_features)
        if probe is None:
            raise RuntimeError("Не удалось сгенерировать пробный фингерпринт.")
        self.n_features_ = probe.shape[0]
        self.n_invalid_ = 0
        return self

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, ["n_features_"])

        smiles_list = self._extract_smiles(X)
        n_features = self.n_features_

        if self.verbose:
            logger.info("Генерация %s FP для %d молекул...", self.fp_type, len(smiles_list))

        results = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(_smiles_to_fp)(
                s, self.fp_type, self.radius, self.n_bits, self.use_features
            )
            for s in smiles_list
        )

        n_invalid = sum(r is None for r in results)
        self.n_invalid_ = n_invalid
        if n_invalid and self.verbose:
            logger.warning("Невалидных SMILES: %d / %d", n_invalid, len(results))

        if n_invalid and self.on_error == "raise":
            raise ValueError(f"{n_invalid} SMILES не удалось распарсить.")

        # Собираем матрицу
        out = np.zeros((len(results), n_features), dtype=np.uint8)
        for i, r in enumerate(results):
            if r is None:
                if self.on_error == "drop":
                    out[i, :] = np.nan  # помечаем строку как «удалить»
                # 'zero' и 'drop' для невалидных оставляют нули/nan
                continue
            out[i, :] = r.astype(np.uint8, copy=False)

        return out

    # ---- удобный доступ к именам признаков --------------------------------
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, ["n_features_"])
        return np.array([f"{self.fp_type}_{i}" for i in range(self.n_features_)], dtype=object)