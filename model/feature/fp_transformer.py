
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .fp_generators import (
    FpCallable,
    FpFactory,
    make_atompair,
    make_avalon,
    make_maccs,
    make_morgan,
    make_rdkit,
    make_torsion,
)

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)

FP_GENERATORS: dict[str, FpFactory] = {
    "morgan": make_morgan,
    "rdkit": make_rdkit,
    "atompair": make_atompair,
    "torsion": make_torsion,
    "maccs": make_maccs,
    "avalon": make_avalon,
}


def _resolve_fp_callable(fp_type: str, fp_params: dict) -> FpCallable:
    try:
        factory = FP_GENERATORS[fp_type]
    except KeyError as exc:
        known = ", ".join(sorted(FP_GENERATORS))
        raise ValueError(
            f"Unknown fp_type: {fp_type!r}. Known: {known}"
        ) from exc
    return factory(fp_params)


def _smiles_to_fp(smiles: str, fp_type: str, fp_params: dict) -> Optional[np.ndarray]:
    if not isinstance(smiles, str) or not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = _resolve_fp_callable(fp_type, fp_params)(mol)

    arr = np.zeros((fp.GetNumBits(),), dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


class SmilesToFingerprint(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        smiles_col: str = "std_smiles",
        fp_type: str = "morgan",
        n_jobs: int = -1,
        on_error: str = "zero",
        verbose: int = 0,
        fp_params: Optional[dict] = None,
    ):
        self.smiles_col = smiles_col
        self.fp_type = fp_type
        self.n_jobs = n_jobs
        self.on_error = on_error
        self.verbose = verbose
        self.fp_params = fp_params

    def _extract_smiles(self, X) -> Sequence[str]:
        # pandas DataFrame
        if hasattr(X, "columns") and self.smiles_col in getattr(X, "columns", []):
            return X[self.smiles_col].tolist()
        # pandas Series / ndarray / list
        if hasattr(X, "tolist"):
            return X.tolist()
        return list(X)

    def _fp_params(self) -> dict:
        return {} if self.fp_params is None else dict(self.fp_params)

    def fit(self, X, y=None):
        probe = _smiles_to_fp("C", self.fp_type, self._fp_params())
        if probe is None:
            raise RuntimeError("Не удалось сгенерировать пробный фингерпринт.")
        self.n_features_ = probe.shape[0]
        self.n_invalid_ = 0
        return self

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, ["n_features_"])

        smiles_list = self._extract_smiles(X)
        n_features = self.n_features_
        fp_params = self._fp_params()

        if self.verbose:
            logger.info("Генерация %s FP для %d молекул...", self.fp_type, len(smiles_list))

        results = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(_smiles_to_fp)(s, self.fp_type, fp_params)
            for s in smiles_list
        )

        n_invalid = sum(r is None for r in results)
        self.n_invalid_ = n_invalid
        if n_invalid and self.verbose:
            logger.warning("Невалидных SMILES: %d / %d", n_invalid, len(results))

        if n_invalid and self.on_error == "raise":
            raise ValueError(f"{n_invalid} SMILES не удалось распарсить.")
        out = np.zeros((len(results), n_features), dtype=np.uint8)
        for i, r in enumerate(results):
            if r is None:
                if self.on_error == "drop":
                    out[i, :] = np.nan  # помечаем строку как «удалить»
                # 'zero' и 'drop' для невалидных оставляют нули/nan
                continue
            out[i, :] = r.astype(np.uint8, copy=False)

        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, ["n_features_"])
        return np.array([f"{self.fp_type}_{i}" for i in range(self.n_features_)], dtype=object)
