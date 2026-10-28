"""
Скрипт для генерации фингерпринтов и объединения их с исходными датасетами.
Требует: pandas, openpyxl, rdkit, scikit-learn, joblib.

Перед запуском убедитесь, что файл fp_transformer.py с классом SmilesToFingerprint
лежит в той же директории (или установлен как модуль).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from model.feature.fp_transformer import SmilesToFingerprint

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
DATASETS = [
    "egfr_activities_std.xlsx",
    "egfr_qsar_dataset_all.xlsx",
    "egfr_qsar_dataset_long.xlsx",
    "egfr_qsar_dataset_wt.xlsx",
]

INPUT_DIR = Path(".")          # Папка, где лежат файлы (по умолчанию текущая)
OUTPUT_DIR = Path("./fp_output")  # Куда складывать результаты
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Возможные названия колонки со SMILES (проверяются по порядку)
SMILES_CANDIDATES = ["std_smiles", "canonical_smiles", "smiles", "SMILES"]

# Параметры фингерпринта
FP_PARAMS = dict(
    fp_type="morgan",   # ECFP4
    n_jobs=2,
    on_error="zero",    # невалидные SMILES -> нулевой вектор
    verbose=1,
    fp_params={
        "radius": 2,
        "n_bits": 2048,
        "use_features": False,
    },
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------
def find_smiles_column(df: pd.DataFrame) -> Optional[str]:
    """Ищет колонку со SMILES среди известных названий."""
    for col in SMILES_CANDIDATES:
        if col in df.columns:
            return col
    # fallback: ищем любую колонку, в названии которой есть 'smiles'
    for col in df.columns:
        if "smiles" in str(col).lower():
            return col
    return None


def process_one_file(filepath: Path, output_dir: Path) -> Optional[Path]:
    """Обрабатывает один Excel-файл: FP + merge + save."""
    logger.info("=" * 70)
    logger.info("Обработка файла: %s", filepath.name)

    if not filepath.exists():
        logger.error("Файл не найден: %s", filepath)
        return None

    t0 = time.time()

    # 1. Читаем Excel
    try:
        df = pd.read_excel(filepath, engine="openpyxl")
    except Exception as e:
        logger.exception("Не удалось прочитать %s: %s", filepath.name, e)
        return None

    logger.info("Загружено строк: %d, колонок: %d", len(df), len(df.columns))

    # 2. Ищем SMILES-колонку
    smiles_col = find_smiles_column(df)
    if smiles_col is None:
        logger.error("Колонка со SMILES не найдена в %s. Пропускаем.", filepath.name)
        return None
    logger.info("Используем колонку SMILES: '%s'", smiles_col)

    # 3. Убираем строки без SMILES
    n_before = len(df)
    df = df.dropna(subset=[smiles_col]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning("Удалено строк с пустыми SMILES: %d", n_dropped)

    if df.empty:
        logger.error("Датафрейм пуст после фильтрации. Пропускаем.")
        return None

    # 4. Генерируем фингерпринты
    logger.info(
        "Генерация фингерпринтов (%s, %s)...",
        FP_PARAMS["fp_type"],
        FP_PARAMS["fp_params"],
    )
    transformer = SmilesToFingerprint(
        smiles_col=smiles_col,
        **FP_PARAMS,
    )
    X_fp = transformer.fit_transform(df)  # numpy.ndarray (N, n_bits)
    logger.info("Готово. Форма матрицы FP: %s. Невалидных SMILES: %d",
                X_fp.shape, transformer.n_invalid_)

    # 5. Оборачиваем FP в DataFrame с осмысленными именами колонок
    fp_cols = [f"fp_{i}" for i in range(X_fp.shape[1])]
    df_fp = pd.DataFrame(X_fp, columns=fp_cols)

    # 6. Объединяем с исходными данными
    df_combined = pd.concat([df, df_fp], axis=1)

    # 7. Сохраняем
    out_name = filepath.stem + "_fp.xlsx"
    out_path = output_dir / out_name
    logger.info("Сохранение в %s ...", out_path)
    try:
        df_combined.to_parquet(out_path.with_suffix(".parquet"), index=False)
    except Exception as e:
        logger.exception("Ошибка сохранения %s: %s", out_path, e)
        return None

    elapsed = time.time() - t0
    logger.info("✅ Готово за %.1f сек. Файл: %s", elapsed, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("Старт обработки %d датасетов", len(DATASETS))
    results = []

    for fname in DATASETS:
        filepath = INPUT_DIR / fname
        out = process_one_file(filepath, OUTPUT_DIR)
        results.append((fname, out))

    logger.info("=" * 70)
    logger.info("ИТОГИ:")
    for fname, out in results:
        status = "✅ OK" if out else "❌ FAILED"
        logger.info("  %-40s -> %s", fname, status)

    logger.info("Все результаты в папке: %s", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()