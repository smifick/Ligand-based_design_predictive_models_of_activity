"""
Курирование активностей EGFR:
- merge с confidence_score
- фильтры: =, nM, B, confidence >= 8
- расчёт pIC50 и сверка с pchembl_value
- выделение мутантов
- сохранение supplementary-файлов
"""
import json
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
REPORTS_DIR = ROOT / "reports"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_ACTIVITIES = RAW_DIR / "egfr_raw_activities.csv"
RAW_ASSAYS = RAW_DIR / "egfr_assays.csv"

FILTERS = {
    "standard_relation": "=",
    "standard_units": "nM",
    "assay_type": "B",
    "confidence_score_min": 8,
    "pIC50_range": (3.0, 12.0),
    "pIC50_diff_threshold": 0.05,
}

ACTIVITY_COLS = [
    "activity_id",
    "molecule_chembl_id",
    "parent_molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "assay_chembl_id",
    "assay_type",
    "assay_description",
    "assay_variant_mutation",
    "target_chembl_id",
    "target_organism",
    "document_chembl_id",
    "document_year",
]

CENSORED_RELATIONS = {">", "<", "<=", ">=", ">>", "~"}


def variant_status(x):
    if pd.isna(x) or str(x).strip() == "":
        return "unspecified"
    return f"mutant:{str(x).strip()}"


def main():
    log = []

    # ---------- 1. Загружаем активность ----------
    print("=" * 60)
    print("1. Загрузка активностей")
    print("=" * 60)
    df = pd.read_csv(RAW_ACTIVITIES, low_memory=False)
    print(f"Строк: {len(df)}, молекул: {df['molecule_chembl_id'].nunique()}")

    # ---------- 2. Загружаем assay ----------
    print("\n" + "=" * 60)
    print("2. Загрузка assay и merge")
    print("=" * 60)
    assays = pd.read_csv(RAW_ASSAYS, low_memory=False)
    assays = assays[["assay_chembl_id", "confidence_score"]].drop_duplicates("assay_chembl_id")
    assays["confidence_score"] = pd.to_numeric(assays["confidence_score"], errors="coerce")
    print(f"assay: {len(assays)}")

    df = df.merge(assays, on="assay_chembl_id", how="left")
    n_no_conf = df["confidence_score"].isna().sum()
    print(f"После merge: {len(df)}, без confidence_score: {n_no_conf}")

    # ---------- 3. Отбор колонок ----------
    existing = [c for c in ACTIVITY_COLS if c in df.columns]
    df = df[existing + ["confidence_score"]].copy()

    # ---------- 4. variant_status ----------
    df["variant_status"] = df["assay_variant_mutation"].apply(variant_status)

    # ---------- 5. Приведение типов ----------
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")

    # ---------- 6. pIC50_calc (для nM и =) ----------
    mask_calc = (
        (df["standard_units"] == "nM") &
        (df["standard_relation"] == "=") &
        (df["standard_value"].notna()) &
        (df["standard_value"] > 0)
    )
    df["pIC50_calc"] = np.nan
    df.loc[mask_calc, "pIC50_calc"] = 9.0 - np.log10(df.loc[mask_calc, "standard_value"])

  
    print("\n" + "=" * 60)
    print("3. Supplementary-файлы")
    print("=" * 60)

    censored = df[df["standard_relation"].isin(CENSORED_RELATIONS)].copy()
    censored_path = INTERIM_DIR / "egfr_activities_censored.parquet"
    censored.to_parquet(censored_path, index=False)
    print(f"Цензурированные ({sorted(CENSORED_RELATIONS)}): {len(censored)} -> {censored_path.name}")

    functional = df[df["assay_type"] == "F"].copy()
    functional_path = INTERIM_DIR / "egfr_activities_functional.parquet"
    functional.to_parquet(functional_path, index=False)
    print(f"Functional (F): {len(functional)} -> {functional_path.name}")


    print("\n" + "=" * 60)
    print("4. Основные фильтры")
    print("=" * 60)

    steps = [
        ("standard_relation == '='",
         lambda d: d[d["standard_relation"] == FILTERS["standard_relation"]],
         "точные IC50"),
        ("standard_units == 'nM'",
         lambda d: d[d["standard_units"] == FILTERS["standard_units"]],
         "молярные nM"),
        ("assay_type == 'B'",
         lambda d: d[d["assay_type"] == FILTERS["assay_type"]],
         "binding assay"),
        ("confidence_score >= 8",
         lambda d: d[d["confidence_score"] >= FILTERS["confidence_score_min"]],
         "надёжный маппинг на мишень"),
        ("standard_value > 0",
         lambda d: d[d["standard_value"].notna() & (d["standard_value"] > 0)],
         "убираем нули и NaN"),
        ("canonical_smiles not NaN",
         lambda d: d[d["canonical_smiles"].notna()],
         "нужна структура"),
    ]

    for name, fn, reason in steps:
        n_before = len(df)
        df = fn(df)
        removed = n_before - len(df)
        log.append({
            "step": name, "before": n_before, "after": len(df),
            "removed": removed, "reason": reason,
        })
        print(f"{name}: {n_before} -> {len(df)} (-{removed})")

    # ---------- 9. Финальный pIC50 и сверка ----------
    print("\n" + "=" * 60)
    print("5. pIC50 и сверка с pchembl_value")
    print("=" * 60)

    df["pIC50"] = np.where(
        df["pchembl_value"].notna(),
        df["pchembl_value"],
        df["pIC50_calc"],
    )

    both = df["pchembl_value"].notna() & df["pIC50_calc"].notna()
    df["pIC50_diff"] = np.nan
    df.loc[both, "pIC50_diff"] = (
        df.loc[both, "pIC50_calc"] - df.loc[both, "pchembl_value"]
    ).abs()
    df["pIC50_inconsistent"] = df["pIC50_diff"] > FILTERS["pIC50_diff_threshold"]

    n_both = int(both.sum())
    n_inc = int(df["pIC50_inconsistent"].sum())
    print(f"Сверено: {n_both}, расхождений > {FILTERS['pIC50_diff_threshold']}: {n_inc}")

    # ---------- 10. Обрезка выбросов ----------
    lo, hi = FILTERS["pIC50_range"]
    n_before = len(df)
    mask_outlier = (df["pIC50"] < lo) | (df["pIC50"] > hi)
    df = df[~mask_outlier]
    removed = n_before - len(df)
    log.append({
        "step": f"pIC50 in [{lo}, {hi}]",
        "before": n_before, "after": len(df),
        "removed": removed, "reason": "физиологически разумный диапазон",
    })
    print(f"pIC50 in [{lo}, {hi}]: {n_before} -> {len(df)} (-{removed})")

    # ---------- 11. Сохранение ----------
    print("\n" + "=" * 60)
    print("6. Сохранение")
    print("=" * 60)

    out = INTERIM_DIR / "egfr_activities_clean.parquet"
    df.to_parquet(out, index=False)
    print(f"Файл: {out}")
    print(f"Записей: {len(df)}")
    print(f"Уникальных molecule_chembl_id: {df['molecule_chembl_id'].nunique()}")
    print(f"Уникальных parent_molecule_chembl_id: {df['parent_molecule_chembl_id'].nunique()}")
    print(f"Уникальных assay_chembl_id: {df['assay_chembl_id'].nunique()}")

    # ---------- 12. Лог фильтрации ----------
    log_df = pd.DataFrame(log)
    log_path = REPORTS_DIR / "filter_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"\nЛог: {log_path}")

    # ---------- 13. Метаданные ----------
    metadata = {
        "date": str(date.today()),
        "target_chembl_id": "CHEMBL203",
        "filters": {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in FILTERS.items()},
        "n_final": int(len(df)),
        "n_unique_molecules": int(df["molecule_chembl_id"].nunique()),
    }
    meta_path = REPORTS_DIR / "clean_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Метаданные: {meta_path}")

    # ---------- 14. Статистика ----------
    print("\n" + "=" * 60)
    print("7. Итоговая статистика")
    print("=" * 60)
    print("\nРаспределение pIC50:")
    print(df["pIC50"].describe().round(3))

    print("\nvariant_status (топ-15):")
    print(df["variant_status"].value_counts().head(15))

    print("\nassay_type:")
    print(df["assay_type"].value_counts())

    print("\ndocument_year:")
    print(df["document_year"].describe().round(1))


if __name__ == "__main__":
    main()