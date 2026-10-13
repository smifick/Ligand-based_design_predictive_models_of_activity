"""
Дедупликация по (InChIKey, variant_status).
Агрегация pIC50. Два основных выходных файла + long format.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

IN_PATH = INTERIM_DIR / "egfr_activities_std.parquet"
OUT_ALL = PROCESSED_DIR / "egfr_qsar_dataset_all.parquet"
OUT_WT = PROCESSED_DIR / "egfr_qsar_dataset_wt.parquet"
OUT_LONG = PROCESSED_DIR / "egfr_qsar_dataset_long.parquet"

CONTRADICTION_STD = 1.0
ACTIVE_THR = 6.0
INACTIVE_THR = 5.0
HIGH_CONF_N = 3


def variant_class(vs):
    """wt — не указан; mutant_undefined — 'UNDEFINED MUTATION'; иначе mutant."""
    if vs == "unspecified":
        return "wt"
    if "UNDEFINED" in str(vs).upper():
        return "mutant_undefined"
    return "mutant"


def classify(pIC50):
    if pIC50 >= ACTIVE_THR:
        return "active"
    if pIC50 <= INACTIVE_THR:
        return "inactive"
    return "gray"


def aggregate(df):
    agg = df.groupby(["inchi_key", "variant_status"], as_index=False).agg(
        variant_class=("variant_status", lambda x: variant_class(x.iloc[0])),
        std_smiles=("std_smiles", "first"),
        scaffold=("scaffold", "first"),
        mol_weight=("mol_weight", "first"),
        molecule_chembl_id=("molecule_chembl_id", "first"),
        parent_molecule_chembl_id=("parent_molecule_chembl_id", "first"),

        pIC50=("pIC50", "median"),
        mean_pIC50=("pIC50", "mean"),
        std_pIC50=("pIC50", "std"),
        min_pIC50=("pIC50", "min"),
        max_pIC50=("pIC50", "max"),
        n_measurements=("pIC50", "size"),

        n_assays=("assay_chembl_id", "nunique"),
        n_documents=("document_chembl_id", "nunique"),
        assay_ids=("assay_chembl_id", lambda x: ",".join(sorted(set(map(str, x))))),

        document_year_min=("document_year", "min"),
        document_year_max=("document_year", "max"),
    )

    agg["range_pIC50"] = agg["max_pIC50"] - agg["min_pIC50"]
    agg["iqr_pIC50"] = agg["max_pIC50"] - agg["min_pIC50"]  # placeholder, посчитаем ниже
    agg["std_pIC50"] = agg["std_pIC50"].fillna(0.0)
    agg["contradiction"] = agg["std_pIC50"] > CONTRADICTION_STD
    agg["high_confidence"] = agg["n_measurements"] >= HIGH_CONF_N
    agg["label"] = agg["pIC50"].apply(classify)

    # пересчитываем IQR корректно через отдельную группировку
    iqr = df.groupby(["inchi_key", "variant_status"])["pIC50"].quantile(0.75) - \
          df.groupby(["inchi_key", "variant_status"])["pIC50"].quantile(0.25)
    iqr = iqr.rename("iqr_pIC50").reset_index()
    agg = agg.drop(columns=["iqr_pIC50"]).merge(iqr, on=["inchi_key", "variant_status"], how="left")

    return agg


def main():
    print("=" * 60)
    print("1. Загрузка")
    print("=" * 60)
    df = pd.read_parquet(IN_PATH)
    print(f"Строк: {len(df)}")
    print(f"Уникальных InChIKey: {df['inchi_key'].nunique()}")

    # ---------- long format ----------
    print("\n" + "=" * 60)
    print("2. Long format (все измерения)")
    print("=" * 60)
    long_cols = [
        "inchi_key", "variant_status", "std_smiles", "scaffold", "mol_weight",
        "molecule_chembl_id", "parent_molecule_chembl_id",
        "assay_chembl_id", "document_chembl_id", "document_year",
        "pIC50", "pchembl_value", "pIC50_calc", "pIC50_diff", "pIC50_inconsistent",
        "standard_value", "standard_units",
    ]
    long_cols = [c for c in long_cols if c in df.columns]
    df[long_cols].to_parquet(OUT_LONG, index=False)
    print(f"Сохранено: {OUT_LONG} ({len(df)} строк)")

    # ---------- агрегация ----------
    print("\n" + "=" * 60)
    print("3. Дедупликация по (InChIKey, variant_status)")
    print("=" * 60)
    agg = aggregate(df)
    print(f"Строк после дедупа: {len(agg)}")
    print(f"Уникальных InChIKey: {agg['inchi_key'].nunique()}")
    print(f"Уникальных (InChIKey, variant): {len(agg)}")

    print("\nРаспределение n_measurements:")
    print(agg["n_measurements"].describe().round(2))

    print("\nРаспределение variant_class:")
    print(agg["variant_class"].value_counts())

    print("\nРаспределение label:")
    print(agg["label"].value_counts())

    n_contr = int(agg["contradiction"].sum())
    print(f"\nПротиворечий (std > {CONTRADICTION_STD}): {n_contr}")

    print("\nРаспределение n_assays:")
    print(agg["n_assays"].describe().round(2))

    # ---------- Сохранение ALL ----------
    agg.to_parquet(OUT_ALL, index=False)
    print(f"\nСохранено ALL: {OUT_ALL}  ({len(agg)} строк)")

    # ---------- Сохранение WT-only ----------
    wt = agg[agg["variant_class"] == "wt"].copy()
    wt.to_parquet(OUT_WT, index=False)
    print(f"Сохранено WT:  {OUT_WT}  ({len(wt)} строк)")

    # ---------- Лог противоречий ----------
    contr = agg[agg["contradiction"]].sort_values("std_pIC50", ascending=False)
    contr.to_csv(REPORTS_DIR / "contradictions.csv", index=False)
    print(f"\nПротиворечия: {REPORTS_DIR / 'contradictions.csv'} ({len(contr)})")

    # ---------- Сводка ----------
    print("\n" + "=" * 60)
    print("4. Сводка")
    print("=" * 60)
    print(f"ALL: {len(agg)} строк, {agg['inchi_key'].nunique()} уникальных молекул")
    print(f"     {agg['scaffold'].nunique()} уникальных scaffold")
    print(f"WT:  {len(wt)} строк, {wt['inchi_key'].nunique()} уникальных молекул")
    print(f"     {wt['scaffold'].nunique()} уникальных scaffold")

    print("\npIC50 ALL:")
    print(agg["pIC50"].describe().round(3))
    print("\npIC50 WT:")
    print(wt["pIC50"].describe().round(3))

    print("\nlabel WT:")
    print(wt["label"].value_counts())


if __name__ == "__main__":
    main()