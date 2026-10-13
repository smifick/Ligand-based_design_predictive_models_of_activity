"""
Стандартизация SMILES через RDKit + chembl_structure_pipeline.
Считаем: канонический SMILES, InChIKey, Bemis-Murcko scaffold, MW.
"""
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from chembl_structure_pipeline import standardizer

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
INTERIM_DIR = ROOT / "data" / "interim"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

IN_PATH = INTERIM_DIR / "egfr_activities_clean.parquet"
OUT_PATH = INTERIM_DIR / "egfr_activities_std.parquet"


def standardize_one(smiles):
    """Возвращает dict с полями или None-полями при ошибке."""
    empty = {"std_smiles": None, "inchi_key": None, "scaffold": None,
             "mol_weight": None, "error": None}

    if not isinstance(smiles, str) or smiles.strip() == "":
        empty["error"] = "empty"
        return empty

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        empty["error"] = "parse"
        return empty

    try:
        mol = standardizer.standardize_mol(mol)
    except Exception as e:
        empty["error"] = f"standardize_mol:{type(e).__name__}"
        return empty

    # ВАЖНО: в новых версиях chembl_structure_pipeline
    # get_parent_mol возвращает (mol, exclude_flag)
    try:
        result = standardizer.get_parent_mol(mol)
    except Exception as e:
        empty["error"] = f"get_parent_mol:{type(e).__name__}"
        return empty

    if isinstance(result, tuple):
        parent, _exclude = result
    else:
        parent = result

    if parent is None:
        empty["error"] = "no_parent"
        return empty

    try:
        can = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    except Exception as e:
        empty["error"] = f"MolToSmiles:{type(e).__name__}"
        return empty

    try:
        inchi_key = Chem.MolToInchiKey(parent)
    except Exception as e:
        empty["error"] = f"MolToInchiKey:{type(e).__name__}"
        return empty

    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(can)
    except Exception as e:
        print(f"[WARN scaffold] {type(e).__name__}: {e} | smiles={can}")
        scaffold = None

    try:
        mw = Descriptors.MolWt(parent)
    except Exception:
        mw = None

    return {"std_smiles": can, "inchi_key": inchi_key, "scaffold": scaffold,
            "mol_weight": mw, "error": None}


def main():
    print("=" * 60)
    print("1. Загрузка")
    print("=" * 60)
    df = pd.read_parquet(IN_PATH)
    print(f"Строк: {len(df)}")
    print(f"Уникальных исходных SMILES: {df['canonical_smiles'].nunique()}")

    print("\n" + "=" * 60)
    print("2. Стандартизация SMILES")
    print("=" * 60)

    n_total = len(df)
    rows = []
    for i, smi in enumerate(df["canonical_smiles"], 1):
        rows.append(standardize_one(smi))
        if i % 3000 == 0:
            print(f"  обработано {i}/{n_total}")

    std_df = pd.DataFrame(rows)
    df = pd.concat([df.reset_index(drop=True), std_df], axis=1)

    n_before = len(df)

    print("\n--- Ошибки парсинга ---")
    print(df["error"].value_counts(dropna=False))

    n_changed = (df["canonical_smiles"] != df["std_smiles"]).sum()
    print(f"\nSMILES изменились после стандартизации: {n_changed} из {n_before}")

    n_no_std = df["std_smiles"].isna().sum()
    n_no_inchi = df["inchi_key"].isna().sum()
    n_no_scaf = df["scaffold"].isna().sum()
    print(f"\nБез std_smiles:  {n_no_std}")
    print(f"Без inchi_key:   {n_no_inchi}")
    print(f"Без scaffold:    {n_no_scaf}")

    df = df[df["std_smiles"].notna() & df["inchi_key"].notna()].copy()
    print(f"\nПосле отбрасывания: {len(df)} (убрано {n_before - len(df)})")

    print(f"\nУникальных InChIKey: {df['inchi_key'].nunique()}")
    print(f"Уникальных scaffold: {df['scaffold'].nunique()}")

    # Проверка аномалий: один std_smiles, разные inchi_key
    dup_check = df.groupby("std_smiles")["inchi_key"].nunique()
    n_dup_anomaly = (dup_check > 1).sum()
    print(f"Аномалий (один SMILES -> разные InChIKey): {n_dup_anomaly}")

    # Проверка: один inchi_key, разные std_smiles
    dup_check2 = df.groupby("inchi_key")["std_smiles"].nunique()
    n_dup_anomaly2 = (dup_check2 > 1).sum()
    print(f"Аномалий (один InChIKey -> разные SMILES): {n_dup_anomaly2}")
    if n_dup_anomaly2 > 0:
        examples = dup_check2[dup_check2 > 1].head(5).index.tolist()
        print(f"  примеры InChIKey: {examples}")

    print("\nMW распределение:")
    print(df["mol_weight"].describe().round(2))

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nСохранено: {OUT_PATH}")


if __name__ == "__main__":
    main()