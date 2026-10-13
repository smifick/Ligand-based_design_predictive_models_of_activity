"""
Scaffold split по Bemis-Murcko scaffold.
Одна split-карта применяется и к ALL, и к WT.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

IN_ALL = PROCESSED_DIR / "egfr_qsar_dataset_all.parquet"
IN_WT = PROCESSED_DIR / "egfr_qsar_dataset_wt.parquet"
OUT_ALL = PROCESSED_DIR / "egfr_qsar_dataset_all.parquet"
OUT_WT = PROCESSED_DIR / "egfr_qsar_dataset_wt.parquet"

FRAC_TRAIN = 0.8
FRAC_VAL = 0.1
FRAC_TEST = 0.1
SEED = 42


def scaffold_split(scaffolds, frac_train=0.8, frac_val=0.1, seed=42):
    """
    Возвращает массив split-меток ('train'/'val'/'test') той же длины,
    что и входной список scaffolds.

    Логика: жадно по убыванию размера scaffold-группы.
    """
    rng = np.random.RandomState(seed)

    # Группируем индексы по scaffold
    scaffold_to_idx = defaultdict(list)
    for i, s in enumerate(scaffolds):
        scaffold_to_idx[s].append(i)

    # Сортируем: большие группы первыми, тайбрейк случайный
    groups = list(scaffold_to_idx.values())
    for g in groups:
        rng.shuffle(g)
    groups.sort(key=len, reverse=True)

    n = len(scaffolds)
    target_train = int(frac_train * n)
    target_val = int(frac_val * n)

    train_idx, val_idx, test_idx = [], [], []
    for g in groups:
        if len(train_idx) + len(g) <= target_train:
            train_idx.extend(g)
        elif len(val_idx) + len(g) <= target_val:
            val_idx.extend(g)
        else:
            test_idx.extend(g)

    split = np.empty(n, dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"
    return split


def add_split(df, split_map_by_scaffold):
    df = df.copy()
    df["split"] = df["scaffold"].map(split_map_by_scaffold)
    assert df["split"].notna().all(), "Некоторые scaffold не получили split"
    return df


def report_distributions(df, name):
    print(f"\n--- {name} ---")
    n = len(df)
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        print(f"\n[{split}] n={len(sub)} ({100*len(sub)/n:.1f}%)")
        print(f"  уникальных scaffold:  {sub['scaffold'].nunique()}")
        print(f"  уникальных InChIKey:  {sub['inchi_key'].nunique()}")
        if "label" in sub.columns:
            print(f"  label: {sub['label'].value_counts().to_dict()}")
        if "variant_class" in sub.columns:
            print(f"  variant_class: {sub['variant_class'].value_counts().to_dict()}")
        print(f"  pIC50: mean={sub['pIC50'].mean():.2f}, "
              f"std={sub['pIC50'].std():.2f}, "
              f"min={sub['pIC50'].min():.2f}, "
              f"max={sub['pIC50'].max():.2f}")


def main():
    print("=" * 60)
    print("1. Загрузка")
    print("=" * 60)
    all_df = pd.read_parquet(IN_ALL)
    wt_df = pd.read_parquet(IN_WT)
    print(f"ALL: {len(all_df)} строк, {all_df['scaffold'].nunique()} scaffold")
    print(f"WT:  {len(wt_df)} строк, {wt_df['scaffold'].nunique()} scaffold")

    # ---------- 2. Split на ALL ----------
    print("\n" + "=" * 60)
    print("2. Scaffold split на ALL")
    print("=" * 60)
    split = scaffold_split(
        all_df["scaffold"].tolist(),
        frac_train=FRAC_TRAIN,
        frac_val=FRAC_VAL,
        seed=SEED,
    )
    all_df["split"] = split

    # Проверка утечки
    leak = all_df.groupby("scaffold")["split"].nunique()
    n_leak = (leak > 1).sum()
    print(f"Scaffold, попавших в >1 split: {n_leak}")
    assert n_leak == 0, "Scaffold leakage!"

    print("\nРазмеры ALL по split:")
    print(all_df["split"].value_counts())

    # ---------- 3. Применяем ту же карту к WT ----------
    print("\n" + "=" * 60)
    print("3. Применение карты к WT")
    print("=" * 60)

    # Карта: scaffold -> split
    split_map = all_df.groupby("scaffold")["split"].first().to_dict()

    # Для WT scaffold должен быть в карте, но на всякий случай проверим
    wt_df = add_split(wt_df, split_map)

    leak_wt = wt_df.groupby("scaffold")["split"].nunique()
    n_leak_wt = (leak_wt > 1).sum()
    print(f"WT scaffold, попавших в >1 split: {n_leak_wt}")
    assert n_leak_wt == 0, "WT scaffold leakage!"

    print("\nРазмеры WT по split:")
    print(wt_df["split"].value_counts())

    # ---------- 4. Проверка согласованности ----------
    print("\n" + "=" * 60)
    print("4. Проверка согласованности ALL vs WT")
    print("=" * 60)
    common_keys = set(zip(all_df["inchi_key"], all_df["variant_status"])) & \
                  set(zip(wt_df["inchi_key"], wt_df["variant_status"]))
    merged = all_df.merge(
        wt_df[["inchi_key", "variant_status", "split"]],
        on=["inchi_key", "variant_status"],
        suffixes=("_all", "_wt"),
    )
    n_mismatch = (merged["split_all"] != merged["split_wt"]).sum()
    print(f"Молекул с расхождением split между ALL и WT: {n_mismatch}")
    assert n_mismatch == 0, "Split не согласован между ALL и WT!"

    # ---------- 5. Отчёт по распределениям ----------
    print("\n" + "=" * 60)
    print("5. Распределения")
    print("=" * 60)
    report_distributions(all_df, "ALL")
    report_distributions(wt_df, "WT")

    # ---------- 6. Сохранение ----------
    print("\n" + "=" * 60)
    print("6. Сохранение")
    print("=" * 60)
    all_df.to_parquet(OUT_ALL, index=False)
    wt_df.to_parquet(OUT_WT, index=False)

    # CSV-копии для людей
    all_df.to_csv(OUT_ALL.with_suffix(".csv"), index=False)
    wt_df.to_csv(OUT_WT.with_suffix(".csv"), index=False)

    print(f"ALL: {OUT_ALL}")
    print(f"WT:  {OUT_WT}")

    # ---------- 7. Split summary ----------
    summary = []
    for name, df in [("ALL", all_df), ("WT", wt_df)]:
        for split in ["train", "val", "test"]:
            sub = df[df["split"] == split]
            summary.append({
                "dataset": name,
                "split": split,
                "n_rows": len(sub),
                "n_scaffolds": sub["scaffold"].nunique(),
                "n_inchikeys": sub["inchi_key"].nunique(),
                "pIC50_mean": sub["pIC50"].mean(),
                "pIC50_std": sub["pIC50"].std(),
                "n_active": (sub["label"] == "active").sum(),
                "n_inactive": (sub["label"] == "inactive").sum(),
                "n_gray": (sub["label"] == "gray").sum(),
            })
    pd.DataFrame(summary).to_csv(
        REPORTS_DIR / "split_summary.csv", index=False
    )
    print(f"\nСводка: {REPORTS_DIR / 'split_summary.csv'}")


if __name__ == "__main__":
    main()