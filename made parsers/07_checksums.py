"""Считает SHA256 всех артефактов датасета и сохраняет в reports/checksums.txt."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

ARTIFACTS = [
    "data/raw/egfr_raw_activities.csv",
    "data/raw/egfr_assays.csv",
    "data/interim/egfr_activities_clean.parquet",
    "data/interim/egfr_activities_censored.parquet",
    "data/interim/egfr_activities_functional.parquet",
    "data/interim/egfr_activities_std.parquet",
    "data/processed/egfr_qsar_dataset_long.parquet",
    "data/processed/egfr_qsar_dataset_all.parquet",
    "data/processed/egfr_qsar_dataset_all.csv",
    "data/processed/egfr_qsar_dataset_wt.parquet",
    "data/processed/egfr_qsar_dataset_wt.csv",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    lines = []
    for rel in ARTIFACTS:
        p = ROOT / rel
        if not p.exists():
            print(f"[skip] {rel}")
            continue
        digest = sha256(p)
        lines.append(f"{digest}  {rel}")
        print(f"{digest}  {rel}")

    out = REPORTS / "checksums.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()