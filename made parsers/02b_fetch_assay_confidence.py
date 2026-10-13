"""
Скачивает метаданные assay по EGFR (CHEMBL203) из ChEMBL API.
Нужно для получения confidence_score, которого нет в activity-эндпоинте.
"""

import os
import time
import json
import requests
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/assay.json"
STATUS_URL = "https://www.ebi.ac.uk/chembl/api/data/status.json"
TARGET_CHEMBL_ID = "CHEMBL203"
LIMIT = 1000
MAX_RETRIES = 3


def request_with_retry(url, params=None):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r
        except Exception as e:
            print(f"  [retry {attempt}/{MAX_RETRIES}] {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Не удалось получить {url} после {MAX_RETRIES} попыток")


def fetch_all_assays():
    params = {
        "target_chembl_id": TARGET_CHEMBL_ID,
        "limit": LIMIT,
        "offset": 0,
    }
    all_records = []
    page = 1

    while True:
        r = request_with_retry(BASE_URL, params=params)
        data = r.json()
        assays = data.get("assays", [])
        all_records.extend(assays)
        print(f"Страница {page}: {len(assays)} записей, всего {len(all_records)}")

        next_page = data.get("page_meta", {}).get("next")
        if not next_page:
            break

        params["offset"] += LIMIT
        page += 1
        time.sleep(0.3)

    return all_records


def try_fetch_chembl_version():
    try:
        r = request_with_retry(STATUS_URL)
        data = r.json()
        return data
    except Exception as e:
        print(f"[warn] не удалось получить версию ChEMBL: {e}")
        return None


def main():
    print("=" * 60)
    print("1. Метаданные ChEMBL")
    print("=" * 60)
    status = try_fetch_chembl_version()
    if status:
        print(json.dumps(status, indent=2, ensure_ascii=False)[:500])

    print("\n" + "=" * 60)
    print("2. Скачиваем assay по CHEMBL203")
    print("=" * 60)
    records = fetch_all_assays()
    df = pd.DataFrame(records)
    print(f"\nВсего assay: {len(df)}")

    keep = ["assay_chembl_id", "confidence_score", "assay_type", "description",
            "target_chembl_id", "assay_organism"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")

    out = RAW_DIR / "egfr_assays.csv"
    df.to_csv(out, index=False)
    print(f"\nСохранено в {out}")

    print("\nРаспределение confidence_score:")
    print(df["confidence_score"].value_counts(dropna=False).sort_index())

    print("\nРаспределение assay_type:")
    print(df["assay_type"].value_counts(dropna=False))


if __name__ == "__main__":
    main()