import os
import requests
import pandas as pd
import time

BASE_URL = 'https://www.ebi.ac.uk/chembl/api/data/activity.json'


def fetch_egfr_activities():
    params = {
        "target_chembl_id": "CHEMBL203",
        "standard_type": "IC50",
        "limit": 100,
        "offset": 0,
    }
    all_records = []
    page = 1

    while True:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        activities = data.get("activities", [])
        all_records.extend(activities)

        print(f"Страница {page}: получено {len(activities)} записей, всего {len(all_records)}")

        next_page = data.get("page_meta", {}).get("next")
        if not next_page:
            break

        params["offset"] += params["limit"]
        page += 1
        time.sleep(0.2)

    return all_records


if __name__ == "__main__":
    all_records = fetch_egfr_activities()

    df = pd.DataFrame(all_records)
    print(f"\nВсего получено записей: {len(df)}")

    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/egfr_raw_activities.csv"
    df.to_csv(output_path, index=False)
    print(f"Сохранено в {output_path}")