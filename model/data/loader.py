import pandas as pd
from pathlib import Path


DATA_DIR = Path('../data/processed')
FEATURE_COLUMNS = ["std_smiles"]
TARGET_COLUMN = "pIC50"


def load_data(data_dir: str = DATA_DIR):
    data_dir = Path(data_dir)
    data = pd.read_excel(data_dir / 'egfr_qsar_dataset_all.xlsx')

    train = data[data['split'] == 'train'].copy()
    valid = data[data['split'] == 'val'].copy()
    test = data[data['split'] == 'test'].copy()

    return train, valid, test

def get_xy(data, feature_columns=FEATURE_COLUMNS):
    x = data[feature_columns].copy()
    y = data[TARGET_COLUMN].to_numpy()

    return x, y