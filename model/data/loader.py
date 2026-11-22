import pandas as pd
from pathlib import Path


DATA_DIR = Path('../data/processed')
FEATURE_COLUMNS = ["std_smiles"]
TARGET_COLUMN = "pIC50"


def load_data(data_dir: str = DATA_DIR):
    data_dir = Path(data_dir)
    data = pd.read_excel(data_dir / 'egfr_qsar_dataset_all.xlsx')

    train_proc = data[data['split'] == 'train'].copy()
    valid_proc = data[data['split'] == 'val'].copy()
    test_proc = data[data['split'] == 'test'].copy()

    data_catboost_test = pd.read_csv('../data/pred_cat/test_predictions.csv')
    data_catboost_val = pd.read_csv('../data/pred_cat/val_predictions.csv')
    data_catboost_train = pd.read_csv('../data/pred_cat/train_predictions.csv')

    train = join_df(train_proc, data_catboost_train)
    valid = join_df(valid_proc, data_catboost_val)
    test = join_df(test_proc, data_catboost_test)

    return train, valid, test

def get_xy(data, feature_columns=FEATURE_COLUMNS):
    x = data[feature_columns].copy()
    y = data[TARGET_COLUMN].to_numpy()

    return x, y

def join_df(df1, df2):
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    df2 = df2.drop(columns=['pIC50'])

    return pd.concat([df1, df2], axis=1)