from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR


MODEL_REGISTRY = {
    "catboost": CatBoostRegressor,
    "xgboost": XGBRegressor,
    "lightgbm": LGBMRegressor,
    "random_forest": RandomForestRegressor,
    "svm": SVR,
    "ridge": Ridge,
}

def create_model(config):

    name = config["name"]
    config = config["config"]

    model_class = MODEL_REGISTRY[name]

    params = {
        key: value
        for key, value in config.items()
    }

    return model_class(**params)