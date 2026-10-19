from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from ..clearML.resolve_model_config import resolve_model_config


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
    config = config["overrides"]

    resolved_config = resolve_model_config(name, config)
    name = resolved_config["name"]
    config = resolved_config["config"]

    model_class = MODEL_REGISTRY[name]

    params = {
        key: value
        for key, value in config.items()
    }

    return model_class(**params)