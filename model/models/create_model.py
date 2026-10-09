from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor


def create_model(config):
    name = config["name"]

    params = {
        k: v for k, v in config.items()
        if k != "name"
    }

    if name == "catboost":
        return CatBoostRegressor(**params)

    if name == "random_forest":
        return RandomForestRegressor(**params)

    raise ValueError(f"Unknown model: {name}")