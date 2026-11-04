from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
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
    "svr": SVR,
    "ridge": Ridge,
}

def create_model(config):
    name = config["name"]
    name_if_gs = name[2:].lower()

    if name_if_gs in MODEL_REGISTRY:
        return create_model_with_gs(config)
    else:
        config = config["config"]
        model_class = MODEL_REGISTRY[name]

        params = {
            key: value
            for key, value in config.items()
        }

    return model_class(**params)


def create_model_with_gs(config):
    gs_list = ['cv', 'scoring', 'n_jobs', 'verbose', 'refit']

    name = config['name']
    config = config['config']
    name_of_model = name[2:].lower()
    model_class = MODEL_REGISTRY[name_of_model]

    param_grid = {
        k : v for k, v in config.items()
        if isinstance(v, (list, tuple, range)) and k not in gs_list
    }
    
    param_gs = {k: config[k] for k in gs_list if k in config}

    param_for_model = {k: v for k, v in config.items()
                       if not isinstance(v, (list, tuple, range)) and k not in gs_list
                       }

    return GridSearchCV(
        estimator=model_class(**param_for_model),
        param_grid=param_grid,
        **param_gs
    )

# isinstance(config[k], (list, tuple, range))




