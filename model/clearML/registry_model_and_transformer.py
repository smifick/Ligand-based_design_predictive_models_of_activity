from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


MODEL_REGISTRY = {
    'catboost':{
        'model': CatBoostRegressor,
        'default': {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'border_count': 254,
            'bagging_temperature': 1,
            'bootstrap_type': 'Bayesian',
            'random_strength': 1,
            'random_seed': 0,
            'boosting_type': 'Ordered',
            'grow_policy': 'SymmetricTree',
            'model_size_reg': 0.5,
            'rsm': 1,
            'nan_mode': 'Min',
            'feature_border_type': 'GreedyLogSum',
            'leaf_estimation_method': 'Newton',
            'leaf_estimation_iterations': 1,
            'leaf_estimation_backtracking': 'AnyImprovement',
        },
    },
    'random_forest':{
        'model': RandomForestRegressor,
        'default': {
            'n_estimators': 100,
            'criterion': 'squared_error',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_features': 1.0,
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0,
            'bootstrap': True,
            'oob_score': False,
            'n_jobs': None,
            'random_state': None,
            'verbose': 0,
            'warm_start': False,
            'ccp_alpha': 0.0,
            'max_samples': None
        },
    },
    #TODO остальные модели
}

COMMON_DESCRIPTOR_DEFAULTS = {
    "smiles_col": "std_smiles",
    "n_jobs": 4,
    "on_error": "raise",
    "verbose": 0,
}


DESCRIPTOR_REGISTRY = {
    "morgan": {

        'name': 'morgan',
        'default':{
            "radius": 2,
            "n_bits": 2048,
            "use_features": False,
        }
    },
}
