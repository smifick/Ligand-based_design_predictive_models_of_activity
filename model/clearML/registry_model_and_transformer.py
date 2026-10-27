from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR


#СЛИШКОМ ПОЗДНО понял что можно просто было параметры у моделей тащить, а не вручную переписывать. Блин. Ну так можно хотя бы дефолтные настройки подогнать под наш проект
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
    'xgboost':{
        'model': XGBRegressor,
        'default':{
            "objective": "reg:squarederror",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "min_child_weight": 1,
            "gamma": 0.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "colsample_bylevel": 1.0,
            "colsample_bynode": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "booster": "gbtree",
            "tree_method": "hist",
            "max_bin": 256,
            "max_leaves": 0,
            "grow_policy": "depthwise",
            "max_delta_step": 0,
            "base_score": None,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 1,
            "device": "cpu",
            "sampling_method": "uniform",
            "scale_pos_weight": None,
            "num_parallel_tree": 1,
            "monotone_constraints": None,
            "interaction_constraints": None,
            "importance_type": "gain",
            "eval_metric": "rmse",
            "early_stopping_rounds": None,
            "callbacks": None,
            "enable_categorical": False,
            "feature_types": None,
            "feature_weights": None,
            "max_cat_to_onehot": None,
            "max_cat_threshold": None,
            "multi_strategy": None,
            "validate_parameters": True,
            "missing": float("nan"),
        },
    },
    'lightgbm':{
        'model': LGBMRegressor,
        'default':{
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": -1,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample_for_bin": 200000,
            "objective": "regression",
            "class_weight": None,
            "min_split_gain": 0.0,
            "min_child_weight": 0.001,
            "min_child_samples": 20,
            "subsample": 1.0,
            "subsample_freq": 0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "random_state": 42,
            "n_jobs": -1,
            "importance_type": "split",
        },
    },

    'ridge':{
        'model': Ridge,
        'default': {
            "alpha": 1.0,
            "fit_intercept": True,
            "copy_X": True,
            "max_iter": 1000,
            "tol": 1e-4,
            "solver": "auto",
            "positive": False,
            "random_state": 42,
        },
    },
    'svr':{
        'model': SVR,
        'default': {
            "kernel": "rbf",
            "degree": 3,
            "gamma": "scale",
            "coef0": 0.0,
            "tol": 1e-3,
            "C": 1.0,
            "epsilon": 0.1,
            "shrinking": True,
            "cache_size": 200,
            "verbose": False,
            "max_iter": -1,
        },
    }
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
