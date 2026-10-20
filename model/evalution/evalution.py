import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def calculate_metrics(targets, predictions):

    results = {
        'train': dict(),
        'val': dict(),
        'test': dict()
    }

    for key in results.keys():
        results[key] = {
            "mae": float(mean_absolute_error(targets[key], predictions[key])),
            "rmse": float(np.sqrt(mean_squared_error(targets[key], predictions[key]))),
            "r2": float(r2_score(targets[key], predictions[key])),
        }

    return results