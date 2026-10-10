import pandas as pd

from model.training import run_pipeline
from model.clearML.clearML import init_experiment, log_results
from model.data import load_data

def main():
    config = {
        "descriptor": {
            "fp_type": "morgan",
            "radius": 2,
            "n_bits": 2048,
            "n_jobs": 4,
            "on_error": "raise",
        },
        "model": {
            "name": "catboost",
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
        },
        "training": {
            "seed": 67,
        },
    }

    task, config = init_experiment(config)

    train_data,val_data, test_data = load_data()

    print("Train:", train_data.shape)
    print("Validation:", val_data.shape)
    print("Test:", test_data.shape)

    pipeline, metrics, predictions, y_true = run_pipeline(
        config,
        train_data,
        val_data
    )

    log_results(task, metrics, y_true, predictions)

    print(metrics)

    task.close()


if __name__ == "__main__":
    main()