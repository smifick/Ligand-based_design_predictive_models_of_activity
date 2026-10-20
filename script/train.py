import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.training import run_pipeline
from model.clearML.clearML import init_experiment, log_results
from model.data import load_data
from model.clearML.resolve_model_config import resolve_model_config
from model.clearML.resolve_descriptor_config import resolve_descriptor_config

def main():

    task, selection, overrides = init_experiment()

    model_config = resolve_model_config(
        selection["model"],
        overrides.get("model", {}),
    )

    descriptor_config = resolve_descriptor_config(
        selection["descriptor"],
        overrides.get("descriptor", {}),
    )

    config = {
        "model": model_config,
        "descriptor": descriptor_config,
        "training": {
            "seed": selection["seed"],
        },
    }

    train_data, val_data, test_data = load_data()

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