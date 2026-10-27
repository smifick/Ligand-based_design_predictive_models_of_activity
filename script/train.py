import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.training import run_pipeline
from copy import deepcopy
import joblib
from model.clearML.clearML import init_experiment, log_results
from clearml import Task
from model.data import load_data
from model.clearML.resolve_model_config import resolve_model_config
from model.clearML.resolve_descriptor_config import resolve_descriptor_config

REPO_ROOT = Path(__file__).resolve().parents[1]

requirements = REPO_ROOT / "requirements.txt"

TASK_NAME = 'МОДЕЛЬ: random_forest ДЕСКРИПТОР: morgan (((ШАБЛОН)))'
PROJECT_NAME = "Ligand-Activity"
#МОДЕЛИ: catboost xgboost lightgbm random_forest svr ridge
SELECTION = {
    "model": "random_forest",
    "descriptor": "morgan",
    "seed": 67
}


def main():
    Task.force_requirements_env_freeze(
        requirements_file=requirements
    )
    task, selection, overrides = init_experiment(task_name= TASK_NAME,project_name= PROJECT_NAME, config = SELECTION)

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
    model_params = task.connect(
        deepcopy(config["model"]["config"]),
        name="Model"
    )
    descriptor_params = task.connect(
        deepcopy(config["descriptor"]["config"]),
        name="Descriptor"
    )

    config["model"]["config"] = dict(model_params)
    config["descriptor"]["config"] = dict(descriptor_params)

    train_data, val_data, test_data = load_data()

    print("Train:", train_data.shape)
    print("Validation:", val_data.shape)
    print("Test:", test_data.shape)

    pipeline, metrics, predictions, targets = run_pipeline(
        config,
        train_data,
        val_data,
        test_data,
    )

    log_results(task, metrics, targets, predictions)

    model_path = Path("artifacts/model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)

    task.update_output_model(
        model_path=str(model_path),
        name="Pipeline_model",
        model_name=f"{SELECTION['model']}_model;",
    )

    task.flush(wait_for_uploads=True)
    task.close()


if __name__ == "__main__":
    main()