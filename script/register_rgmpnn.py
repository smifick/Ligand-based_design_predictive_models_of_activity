"""Register a trained RG-MPNN pIC50 pipeline in ClearML."""
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PIPELINE_PATH = REPO_ROOT / "artifacts" / "rg_mpnn.joblib"
PROJECT_NAME = "Ligand-Activity"
MODEL_NAME = "RG-MPNN pIC50"
EPOCHS = 50
DEVICE = "cpu"
UPLOAD_URI = True


def validate_pipeline(path):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.utils.validation import check_is_fitted

    from model.models.RGMPNN import RGMPNNRegressor

    pipeline = joblib.load(path)
    if not isinstance(pipeline, Pipeline):
        raise ValueError(f"Expected a fitted RG-MPNN pipeline in {path}")
    estimator = pipeline.named_steps.get("model")
    if isinstance(estimator, GridSearchCV):
        check_is_fitted(estimator, "best_estimator_")
        estimator = estimator.best_estimator_
    if not isinstance(estimator, RGMPNNRegressor):
        raise ValueError(f"The pipeline in {path} does not contain RG-MPNN")
    check_is_fitted(estimator, "model_")

    sample = pd.DataFrame({estimator.smiles_col: ["CCO", "c1ccccc1", "CCO"]})
    predictions = np.asarray(pipeline.predict(sample))
    if predictions.shape != (3,) or not np.isfinite(predictions).all():
        raise ValueError("RG-MPNN must return one finite pIC50 prediction per row")
    if not np.isclose(predictions[0], predictions[2], rtol=1e-5, atol=1e-6):
        raise ValueError("RG-MPNN returned inconsistent predictions for repeated SMILES")

    return {
        "model": "rg_mpnn",
        "target": "pIC50",
        "input_column": estimator.smiles_col,
        "checkpoint": path.name,
        "model_params": estimator.get_params(),
    }


def train_pipeline(path, epochs, device):
    import joblib

    from model.clearML.resolve_descriptor_config import resolve_descriptor_config
    from model.clearML.resolve_model_config import resolve_model_config
    from model.data import load_data
    from model.training import run_pipeline

    config = {
        "model": resolve_model_config("rg_mpnn", {"epochs": epochs, "device": device}),
        "descriptor": resolve_descriptor_config("identity", {}),
    }
    print(f"Training RG-MPNN for {epochs} epochs on {device}", flush=True)
    pipeline, metrics, _, _ = run_pipeline(config, *load_data())
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        candidate = Path(directory) / path.name
        joblib.dump(pipeline, candidate)
        metadata = validate_pipeline(candidate)
        # Publish the complete artifact without replacing an existing file.
        os.link(candidate, path)
    print(f"Metrics: {metrics}")
    print(f"Saved RG-MPNN pipeline: {path}")
    return metadata


def main():
    from clearml import OutputModel, Task

    path = PIPELINE_PATH
    if path.is_file():
        config = validate_pipeline(path)
    else:
        config = train_pipeline(path, EPOCHS, DEVICE)

    task = Task.init(
        project_name=PROJECT_NAME,
        task_name=f"REGISTER {MODEL_NAME}",
        task_type=Task.TaskTypes.data_processing,
        reuse_last_task_id=False,
        output_uri=UPLOAD_URI,
        auto_connect_frameworks=False,
        auto_connect_arg_parser=False,
    )
    try:
        model = OutputModel(task=task, name=MODEL_NAME, config_dict=config)
        uri = model.update_weights(
            weights_filename=str(path), async_enable=False, auto_delete_file=False,
        )
        task.flush(wait_for_uploads=True)
        if not uri:
            raise RuntimeError("ClearML did not return a URI for the uploaded pipeline")
        print(f"ClearML Model ID: {model.id}")
        return model.id
    except Exception as error:
        task.mark_failed(status_reason="RG-MPNN registration failed", status_message=str(error))
        raise
    finally:
        task.close()


if __name__ == "__main__":
    main()

