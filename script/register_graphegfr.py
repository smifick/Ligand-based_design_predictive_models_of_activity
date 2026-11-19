"""Register the configured GraphEGFR checkpoint in ClearML."""

import csv
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

WEIGHTS_DIR = REPO_ROOT / "data" / "weights" / "graphegfr"
CHECKPOINT_NAME = "MTL_HER1_ALL_MT"
OUTPUT_INDEX = 0  # HER1 (EGFR WT) in MTL_HER1_ALL_MT.
BATCH_SIZE = 32
UPLOAD_URI = True  # Use the ClearML fileserver unless a shared storage URI is set.


def read_columns(path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        columns = next(csv.reader(source), [])
    if not columns or len(set(columns)) != len(columns):
        raise ValueError(f"Expected nonempty, unique fingerprint columns in {path}")
    return columns


def main():
    weights = WEIGHTS_DIR / f"{CHECKPOINT_NAME}.pt"
    config_path = weights.with_suffix(".json")
    hash_csv = weights.with_name(f"{weights.stem}hash.csv")
    nonhash_csv = weights.with_name(f"{weights.stem}nonhash.csv")
    print(f"Selected GraphEGFR checkpoint: {weights}")
    missing = [path.name for path in (weights, config_path, hash_csv, nonhash_csv) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files for {weights.name} in {weights.parent}: {', '.join(missing)}")

    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from clearml import OutputModel, Task

    from model.models.GraphEGFR.adapter import GraphEGFRAdapter
    from model.models.GraphEGFR.predictor import GraphEGFRPredictor

    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    hyperparams = config["hyperparam"]
    # Read dimensions from the same state dict that the adapter will restore.
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    n_tasks = state["predict_property.weight"].shape[0]
    del state, checkpoint
    if n_tasks != 1 and OUTPUT_INDEX is None:
        raise ValueError("Set OUTPUT_INDEX in the script for this multi-target checkpoint.")
    output_index = OUTPUT_INDEX if OUTPUT_INDEX is not None else 0

    adapter = GraphEGFRAdapter(
        path_weights=weights,
        node_feat_size=75,
        edge_feat_size=11,
        graph_feat_size=hyperparams["fingerprint_dim"],
        num_layers=hyperparams["num_layers"],
        num_timesteps=hyperparams["num_timesteps"],
        dropout=hyperparams.get("dropout", 0.0),
        configs={
            "n_tasks": n_tasks,
            "enable_fea": config.get("enable_fea", True),
            "enable_feaf": config.get("enable_feaf", True),
            "enable_feac": config.get("enable_feac", True),
        },
    )
    predictor = GraphEGFRPredictor(
        adapter, read_columns(hash_csv), read_columns(nonhash_csv),
        output_index, BATCH_SIZE,
    )

    with TemporaryDirectory() as directory:
        path = Path(directory) / "model.joblib"
        joblib.dump(predictor, path)
        # Exercise raw SMILES, batching, serialization, and checkpoint compatibility before upload.
        sample = pd.DataFrame({"std_smiles": ["CCO", "c1ccccc1", "CCO"]})
        predictions = joblib.load(path).predict(sample)
        if predictions.shape != (3,) or not np.isfinite(predictions).all():
            raise ValueError("GraphEGFR smoke check did not return one finite prediction per row.")
        np.testing.assert_allclose(predictions[0], predictions[2], rtol=1e-5, atol=1e-6)

        task = Task.init(
            project_name="Ligand-Activity",
            task_name=f"REGISTER GraphEGFR {weights.stem}",
            task_type=Task.TaskTypes.data_processing,
            reuse_last_task_id=False,
            output_uri=UPLOAD_URI,
            auto_connect_frameworks=False,
            auto_connect_arg_parser=False,
        )
        try:
            model = OutputModel(
                task=task,
                name=f"GraphEGFR {weights.stem} [output {output_index}]",
                config_dict={
                    "training_config": config,
                    "checkpoint": weights.name,
                    "output_index": output_index,
                    "input_column": "std_smiles",
                },
            )
            model.update_weights(
                weights_filename=str(path), async_enable=False, auto_delete_file=False,
            )
            task.flush(wait_for_uploads=True)
            print(f"ClearML Model ID: {model.id}")
            return model.id
        except Exception as error:
            task.mark_failed(status_reason="GraphEGFR registration failed", status_message=str(error))
            raise
        finally:
            task.close()


if __name__ == "__main__":
    main()
