"""Inference for an already trained sklearn Pipeline registered in ClearML.

Run from the repository root:
    python -m model.inference --model-id MODEL_ID --data data/processed/egfr_qsar_dataset_all.parquet

The model is *never trained here*. The existing project get_xy() and
calculate_metrics() functions are reused; the scaffold splits are not rebuilt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate existing ClearML Pipeline")
    parser.add_argument("--model-id", required=True, help="ID from ClearML Model Registry (not Task ID)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="Single CSV/XLSX/Parquet with a split column")
    source.add_argument("--train", type=Path, help="Train split file; requires --val and --test")
    parser.add_argument("--val", type=Path, help="Validation file, used with --train")
    parser.add_argument("--test", type=Path, help="Test file, used with --train")
    parser.add_argument("--split-col", default="split", help="Existing split-label column in --data")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/inference"))
    parser.add_argument("--project-name", default="EGFR-QSAR")
    parser.add_argument("--task-name", default=None)
    args = parser.parse_args(argv)
    if args.train and (args.val is None or args.test is None):
        parser.error("--train requires both --val and --test")
    if args.data and (args.val is not None or args.test is not None):
        parser.error("--val/--test can only be used with --train")
    return args


def read_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format {suffix}. Use CSV, XLSX or Parquet.")


def load_splits(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """Consume the *existing* split; do not call train_test_split/scaffold splitter."""
    if args.data is not None:
        frame = read_frame(args.data)
        if args.split_col not in frame:
            raise KeyError(f"Missing split column {args.split_col!r}; available: {list(frame.columns)}")
        split_labels = frame[args.split_col].astype("string").str.strip().str.lower()
        if split_labels.isna().any():
            raise ValueError(f"Null split labels in column {args.split_col!r}; refusing to drop records silently.")
        unknown = sorted(set(split_labels.dropna()) - set(SPLITS))
        if unknown:
            raise ValueError(f"Unknown split labels {unknown}. Expected: {SPLITS}. No split is reconstructed.")
        result = {split: frame.loc[split_labels == split].copy() for split in SPLITS}
    else:
        result = {split: read_frame(path) for split, path in (
            ("train", args.train), ("val", args.val), ("test", args.test)
        )}
    missing = [split for split, subset in result.items() if subset.empty]
    if missing:
        raise ValueError(f"Empty or missing splits: {missing}. Expected existing train/val/test.")
    return result


def obtain_pipeline(model_id: str, task: Any):
    """Download a serialized *whole Pipeline* by ClearML Model ID."""
    from clearml import InputModel

    model = InputModel(model_id=model_id)
    model.connect(task, name="source_pipeline")
    path_string = model.get_local_copy(raise_on_error=True)
    if not path_string or not Path(path_string).is_file():
        url = getattr(model, "url", "<unknown>")
        raise FileNotFoundError(
            "ClearML model registered, but its file is not accessible here. "
            f"Model ID: {model_id}; stored URL/path: {url!r}. "
            "Check that the training Task uploaded the model to shared storage/fileserver."
        )
    try:
        pipeline = joblib.load(path_string)
    except Exception as error:
        raise RuntimeError(
            f"Could not deserialize model {model_id} from {path_string!r}. "
            "Ensure it was saved as a full joblib/pickle Pipeline and use the "
            "same library versions and importable project modules as training."
        ) from error
    if not callable(getattr(pipeline, "predict", None)):
        raise TypeError(f"Loaded object is {type(pipeline).__name__}, not a predictor with predict().")
    return pipeline


def evaluate_pipeline(
    pipeline: Any,
    subsets: dict[str, pd.DataFrame],
    get_xy: Callable,
    calculate_metrics: Callable,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Use the same get_xy and calculate_metrics as model.training.run_pipeline."""
    targets: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}
    frames = []
    for split in SPLITS:
        df = subsets[split]
        X, y = get_xy(df)
        y_true = np.asarray(y, dtype=float).reshape(-1)
        y_pred = np.asarray(pipeline.predict(X), dtype=float).reshape(-1)
        if len(y_true) != len(df) or len(y_pred) != len(df):
            raise ValueError(
                f"{split}: dataset size {len(df)}, labels {len(y_true)}, "
                f"predictions {len(y_pred)}. Check get_xy() and preprocessing."
            )
        if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
            raise ValueError(f"{split}: NaN/Inf in targets or predictions; cannot report trustworthy metrics.")
        targets[split] = y_true
        predictions[split] = y_pred
        out = df.reset_index(drop=False).rename(columns={"index": "source_row_index"}).copy()
        out["eval_split"] = split
        out["y_true"] = y_true
        out["y_pred"] = y_pred
        frames.append(out)

    # Exactly the existing project metric function; no duplicate implementation.
    metrics = calculate_metrics(targets, predictions)
    if not isinstance(metrics, dict):
        raise TypeError("calculate_metrics() must return a dict for ClearML logging.")
    return metrics, pd.concat(frames, ignore_index=True)


def to_jsonable(obj: Any):
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(value) for value in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if np.isfinite(x) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_results(metrics: dict, predictions: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = out_dir / "metrics.json"
    predictions_file = out_dir / "predictions.csv"
    metrics_file.write_text(json.dumps(to_jsonable(metrics), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    predictions.to_csv(predictions_file, index=False)
    return metrics_file, predictions_file


def report_to_clearml(task: Any, metrics: dict, files: tuple[Path, Path]) -> None:
    logger = task.get_logger()
    for outer_key, values in metrics.items():
        if isinstance(values, dict):
            for inner_key, val in values.items():
                if isinstance(val, (float, int, np.number)) and np.isfinite(val):
                    # Works with both split->metric and metric->split nesting.
                    if outer_key in SPLITS:
                        title, series = str(inner_key), str(outer_key)
                    else:
                        title, series = str(outer_key), str(inner_key)
                    logger.report_scalar(title=title, series=series, value=float(val), iteration=0)
        elif isinstance(values, (float, int, np.number)) and np.isfinite(values):
            logger.report_scalar(title="Evaluation", series=str(outer_key), value=float(values), iteration=0)
    task.upload_artifact(name="inference_metrics", artifact_object=str(files[0]), wait_on_upload=True)
    task.upload_artifact(name="inference_predictions", artifact_object=str(files[1]), wait_on_upload=True)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    from clearml import Task
    from model.data.loader import get_xy
    from model.evalution import calculate_metrics

    task = Task.init(
        project_name=args.project_name,
        task_name=args.task_name or f"inference/{args.model_id[:12]}",
        task_type=Task.TaskTypes.testing,
        reuse_last_task_id=False,
        output_uri=True,
        auto_connect_frameworks=False,
        auto_connect_arg_parser=False,
    )
    try:
        task.connect({
            "model_id": args.model_id,
            "data": str(args.data) if args.data else None,
            "train": str(args.train) if args.train else None,
            "val": str(args.val) if args.val else None,
            "test": str(args.test) if args.test else None,
            "split_col": args.split_col,
        }, name="Inference")
        pipeline = obtain_pipeline(args.model_id, task)
        subsets = load_splits(args)
        metrics, rows = evaluate_pipeline(pipeline, subsets, get_xy, calculate_metrics)
        rows.insert(0, "std_smiles", rows.pop("std_smiles"))
        rows.insert(1, "clearml_model_id", args.model_id)
        files = save_results(metrics, rows, args.output_dir)
        report_to_clearml(task, metrics, files)
        print(json.dumps(to_jsonable(metrics), ensure_ascii=False, indent=2))
        print(f"ClearML Input Model: {args.model_id}")
        print(f"Metrics: {files[0]} | Predictions: {files[1]}")
        return metrics
    except Exception as error:
        task.mark_failed(status_reason="Inference failed", status_message=str(error))
        raise
    finally:
        task.close()


if __name__ == "__main__":
    main()
