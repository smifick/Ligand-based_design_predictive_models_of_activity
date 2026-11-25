"""Inference сохранённого предиктора из ClearML Model Registry.

Разместить рядом с train.py. Запускать аналогично train.py:
    python script/inference_clearml.py

Задайте Model ID в SELECTION или через Inference/model_id в клоне задачи ClearML.
Обучение не выполняется. Используется проектная логика подготовки данных,
вычисления метрик и их логирования в ClearML.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from clearml import InputModel, StorageManager, Task
from model.clearML.clearML import log_results
from model.data import load_data
from model.data.loader import get_xy
from model.evalution import calculate_metrics

PROJECT_NAME = "Ligand-Activity"
TASK_NAME = "катбуст инференс"
SELECTION = {
    "model_id": "a7853dc7dca44d5abdd4bed75f2a817b",
    "save_predictions": True,
    "save_metrics": True,
}
requirements = PROJECT_ROOT / "requirements.txt"


def load_pipeline(model_id: str, task: Task):
    """Подключить к Task существующий предиктор и загрузить его из joblib."""
    input_model = InputModel(model_id=model_id)
    input_model.connect(task, name="Pipeline", ignore_remote_overrides=True)

    try:
        model_path = input_model.get_local_copy(raise_on_error=True)
    except Exception as exc:
        model_path = None
        url = input_model.url or ""
        if url.startswith(("http://", "https://")) and "/REGISTER%3A" in url:
            # The fileserver stored the old task name with a literal %3A.
            corrected_url = url.replace("/REGISTER%3A", "/REGISTER%253A", 1)
            try:
                model_path = StorageManager.get_local_copy(corrected_url)
            except Exception as retry_error:
                exc = retry_error
        if not model_path:
            raise RuntimeError(
                f"Не удалось скачать модель ClearML {input_model.id}. "
                "Проверьте URL модели и доступность файлового хранилища из среды inference."
            ) from exc

    if not model_path or not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Файл для модели {input_model.id} не найден: {model_path!r}. "
            "Возможно, в ClearML зарегистрирован локальный путь агента."
        )

    pipeline = joblib.load(model_path)
    if not callable(getattr(pipeline, "predict", None)):
        raise TypeError(f"Загруженный объект {type(pipeline).__name__} не имеет predict().")
    return pipeline, input_model.id


def predict_splits(pipeline, datasets):
    predictions = {}
    targets = {}
    for split, frame in datasets.items():
        if frame.empty:
            raise ValueError(f"Выборка {split!r} пуста — метрики не рассчитываются.")
        X, y = get_xy(frame)
        y_true = np.asarray(y, dtype=float).reshape(-1)
        y_pred = np.asarray(pipeline.predict(X), dtype=float)
        if y_pred.shape not in {(len(frame),), (len(frame), 1)}:
            raise ValueError(
                f"{split}: ожидалось одно предсказание на строку, получено {y_pred.shape}."
            )
        y_pred = y_pred.reshape(-1)
        if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
            raise ValueError(f"{split}: в целях или предсказаниях есть NaN/Inf.")
        predictions[split] = y_pred
        targets[split] = y_true
    return predictions, targets


def build_prediction_table(datasets, targets, predictions, model_id):
    prediction_tables = []
    for split, frame in datasets.items():
        # Сохраняем исходные поля (SMILES, тип мутации и другие метаданные).
        report = frame.reset_index(drop=True).copy()
        report.insert(0, "std_smiles", report.pop("std_smiles"))
        report.insert(1, "clearml_model_id", model_id)
        report.insert(2, "inference_split", split)
        report["target_pIC50"] = targets[split]
        report["prediction_pIC50"] = predictions[split]
        prediction_tables.append(report)

    return pd.concat(prediction_tables, ignore_index=True)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Объект типа {type(value).__name__} не сериализуется в JSON")


def upload_reports(task: Task, metrics, prediction_table, settings: dict):
    """Загрузить отчёты в ClearML и удалить временные файлы после загрузки."""
    if not settings["save_metrics"] and not settings["save_predictions"]:
        return
    with TemporaryDirectory(prefix="egfr-inference-") as staging_dir:
        output_dir = Path(staging_dir)
        if settings["save_metrics"]:
            metrics_path = output_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(metrics, default=_json_default, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            task.upload_artifact(
                name="inference_metrics", artifact_object=str(metrics_path), wait_on_upload=True
            )

        if settings["save_predictions"]:
            predictions_path = output_dir / "predictions.csv"
            prediction_table.to_csv(predictions_path, index=False)
            task.upload_artifact(
                name="inference_predictions", artifact_object=str(predictions_path), wait_on_upload=True
            )


def main():
    Task.force_requirements_env_freeze(
        requirements_file=requirements
    )

    task = Task.init(
        project_name=PROJECT_NAME,
        task_name=TASK_NAME,
        task_type=Task.TaskTypes.testing,
        reuse_last_task_id=False,
        output_uri=True,
        auto_connect_frameworks=False,
        auto_connect_arg_parser=False,
    )
    try:
        # Позволяет менять Model ID прямо в UI клонированного ClearML Task.
        settings = dict(task.connect(deepcopy(SELECTION), name="Inference"))
        model_id = str(settings["model_id"]).strip()
        if not model_id or model_id == "PASTE_CLEARML_MODEL_ID_HERE":
            raise ValueError("Укажите ClearML Model ID в SELECTION или Inference/model_id.")

        pipeline, resolved_model_id = load_pipeline(model_id, task)
        train, val, test = load_data()
        print(f"Train: {train.shape}, Validation: {val.shape}, Test: {test.shape}")

        datasets = {"train": train, "val": val, "test": test}
        predictions, targets = predict_splits(pipeline, datasets)
        metrics = calculate_metrics(targets, predictions)
        log_results(task, metrics, targets, predictions)
        prediction_table = (
            build_prediction_table(datasets, targets, predictions, resolved_model_id)
            if settings["save_predictions"] else None
        )
        upload_reports(task, metrics, prediction_table, settings)

        print(f"ClearML Input Model ID: {resolved_model_id}")
        print(json.dumps(metrics, default=_json_default, ensure_ascii=False, indent=2))
        task.flush(wait_for_uploads=True)
        return metrics
    except Exception as exc:
        task.mark_failed(status_reason="Inference failed", status_message=str(exc))
        raise
    finally:
        task.close()


if __name__ == "__main__":
    main()
