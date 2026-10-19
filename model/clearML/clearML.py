import numpy as np
from clearml import Task
import pandas as pd

def init_experiment(config: dict = None):
    task = Task.init(
        project_name="Molecules",
        task_name="OneShot",
        output_uri=True,
    )

    selection = task.connect(
        {
            "model": "catboost",
            "descriptor": "morgan",
            "seed": 67,
        },
        name="Selection",
    )

    overrides = task.connect_configuration(
        {
            "model": {},
            "descriptor": {},
        },
        name="Overrides",
    )

    return task, selection, overrides


def log_results(task, metrics, y_true, y_pred):
    logger = task.get_logger()

    for name, value in metrics.items():
        logger.report_single_value(
            name=f"val_{name}",
            value=float(value),
        )

    logger.report_scatter2d(
        title="True vs Predicted",
        series="Validation",
        scatter=np.column_stack([y_true, y_pred]),
        iteration=0,
        xaxis="True pIC50",
        yaxis="Predicted pIC50",
        mode="markers",
    )

    predictions_df = pd.DataFrame({
        "pIC50": y_true,
        "predicted_pIC50": y_pred,
    })

    task.upload_artifact(
        name="validation_predictions",
        artifact_object=predictions_df,
    )