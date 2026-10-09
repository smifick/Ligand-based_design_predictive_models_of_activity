import numpy as np
from clearml import Task


def init_experiment(config: dict):
    task = Task.init(
        project_name="",
        task_name="",
        output_uri=True,
    )

    config["descriptor"] = task.connect(
        config["descriptor"],
        name="Descriptor",
    )

    config["model"] = task.connect(
        config["model"],
        name="Model",
    )

    config["training"] = task.connect(
        config["training"],
        name="Training",
    )

    return task, config


def log_results(task, metrics: dict, predictions):
    logger = task.get_logger()

    for name, value in metrics.items():
        logger.report_single_value(
            name=f"val_{name}",
            value=float(value),
        )

    logger.report_scatter2d(
        title="True vs Predicted",
        series="Validation",
        scatter=np.column_stack([
            predictions["pIC50"].to_numpy(),
            predictions["predicted_pIC50"].to_numpy(),
        ]),
        iteration=0,
        xaxis="True pIC50",
        yaxis="Predicted pIC50",
        mode="markers",
    )

    task.upload_artifact(
        name="validation_predictions",
        artifact_object=predictions,
    )