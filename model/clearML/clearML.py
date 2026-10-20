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


def log_results(task, metrics, targets, predictions):
    logger = task.get_logger()

    #Просто графики рассеяния
    logger.report_scatter2d(
        title="True vs Predicted",
        series="Validation",
        scatter=np.column_stack([targets['val'],predictions['val'] ]),
        iteration=0,
        xaxis="True pIC50",
        yaxis="Predicted pIC50",
        mode="markers",
    )
    logger.report_scatter2d(
        title="True vs Predicted",
        series="Test",
        scatter=np.column_stack([targets['test'], predictions['test']]),
        iteration=0,
        xaxis="True pIC50",
        yaxis="Predicted pIC50",
        mode="markers",
    )
    logger.report_scatter2d(
        title="True vs Predicted",
        series="Training",
        scatter=np.column_stack([targets['train'], predictions['train']]),
        iteration=0,
        xaxis="True pIC50",
        yaxis="Predicted pIC50",
        mode="markers",
    )
    #Графики Истинное значение против отклонения от него на предикте
    logger.report_scatter2d(
        title="True vs Residuals",
        series="Validation",
        scatter=np.column_stack([targets['val'] ,targets['val'] -  predictions['val']]),
        iteration=0,
        xaxis="True pIC50",
        yaxis="Residual pIC50",
        mode="markers",
    )
    logger.report_scatter2d(
        title="True vs Residuals",
        series="Test",
        scatter=np.column_stack([targets['test'] ,targets['test'] -  predictions['test']]),
        iteration=0,
        xaxis="True pIC50",
        yaxis="Residual pIC50",
        mode="markers",
    )
    logger.report_scatter2d(
        title="True vs Residuals",
        series="Training",
        scatter=np.column_stack([targets['train'] ,targets['train'] -  predictions['train']]),
        iteration=0,
        xaxis="True pIC50",
        yaxis="Residual pIC50",
        mode="markers",
    )
    #Динамика метрик по сплитам

    logger.report_histogram(
        title="Metrics",
        series="Validation",
        values = list(metrics['val'].values()) ,
        xlabels = list(metrics['val'].keys()),
        mode="group"
    )
    logger.report_histogram(
        title="Metrics",
        series="Test",
        values = list(metrics['test'].values()),
        xlabels= list(metrics['test'].keys()),
        mode="group"
    )
    logger.report_histogram(
        title="Metrics",
        series="Training",
        values = list(metrics['train'].values()),
        xlabels=list(metrics['train'].keys()),
        mode="group"
    )

    for split, split_metrics in metrics.items():
        for name, value in split_metrics.items():
            logger.report_single_value(
                name=f"{split}_{name}",
                value=float(value),
            )

    for split in predictions:
        predictions_df = pd.DataFrame({
            "pIC50": np.asarray(targets[split]).ravel(),
            "predicted_pIC50": np.asarray(predictions[split]).ravel(),
        })

        task.upload_artifact(
            name=f"{split}_predictions",
            artifact_object=predictions_df,
        )

