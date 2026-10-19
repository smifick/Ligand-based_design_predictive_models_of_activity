from model.pipeline import build_pipeline
from model.data.loader import get_xy
from model.evalution import calculate_metrics


def run_pipeline(config, train, val):
    pipeline = build_pipeline(config)
    X_train, y_train = get_xy(train)
    X_val, y_val = get_xy(val)

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)
    metrics = calculate_metrics(y_val, predictions)

    return pipeline, metrics, predictions, y_val