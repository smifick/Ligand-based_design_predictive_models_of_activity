from model.pipeline import build_pipeline
from model.models.RGMPNN import RGMPNNRegressor
from model.data.loader import get_xy
from model.evalution import calculate_metrics


def run_pipeline(config, train, val, test, *, on_validation=None):
    pipeline = build_pipeline(config)

    X_train, y_train = get_xy(train)
    X_val, y_val = get_xy(val)
    X_test, y_test = get_xy(test)


    fit_params = {}
    if isinstance(pipeline.named_steps["model"], RGMPNNRegressor):
        fit_params["model__validation_data"] = (X_val, y_val)
        fit_params["model__on_validation"] = on_validation
    pipeline.fit(X_train, y_train, **fit_params)

    predictions = {
        'train': pipeline.predict(X_train),
        'val': pipeline.predict(X_val),
        'test': pipeline.predict(X_test)
    }
    targets = {
        'train': y_train,
        'val': y_val,
        'test': y_test,
    }
    metrics = calculate_metrics(targets, predictions)

    return pipeline, metrics, predictions, targets