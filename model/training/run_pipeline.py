from model.pipeline import build_pipeline
from model.data.loader import get_xy
from model.evalution import calculate_metrics


def run_pipeline(config, train, val, test):
    pipeline = build_pipeline(config)

    X_train, y_train = get_xy(train)
    X_val, y_val = get_xy(val)
    X_test, y_test = get_xy(test)


    preprocessing = pipeline.named_steps["descriptor"]
    X_train_fp = preprocessing.fit_transform(X_train, y_train)
    X_val_fp = preprocessing.transform(X_val)

    pipeline.fit(
        X_train,
        y_train,
    )

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