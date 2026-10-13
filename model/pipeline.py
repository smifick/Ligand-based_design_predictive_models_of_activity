from sklearn.pipeline import Pipeline

from model.models import create_model
from model.feature import SmilesToFingerprint


def build_pipeline(config):
    descriptor = SmilesToFingerprint(
        **config["descriptor"]
    )

    model = create_model(config["model"])

    return Pipeline([
        ("descriptor", descriptor),
        ("model", model)
    ])