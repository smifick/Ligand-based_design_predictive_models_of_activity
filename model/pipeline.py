from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from model.models import create_model
from model.feature import SmilesToFingerprint


def build_pipeline(config):

    #preprocessing = ... - препроцессинг данных, конфигурируемый датасет

    descriptor_config = config["descriptor"]

    descriptor = SmilesToFingerprint(
        fp_type = descriptor_config["name"],
        **descriptor_config["config"]
    )

    model = create_model(config["model"])

    return Pipeline([
        #('preprocessing', preprocessing)
        ("descriptor", descriptor),
        ("model", model),

        ]
    )