from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from model.models import create_model
from model.feature import SmilesToFingerprint


_DESCRIPTOR_RUNTIME = ("smiles_col", "n_jobs", "on_error", "verbose")


def build_pipeline(config):

    #preprocessing = ... - препроцессинг данных, конфигурируемый датасет

    descriptor_config = config["descriptor"]
    raw = dict(descriptor_config["config"])
    runtime = {key: raw.pop(key) for key in _DESCRIPTOR_RUNTIME if key in raw}

    descriptor = SmilesToFingerprint(
        fp_type=descriptor_config["name"],
        fp_params=raw,
        **runtime,
    )

    model = create_model(config["model"])

    return Pipeline([
        #('preprocessing', preprocessing)
        ("descriptor", descriptor),
        ("model", model),

        ]
    )
