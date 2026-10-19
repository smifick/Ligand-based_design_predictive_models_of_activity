from model.clearML.registry_model_and_transformer import MODEL_REGISTRY


def resolve_model_config(name, overrides):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")

    default = MODEL_REGISTRY[name]["default"]

    config = default.copy()

    for key, value in overrides.items():
        if key not in config:
            config[key] = value
        else:
            config[key] = overrides[key]

    return {
        'name': name,
        'config': config,
    }