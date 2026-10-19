from model.clearML.registry_model_and_transformer import COMMON_DESCRIPTOR_DEFAULTS, DESCRIPTOR_REGISTRY


def resolve_descriptor_config(name, overrides):
    if name not in DESCRIPTOR_REGISTRY:
        raise ValueError(f"Unknown descriptor: {name}")

    default = DESCRIPTOR_REGISTRY[name]["default"]

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