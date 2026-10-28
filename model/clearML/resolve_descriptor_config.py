from model.clearML.registry_model_and_transformer import COMMON_DESCRIPTOR_DEFAULTS, DESCRIPTOR_REGISTRY


def resolve_descriptor_config(name, overrides):
    if name not in DESCRIPTOR_REGISTRY:
        raise ValueError(f"Unknown descriptor: {name}")

    config = {
        **COMMON_DESCRIPTOR_DEFAULTS,
        **DESCRIPTOR_REGISTRY[name]["default"],
    }
    config.update(overrides)

    return {
        'name': name,
        'config': config,
    }
