def model_registry(*args, **kwargs):
    from .model_config import model_registry as _model_registry

    return _model_registry(*args, **kwargs)


def worldmodel(*args, **kwargs):
    from .config_registry import worldmodel as _worldmodel

    return _worldmodel(*args, **kwargs)


def worldmodel_wan(*args, **kwargs):
    from .config_registry import worldmodel_wan as _worldmodel_wan

    return _worldmodel_wan(*args, **kwargs)


def worldmodel_wan_debug(*args, **kwargs):
    from .config_registry import worldmodel_wan_debug as _worldmodel_wan_debug

    return _worldmodel_wan_debug(*args, **kwargs)


__all__ = [
    "model_registry",
    "worldmodel",
    "worldmodel_wan",
    "worldmodel_wan_debug",
]
