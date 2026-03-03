import importlib

_LAZY_ATTRS = {
    "NeMoAutoDiffusionPipeline": (".auto_diffusion_pipeline", "NeMoAutoDiffusionPipeline"),
    "PipelineSpec": (".auto_diffusion_pipeline", "PipelineSpec"),
}

__all__ = sorted(_LAZY_ATTRS.keys())


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_path, __name__)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
