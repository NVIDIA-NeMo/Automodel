"""Llama Nemotron VL model for multimodal embedding and retrieval tasks."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "LlamaNemotronVLModel",
    "LlamaNemotronVLConfig",
    "LlamaNemotronVLProcessor",
]


if TYPE_CHECKING:
    from nemo_automodel.components.models.llama_nemotron_vl.model import (
        LlamaNemotronVLConfig,
        LlamaNemotronVLModel,
    )
    from nemo_automodel.components.models.llama_nemotron_vl.processor import LlamaNemotronVLProcessor


def __getattr__(name: str) -> Any:
    """Load public model exports only when requested, keeping sidecars lightweight."""
    if name in ("LlamaNemotronVLConfig", "LlamaNemotronVLModel"):
        return getattr(import_module("nemo_automodel.components.models.llama_nemotron_vl.model"), name)
    if name in ("LlamaNemotronVLProcessor",):
        return getattr(import_module("nemo_automodel.components.models.llama_nemotron_vl.processor"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
