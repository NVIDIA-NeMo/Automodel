"""Convenience model builders for NeMo Automodel.

Currently includes:
    • build_gpt2_model – returns a GPT-2 causal language model (Flash-Attention-2 by default).
"""
from .gpt2 import build_gpt2_model  # noqa: F401

__all__ = [
    "build_gpt2_model",
] 