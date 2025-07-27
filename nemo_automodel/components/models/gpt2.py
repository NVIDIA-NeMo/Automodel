"""GPT-2 model utility wrappers for NeMo Automodel.

The canonical way to instantiate a GPT-2 with custom sizes is to pass a
`transformers.GPT2Config` into `NeMoAutoModelForCausalLM.from_config`.  For
YAML-driven workflows, however, specifying the entire nested config can be
verbose.  This module provides a *single-level* builder function that exposes
the most common GPT-2 hyper-parameters directly.

Example (YAML):

```yaml
model:
  _target_: nemo_automodel.components.models.gpt2.build_gpt2_model
  n_layer: 24           # GPT-2 Medium
  n_embd: 1024
  n_head: 16
  vocab_size: 50257
  n_positions: 2048
```
"""
from __future__ import annotations

from typing import Any

from transformers import GPT2Config

from nemo_automodel import NeMoAutoModelForCausalLM

__all__ = ["build_gpt2_model"]


def build_gpt2_model(
    *,
    vocab_size: int = 50257,
    n_positions: int = 2048,
    n_ctx: int | None = None,
    n_embd: int = 768,
    n_layer: int = 12,
    n_head: int = 12,
    bos_token_id: int = 50256,
    eos_token_id: int = 50256,
    attn_implementation: str = "flash_attention_2",
    **extra_cfg: Any,
):
    """Return a `NeMoAutoModelForCausalLM` GPT-2 with the given hyper-parameters.

    Parameters mirror the public attributes of `GPT2Config`.  Any additional
    keyword arguments are forwarded verbatim, enabling advanced tweaks (e.g.,
    `scale_attn_by_inverse_layer_idx`).
    """

    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_ctx=n_ctx if n_ctx is not None else n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        **extra_cfg,
    )

    return NeMoAutoModelForCausalLM.from_config(cfg, attn_implementation=attn_implementation) 