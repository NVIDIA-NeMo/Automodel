# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom DeciLM model implementation for NeMo AutoModel.

Supports:
- Sequence packing via THD (Total-tokens, Hidden, Depth) format
- Context parallelism (CP) via TE DotProductAttention.set_context_parallel_group
- Tensor parallelism (TP) via separate QKV / gate_up projections with
  ColwiseParallel / RowwiseParallel
- HF checkpoint loading / saving via DeciLMStateDictAdapter (pass-through)

Architecture:
- Stripped-down Llama with per-layer variable GQA via block_configs
- Some layers may be no-op (skipped) or replaced with linear projections
- Per-layer variable MLP width via ffn_mult
- Llama3-style RoPE scaling
"""

import functools
import math
from typing import Any, Optional

import torch
import torch.nn as nn

from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.decilm.layers import Block
from nemo_automodel.components.models.decilm.state_dict_adapter import DeciLMStateDictAdapter
from nemo_automodel.components.models.gpt_oss.rope_utils import position_ids_to_freqs_cis
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class DeciLMRotaryEmbedding(torch.nn.Module):
    """Llama3-style RoPE compatible with ``position_ids_to_freqs_cis``.

    Computes Llama3 smooth-interpolation inverse frequencies and exposes them
    via ``_compute_concentration_and_inv_freq()`` so the shared
    ``position_ids_to_freqs_cis`` utility (from gpt_oss/rope_utils) can
    produce the correct freqs_cis tensor for TE fused rope / non-fused rope.
    """

    def __init__(self, config, device: Optional[torch.device] = None):
        super().__init__()
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.rotary_dim = self.head_dim
        self.device = device

        base, rope_scaling = _get_rope_config(config)
        self.base = base

        inv_freq, self.attention_scaling = _compute_llama3_inv_freq(config, device, base, rope_scaling)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @functools.cache
    def _compute_concentration_and_inv_freq(self):
        """API expected by ``position_ids_to_freqs_cis``."""
        return self.attention_scaling, self.inv_freq


def _get_rope_config(config) -> tuple[float, dict]:
    """Extract rope parameters from config (handles both rope_parameters and rope_theta/rope_scaling)."""
    if hasattr(config, "rope_parameters") and config.rope_parameters:
        rope_params = config.rope_parameters
        base = rope_params.get("rope_theta", 10000.0)
        rope_scaling = rope_params
    else:
        base = getattr(config, "rope_theta", 10000.0)
        rope_scaling = getattr(config, "rope_scaling", {}) or {}
    return base, rope_scaling


def _compute_llama3_inv_freq(
    config, device: Optional[torch.device], base: float, rope_scaling: dict
) -> tuple[torch.Tensor, float]:
    """Compute Llama3-style smooth-interpolation inverse frequencies."""
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32) / head_dim)
    )

    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
    if rope_type not in ("llama3", "longrope"):
        return inv_freq, 1.0

    factor = rope_scaling.get("factor", 1.0)
    low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
    high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
    old_context_len = rope_scaling.get("original_max_position_embeddings", config.max_position_embeddings)

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = (~(wavelen < high_freq_wavelen)) & (~(wavelen > low_freq_wavelen))
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, 1.0


class DeciLMModel(nn.Module):
    """DeciLM transformer body: embeddings + variable decoder stack + final norm + RoPE."""

    def __init__(self, config, backend: BackendConfig):
        super().__init__()
        self.config = config
        self.backend = backend

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=get_dtype(config.torch_dtype, torch.bfloat16)
        )
        self.layers = nn.ModuleList(Block(layer_id, config, backend) for layer_id in range(config.num_hidden_layers))
        self.norm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = DeciLMRotaryEmbedding(config, device=torch.device(f"cuda:{torch.cuda.current_device()}"))

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = (
                torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
            )

        freqs_cis = position_ids_to_freqs_cis(
            self.rotary_emb,
            position_ids,
            qkv_format=attn_kwargs.get("qkv_format", "bshd"),
            for_fused_rope=self.backend.rope_fusion,
            cp_size=attn_kwargs.get("cp_size", 1),
        )

        h = self.embed_tokens(input_ids) if self.embed_tokens is not None else input_ids

        for layer in self.layers:
            h = layer(
                x=h,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                **attn_kwargs,
            )

        return self.norm(h) if self.norm is not None else h

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()
            self.rotary_emb.device = buffer_device
        for layer in self.layers:
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)


class DeciLMForCausalLM(HFCheckpointingMixin, nn.Module):
    """DeciLM causal LM with THD packing, CP, and TP support.

    Instantiate via ``from_config`` or ``from_pretrained``.  Pass a custom
    ``BackendConfig`` to control attention backend (te / sdpa), linear backend
    (te / torch), and whether to create the HF state dict adapter.
    """

    @classmethod
    def from_config(
        cls,
        config,
        backend: BackendConfig | None = None,
        **kwargs,
    ) -> "DeciLMForCausalLM":
        return cls(config, backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ) -> "DeciLMForCausalLM":
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        self.model = DeciLMModel(config, backend=self.backend)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

        # Native key names match HF format; adapter is a pass-through.
        self.state_dict_adapter = DeciLMStateDictAdapter(config)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        # Sequence packing: collapse [1, T] batch dim to [T] for THD format
        if attn_kwargs.get("qkv_format") == "thd":
            input_ids, position_ids, _, attn_kwargs = squeeze_input_for_thd(input_ids, position_ids, None, attn_kwargs)
            attention_mask = None

        hidden = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **attn_kwargs,
        )
        logits = self.lm_head(hidden) if self.lm_head is not None else hidden

        # Restore batch dim for THD output
        if attn_kwargs.get("qkv_format") == "thd":
            logits = logits.unsqueeze(0)
        return logits

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            self.model.init_weights(buffer_device=buffer_device)
            final_out_std = self.config.hidden_size**-0.5
            cutoff_factor = 3
            if self.lm_head is not None and not getattr(self.config, "tie_word_embeddings", False):
                nn.init.trunc_normal_(
                    self.lm_head.weight,
                    mean=0.0,
                    std=final_out_std,
                    a=-cutoff_factor * final_out_std,
                    b=cutoff_factor * final_out_std,
                )

        cast_model_to_dtype(self, dtype)
        with buffer_device:
            self.model.rotary_emb.device = buffer_device


ModelClass = DeciLMForCausalLM
