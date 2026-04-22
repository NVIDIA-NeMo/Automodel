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

"""Custom Qwen3 dense model implementation for NeMo AutoModel.

Supports:
- Sequence packing via THD (Total-tokens, Hidden, Depth) format
- Context parallelism (CP) via TE DotProductAttention.set_context_parallel_group
- Tensor parallelism (TP) via combined QKV / gate_up projections with
  KV-head-grouped interleaved layout (ColwiseParallel-safe)
- FP8 via TE linear backend
- HF checkpoint loading / saving via Qwen3StateDictAdapter

Architecture highlights vs Qwen2:
- Per-head RMSNorm on Q and K (q_norm / k_norm) before RoPE
- No attention bias
- No sliding window (full attention on all layers)
"""

from typing import Any

import torch
import torch.nn as nn
from transformers import Qwen3Config

from nemo_automodel.components.models.common import (
    BackendConfig,
    get_rope_config,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding, position_ids_to_freqs_cis
from nemo_automodel.components.models.qwen3.layers import Qwen3Attention
from nemo_automodel.components.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from nemo_automodel.components.moe.layers import MLP
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class Block(nn.Module):
    """Single Qwen3 decoder layer.

    Residual structure: x + Attn(LN(x)) + MLP(LN(Attn_out))
    """

    def __init__(self, layer_idx: int, config: Qwen3Config, backend: BackendConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Qwen3Attention(config, backend)
        # Use "torch" backend (nn.Linear) so ColwiseParallel / RowwiseParallel
        # can shard gate_proj / up_proj / down_proj without errors.
        self.mlp = MLP(config.hidden_size, config.intermediate_size, "torch")
        self.input_layernorm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        attn_out = self.self_attn(
            x=self.input_layernorm(x),
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            **attn_kwargs,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.input_layernorm, self.post_attention_layernorm):
            norm.reset_parameters()
        self.self_attn.init_weights(buffer_device)
        self.mlp.init_weights(buffer_device)


class Qwen3Model(nn.Module):
    """Qwen3 transformer body: embeddings + decoder stack + final norm + RoPE."""

    def __init__(self, config: Qwen3Config, backend: BackendConfig):
        super().__init__()
        self.config = config
        self.backend = backend

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=get_dtype(config.torch_dtype, torch.bfloat16)
        )
        self.layers = nn.ModuleList(Block(layer_id, config, backend) for layer_id in range(config.num_hidden_layers))
        self.norm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps)

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        base, rope_scaling, _ = get_rope_config(config)

        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            base=base,
            dtype=torch.float32,
            initial_context_length=rope_scaling.get("original_max_position_embeddings", 4096),
            scaling_factor=rope_scaling.get("factor", 1.0),
            ntk_alpha=rope_scaling.get("beta_slow", 1.0),
            ntk_beta=rope_scaling.get("beta_fast", 32.0),
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        )

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


class Qwen3ForCausalLM(HFCheckpointingMixin, nn.Module):
    """Qwen3 dense causal LM with THD packing, CP, and TP support.

    Instantiate via ``from_config`` or ``from_pretrained``.  Pass a custom
    ``BackendConfig`` to control attention backend (te / sdpa), linear backend
    (te / torch), and whether to create the HF state dict adapter.
    """

    @classmethod
    def from_config(
        cls,
        config: Qwen3Config,
        backend: BackendConfig | None = None,
        **kwargs,
    ) -> "Qwen3ForCausalLM":
        return cls(config, backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ) -> "Qwen3ForCausalLM":
        config = Qwen3Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Qwen3Config,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        self.model = Qwen3Model(config, backend=self.backend)
        # nn.Linear (not TE linear) so ColwiseParallel can shard lm_head.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

        # Native key names already match HF format; adapter is a pass-through.
        self.state_dict_adapter = Qwen3StateDictAdapter(config)

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


ModelClass = Qwen3ForCausalLM
