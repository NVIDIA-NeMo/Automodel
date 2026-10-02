# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Test-only custom-model MoE diffusion transformer ("toy MoE DiT").

The model follows the custom-model contract (``HFCheckpointingMixin``,
``MoEFSDPSyncMixin``, ``ModelCapabilities``, ``state_dict_adapter``, ``ModelClass``)
and places an Automodel :class:`~nemo_automodel.components.moe.layers.MoE` in
``model.model.layers[*].mlp`` so the MoE parallelizer applies expert parallelism.
Its forward follows the :class:`SimpleAdapter` calling convention used by the
diffusion recipe: ``model(hidden_states, timestep, encoder_hidden_states,
attention_kwargs=None, return_dict=False)`` returning a one-element tuple whose
tensor has the shape of ``hidden_states``.

Both the unit tests and the multi-GPU functional test import this module.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin
from nemo_automodel.shared.utils import dtype_from_str

TOY_MOE_DIT_MODEL_TYPE = "toy_moe_dit"
TOY_MOE_DIT_ARCHITECTURE = "ToyMoEDiTForDiffusion"


def toy_backend(**overrides: Any) -> BackendConfig:
    """Return a pure-PyTorch backend (no TE / DeepEP) that runs on CPU and GPU."""
    fields = {
        "attn": "sdpa",
        "linear": "torch",
        "rms_norm": "torch",
        "rope_fusion": False,
        "experts": "torch",
        "dispatcher": "torch",
        "enable_hf_state_dict_adapter": True,
    }
    fields.update(overrides)
    return BackendConfig(**fields)


class ToyMoEDiTConfig(PretrainedConfig):
    """Configuration of the toy MoE diffusion transformer."""

    model_type = TOY_MOE_DIT_MODEL_TYPE

    def __init__(
        self,
        in_channels: int = 4,
        hidden_size: int = 64,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        text_embed_dim: int = 32,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        moe_intermediate_size: int = 32,
        router_aux_loss_coef: float = 0.01,
        score_func: str = "softmax",
        gate_bias_update_factor: float = 0.0,
        norm_eps: float = 1e-6,
        **kwargs: Any,
    ):
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.text_embed_dim = text_embed_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.router_aux_loss_coef = router_aux_loss_coef
        self.score_func = score_func
        self.gate_bias_update_factor = gate_bias_update_factor
        self.norm_eps = norm_eps
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)


def _model_dtype(config: ToyMoEDiTConfig) -> torch.dtype:
    return dtype_from_str(config.torch_dtype, default=torch.bfloat16)


def _timestep_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding ``[B] -> [B, dim]`` computed in fp32."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=timestep.device, dtype=torch.float32) / half)
    args = timestep.float().reshape(-1, 1) * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ToyMoEDiTBlock(nn.Module):
    """Joint (text + latent) self-attention followed by a routed MoE MLP."""

    def __init__(self, config: ToyMoEDiTConfig, moe_config: MoEConfig, backend: BackendConfig, dtype: torch.dtype):
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.norm1 = nn.LayerNorm(dim, eps=config.norm_eps, dtype=dtype)
        self.qkv = nn.Linear(dim, 3 * dim, dtype=dtype)
        self.proj = nn.Linear(dim, dim, dtype=dtype)
        self.norm2 = nn.LayerNorm(dim, eps=config.norm_eps, dtype=dtype)
        self.mlp = MoE(moe_config, backend)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """Run one block.

        Args:
            x: ``[batch, tokens, hidden]`` joint sequence; text tokens come first, latent tokens after them.
            temb: ``[batch, hidden]`` timestep embedding added to every token before attention.

        Returns:
            ``[batch, tokens, hidden]`` updated sequence.
        """
        batch, seq, dim = x.shape
        h = self.norm1(x) + temb[:, None, :]
        q, k, v = self.qkv(h).view(batch, seq, 3, self.num_heads, dim // self.num_heads).unbind(2)
        attn = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = x + self.proj(attn.transpose(1, 2).reshape(batch, seq, dim))
        return x + self.mlp(self.norm2(x))

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        for norm in (self.norm1, self.norm2):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        for linear in (self.qkv, self.proj):
            nn.init.normal_(linear.weight, std=init_std)
            nn.init.zeros_(linear.bias)
        self.mlp.init_weights(buffer_device, init_std=init_std)


class ToyMoEDiTModel(nn.Module):
    """Backbone exposing ``layers`` and ``moe_config`` for the MoE parallelizer."""

    def __init__(self, config: ToyMoEDiTConfig, backend: BackendConfig, moe_config: MoEConfig | None = None):
        super().__init__()
        dtype = _model_dtype(config)
        dim = config.hidden_size
        self.config = config
        self.moe_config = moe_config or MoEConfig(
            dim=dim,
            inter_dim=config.moe_intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_routed_experts=config.num_experts,
            n_shared_experts=0,
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=config.gate_bias_update_factor,
            score_func=config.score_func,
            route_scale=1.0,
            aux_loss_coeff=config.router_aux_loss_coef,
            norm_topk_prob=True,
            expert_bias=False,
            router_bias=False,
            expert_activation="swiglu",
            softmax_before_topk=config.score_func == "softmax",
            dtype=dtype,
        )
        self.proj_in = nn.Linear(config.in_channels, dim, dtype=dtype)
        self.time_proj = nn.Linear(dim, dim, dtype=dtype)
        self.text_proj = nn.Linear(config.text_embed_dim, dim, dtype=dtype)
        self.layers = nn.ModuleDict(
            {str(i): ToyMoEDiTBlock(config, self.moe_config, backend, dtype) for i in range(config.num_hidden_layers)}
        )
        self.norm_out = nn.LayerNorm(dim, eps=config.norm_eps, dtype=dtype)
        self.proj_out = nn.Linear(dim, config.in_channels, dtype=dtype)

    def forward(
        self, hidden_states: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Predict the flow velocity of the latents.

        Args:
            hidden_states: ``[batch, channels, *spatial]`` noisy latents; any number of trailing spatial dims.
            timestep: ``[batch]`` diffusion timesteps.
            encoder_hidden_states: ``[batch, text_sequence, text_embed_dim]`` text conditioning.

        Returns:
            ``[batch, channels, *spatial]`` prediction with the layout of ``hidden_states``.
        """
        latent_shape = hidden_states.shape
        batch, channels = latent_shape[:2]
        # [B, C, *spatial] -> [B, N, C] latent tokens.
        latent_tokens = hidden_states.reshape(batch, channels, -1).transpose(1, 2)
        x = self.proj_in(latent_tokens.to(self.proj_in.weight.dtype))
        text = self.text_proj(encoder_hidden_states.to(x.dtype))
        temb = self.time_proj(_timestep_embedding(timestep, x.shape[-1]).to(x.dtype))
        num_text = text.shape[1]
        h = torch.cat([text, x], dim=1)
        for layer in self.layers.values():
            h = layer(h, temb)
        out = self.proj_out(self.norm_out(h[:, num_text:]))
        return out.transpose(1, 2).reshape(latent_shape)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        for linear in (self.proj_in, self.time_proj, self.text_proj, self.proj_out):
            nn.init.normal_(linear.weight, std=init_std)
            nn.init.zeros_(linear.bias)
        nn.init.ones_(self.norm_out.weight)
        nn.init.zeros_(self.norm_out.bias)
        for layer in self.layers.values():
            layer.init_weights(buffer_device, init_std=init_std)


class ToyMoEDiTStateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Grouped native experts <-> per-expert HF ``gate_proj``/``up_proj``/``down_proj`` keys."""

    _supports_low_memory_dcp_load = True

    def __init__(self, config: ToyMoEDiTConfig, moe_config: MoEConfig, backend: BackendConfig, dtype: torch.dtype):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True

    def from_hf(self, hf_state_dict: dict[str, torch.Tensor], device_mesh=None, **kwargs) -> dict[str, torch.Tensor]:
        for key in hf_state_dict:
            if ".mlp.experts." in key and key.endswith(".weight"):
                self._uses_model_prefix = key.startswith("model.")
                break
        return self._from_hf_w_merged_experts(hf_state_dict, device_mesh)

    def to_hf(
        self, state_dict: dict[str, torch.Tensor], exclude_key_regex: str | None = None, **kwargs
    ) -> dict[str, torch.Tensor]:
        hf_state_dict: dict[str, torch.Tensor] = {}
        for fqn, tensor in state_dict.items():
            for key, value in self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, **kwargs
            ):
                hf_state_dict[key] = value
        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: torch.Tensor, **kwargs) -> list[tuple[str, torch.Tensor]]:
        exclude_key_regex = kwargs.get("exclude_key_regex", None)
        pairs = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if pairs is None:
            pairs = [(fqn, tensor)]
        if exclude_key_regex:
            pairs = [(key, value) for key, value in pairs if not re.match(exclude_key_regex, key)]
        return pairs


class ToyMoEDiTForDiffusion(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Custom-model MoE diffusion transformer with the ``SimpleAdapter`` forward signature."""

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = True

    @classmethod
    def from_config(cls, config: ToyMoEDiTConfig, moe_config: MoEConfig | None = None, backend=None, **kwargs):
        return cls(config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        config = ToyMoEDiTConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: ToyMoEDiTConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.config = config
        self.backend = backend or toy_backend()
        self.model = ToyMoEDiTModel(config, self.backend, moe_config=moe_config)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = ToyMoEDiTStateDictAdapter(
                config, self.model.moe_config, self.backend, dtype=_model_dtype(config)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_kwargs: dict | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ):
        """Diffusers-style forward used by ``SimpleAdapter``.

        Args:
            hidden_states: ``[batch, channels, *spatial]`` noisy latents.
            timestep: ``[batch]`` diffusion timesteps.
            encoder_hidden_states: ``[batch, text_sequence, text_embed_dim]`` text conditioning.
            attention_kwargs: Ignored; accepted for interface compatibility.
            return_dict: Return ``{"sample": prediction}`` instead of ``(prediction,)``.

        Returns:
            A one-element tuple or a dict with ``sample``: ``[batch, channels, *spatial]`` prediction with the layout
            of ``hidden_states``.
        """
        sample = self.model(hidden_states, timestep, encoder_hidden_states)
        return {"sample": sample} if return_dict else (sample,)

    def update_moe_gate_bias(self) -> None:
        with torch.no_grad():
            for block in self.model.layers.values():
                if isinstance(block.mlp, MoE) and block.mlp.gate.bias_update_factor > 0:
                    block.mlp.gate.update_bias()

    @torch.no_grad()
    def initialize_weights(self, buffer_device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        if buffer_device is None:
            buffer_device = (
                torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            )
        with buffer_device:
            self.model.init_weights(buffer_device)
        if dtype is not None:
            for param in self.parameters():
                if param.is_floating_point() and param.dtype != dtype:
                    param.data = param.data.to(dtype)


ModelClass = ToyMoEDiTForDiffusion


def register_toy_moe_dit() -> None:
    """Register the toy config with ``AutoConfig`` and the architecture with Automodel (idempotent)."""
    from transformers import AutoConfig

    from nemo_automodel._transformers.registry import register_architecture

    try:
        AutoConfig.register(TOY_MOE_DIT_MODEL_TYPE, ToyMoEDiTConfig)
    except ValueError:
        pass  # already registered in this process
    register_architecture(TOY_MOE_DIT_ARCHITECTURE, ToyMoEDiTForDiffusion, exist_ok=True)


def write_toy_moe_dit_checkpoint(
    path: str,
    *,
    seed: int = 1234,
    diffusers_layout: bool = True,
    dtype: torch.dtype = torch.float32,
    **config_kwargs: bool | int | float | str,
) -> str:
    """Write an HF-format toy checkpoint (config.json + model.safetensors) from a seeded random init.

    Args:
        path: Output root directory.
        seed: Seed for the random initialization.
        diffusers_layout: Write ``model_index.json`` plus a ``transformer/`` subfolder (diffusers repo
            layout) instead of a single-model checkpoint at ``path``.
        dtype: Stored tensor dtype.
        **config_kwargs: Overrides for :class:`ToyMoEDiTConfig`.

    Returns:
        The directory to pass as ``model.pretrained_model_name_or_path``.
    """
    from safetensors.torch import save_file

    config = ToyMoEDiTConfig(**config_kwargs)
    config.architectures = [TOY_MOE_DIT_ARCHITECTURE]
    config.torch_dtype = dtype
    model_dir = os.path.join(path, "transformer") if diffusers_layout else path
    os.makedirs(model_dir, exist_ok=True)

    generator_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        model = ToyMoEDiTForDiffusion(config, backend=toy_backend())
        model.initialize_weights(buffer_device=torch.device("cpu"), dtype=dtype)
    finally:
        torch.random.set_rng_state(generator_state)

    hf_state_dict = model.state_dict_adapter.to_hf(model.state_dict())
    save_file(
        {key: value.detach().contiguous() for key, value in hf_state_dict.items()},
        os.path.join(model_dir, "model.safetensors"),
    )
    config.save_pretrained(model_dir)
    if diffusers_layout:
        with open(os.path.join(path, "model_index.json"), "w") as f:
            json.dump(
                {
                    "_class_name": "ToyMoEDiTPipeline",
                    "_diffusers_version": "0.0.0",
                    "transformer": ["toy_moe_dit", TOY_MOE_DIT_ARCHITECTURE],
                },
                f,
            )
    return path
