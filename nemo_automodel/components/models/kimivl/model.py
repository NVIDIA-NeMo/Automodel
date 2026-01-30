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

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.activations import GELUActivation
from transformers.configuration_utils import PretrainedConfig
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast

from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Configuration Classes
# =============================================================================


class MoonViTConfig(PretrainedConfig):
    """Configuration for MoonVit vision encoder."""

    model_type = "moonvit"

    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 27,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        merge_kernel_size: Tuple[int, int] = (2, 2),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.merge_kernel_size = list(merge_kernel_size) if isinstance(merge_kernel_size, tuple) else merge_kernel_size


class KimiVLConfig(PretrainedConfig):
    """Configuration for KimiVL model."""

    model_type = "kimi_vl"

    def __init__(
        self,
        vision_config: Optional[Union[Dict, MoonViTConfig]] = None,
        text_config: Optional[Union[Dict, DeepseekV3Config]] = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        architectures: Optional[List[str]] = None,
        **kwargs,
    ):
        if vision_config is None:
            vision_config = MoonViTConfig()
        elif isinstance(vision_config, dict):
            vision_config = MoonViTConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = DeepseekV3Config()
        elif isinstance(text_config, dict):
            text_config = DeepseekV3Config(**text_config)
        self.text_config = text_config

        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id

        # Ensure architectures is set for ModelRegistry matching
        if architectures is None:
            architectures = ["KimiVLForConditionalGeneration"]

        super().__init__(pad_token_id=pad_token_id, architectures=architectures, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        return output


from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.deepseek_v3.model import DeepseekV3Model
from nemo_automodel.components.models.deepseek_v3.rope_utils import freqs_cis_from_position_ids
from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

# Check for flash attention
try:
    from flash_attn import flash_attn_varlen_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


# =============================================================================
# Vision Tower Components (MoonVit)
# =============================================================================


def _apply_rope_vision(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding for vision."""
    freqs_cis = freqs_cis.unsqueeze(-2)
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def vision_attention_flash(q, k, v, q_cu_seqlens, k_cu_seqlens):
    """Flash attention for vision."""
    max_seqlen_q = (q_cu_seqlens[1:] - q_cu_seqlens[:-1]).max().item()
    max_seqlen_k = (k_cu_seqlens[1:] - k_cu_seqlens[:-1]).max().item()
    attn_out = flash_attn_varlen_func(q, k, v, q_cu_seqlens, k_cu_seqlens, max_seqlen_q, max_seqlen_k, causal=False)
    return attn_out.flatten(start_dim=-2)


def vision_attention_sdpa(q, k, v, q_cu_seqlens, k_cu_seqlens):
    """SDPA attention for vision."""
    seq_length = q.shape[0]
    attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[..., q_cu_seqlens[i - 1] : q_cu_seqlens[i], q_cu_seqlens[i - 1] : q_cu_seqlens[i]] = True
    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
    return attn_output.transpose(0, 1).reshape(seq_length, -1)


class Learnable2DInterpPosEmb(nn.Module):
    """Learnable 2D interpolatable position embedding."""

    def __init__(self, height: int, width: int, dim: int, interpolation_mode: str = "bicubic"):
        super().__init__()
        self.height = height
        self.width = width
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for shape in grid_hws.tolist():
            if shape == [self.height, self.width]:
                pos_embs.append(self.weight.flatten(end_dim=1))
            else:
                pos_embs.append(
                    F.interpolate(self.weight.permute(2, 0, 1).unsqueeze(0), size=shape, mode=self.interpolation_mode)
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .flatten(end_dim=1)
                )
        return x + torch.cat(pos_embs)


class Rope2DPosEmb(nn.Module):
    def __init__(self, dim: int, max_height: int, max_width: int, theta_base: float = 10000):
        super().__init__()
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base
        self.freqs_cis = None

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()
        y_freqs = torch.outer(y_pos, freqs).float()
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
        freqs_cis = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)
        return freqs_cis.reshape(self.max_height, self.max_width, -1)

    def get_freqs_cis(self, grid_hws: torch.Tensor) -> torch.Tensor:
        if self.freqs_cis is None:
            self.freqs_cis = self._precompute_freqs_cis(grid_hws.device)
        shapes = grid_hws.tolist()
        return torch.cat([self.freqs_cis[:h, :w].reshape(-1, self.dim // 2) for h, w in shapes], dim=0)


class MoonVitMLP(nn.Module):
    """MLP for MoonVit."""

    def __init__(self, dims: List[int], activation, bias: bool = True):
        super().__init__()
        self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)
        self.activation = activation
        for m in [self.fc0, self.fc1]:
            nn.init.trunc_normal_(m.weight, std=math.sqrt(2 / m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(self.activation(self.fc0(x)))


class MoonVitEncoderLayer(nn.Module):
    """Single encoder layer for MoonVit."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        *,
        activation=F.gelu,
        attn_bias: bool = False,
        attn_implementation: str = "flash_attention_2",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.attn_implementation = attn_implementation

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = MoonVitMLP([hidden_dim, mlp_dim, hidden_dim], activation)
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=attn_bias)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rope_freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)

        xqkv = self.wqkv(hidden_states)
        qkv_shape = xqkv.size()[:-1] + (3, self.num_heads, self.head_dim)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)
        xq, xk = _apply_rope_vision(xq, xk, rope_freqs_cis)

        if self.attn_implementation == "flash_attention_2" and FLASH_ATTN_AVAILABLE:
            attn_out = vision_attention_flash(xq, xk, xv, cu_seqlens, cu_seqlens)
        else:
            attn_out = vision_attention_sdpa(xq, xk, xv, cu_seqlens, cu_seqlens)

        hidden_states = residual + self.wo(attn_out)
        hidden_states = hidden_states + self.mlp(self.norm1(hidden_states))
        return hidden_states


class MoonVitEncoder(nn.Module):
    """MoonVit encoder."""

    def __init__(self, hidden_dim: int, num_layers: int, block_cfg: dict):
        super().__init__()
        self.rope_2d = Rope2DPosEmb(block_cfg["hidden_dim"] // block_cfg["num_heads"], 512, 512)
        self.blocks = nn.ModuleList([MoonVitEncoderLayer(**block_cfg) for _ in range(num_layers)])
        self.final_layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_hws)
        lengths = torch.cat(
            (torch.zeros(1, device=hidden_states.device, dtype=grid_hws.dtype), grid_hws[:, 0] * grid_hws[:, 1])
        )
        cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, rope_freqs_cis)
        return self.final_layernorm(hidden_states)


class MoonVisionPatchEmbed(nn.Module):
    """Patch embedding for MoonVit."""

    def __init__(
        self, out_dim: int, in_dim: int = 3, patch_size: int = 14, pos_emb_height: int = 64, pos_emb_width: int = 64
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_emb = Learnable2DInterpPosEmb(pos_emb_height, pos_emb_width, out_dim)

    def forward(self, x: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).view(x.size(0), -1)
        return self.pos_emb(x, grid_hws)


def patch_merger(x: torch.Tensor, grid_hws: torch.Tensor, merge_kernel_size: List[int] = [2, 2]) -> List[torch.Tensor]:
    """Merge patches."""
    d_model = x.size(-1)
    outputs = []
    pre_sum = 0
    for h, w in grid_hws.tolist():
        seq = x[pre_sum : pre_sum + h * w]
        kh, kw = merge_kernel_size
        new_h, new_w = h // kh, w // kw
        reshaped = seq.view(new_h, kh, new_w, kw, d_model).permute(0, 2, 1, 3, 4).contiguous()
        outputs.append(reshaped.view(new_h * new_w, kh * kw, -1))
        pre_sum += h * w
    return outputs


class MoonVitPretrainedModel(nn.Module):
    """MoonVit vision encoder."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed = MoonVisionPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
        )

        # GELU with tanh approximation
        activation = lambda x: F.gelu(x, approximate="tanh")

        attn_impl = getattr(config, "_attn_implementation", "flash_attention_2")
        block_cfg = {
            "num_heads": config.num_attention_heads,
            "hidden_dim": config.hidden_size,
            "mlp_dim": config.intermediate_size,
            "activation": activation,
            "attn_bias": True,
            "attn_implementation": attn_impl,
        }
        self.encoder = MoonVitEncoder(config.hidden_size, config.num_hidden_layers, block_cfg)
        self.merge_kernel_size = config.merge_kernel_size

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, pixel_values: torch.Tensor, grid_hws: torch.Tensor) -> List[torch.Tensor]:
        hidden_states = self.patch_embed(pixel_values, grid_hws)
        hidden_states = self.encoder(hidden_states, grid_hws)
        return patch_merger(hidden_states, grid_hws, self.merge_kernel_size)


# =============================================================================
# Multi-Modal Projector
# =============================================================================


class KimiVLMultiModalProjector(nn.Module):
    """Projects vision features to language model dimension."""

    def __init__(self, config):
        super().__init__()
        vision_config = config.vision_config
        text_config = config.text_config

        self.hidden_size = (
            vision_config.hidden_size * vision_config.merge_kernel_size[0] * vision_config.merge_kernel_size[1]
        )
        self.pre_norm = nn.LayerNorm(vision_config.hidden_size, eps=1e-05)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = GELUActivation()
        self.linear_2 = nn.Linear(self.hidden_size, text_config.hidden_size, bias=True)

    def forward(self, image_features: List[torch.Tensor]) -> torch.Tensor:
        image_features = torch.cat(image_features, dim=0)
        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.linear_2(hidden_states)


# =============================================================================
# Rotary Embedding Adapter (Non-Module callable for PP/FSDP compatibility)
# =============================================================================


class DeepSeekV3RotaryEmbeddingAdapter:
    """Callable adapter that wraps DeepseekV3's freqs_cis-based RoPE.

    This is NOT an nn.Module to avoid being pruned during PP split.
    It holds a reference to the parent module's freqs_cis buffer and computes
    position embeddings on demand.

    The parent module (KimiVLLanguageModelBackend) owns the freqs_cis buffer,
    and this adapter accesses it via the reference.
    """

    def __init__(self, parent_module: nn.Module, rope_fusion: bool = False):
        # Store weak reference to avoid circular dependencies
        self._parent = parent_module
        self.rope_fusion = rope_fusion

    @property
    def freqs_cis(self):
        """Access freqs_cis from the parent module."""
        return self._parent.freqs_cis

    def __call__(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Compute position embeddings from pre-computed freqs_cis.

        Args:
            hidden_states: Input tensor (used only for device/dtype inference)
            position_ids: Position indices tensor

        Returns:
            Position embeddings tensor compatible with DeepseekV3 Block layers
        """
        freqs = self.freqs_cis
        if freqs is None:
            raise RuntimeError(
                "freqs_cis is None on parent module. This can happen if the model "
                "was pruned during PP split and freqs_cis buffer was not preserved."
            )
        return freqs_cis_from_position_ids(
            position_ids,
            freqs,
            qkv_format="bshd",
            for_fused_rope=self.rope_fusion,
            cp_size=1,
        )


# =============================================================================
# Language Model Backend
# =============================================================================


class KimiVLLanguageModelBackend(nn.Module):
    """Backend-aware language model wrapper using DeepseekV3 architecture.

    Note: lm_head is NOT included here - it's at the top level of
    KimiVLForConditionalGeneration to match HF checkpoint structure.
    """

    def __init__(self, config, backend: BackendConfig, *, moe_config: MoEConfig | None = None):
        super().__init__()
        self.config = config
        self.backend = backend
        # Use DeepseekV3Model (backbone only, no lm_head). lm_head is created at KimiVLForConditionalGeneration level.
        self.model = DeepseekV3Model(config, backend, moe_config=moe_config)
        self.moe_config = self.model.moe_config
        self.register_buffer("freqs_cis", self.model.freqs_cis, persistent=False)

        self.rotary_emb = DeepSeekV3RotaryEmbeddingAdapter(parent_module=self, rope_fusion=backend.rope_fusion)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        text_model = self._text_model()
        if text_model is not None:
            text_model.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        padding_mask=None,
        **kwargs,
    ):
        return self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            padding_mask=padding_mask,
            **kwargs,
        )

    @torch.no_grad()
    def init_weights(self, buffer_device=None):
        if self.model is not None:
            self.model.init_weights(buffer_device=buffer_device)
            self.freqs_cis = self.model.freqs_cis

    @property
    def embed_tokens(self):
        return self.model.embed_tokens if self.model is not None else None

    @property
    def layers(self):
        return self.model.layers if self.model is not None else None

    @property
    def norm(self):
        return self.model.norm if self.model is not None else None


# =============================================================================
# Main Model
# =============================================================================


class KimiVLModel(nn.Module):
    """KimiVL multimodal backbone with a DeepseekV3 text decoder."""

    def __init__(self, config, moe_config: MoEConfig | None = None, backend: BackendConfig | None = None):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()

        self.vision_tower = MoonVitPretrainedModel(config.vision_config)
        self.multi_modal_projector = KimiVLMultiModalProjector(config)
        self.language_model = KimiVLLanguageModelBackend(
            config.text_config, backend=self.backend, moe_config=moe_config
        )

        self.moe_config = self.language_model.moe_config

        self.media_placeholder_token_id = config.media_placeholder_token_id

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def embed_tokens(self):
        return self.language_model.embed_tokens

    @property
    def norm(self):
        return self.language_model.norm

    def _merge_with_image_features(self, inputs_embeds, input_ids, image_features):
        """Merge image features into input embeddings."""
        batch_size, seq_len, embed_dim = inputs_embeds.shape
        inputs_embeds = inputs_embeds.reshape(-1, embed_dim)
        input_ids_flat = input_ids.flatten()
        inputs_embeds[input_ids_flat == self.media_placeholder_token_id] = image_features
        return inputs_embeds.reshape(batch_size, seq_len, embed_dim)

    def _extract_image_features(self, pixel_values, image_grid_hws):
        """Extract and project image features."""
        image_features = self.vision_tower(pixel_values, image_grid_hws)
        return self.multi_modal_projector(image_features)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        pixel_values=None,
        image_grid_hws=None,
        padding_mask=None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if "qkv_format" in kwargs and kwargs["qkv_format"] == "thd":
            input_ids, position_ids, padding_mask, kwargs = squeeze_input_for_thd(
                input_ids, position_ids, padding_mask, kwargs
            )
            attention_mask = None
            if padding_mask is not None:
                kwargs["padding_mask"] = padding_mask

        if inputs_embeds is None:
            embed_tokens = self.language_model.get_input_embeddings()
            if embed_tokens is None:
                if (
                    input_ids is not None
                    and isinstance(input_ids, torch.Tensor)
                    and input_ids.dtype in (torch.float16, torch.bfloat16, torch.float32)
                ):
                    # PP middle stages may receive hidden states as input_ids
                    inputs_embeds = input_ids
                else:
                    raise ValueError("inputs_embeds must be provided for pipeline stages without embed_tokens")
            else:
                inputs_embeds = embed_tokens(input_ids)

        if (
            pixel_values is not None
            and pixel_values.size(0) > 0
            and self.vision_tower is not None
            and self.multi_modal_projector is not None
        ):
            pixel_values = pixel_values.to(self.vision_tower.dtype)
            image_features = self._extract_image_features(pixel_values, image_grid_hws)
            inputs_embeds = inputs_embeds.to(image_features.dtype)
            inputs_embeds = self._merge_with_image_features(inputs_embeds, input_ids, image_features)

        hidden_states = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            padding_mask=padding_mask,
            **kwargs,
        )
        return hidden_states


class KimiVLForConditionalGeneration(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """KimiVL model with backend-aware DeepseekV3 language model."""

    @classmethod
    def from_config(cls, config, moe_config: MoEConfig | None = None, backend: BackendConfig | None = None, **kwargs):
        return cls(config, moe_config=moe_config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        config = KimiVLConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(self, config, moe_config: MoEConfig | None = None, backend: BackendConfig | None = None, **kwargs):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()

        self.model = KimiVLModel(config, moe_config=moe_config, backend=self.backend)
        self.moe_config = self.model.moe_config

        self.model.language_model.lm_head = initialize_linear_module(
            self.backend.linear, config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )

        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = getattr(config.text_config, "pad_token_id", -1) or -1
        self.media_placeholder_token_id = config.media_placeholder_token_id

        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = KimiVLStateDictAdapter(
                config,
                self.moe_config,
                self.backend,
                dtype=get_dtype(getattr(config.text_config, "torch_dtype", None), torch.bfloat16),
            )

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.language_model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.language_model.lm_head = new_embeddings

    @property
    def lm_head(self):
        """Convenience property to access lm_head from top level."""
        return self.model.language_model.lm_head

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        image_grid_hws=None,
        padding_mask=None,
        **kwargs,
    ):
        # Retrieve pre-chunked VLM inputs from model attributes
        # This allows native forward to work with pipeline parallelism
        # finetune.py stores chunks on model, we retrieve them here per microbatch
        if (
            pixel_values is None
            and hasattr(self, "_vlm_pixel_values_chunks")
            and self._vlm_pixel_values_chunks is not None
        ):
            # Check if we have media tokens to process
            has_media_tokens = (
                input_ids is not None
                and self.media_placeholder_token_id is not None
                and (input_ids == self.media_placeholder_token_id).any()
            )
            if has_media_tokens:
                chunk_idx = getattr(self, "_vlm_chunk_idx", 0)
                if chunk_idx < len(self._vlm_pixel_values_chunks):
                    pixel_values = self._vlm_pixel_values_chunks[chunk_idx]
                    image_grid_hws = self._vlm_image_grid_hws_chunks[chunk_idx]
                    self._vlm_chunk_idx = chunk_idx + 1

        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_hws=image_grid_hws,
            padding_mask=padding_mask,
            **kwargs,
        )

        logits = self.lm_head(hidden_states) if self.lm_head is not None else hidden_states

        loss = None
        if labels is not None and self.lm_head is not None:
            if attention_mask is not None:
                shift_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if return_dict is None:
            return_dict = False
        if not return_dict:
            return logits

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=None,
        )

    @torch.no_grad()
    def initialize_weights(self, buffer_device=None, dtype=torch.bfloat16):
        self.model.language_model.init_weights(buffer_device=buffer_device)


class KimiVLStateDictAdapter:
    """State dict adapter for KimiVL checkpoints."""

    def __init__(self, config, moe_config: MoEConfig, backend: BackendConfig, dtype: torch.dtype = torch.float32):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self.llm_adapter = DeepSeekV3StateDictAdapter(config.text_config, moe_config, backend, dtype)
        self._last_expected_hf_keys: set[str] | None = None

    def to_hf(self, state_dict: dict, **kwargs) -> dict:
        import re

        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        hf_state_dict = {}
        llm_keys = {}
        for key, value in state_dict.items():
            if exclude_key_regex and re.match(exclude_key_regex, key):
                continue

            if key.startswith("model.model."):
                continue
            if key.startswith("model.language_model."):
                key = key.replace("model.", "", 1)
            if key.startswith("model.") and not key.startswith("model.language_model."):
                key = key.replace("model.", "", 1)

            if key.startswith("language_model.model."):
                llm_key = key.replace("language_model.model.", "model.")
                llm_keys[llm_key] = value
            elif (
                key.startswith("language_model.embed_tokens.")
                or key.startswith("language_model.layers.")
                or key.startswith("language_model.norm.")
            ):
                llm_key = "model." + key.replace("language_model.", "", 1)
                llm_keys[llm_key] = value
            elif key.startswith("lm_head."):
                hf_state_dict["language_model." + key] = value
            else:
                hf_state_dict[key] = value

        if llm_keys:
            for k, v in self.llm_adapter.to_hf(llm_keys, **kwargs).items():
                hf_state_dict[k.replace("model.", "language_model.model.")] = v
        return hf_state_dict

    def from_hf(self, state_dict: dict, **kwargs) -> dict:
        native_state_dict = {}

        llm_keys = {}
        for key, value in state_dict.items():
            if key.startswith("language_model.model."):
                llm_key = key.replace("language_model.model.", "model.")
                llm_keys[llm_key] = value
            elif key.startswith("language_model.lm_head."):
                # Map HF format language_model.lm_head to top-level lm_head
                native_key = key.replace("language_model.lm_head.", "lm_head.")
                native_state_dict[native_key] = value
            elif key.startswith(("vision_tower.", "multi_modal_projector.", "language_model.")):
                native_state_dict["model." + key] = value
            else:
                native_state_dict[key] = value

        if llm_keys:
            converted_llm = self.llm_adapter.from_hf(llm_keys, **kwargs)
            for k, v in converted_llm.items():
                native_state_dict[k.replace("model.", "model.language_model.model.")] = v

        return native_state_dict


ModelClass = KimiVLForConditionalGeneration


def _register_kimi_vl_with_transformers():
    """
    Register KimiVLConfig and model with transformers Auto classes.

    This uses the official transformers registration API. When registered,
    AutoModelForImageTextToText.from_pretrained will use our local implementation
    directly, bypassing the trust_remote_code mechanism entirely.
    """
    import logging

    from transformers import AutoModelForImageTextToText
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    _logger = logging.getLogger(__name__)

    # Register config with AutoConfig
    if "kimi_vl" not in CONFIG_MAPPING:
        try:
            AutoConfig.register("kimi_vl", KimiVLConfig)
        except ValueError as e:
            _logger.debug(f"KimiVLConfig registration skipped: {e}")

    try:
        AutoModelForImageTextToText.register(KimiVLConfig, KimiVLForConditionalGeneration)
    except ValueError as e:
        _logger.debug(f"KimiVLForConditionalGeneration registration skipped: {e}")


# Perform registration at module import time
_register_kimi_vl_with_transformers()
