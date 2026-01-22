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

"""KimiK25VL model with backend-aware DeepseekV3 language model.

This is a self-contained implementation that includes all necessary components:
- Configuration classes
- Vision tower (MoonViT3d with temporal dimension)
- Multi-modal projector (PatchMergerMLP)
- Language model backend (DeepseekV3)
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
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


class MoonViT3dConfig(PretrainedConfig):
    """Configuration for MoonViT3d vision encoder with temporal support."""

    model_type = "moonvit3d"

    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        num_attention_heads: int = 16,
        num_hidden_layers: int = 27,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        merge_kernel_size: Tuple[int, int] = (2, 2),
        video_attn_type: str = "spatial_temporal",
        merge_type: str = "sd2_tpool",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.merge_kernel_size = list(merge_kernel_size) if isinstance(merge_kernel_size, tuple) else merge_kernel_size
        self.video_attn_type = video_attn_type
        self.merge_type = merge_type


class KimiK25VLConfig(PretrainedConfig):
    """Configuration for KimiK25VL model.

    Supports both 'kimi_k25_vl' and 'kimi_k25' model types for compatibility
    with original checkpoints.
    """

    model_type = "kimi_k25"  # Use kimi_k25 to match original checkpoint

    def __init__(
        self,
        vision_config: Optional[Union[Dict, MoonViT3dConfig]] = None,
        text_config: Optional[Union[Dict, DeepseekV3Config]] = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        tie_word_embeddings: bool = False,  # Must be False for pipeline parallelism
        # MM Projector parameters
        mm_projector_type: str = "patchmerger",
        mm_hidden_size: Optional[int] = None,
        projector_hidden_act: str = "gelu",
        projector_ln_eps: float = 1e-5,
        architectures: Optional[List[str]] = None,
        **kwargs,
    ):
        if vision_config is None:
            vision_config = MoonViT3dConfig()
        elif isinstance(vision_config, dict):
            vision_config = MoonViT3dConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = DeepseekV3Config()
        elif isinstance(text_config, dict):
            text_config = DeepseekV3Config(**text_config)
        self.text_config = text_config

        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id

        # MM Projector config
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size if mm_hidden_size is not None else vision_config.hidden_size
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps

        # Ensure architectures is set for ModelRegistry matching
        # Include both original and our architecture names
        if architectures is None:
            architectures = ["KimiK25ForConditionalGeneration", "KimiK25VLForConditionalGeneration"]

        super().__init__(
            pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, architectures=architectures, **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        return output


from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.deepseek_v3.model import DeepseekV3Model
from nemo_automodel.components.models.deepseek_v3.rope_utils import freqs_cis_from_position_ids
from nemo_automodel.components.models.kimi_k25_vl.state_dict_adapter import KimiK25VLStateDictAdapter
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

# Check for flash attention
try:
    from flash_attn import flash_attn_varlen_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


# =============================================================================
# Vision Tower Components (MoonViT3d with temporal support)
# =============================================================================


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate 1D sinusoidal positional embedding from grid positions."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_1d_sincos_pos_embed(embed_dim: int, t_size: int, cls_token: bool = False) -> np.ndarray:
    """Generate 1D sinusoidal positional embedding for temporal dimension."""
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


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


def vision_attention_flash(q, k, v, q_cu_seqlens, k_cu_seqlens, max_seqlen_q=None, max_seqlen_k=None):
    """Flash attention for vision."""
    if max_seqlen_q is None:
        max_seqlen_q = (q_cu_seqlens[1:] - q_cu_seqlens[:-1]).max().item()
    if max_seqlen_k is None:
        max_seqlen_k = (k_cu_seqlens[1:] - k_cu_seqlens[:-1]).max().item()
    attn_out = flash_attn_varlen_func(q, k, v, q_cu_seqlens, k_cu_seqlens, max_seqlen_q, max_seqlen_k, causal=False)
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    return attn_out.flatten(start_dim=-2)


def vision_attention_sdpa(q, k, v, q_cu_seqlens, k_cu_seqlens, **kwargs):
    """SDPA attention for vision."""
    seq_length = q.shape[0]
    attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[..., q_cu_seqlens[i - 1] : q_cu_seqlens[i], q_cu_seqlens[i - 1] : q_cu_seqlens[i]] = True
    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
    return attn_output.transpose(0, 1).reshape(seq_length, -1)


class Learnable2DInterpPosEmbDividedFixed(nn.Module):
    """Learnable 2D interpolatable position embedding with fixed temporal sincos embedding."""

    def __init__(self, height: int, width: int, num_frames: int, dim: int, interpolation_mode: str = "bicubic"):
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.register_buffer(
            "time_weight",
            torch.from_numpy(get_1d_sincos_pos_embed(dim, num_frames)).float().unsqueeze(1),
            persistent=False,
        )
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            assert t <= self.num_frames, f"t:{t} > self.num_frames:{self.num_frames}"

            if (h, w) == (self.height, self.width):
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = (
                    F.interpolate(self.weight.permute(2, 0, 1).unsqueeze(0), size=(h, w), mode=self.interpolation_mode)
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .flatten(end_dim=1)
                )

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        return x + torch.cat(pos_embs)


class Rope2DPosEmbRepeated(nn.Module):
    """2D rotary position embedding repeated for temporal dimension."""

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

    def get_freqs_cis(self, grid_thws: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self.freqs_cis is None:
            self.freqs_cis = self._precompute_freqs_cis(device)
        shapes = grid_thws.tolist()
        # Repeat spatial RoPE for each temporal frame
        freqs_cis = torch.cat(
            [self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1) for t, h, w in shapes], dim=0
        )
        return freqs_cis


class MoonViT3dMLP(nn.Module):
    """MLP for MoonViT3d."""

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


class MoonViT3dEncoderLayer(nn.Module):
    """Single encoder layer for MoonViT3d."""

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
        self.mlp = MoonViT3dMLP([hidden_dim, mlp_dim, hidden_dim], activation)
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=attn_bias)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int, rope_freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)

        xqkv = self.wqkv(hidden_states)
        qkv_shape = xqkv.size()[:-1] + (3, self.num_heads, self.head_dim)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)
        xq, xk = _apply_rope_vision(xq, xk, rope_freqs_cis)

        if self.attn_implementation == "flash_attention_2" and FLASH_ATTN_AVAILABLE:
            attn_out = vision_attention_flash(xq, xk, xv, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
        else:
            attn_out = vision_attention_sdpa(xq, xk, xv, cu_seqlens, cu_seqlens)

        hidden_states = residual + self.wo(attn_out)
        hidden_states = hidden_states + self.mlp(self.norm1(hidden_states))
        return hidden_states


class MoonViT3dEncoder(nn.Module):
    """MoonViT3d encoder with temporal support."""

    def __init__(self, hidden_dim: int, num_layers: int, block_cfg: dict):
        super().__init__()
        self.rope_2d = Rope2DPosEmbRepeated(block_cfg["hidden_dim"] // block_cfg["num_heads"], 512, 512)
        self.blocks = nn.ModuleList([MoonViT3dEncoderLayer(**block_cfg) for _ in range(num_layers)])
        self.final_layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_thws, device=hidden_states.device)

        # Compute cumulative sequence lengths: t * h * w for each sample
        lengths = torch.cat(
            (
                torch.zeros(1, device=hidden_states.device, dtype=grid_thws.dtype),
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
            )
        )
        max_seqlen = lengths.max().item()
        cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis)
        return self.final_layernorm(hidden_states)


class MoonVision3dPatchEmbed(nn.Module):
    """Patch embedding for MoonViT3d."""

    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int = 14,
        pos_emb_height: int = 64,
        pos_emb_width: int = 64,
        pos_emb_time: int = 4,
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_emb = Learnable2DInterpPosEmbDividedFixed(pos_emb_height, pos_emb_width, pos_emb_time, out_dim)

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).view(x.size(0), -1)
        return self.pos_emb(x, grid_thws)


def tpool_patch_merger(
    x: torch.Tensor, grid_thws: torch.Tensor, merge_kernel_size: List[int] = [2, 2]
) -> List[torch.Tensor]:
    """Merge patches with temporal pooling."""
    d_model = x.size(-1)
    outputs = []
    pre_sum = 0

    for t, h, w in grid_thws.tolist():
        seq = x[pre_sum : pre_sum + t * h * w]
        kh, kw = merge_kernel_size
        new_h, new_w = h // kh, w // kw

        # Reshape: (t, h, w, d) -> (t, new_h, kh, new_w, kw, d)
        reshaped = seq.view(t, new_h, kh, new_w, kw, d_model)
        # Permute and temporal pooling (mean over temporal dimension)
        reshaped = reshaped.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        # Output: (new_h * new_w, kh * kw, d)
        padded_seq = reshaped.view(new_h * new_w, kh * kw, -1)
        outputs.append(padded_seq)
        pre_sum += t * h * w

    return outputs


class MoonViT3dPretrainedModel(nn.Module):
    """MoonViT3d vision encoder with temporal support."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.merge_kernel_size = config.merge_kernel_size
        self.merge_type = config.merge_type

        self.patch_embed = MoonVision3dPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            pos_emb_time=config.init_pos_emb_time,
        )

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
        self.encoder = MoonViT3dEncoder(config.hidden_size, config.num_hidden_layers, block_cfg)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> List[torch.Tensor]:
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)

        if self.merge_type == "sd2_tpool":
            return tpool_patch_merger(hidden_states, grid_thws, self.merge_kernel_size)
        else:
            raise NotImplementedError(f"Unsupported merge_type: {self.merge_type}")


# =============================================================================
# Multi-Modal Projector (PatchMergerMLP style)
# =============================================================================


class KimiK25VLMultiModalProjector(nn.Module):
    """Projects vision features to language model dimension using patch merger MLP."""

    def __init__(self, config):
        super().__init__()
        vision_config = config.vision_config
        text_config = config.text_config

        mm_hidden_size = config.mm_hidden_size
        merge_kernel_size = vision_config.merge_kernel_size

        self.hidden_size = mm_hidden_size * merge_kernel_size[0] * merge_kernel_size[1]
        self.pre_norm = nn.LayerNorm(mm_hidden_size, eps=config.projector_ln_eps)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = GELUActivation()
        self.linear_2 = nn.Linear(self.hidden_size, text_config.hidden_size, bias=True)

    def forward(self, image_features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Process each feature tensor separately to maintain list structure
        outputs = []
        for item in image_features:
            # item shape: (num_patches, kernel_size, hidden_size)
            hidden = self.pre_norm(item).view(item.shape[0], -1)
            hidden = self.linear_1(hidden)
            hidden = self.act(hidden)
            hidden = self.linear_2(hidden)
            outputs.append(hidden)
        return outputs


# =============================================================================
# Rotary Embedding Adapter (Non-Module callable for PP/FSDP compatibility)
# =============================================================================


class DeepSeekV3RotaryEmbeddingAdapter:
    """Callable adapter that wraps DeepseekV3's freqs_cis-based RoPE."""

    def __init__(self, parent_module: nn.Module, rope_fusion: bool = False):
        self._parent = parent_module
        self.rope_fusion = rope_fusion

    @property
    def freqs_cis(self):
        return self._parent.freqs_cis

    def __call__(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        freqs = self.freqs_cis
        if freqs is None:
            raise RuntimeError("freqs_cis is None on parent module.")
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


class KimiK25VLLanguageModelBackend(nn.Module):
    """Backend-aware language model wrapper using DeepseekV3 architecture."""

    def __init__(self, config, backend: BackendConfig, *, moe_config: MoEConfig | None = None):
        super().__init__()
        self.config = config
        self.backend = backend
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


class KimiK25VLModel(nn.Module):
    """KimiK25VL multimodal backbone with a DeepseekV3 text decoder."""

    def __init__(self, config, moe_config: MoEConfig | None = None, backend: BackendConfig | None = None):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()

        self.vision_tower = MoonViT3dPretrainedModel(config.vision_config)
        self.multi_modal_projector = KimiK25VLMultiModalProjector(config)
        self.language_model = KimiK25VLLanguageModelBackend(
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

    def _compute_num_image_tokens_from_grid(self, grid_thws: torch.Tensor) -> List[int]:
        """Pre-compute number of image tokens from grid_thws without running vision tower.

        For 1 image per sample: num_tokens = (h // merge_h) * (w // merge_w)
        With default merge_kernel_size=(2,2): num_tokens = (h // 2) * (w // 2)

        Args:
            grid_thws: Tensor of shape (batch_size, 3) with [t, h, w] per sample

        Returns:
            List of expected image token counts per sample
        """
        merge_h, merge_w = self.config.vision_config.merge_kernel_size
        token_counts = []
        for t, h, w in grid_thws.tolist():
            # After patch merger: new_h = h // merge_h, new_w = w // merge_w
            num_tokens = (h // merge_h) * (w // merge_w)
            token_counts.append(num_tokens)
        return token_counts

    def _merge_input_ids_with_image_features(
        self,
        image_features: List[torch.Tensor],
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        target_seq_length: Optional[int] = None,
    ):
        """Merge image features into input embeddings.

        Supports two modes:
        1. Pre-expanded (PP mode): input_ids already has N placeholder tokens per image,
           where N = number of image features. Does simple 1:1 replacement.
        2. Dynamic expansion: input_ids has 1 placeholder per image, expands to N tokens.

        Args:
            image_features: List of image feature tensors, one per image
            inputs_embeds: Text embeddings (batch_size, seq_len, embed_dim)
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Optional labels for training
            target_seq_length: Optional fixed output length for pipeline parallelism.
        """
        _, embed_dim = image_features[0].shape
        feature_lengths = [x.shape[0] for x in image_features]
        total_image_features = sum(feature_lengths)
        image_features_cat = torch.cat(image_features, dim=0)

        image_token_index = self.media_placeholder_token_id
        pad_token_id = self.config.pad_token_id
        ignore_index = self.config.ignore_index

        batch_size, sequence_length = input_ids.shape

        # Count placeholder tokens in input_ids
        num_placeholders = (input_ids == image_token_index).sum().item()

        # Check if tokens are pre-expanded (PP mode with collate-time expansion)
        if num_placeholders == total_image_features:
            # Pre-expanded mode: simple 1:1 replacement, no sequence length change
            final_embedding = inputs_embeds.clone()
            image_mask = input_ids == image_token_index

            # Replace placeholder embeddings with image features
            final_embedding[image_mask] = image_features_cat.to(inputs_embeds.dtype)

            # Attention mask and labels stay the same (no expansion)
            final_attention_mask = attention_mask
            position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_((attention_mask == 0), 1)

            if labels is not None:
                # Mask out image positions in labels (don't compute loss on image tokens)
                final_labels = labels.clone()
                final_labels[image_mask] = ignore_index
            else:
                final_labels = None

            return final_embedding, final_attention_mask, final_labels, position_ids

        # Dynamic expansion mode (original behavior)
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(pad_token_id))

        # Create token occupation table
        _token_occupation_table = torch.ones_like(input_ids.flatten())
        _token_occupation_table[input_ids.flatten() == image_token_index] = torch.tensor(
            feature_lengths, dtype=torch.long, device=input_ids.device
        )
        _token_occupation_table = _token_occupation_table.reshape(input_ids.shape)

        # Calculate natural expanded length, but use target if provided (for PP)
        natural_max_embed_dim = _token_occupation_table.sum(-1).max().item()
        max_embed_dim = target_seq_length if target_seq_length is not None else natural_max_embed_dim

        batch_indices, non_image_indices = torch.where(input_ids != image_token_index)

        # Compute new positions for text tokens
        new_token_positions = torch.cumsum(_token_occupation_table, -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # Create final embeddings (with target_seq_length for PP consistency)
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )

        target_device = inputs_embeds.device
        batch_indices = batch_indices.to(target_device)
        non_image_indices = non_image_indices.to(target_device)
        text_to_overwrite = text_to_overwrite.to(target_device)
        attention_mask = attention_mask.to(target_device)

        # Fill text embeddings
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # Fill image embeddings
        image_to_overwrite = torch.full((batch_size, max_embed_dim), True, dtype=torch.bool, device=target_device)
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        final_embedding[image_to_overwrite] = image_features_cat.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # Mask out padding positions
        batch_indices_pad, pad_indices = torch.where(input_ids == pad_token_id)
        indices_to_mask = new_token_positions[batch_indices_pad, pad_indices]
        final_embedding[batch_indices_pad, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def _extract_image_features(self, pixel_values, grid_thws):
        """Extract and project image features."""
        image_features = self.vision_tower(pixel_values, grid_thws)
        return self.multi_modal_projector(image_features)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        pixel_values=None,
        grid_thws=None,
        labels=None,
        padding_mask=None,
        target_seq_length=None,  # For PP: fixed output sequence length after image expansion
        **kwargs,
    ):
        """Forward pass with optional fixed sequence length for pipeline parallelism.

        Args:
            target_seq_length: If provided, the output after image token expansion will be
                              padded to this fixed length. Required for PP with varying image sizes.
                              Can be pre-computed as: max_text_len - 1 + max_image_tokens
                              where max_image_tokens = (h // 2) * (w // 2) for each image.
        """

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
                    inputs_embeds = input_ids
                else:
                    raise ValueError("inputs_embeds must be provided for pipeline stages without embed_tokens")
            else:
                inputs_embeds = embed_tokens(input_ids)

        # Check if we should process vision
        has_vision = self.vision_tower is not None and self.multi_modal_projector is not None
        has_pixels = pixel_values is not None and pixel_values.size(0) > 0
        not_generation = input_ids.shape[1] != 1

        if has_pixels and has_vision and not_generation:
            pixel_values = pixel_values.to(self.vision_tower.dtype)
            image_features = self._extract_image_features(pixel_values, grid_thws)

            inputs_embeds = inputs_embeds.to(image_features[0].dtype)

            inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                image_features,
                inputs_embeds,
                input_ids,
                attention_mask,
                labels,
                target_seq_length=target_seq_length,
            )

        hidden_states = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            padding_mask=padding_mask,
            **kwargs,
        )

        return hidden_states


class KimiK25VLForConditionalGeneration(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """KimiK25VL model with backend-aware DeepseekV3 language model."""

    config_class = KimiK25VLConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    _no_split_modules = ["MoonViT3dEncoderLayer"]
    supports_gradient_checkpointing = True

    @classmethod
    def from_config(cls, config, moe_config: MoEConfig | None = None, backend: BackendConfig | None = None, **kwargs):
        return cls(config, moe_config=moe_config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        """Load model from pretrained path.

        Creates the model structure. Weights are loaded by DCP which calls the
        state_dict_adapter.to_hf() to get checkpoint-format keys (including
        *_packed/*_scale/*_shape for INT4), then from_hf() to dequantize.
        """
        config = kwargs.pop("config", None)
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        num_hidden_layers_override = kwargs.pop("num_hidden_layers", None)

        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" else torch.bfloat16

        if config is None:
            config = KimiK25VLConfig.from_pretrained(pretrained_model_name_or_path)

        # Ensure _name_or_path is set for the adapter to find checkpoint path
        config._name_or_path = pretrained_model_name_or_path

        if num_hidden_layers_override is not None:
            LOGGER.info(
                f"Overriding num_hidden_layers: {config.text_config.num_hidden_layers} -> {num_hidden_layers_override}"
            )
            config.text_config.num_hidden_layers = num_hidden_layers_override

        num_layers = getattr(config.text_config, "num_hidden_layers", 61)
        LOGGER.info(f"Model config has {num_layers} layers")

        config.torch_dtype = torch_dtype
        model = cls.from_config(config, torch_dtype=torch_dtype, *model_args, **kwargs)
        model.name_or_path = pretrained_model_name_or_path
        model = model.to(dtype=torch_dtype)

        LOGGER.info(f"Model created with dtype={torch_dtype}. Weights loaded by DCP via adapter.")
        return model

    def __init__(self, config, moe_config: MoEConfig | None = None, backend: BackendConfig | None = None, **kwargs):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()

        self.model = KimiK25VLModel(config, moe_config=moe_config, backend=self.backend)
        self.moe_config = self.model.moe_config

        self.model.language_model.lm_head = initialize_linear_module(
            self.backend.linear, config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )

        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = getattr(config.text_config, "pad_token_id", -1) or -1
        self.media_placeholder_token_id = config.media_placeholder_token_id

        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = KimiK25VLStateDictAdapter(
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
        grid_thws=None,
        padding_mask=None,
        target_seq_length=None,  # For PP: fixed output length after image expansion
        **kwargs,
    ):
        # Retrieve pre-chunked VLM inputs from model attributes (set by finetune.py for PP)
        if (
            pixel_values is None
            and hasattr(self, "_vlm_pixel_values_chunks")
            and self._vlm_pixel_values_chunks is not None
        ):
            has_media_tokens = (
                input_ids is not None
                and self.media_placeholder_token_id is not None
                and (input_ids == self.media_placeholder_token_id).any()
            )
            if has_media_tokens:
                chunk_idx = getattr(self, "_vlm_chunk_idx", 0)
                if chunk_idx < len(self._vlm_pixel_values_chunks):
                    pixel_values = self._vlm_pixel_values_chunks[chunk_idx]
                    # Recipe stores as image_grid_hws [N, 2], convert to grid_thws [N, 3] (prepend T=1)
                    image_grid_hws = self._vlm_image_grid_hws_chunks[chunk_idx]
                    if image_grid_hws.shape[-1] == 2:
                        # Convert [N, 2] (H, W) -> [N, 3] (T, H, W) with T=1
                        ones = torch.ones(
                            image_grid_hws.shape[0], 1, dtype=image_grid_hws.dtype, device=image_grid_hws.device
                        )
                        grid_thws = torch.cat([ones, image_grid_hws], dim=-1)
                    else:
                        grid_thws = image_grid_hws  # Already [N, 3]
                    self._vlm_chunk_idx = chunk_idx + 1

        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            labels=labels,
            padding_mask=padding_mask,
            target_seq_length=target_seq_length,
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


ModelClass = KimiK25VLForConditionalGeneration


def _register_kimi_k25_vl_with_transformers():
    """Register KimiK25VLConfig and model with transformers Auto classes."""
    import logging

    from transformers import AutoModelForImageTextToText
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    _logger = logging.getLogger(__name__)

    # Register for kimi_k25_vl model type
    if "kimi_k25_vl" not in CONFIG_MAPPING:
        try:
            AutoConfig.register("kimi_k25_vl", KimiK25VLConfig)
        except ValueError as e:
            _logger.debug(f"KimiK25VLConfig registration skipped (kimi_k25_vl): {e}")

    # Also register for kimi_k25 model type (used by the original checkpoint)
    if "kimi_k25" not in CONFIG_MAPPING:
        try:
            AutoConfig.register("kimi_k25", KimiK25VLConfig)
        except ValueError as e:
            _logger.debug(f"KimiK25VLConfig registration skipped (kimi_k25): {e}")

    try:
        AutoModelForImageTextToText.register(KimiK25VLConfig, KimiK25VLForConditionalGeneration)
    except ValueError as e:
        _logger.debug(f"KimiK25VLForConditionalGeneration registration skipped: {e}")


def compute_expanded_seq_length(
    text_seq_length: int,
    grid_thws: torch.Tensor,
    merge_kernel_size: Tuple[int, int] = (2, 2),
    num_images: int = 1,
) -> int:
    """Compute the expanded sequence length after image token insertion.

    For pipeline parallelism, this can be used to pre-compute the target_seq_length
    parameter needed for fixed-shape outputs.

    Args:
        text_seq_length: Original text sequence length (including 1 placeholder per image)
        grid_thws: Tensor of shape (num_images, 3) with [t, h, w] per image
        merge_kernel_size: Vision tower's patch merge kernel size, default (2, 2)
        num_images: Number of images (placeholders) in the sequence

    Returns:
        Expected sequence length after image features are inserted

    Example:
        # For 1 image per sample with grid_thws = [[1, 28, 28]]:
        # num_image_tokens = (28 // 2) * (28 // 2) = 196
        # expanded_length = text_seq_length - 1 + 196
        >>> grid_thws = torch.tensor([[1, 28, 28]])
        >>> compute_expanded_seq_length(82, grid_thws)
        277  # 82 - 1 + 196
    """
    merge_h, merge_w = merge_kernel_size
    total_image_tokens = 0

    for t, h, w in grid_thws.tolist():
        num_tokens = (h // merge_h) * (w // merge_w)
        total_image_tokens += num_tokens

    # Each placeholder (1 token) is replaced by num_image_tokens
    # So: expanded = original - num_placeholders + total_image_tokens
    expanded_length = text_seq_length - num_images + total_image_tokens
    return expanded_length


_register_kimi_k25_vl_with_transformers()
