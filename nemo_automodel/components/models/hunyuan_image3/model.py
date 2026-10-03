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

"""HunyuanImage-3.0 (tencent/HunyuanImage-3.0) for flow-matching text-to-image training.

The released model is a single 80B-total / 13B-active MoE decoder that handles text and images in one sequence.
For text-to-image generation the sequence is::

    <bos> prompt <boi> <img_size_*> <img_ratio_*> <timestep> <img> * (h*w) <eoi>

The ``<timestep>`` slot receives a timestep embedding and the ``<img>`` slots receive the noisy VAE latents through
a UNet patch embedding. Text tokens attend causally; the image tokens attend to each other bidirectionally. The
final hidden states at the image slots (before the final norm) go through a UNet final layer that predicts the
flow velocity ``noise - x0``.

The decoder uses Automodel's :class:`~nemo_automodel.components.moe.layers.MoE` under ``model.layers[*].mlp`` so
the MoE parallelizer can apply FSDP2 and expert parallelism. The VAE and the vision encoder of the release are not
part of this module: latents are precomputed during preprocessing, and image conditioning is not supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.hunyuan_image3.config import HunyuanImage3Config, per_layer
from nemo_automodel.components.models.hunyuan_image3.layers import (
    HunyuanImage3Attention,
    HunyuanRMSNorm,
    HunyuanSharedMLP,
    TimestepEmbedder,
    UNetDown,
    UNetUp,
    init_unet_weights,
)
from nemo_automodel.components.models.hunyuan_image3.rope import image_grid_positions, rope_cos_sin, text_positions
from nemo_automodel.components.models.hunyuan_image3.state_dict_adapter import HunyuanImage3StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.shared.utils import dtype_from_str


def _model_dtype(config: HunyuanImage3Config) -> torch.dtype:
    dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    return dtype if isinstance(dtype, torch.dtype) else dtype_from_str(dtype, default=torch.bfloat16)


def build_moe_config(config: HunyuanImage3Config, overrides: dict[str, Any] | None = None) -> MoEConfig:
    """MoE settings of the release: softmax over all experts, top-k, renormalized, one shared expert."""
    if not config.use_mixed_mlp_moe:
        raise NotImplementedError("HunyuanImage-3.0 checkpoints without the shared expert are not supported.")

    def uniform(field: str) -> int:
        values = {per_layer(getattr(config, field), i) for i in range(config.num_hidden_layers)}
        if len(values) != 1:
            raise NotImplementedError(f"{field} differs across layers ({sorted(values)}); this is not supported.")
        return values.pop()

    if config.hidden_act != "silu":
        raise NotImplementedError(f"Only the SwiGLU (silu) MLP is supported, got {config.hidden_act!r}.")
    fields: dict[str, Any] = dict(
        dim=config.hidden_size,
        inter_dim=config.intermediate_size,
        moe_inter_dim=uniform("moe_intermediate_size"),
        n_routed_experts=uniform("num_experts"),
        # The shared expert is HunyuanSharedMLP (fused [up; gate] layout), attached by HunyuanImage3Block.
        n_shared_experts=0,
        n_activated_experts=uniform("moe_topk"),
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        softmax_before_topk=True,
        route_scale=1.0,
        aux_loss_coeff=config.router_aux_loss_coef,
        norm_topk_prob=config.norm_topk_prob,
        expert_bias=config.mlp_bias,
        router_bias=False,
        expert_activation="swiglu",
        # The release keeps the router in fp32.
        gate_dtype=torch.float32,
        dtype=_model_dtype(config),
    )
    fields.update(overrides or {})
    return MoEConfig(**fields)


class HunyuanImage3Block(nn.Module):
    """Pre-norm decoder layer: GQA attention and a shared-expert MoE."""

    def __init__(self, config: HunyuanImage3Config, moe_config: MoEConfig, backend: BackendConfig):
        super().__init__()
        dtype = _model_dtype(config)
        self.self_attn = HunyuanImage3Attention(config, backend, dtype)
        self.mlp = MoE(moe_config, backend)
        shared_inter = moe_config.moe_inter_dim * per_layer(config.num_shared_expert, 0)
        self.mlp.shared_experts = HunyuanSharedMLP(
            config.hidden_size, shared_inter, backend, bias=config.mlp_bias, dtype=dtype
        )
        self.input_layernorm = HunyuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = HunyuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run one decoder layer.

        Args:
            x: Tensor of shape [batch, sequence, hidden].
            cos: fp32 tensor of shape [batch, sequence, head_dim] rotary table.
            sin: fp32 tensor of shape [batch, sequence, head_dim] rotary table.
            attention_mask: Boolean tensor of shape [batch, 1, sequence, sequence], true where attention is
                allowed, or ``None`` for causal attention.
            padding_mask: Boolean tensor of shape [batch, sequence], true at padding, or ``None``.

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attention_mask)
        return x + self.mlp(self.post_attention_layernorm(x), padding_mask)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        self.self_attn.init_weights(init_std)
        self.mlp.init_weights(buffer_device, init_std=init_std)
        self.mlp.shared_experts.init_weights(init_std)


class HunyuanImage3Model(nn.Module):
    """Decoder backbone: token embedding, MoE layers and the final norm (``wte`` / ``ln_f`` in the release)."""

    def __init__(self, config: HunyuanImage3Config, backend: BackendConfig, moe_config: MoEConfig):
        super().__init__()
        dtype = _model_dtype(config)
        self.config = config
        self.moe_config = moe_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.layers = nn.ModuleDict(
            {str(i): HunyuanImage3Block(config, moe_config, backend) for i in range(config.num_hidden_layers)}
        )
        self.norm = HunyuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the decoder layers.

        Args:
            inputs_embeds: Tensor of shape [batch, sequence, hidden].
            cos: fp32 tensor of shape [batch, sequence, head_dim] rotary table.
            sin: fp32 tensor of shape [batch, sequence, head_dim] rotary table.
            attention_mask: Boolean tensor of shape [batch, 1, sequence, sequence], true where attention is
                allowed, or ``None`` for causal attention.
            padding_mask: Boolean tensor of shape [batch, sequence], true at padding, or ``None``.

        Returns:
            Tensor of shape [batch, sequence, hidden]: the last hidden states *before* the final norm.
        """
        h = inputs_embeds
        for layer in self.layers.values():
            h = layer(h, cos, sin, attention_mask, padding_mask)
        return h

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        nn.init.normal_(self.embed_tokens.weight, std=init_std)
        self.norm.reset_parameters()
        for layer in self.layers.values():
            layer.init_weights(buffer_device, init_std=init_std)


def build_joint_attention_mask(
    seq_len: int, image_starts: torch.Tensor, num_image_tokens: int, valid_lengths: torch.Tensor
) -> torch.Tensor:
    """Boolean ``[batch, 1, seq, seq]`` mask: causal text, bidirectional image span, padded keys masked out.

    Args:
        seq_len: Padded sequence length.
        image_starts: Long tensor of shape [batch], index of the first image token of every sample.
        num_image_tokens: Number of image tokens (same for every sample of a batch).
        valid_lengths: Long tensor of shape [batch], number of real (non-padding) tokens of every sample.

    Returns:
        Boolean tensor of shape [batch, 1, sequence, sequence], indexed [batch, 1, query, key], true where
        attention is allowed.
    """
    device = image_starts.device
    index = torch.arange(seq_len, device=device)
    causal = index[None, :] <= index[:, None]  # [q, k]
    in_image = (index[None, :] >= image_starts[:, None]) & (index[None, :] < (image_starts + num_image_tokens)[:, None])
    image_block = in_image[:, :, None] & in_image[:, None, :]  # [b, q, k]
    key_valid = index[None, :] < valid_lengths[:, None]  # [b, k]
    mask = (causal[None] | image_block) & key_valid[:, None, :]
    # Padded queries would otherwise see no key; let them attend to themselves so softmax stays finite.
    mask = mask | torch.eye(seq_len, dtype=torch.bool, device=device)[None]
    return mask[:, None]


class HunyuanImage3ForCausalMM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """HunyuanImage-3.0 transformer with the text-to-image flow-matching forward.

    ``forward`` predicts the flow velocity of noisy latents given the token sequence; ``forward_text`` returns
    next-token logits (the release's ``gen_text`` mode) and is used for parity checks.
    """

    # The released checkpoint has separate wte / lm_head weights.
    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    # The release keeps the router in fp32.
    _keep_in_fp32_modules_strict = ["mlp.gate"]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = True

    @classmethod
    def from_config(
        cls,
        config: HunyuanImage3Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ) -> "HunyuanImage3ForCausalMM":
        return cls(config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *model_args: Any, **kwargs: Any
    ) -> "HunyuanImage3ForCausalMM":
        config = HunyuanImage3Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: HunyuanImage3Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        reject_unsupported_tie_word_embeddings(type(self), config)
        if config.img_proj_type != "unet" or config.patch_size != 1:
            raise NotImplementedError("Only the released UNet image projection with patch_size=1 is supported.")
        self.config = config
        self.backend = backend or BackendConfig()
        if self.backend.dispatcher == "mok":
            raise NotImplementedError("dispatcher='mok' needs a split shared expert; HunyuanImage-3.0 fuses it.")
        dtype = _model_dtype(config)
        moe_overrides = kwargs.pop("moe_overrides", None)
        if moe_config is not None and moe_overrides is not None:
            raise ValueError("Cannot pass both moe_config and moe_overrides.")
        moe_config = moe_config or build_moe_config(config, moe_overrides)

        self.model = HunyuanImage3Model(config, self.backend, moe_config)
        self.lm_head = initialize_linear_module(
            self.backend.linear, config.hidden_size, config.vocab_size, bias=False, dtype=dtype
        )
        hidden = config.hidden_size
        latent = config.latent_channels
        self.timestep_emb = TimestepEmbedder(hidden, dtype=dtype)
        self.time_embed = TimestepEmbedder(hidden, dtype=dtype)
        self.patch_embed = UNetDown(latent, hidden, config.patch_embed_hidden_dim, hidden, dtype=dtype)
        self.time_embed_2 = TimestepEmbedder(hidden, dtype=dtype)
        self.final_layer = UNetUp(hidden, hidden, config.patch_embed_hidden_dim, latent, dtype=dtype)

        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = HunyuanImage3StateDictAdapter(
                config, self.model.moe_config, self.backend, dtype=dtype
            )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        valid_lengths: torch.Tensor | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor] | dict[str, torch.Tensor]:
        """Predict the flow velocity of the noisy latents.

        Args:
            input_ids: ``[batch, seq]`` token ids laid out as described in the module docstring, right-padded.
                Every row holds exactly ``h*w`` contiguous image tokens, preceded by the ``<timestep>`` token.
            latents: ``[batch, channels, h, w]`` noisy VAE latents.
            timestep: ``[batch]`` flow-matching timesteps in ``[0, 1000]`` (``sigma * 1000``).
            valid_lengths: ``[batch]`` number of non-padding tokens per row; defaults to the full length.
            return_dict: Return ``{"sample": velocity}`` instead of ``(velocity,)``.

        Returns:
            ``[batch, channels, h, w]`` predicted velocity, as a one-element tuple or a dict.
        """
        batch, seq_len = input_ids.shape
        _, _, token_h, token_w = latents.shape
        num_image = token_h * token_w
        device = input_ids.device

        is_image = input_ids == self.config.image_token_id
        counts = is_image.sum(dim=1)
        if not bool((counts == num_image).all()):
            raise ValueError(f"Every row needs {num_image} image tokens for {token_h}x{token_w} latents, got {counts}")
        image_starts = is_image.int().argmax(dim=1)
        if valid_lengths is None:
            valid_lengths = torch.full((batch,), seq_len, device=device, dtype=torch.long)

        embeds = self.model.embed_tokens(input_ids)
        image_tokens = self.patch_embed(latents.to(embeds.dtype), self.time_embed(timestep))
        timestep_tokens = self.timestep_emb(timestep)
        rows = torch.arange(batch, device=device)
        image_index = image_starts[:, None] + torch.arange(num_image, device=device)[None]
        embeds = embeds.index_put((rows[:, None], image_index), image_tokens.to(embeds.dtype))
        embeds = embeds.index_put((rows, image_starts - 1), timestep_tokens.to(embeds.dtype))

        positions = torch.stack(
            [image_grid_positions(seq_len, int(start), token_h, token_w, device=device) for start in image_starts]
        )
        cos, sin = rope_cos_sin(positions, self.config.attention_head_dim, self.config.rope_theta)
        attention_mask = build_joint_attention_mask(seq_len, image_starts, num_image, valid_lengths)
        padding_mask = torch.arange(seq_len, device=device)[None, :] >= valid_lengths[:, None]

        hidden = self.model(embeds, cos, sin, attention_mask=attention_mask, padding_mask=padding_mask)
        image_hidden = hidden[rows[:, None], image_index]
        sample = self.final_layer(image_hidden, self.time_embed_2(timestep), token_h, token_w)
        return {"sample": sample} if return_dict else (sample,)

    def forward_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Next-token logits of a causal text-only sequence (the release's ``gen_text`` mode).

        Args:
            input_ids: Long tensor of shape [batch, sequence].

        Returns:
            fp32 tensor of shape [batch, sequence, vocab].
        """
        _, seq_len = input_ids.shape
        cos, sin = rope_cos_sin(
            text_positions(seq_len, device=input_ids.device)[None],
            self.config.attention_head_dim,
            self.config.rope_theta,
        )
        hidden = self.model(self.model.embed_tokens(input_ids), cos, sin)
        return self.lm_head(self.model.norm(hidden)).float()

    def update_moe_gate_bias(self) -> None:
        with torch.no_grad():
            for block in self.model.layers.values():
                if isinstance(block.mlp, MoE) and block.mlp.gate.bias_update_factor > 0:
                    block.mlp.gate.update_bias()

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        if buffer_device is None:
            buffer_device = (
                torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            )
        with buffer_device:
            self.model.init_weights(buffer_device)
            nn.init.normal_(self.lm_head.weight, std=self.config.hidden_size**-0.5)
            for embedder in (self.timestep_emb, self.time_embed, self.time_embed_2):
                embedder.init_weights()
            init_unet_weights(self.patch_embed)
            init_unet_weights(self.final_layer)
        cast_model_to_dtype(self, dtype)


ModelClass = HunyuanImage3ForCausalMM
