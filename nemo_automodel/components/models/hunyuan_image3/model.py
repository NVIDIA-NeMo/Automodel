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

"""Native HunyuanImage-3.0 (tencent/HunyuanImage-3.0).

The 80B-total / 13B-active MoE language backbone is implemented natively so expert parallelism and FSDP2 can shard
it. The image-side modules (VAE, UNet patch embedding and final layer, timestep embedders, SigLIP2 vision tower) are
built from the checkpoint's own remote code at construction time, so their numerics are identical to the release
and their source is not vendored here.

Two forward modes mirror the release:

* ``gen_text``: causal LM over text tokens; returns logits.
* ``gen_image``: one flow-matching step over a joint sequence. Noisy VAE latents are embedded by ``patch_embed`` and
  scattered into the image slots, a timestep token is scattered in, the backbone runs with a text-causal /
  image-bidirectional mask and 2D RoPE, and the image-slot outputs (without the final norm) are decoded by
  ``final_layer`` into a latent-space prediction.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.hunyuan_image3.config import per_layer
from nemo_automodel.components.models.hunyuan_image3.layers import HunyuanImage3Block, HunyuanImage3RMSNorm
from nemo_automodel.components.models.hunyuan_image3.rope import build_2d_rope_cos_sin
from nemo_automodel.components.models.hunyuan_image3.state_dict_adapter import HunyuanImage3StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

# Image-side modules taken from the checkpoint's remote code: attribute -> remote class.
_IMAGE_GEN_MODULES = ("timestep_emb", "patch_embed", "time_embed", "final_layer", "time_embed_2")
_IMAGE_EXTRA_MODULES = ("vae", "vision_model", "vision_aligner")


@dataclass
class HunyuanImage3Output(CausalLMOutputWithPast):
    """``logits`` for ``gen_text``; ``diffusion_prediction`` ``[B, C, H, W]`` for ``gen_image``."""

    diffusion_prediction: torch.Tensor | None = None


def _remote_code_dir(config: Any) -> str:
    path = getattr(config, "remote_code_dir", None) or getattr(config, "_name_or_path", None)
    if not path:
        raise ValueError(
            "HunyuanImage-3.0 builds its image modules from the checkpoint's remote code; set "
            "`config.remote_code_dir` (or load the config from the checkpoint directory)."
        )
    return path


def build_image_modules(config: Any, include_extra: bool) -> dict[str, nn.Module]:
    """Instantiate the release's image modules from the checkpoint's remote code."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    path = _remote_code_dir(config)

    def cls(name: str):
        return get_class_from_dynamic_module(name, path)

    hidden = config.hidden_size
    latent_channels = config.vae["latent_channels"]
    if config.img_proj_type != "unet":
        raise ValueError(f"Unsupported img_proj_type {config.img_proj_type!r}")
    timestep_embedder = cls("hunyuan.TimestepEmbedder")
    modules = {
        "timestep_emb": timestep_embedder(hidden_size=hidden),
        "patch_embed": cls("hunyuan.UNetDown")(
            patch_size=config.patch_size,
            emb_channels=hidden,
            in_channels=latent_channels,
            hidden_channels=config.patch_embed_hidden_dim,
            out_channels=hidden,
        ),
        "time_embed": timestep_embedder(hidden_size=hidden),
        "final_layer": cls("hunyuan.UNetUp")(
            patch_size=config.patch_size,
            emb_channels=hidden,
            in_channels=hidden,
            hidden_channels=config.patch_embed_hidden_dim,
            out_channels=latent_channels,
            out_norm=True,
        ),
        "time_embed_2": timestep_embedder(hidden_size=hidden),
    }
    if include_extra:
        modules["vae"] = cls("autoencoder_kl_3d.AutoencoderKLConv3D").from_config(config.vae)
        modules["vision_model"] = cls("siglip2.Siglip2VisionTransformer")(config.vit)
        modules["vision_aligner"] = cls("siglip2.LightProjector")(config.vit_aligner)
    return modules


class HunyuanImage3Model(nn.Module):
    """Token embedding, MoE decoder layers and final norm (applied only for text logits)."""

    def __init__(self, config: Any, backend: BackendConfig, moe_config: MoEConfig | None = None):
        super().__init__()
        self.config = config
        self.backend = backend
        self.moe_config = moe_config or MoEConfig(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=per_layer(config.moe_intermediate_size, 0),
            n_routed_experts=config.num_experts,
            # The shared expert is a separate HunyuanImage3MLP so its fused HF layout is kept as is.
            n_shared_experts=0,
            n_activated_experts=per_layer(config.moe_topk, 0),
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            softmax_before_topk=True,
            route_scale=1.0,
            norm_topk_prob=config.norm_topk_prob,
            router_bias=False,
            expert_bias=False,
            expert_activation="swiglu",
            gate_dtype=torch.float32,
            dtype=get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16),
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleDict(
            {str(i): HunyuanImage3Block(i, config, self.moe_config, backend) for i in range(config.num_hidden_layers)}
        )
        self.ln_f = HunyuanImage3RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        h = inputs_embeds
        all_hidden = [h] if output_hidden_states else None
        for layer in self.layers.values():
            h = layer(h, cos=cos, sin=sin, attention_mask=attention_mask, padding_mask=padding_mask)
            if output_hidden_states:
                all_hidden.append(h)
        return h, all_hidden

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device) -> None:
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        self.ln_f.reset_parameters()
        for layer in self.layers.values():
            layer.init_weights(buffer_device)


class HunyuanImage3ForCausalMM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """HunyuanImage-3.0 with the native MoE backbone and the release's image modules."""

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    # The router runs in fp32 like the release; the frozen VAE is stored in fp32 on disk.
    _keep_in_fp32_modules_strict = ["mlp.gate.weight", "vae"]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = True

    @classmethod
    def from_config(
        cls, config: Any, moe_config: MoEConfig | None = None, backend: BackendConfig | None = None, **kwargs
    ):
        return cls(config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=False)
        if not getattr(config, "remote_code_dir", None):
            config.remote_code_dir = str(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        *,
        include_vae_and_vision: bool | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        reject_unsupported_tie_word_embeddings(type(self), config)
        self.backend = backend or BackendConfig()
        self.model = HunyuanImage3Model(config, self.backend, moe_config)
        self.lm_head = initialize_linear_module(self.backend.linear, config.hidden_size, config.vocab_size, bias=False)
        if include_vae_and_vision is None:
            include_vae_and_vision = getattr(config, "include_vae_and_vision", True)
        for name, module in build_image_modules(config, include_extra=include_vae_and_vision).items():
            setattr(self, name, module)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = HunyuanImage3StateDictAdapter(
                config,
                self.model.moe_config,
                self.backend,
                dtype=get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16),
            )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _rope(self, rope_positions: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return build_2d_rope_cos_sin(rope_positions, self.config.attention_head_dim, self.config.rope_theta, device)

    def embed_image_inputs(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        timestep: torch.Tensor,
        image_mask: torch.Tensor,
        timestep_scatter_index: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        """Text embeddings with the noisy-latent image tokens and the timestep token written into their slots."""
        x = self.model.embed_tokens(input_ids)
        bsz, _, hidden = x.shape
        image_tokens, token_h, token_w = self.patch_embed(images, self.time_embed(timestep))
        x = x.masked_scatter(image_mask.bool().unsqueeze(-1), image_tokens.to(x.dtype))
        t_tokens = self.timestep_emb(timestep.reshape(-1)).reshape(bsz, -1, hidden).to(x.dtype)
        x = x.scatter(1, timestep_scatter_index.unsqueeze(-1).expand(-1, -1, hidden), t_tokens)
        return x, token_h, token_w

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        rope_positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        mode: str = "gen_text",
        images: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        timestep_scatter_index: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> HunyuanImage3Output:
        """
        Args:
            input_ids: ``[B, S]`` token ids (image and timestep slots hold placeholder ids).
            rope_positions: ``[B, S, 2]`` integer (row, col) 2D RoPE positions.
            attention_mask: ``[B, 1, S, S]`` boolean (True = attend); None means plain causal.
            mode: ``"gen_text"`` or ``"gen_image"``.
            images: ``[B, C, h, w]`` noisy VAE latents (``gen_image``).
            timestep: ``[B]`` flow-matching time in ``[0, 1000]`` scale used by the release (``gen_image``).
            image_mask: ``[B, S]`` True at the image slots (``gen_image``).
            timestep_scatter_index: ``[B, n]`` positions of the timestep tokens (``gen_image``).
            padding_mask: ``[B, S]`` True at padding tokens; excluded from routing statistics.
        """
        cos, sin = self._rope(rope_positions, input_ids.device)
        if mode == "gen_text":
            x = self.model.embed_tokens(input_ids)
        elif mode == "gen_image":
            for name, value in (
                ("images", images),
                ("timestep", timestep),
                ("image_mask", image_mask),
                ("timestep_scatter_index", timestep_scatter_index),
            ):
                if value is None:
                    raise ValueError(f"`{name}` is required in gen_image mode")
            x, token_h, token_w = self.embed_image_inputs(
                input_ids, images, timestep, image_mask, timestep_scatter_index
            )
        else:
            raise ValueError(f"Unknown mode {mode!r}")

        hidden, all_hidden = self.model(
            x,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = tuple(all_hidden) if all_hidden is not None else None
        if mode == "gen_text":
            logits = self.lm_head(self.model.ln_f(hidden)).float()
            return HunyuanImage3Output(logits=logits, hidden_states=hidden_states)

        bsz, _, dim = hidden.shape
        image_hidden = hidden.masked_select(image_mask.bool().unsqueeze(-1)).reshape(bsz, -1, dim)
        pred = self.final_layer(image_hidden, self.time_embed_2(timestep), token_h, token_w)
        return HunyuanImage3Output(diffusion_prediction=pred, hidden_states=hidden_states)

    def update_moe_gate_bias(self) -> None:
        with torch.no_grad():
            for block in self.model.layers.values():
                if isinstance(block.mlp, MoE) and block.mlp.gate.bias_update_factor > 0:
                    block.mlp.gate.update_bias()

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            self.model.init_weights(buffer_device)
            std = self.config.hidden_size**-0.5
            nn.init.trunc_normal_(self.lm_head.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        cast_model_to_dtype(self, dtype)


ModelClass = HunyuanImage3ForCausalMM
