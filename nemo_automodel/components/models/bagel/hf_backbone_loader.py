# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Helpers for initializing BAGEL pretraining runs from HF backbones."""

from __future__ import annotations

import gc
import json
import logging
import pathlib
from typing import TYPE_CHECKING, Any, Dict

import torch

if TYPE_CHECKING:
    from nemo_automodel.components.models.bagel.backend import BagelBackendConfig

logger = logging.getLogger(__name__)


def _load_siglip_vision_config(vit_path: str):
    """Load a SigLIP vision config from a vision-only or full SigLIP HF folder."""
    from nemo_automodel.components.models.bagel.modeling_siglip_navit import SiglipVisionConfig

    config_path = pathlib.Path(vit_path) / "config.json"
    if config_path.is_file():
        with open(config_path, "r") as f:
            raw_config = json.load(f)
        if "vision_config" in raw_config:
            return SiglipVisionConfig(**raw_config["vision_config"])
    return SiglipVisionConfig.from_pretrained(vit_path)


def _normalize_wrapped_param_name(name: str) -> str:
    """Remove wrapper path fragments that are not part of logical parameter FQNs."""
    for fragment in ("._checkpoint_wrapped_module", "._fsdp_wrapped_module"):
        name = name.replace(fragment, "")
    for fragment in ("_checkpoint_wrapped_module.", "_fsdp_wrapped_module."):
        name = name.replace(fragment, "")
    return name


def _reset_qwen_qk_norms_for_hf_backbone(language_model) -> int:
    """Reset BAGEL-added Q/K norm weights missing from vanilla Qwen checkpoints."""
    reset_count = 0
    qk_norm_suffixes = (
        "self_attn.q_norm",
        "self_attn.k_norm",
        "self_attn.q_norm_moe_gen",
        "self_attn.k_norm_moe_gen",
    )
    for name, module in language_model.named_modules():
        logical_name = _normalize_wrapped_param_name(name)
        if logical_name.endswith(qk_norm_suffixes) and hasattr(module, "weight"):
            module.weight.data.fill_(1.0)
            reset_count += 1
    return reset_count


def _resolve_hf_weight_path(model_path: str) -> str:
    """Resolve a local path or download a HF snapshot containing model weights."""
    if pathlib.Path(model_path).exists():
        return model_path

    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model_path,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.safetensors.index.json",
        ],
    )


def _load_hf_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    """Load a HF safetensors/bin checkpoint as a full CPU state dict."""
    from nemo_automodel.components.checkpoint.checkpointing import _load_hf_checkpoint_preserving_dtype

    resolved_path = _resolve_hf_weight_path(model_path)
    state_dict = _load_hf_checkpoint_preserving_dtype(resolved_path)
    if state_dict is None:
        raise FileNotFoundError(f"No loadable HF checkpoint weights found at {resolved_path}")
    return state_dict


def _copy_qwen_mot_weights_from_und(language_model) -> int:
    """Copy UND Qwen weights into ``*_moe_gen`` siblings after sharding/wrapping."""
    state_by_logical_name = {
        _normalize_wrapped_param_name(name): tensor for name, tensor in language_model.state_dict().items()
    }
    copied = 0
    with torch.no_grad():
        for name, param in language_model.named_parameters():
            if "_moe_gen" not in name:
                continue
            source_name = _normalize_wrapped_param_name(name).replace("_moe_gen", "")
            source = state_by_logical_name.get(source_name)
            if source is None:
                raise KeyError(source_name)
            param.data.copy_(source.data)
            copied += 1
    return copied


def _load_qwen_backbone_into_bagel(model, llm_path: str, *, copy_init_moe: bool) -> None:
    """Load vanilla Qwen weights into BAGEL's language model after AM sharding."""
    from nemo_automodel.components.checkpoint.checkpointing import _load_full_state_dict_into_model

    logger.info("Loading Qwen backbone from %s", llm_path)
    state_dict = _load_hf_state_dict(llm_path)
    _load_full_state_dict_into_model([model.model.language_model], state_dict)
    logger.info("HF-backbone Qwen load: loaded %d tensor(s)", len(state_dict))
    del state_dict
    gc.collect()

    reset_qk_norms = _reset_qwen_qk_norms_for_hf_backbone(model.model.language_model)
    logger.info("HF-backbone Qwen QK-norm reset: %d module(s)", reset_qk_norms)

    if copy_init_moe:
        logger.info("Initializing MoT generation weights from UND weights")
        copied = _copy_qwen_mot_weights_from_und(model.model.language_model)
        logger.info("Initialized %d MoT generation tensor(s) from UND weights", copied)


def _load_siglip_backbone_into_bagel(model, vit_path: str) -> None:
    """Load SigLIP weights into BAGEL's packed-NaViT vision model after AM sharding."""
    from nemo_automodel.components.checkpoint.checkpointing import _load_full_state_dict_into_model

    logger.info("Loading SigLIP vision backbone from %s", vit_path)
    state_dict = _load_hf_state_dict(vit_path)
    patch_weight_key = "vision_model.embeddings.patch_embedding.weight"
    patch_weight = state_dict.get(patch_weight_key)
    if isinstance(patch_weight, torch.Tensor) and patch_weight.ndim == 4:
        state_dict[patch_weight_key] = patch_weight.permute(0, 2, 3, 1).reshape(patch_weight.shape[0], -1)

    _load_full_state_dict_into_model([model.model.vit_model], state_dict)
    logger.info("HF-backbone SigLIP load: loaded %d tensor(s)", len(state_dict))
    del state_dict
    gc.collect()


def initialize_bagel_non_backbone_weights(model: torch.nn.Module) -> None:
    """Initialize BAGEL-owned modules not loaded from Qwen/SigLIP checkpoints."""
    from nemo_automodel.components.models.bagel.embeddings import _get_2d_sincos_pos_embed

    bagel = model.model

    def _reset_linear(module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Linear):
            module.reset_parameters()

    def _init_position_embedding(module: torch.nn.Module) -> None:
        pos_embed = _get_2d_sincos_pos_embed(module.hidden_size, module.max_num_patch_per_side)
        value = torch.from_numpy(pos_embed).to(dtype=module.pos_embed.dtype)
        param_data = module.pos_embed.data
        if type(param_data).__name__ == "DTensor":
            from torch.distributed.tensor import distribute_tensor

            value = distribute_tensor(
                value,
                device_mesh=param_data.device_mesh,
                placements=param_data.placements,
            )
        else:
            value = value.to(device=param_data.device)
        param_data.copy_(value)

    with torch.no_grad():
        if getattr(model.config, "visual_gen", False):
            bagel.time_embedder.apply(_reset_linear)
            bagel.vae2llm.reset_parameters()
            bagel.llm2vae.reset_parameters()
            _init_position_embedding(bagel.latent_pos_embed)

        if getattr(model.config, "visual_und", False):
            bagel.connector.apply(_reset_linear)
            _init_position_embedding(bagel.vit_pos_embed)

        if getattr(model.config, "visual_gen", False):
            torch.nn.init.constant_(bagel.llm2vae.weight, 0.0)
            torch.nn.init.constant_(bagel.llm2vae.bias, 0.0)


def load_bagel_hf_backbone_weights(model: torch.nn.Module, model_cfg: Any) -> None:
    """Load Qwen/SigLIP HF backbone weights into an already-built BAGEL model."""
    llm_path = model_cfg.get("llm_path", None)
    vit_path = model_cfg.get("vit_path", None)
    visual_und = bool(model_cfg.get("visual_und", True))

    if llm_path is None:
        raise ValueError("model.init_mode='hf_backbones' requires model.llm_path")
    if visual_und and vit_path is None:
        raise ValueError("model.init_mode='hf_backbones' with visual_und=True requires model.vit_path")

    _load_qwen_backbone_into_bagel(model, llm_path, copy_init_moe=bool(model_cfg.get("copy_init_moe", True)))
    if visual_und:
        _load_siglip_backbone_into_bagel(model, vit_path)


def build_bagel_from_hf_backbones(
    *,
    model_cfg: Any,
    stage: int,
    vae_config: Dict[str, int] | None,
    meta_init: bool = False,
    load_backbone_weights: bool = True,
    backend: BagelBackendConfig | None = None,
) -> torch.nn.Module:
    """Build BAGEL from upstream Qwen/SigLIP backbone configs."""
    from transformers import Qwen2Config

    from nemo_automodel.components.models.bagel.configuration import BagelConfig
    from nemo_automodel.components.models.bagel.model import BagelForUnifiedMultimodal
    from nemo_automodel.shared.utils import dtype_from_str

    llm_path = model_cfg.get("llm_path", None)
    vit_path = model_cfg.get("vit_path", None)
    if llm_path is None:
        raise ValueError("model.init_mode='hf_backbones' requires model.llm_path")

    visual_gen = stage == 2
    visual_und = bool(model_cfg.get("visual_und", True))
    if visual_und and vit_path is None:
        raise ValueError("model.init_mode='hf_backbones' with visual_und=True requires model.vit_path")
    if hasattr(vae_config, "to_dict"):
        vae_config = vae_config.to_dict()

    llm_config = Qwen2Config.from_pretrained(llm_path)
    default_layer_module = "Qwen2MoTDecoderLayer" if visual_gen else "Qwen2DecoderLayer"
    llm_config.layer_module = model_cfg.get("layer_module", default_layer_module)
    llm_config.qk_norm = bool(model_cfg.get("llm_qk_norm", True))
    llm_config.tie_word_embeddings = bool(model_cfg.get("tie_word_embeddings", False))
    llm_config.freeze_und = bool(model_cfg.get("freeze_und", False))

    vit_config = _load_siglip_vision_config(vit_path) if visual_und else None

    bagel_config = BagelConfig(
        visual_gen=visual_gen,
        visual_und=visual_und,
        stage=stage,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config if visual_gen else None,
        latent_patch_size=int(model_cfg.get("latent_patch_size", 2)),
        max_latent_size=int(model_cfg.get("max_latent_size", 64)),
        vit_patch_size=int(model_cfg.get("vit_patch_size", 14)),
        vit_max_num_patch_per_side=int(model_cfg.get("vit_max_num_patch_per_side", 70)),
        connector_act=model_cfg.get("connector_act", "gelu_pytorch_tanh"),
        interpolate_pos=bool(model_cfg.get("interpolate_pos", False)),
        vit_select_layer=int(model_cfg.get("vit_select_layer", -2)),
        vit_rope=bool(model_cfg.get("vit_rope", False)),
        text_cond_dropout_prob=float(model_cfg.get("text_cond_dropout_prob", 0.1)),
        vae_cond_dropout_prob=float(model_cfg.get("vae_cond_dropout_prob", 0.3)),
        vit_cond_dropout_prob=float(model_cfg.get("vit_cond_dropout_prob", 0.3)),
        timestep_shift=float(model_cfg.get("timestep_shift", 1.0)),
    )

    torch_dtype = model_cfg.get("torch_dtype", "float32")
    torch_dtype = dtype_from_str(torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch_dtype)
        if meta_init:
            from transformers.initialization import no_init_weights

            from nemo_automodel.components.utils.model_utils import init_empty_weights

            with no_init_weights(), init_empty_weights():
                model = BagelForUnifiedMultimodal(bagel_config, backend=backend)
        else:
            model = BagelForUnifiedMultimodal(bagel_config, backend=backend)
    finally:
        torch.set_default_dtype(original_dtype)

    if load_backbone_weights:
        load_bagel_hf_backbone_weights(model, model_cfg)

    return model


__all__ = [
    "build_bagel_from_hf_backbones",
    "initialize_bagel_non_backbone_weights",
    "load_bagel_hf_backbone_weights",
]
