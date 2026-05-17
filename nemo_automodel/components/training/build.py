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

"""LLM model builder.

Constructs the training model from a recipe-style ``cfg_model`` ConfigNode.
Branches on whether the configured ``_target_`` is one of the
``NeMoAutoModelFor*`` classes — those classes apply distributed / PEFT /
quantization / FP8 / compile / QAT / MoE infrastructure internally; for
non-NeMoAutoModel targets we instantiate the bare model and apply the
infrastructure via :func:`apply_model_infrastructure`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from huggingface_hub import constants as hf_constants

from nemo_automodel._transformers import (
    NeMoAutoModelForCausalLM,
    NeMoAutoModelForImageTextToText,
    NeMoAutoModelForMultimodalLM,
    NeMoAutoModelForSequenceClassification,
)
from nemo_automodel._transformers.infrastructure import (
    apply_model_infrastructure,
    instantiate_infrastructure,
)
from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.quantization.fp8 import build_fp8_config
from nemo_automodel.components.training.rng import ScopedRNG
from nemo_automodel.components.utils.compile_utils import build_compile_config

if TYPE_CHECKING:
    from torch.optim import Optimizer

logger = logging.getLogger(__name__)


def build_model(
    cfg_model,
    cfg_peft,
    seed,
    has_packed_sequence=False,
    cfg_fp8=None,
    cfg_compile=None,
    cfg_quantization=None,
    device_mesh=None,
    moe_mesh=None,
    distributed_config=None,
    pipeline_config=None,
    cfg_qat=None,
    cfg_moe=None,
    activation_checkpointing=False,
    unfreeze_modules: list[str] | None = None,
    sdpa_method: list[str] | None = None,
) -> tuple[nn.Module | AutoPipeline, list["Optimizer"]]:  # noqa: F821
    """Build and initialize a model.

    Args:
        cfg_model: Configuration for model instantiation.
        cfg_peft: Configuration for PEFT.
        seed: Random seed.
        has_packed_sequence: Whether using packed sequences.
        cfg_fp8: Configuration for FP8.
        cfg_compile: Configuration for torch.compile.
        cfg_quantization: Configuration for BitsAndBytes quantization.
        device_mesh: Device mesh for distributed training.
        moe_mesh: MOE mesh for expert parallelism.
        distributed_config: Strategy-specific distributed config (FSDP2Config, etc.).
        pipeline_config: Pipeline parallelism config.
        cfg_qat: Configuration for QAT (will be instantiated to QATConfig).
        cfg_moe: MoEParallelizerConfig instance, or ConfigNode to be converted.
        activation_checkpointing: Whether to enable activation checkpointing.
        unfreeze_modules: List of module names/substrings to unfreeze.
        sdpa_method: Explicit list of SDPA backend name strings (e.g.
            ``["flash_attention", "efficient_attention"]``), or ``None`` to
            auto-select based on CP / activation checkpointing.
    """
    with ScopedRNG(seed=seed, ranked=True):
        kwargs = {
            "has_packed_sequence": has_packed_sequence,
            "peft_config": cfg_peft,
            "device_mesh": device_mesh,
            "moe_mesh": moe_mesh,
            "distributed_config": distributed_config,
            "pipeline_config": pipeline_config,
            "sdpa_method": sdpa_method,
        }

        if cfg_qat is not None and cfg_qat.get("enabled", False):
            if cfg_peft is not None:
                raise ValueError("QAT with PEFT is not currently supported")
            qat_config_attr = getattr(cfg_qat, "qat_config", None)
            if qat_config_attr is not None:
                kwargs["qat_config"] = qat_config_attr.instantiate()
            else:
                # Fallback to legacy quantizer format for backward compatibility
                quantizer_attr = getattr(cfg_qat, "quantizer", None)
                if quantizer_attr is not None:
                    kwargs["qat_config"] = quantizer_attr.instantiate()

        if cfg_moe is not None:
            from nemo_automodel.components.moe.config import MoEParallelizerConfig

            if isinstance(cfg_moe, MoEParallelizerConfig):
                kwargs["moe_config"] = cfg_moe
            else:
                moe_dict = cfg_moe.to_dict() if hasattr(cfg_moe, "to_dict") else dict(cfg_moe)
                # activation_checkpointing is handled separately; strip config keys
                moe_dict.pop("activation_checkpointing", None)
                moe_dict.pop("_target_", None)
                kwargs["moe_config"] = MoEParallelizerConfig(**moe_dict)
            kwargs["activation_checkpointing"] = activation_checkpointing

        if cfg_fp8 is not None:
            kwargs["fp8_config"] = build_fp8_config(cfg_fp8)
        if cfg_compile is not None:
            kwargs["compile_config"] = build_compile_config(cfg_compile)
        if cfg_quantization is not None:
            logger.info("Model weight quantization enabled with BitsAndBytes")
            from nemo_automodel.components.quantization.qlora import create_bnb_config

            kwargs["quantization_config"] = create_bnb_config(cfg_quantization)

        is_nemo_auto_model = cfg_model.get("_target_", None) in (
            NeMoAutoModelForCausalLM.from_config,
            NeMoAutoModelForCausalLM.from_pretrained,
            NeMoAutoModelForSequenceClassification.from_config,
            NeMoAutoModelForSequenceClassification.from_pretrained,
        )

        if is_nemo_auto_model:
            # NeMoAutoModel handles infrastructure internally
            model = cfg_model.instantiate(**kwargs)
        else:
            # For non-NemoAutoModel entry points (e.g., build_gpt2_model),
            # instantiate the model first, then apply infrastructure separately.
            # Note: sdpa_method is not supported here — SDPA patching only runs
            # inside NeMoAutoModel._build_model.
            if sdpa_method is not None:
                logger.warning("sdpa_method is ignored for non-NeMoAutoModel targets.")
            # We must convert config objects into runtime objects (model_wrapper,
            # autopipeline, parallelize_fn, etc.) via instantiate_infrastructure,
            # exactly as from_pretrained/from_config do internally.
            model = cfg_model.instantiate()

            mesh = MeshContext.from_meshes(device_mesh, moe_mesh)
            model_wrapper, autopipeline, parallelize_fn, qat_quantizer = instantiate_infrastructure(
                distributed_config=distributed_config,
                pipeline_config=pipeline_config,
                qat_config=kwargs.get("qat_config"),
                moe_config=kwargs.get("moe_config"),
                activation_checkpointing=kwargs.get("activation_checkpointing", False),
                device=torch.device("cuda", torch.cuda.current_device()),
                mesh=mesh,
            )
            loss_fn = pipeline_config.loss_fn if pipeline_config is not None else None

            model = apply_model_infrastructure(
                model,
                is_meta_device=False,
                device=torch.cuda.current_device(),
                mesh=mesh,
                model_wrapper=model_wrapper,
                autopipeline=autopipeline,
                parallelize_fn=parallelize_fn,
                qat_quantizer=qat_quantizer,
                loss_fn=loss_fn,
                peft_config=kwargs.get("peft_config"),
                fp8_config=kwargs.get("fp8_config"),
                compile_config=kwargs.get("compile_config"),
                quantization_config=kwargs.get("quantization_config"),
                pretrained_model_name_or_path=None,
                load_base_model=False,
                cache_dir=hf_constants.HF_HUB_CACHE,
            )

    # Explicitly unfreeze specified modules (e.g. task heads) that need full fine-tuning
    if unfreeze_modules:
        for name, param in model.named_parameters():
            if any(module_name in name for module_name in unfreeze_modules):
                param.requires_grad_(True)
        logging.info(f"Unfroze parameters matching: {unfreeze_modules}")

    return model


def build_vlm_model(
    cfg_model,
    cfg_freeze,
    cfg_peft,
    seed,
    cfg_fp8=None,
    cfg_compile=None,
    device_mesh=None,
    moe_mesh=None,
    distributed_config=None,
    pipeline_config=None,
    cfg_moe=None,
    activation_checkpointing=False,
) -> nn.Module | AutoPipeline:
    """Build and initialize a VLM model.

    Unlike :func:`build_model` (LLM), this requires the configured ``_target_``
    to be one of the ``NeMoAutoModelFor{ImageTextToText,MultimodalLM,CausalLM}``
    classes — there's no fallback path. The classes apply distributed / PEFT /
    FP8 / compile / MoE / freeze logic internally.
    """
    with ScopedRNG(seed=seed, ranked=True):
        kwargs = {
            "peft_config": cfg_peft,
            "device_mesh": device_mesh,
            "moe_mesh": moe_mesh,
            "distributed_config": distributed_config,
            "pipeline_config": pipeline_config,
            "freeze_config": cfg_freeze.to_dict() if cfg_freeze is not None else None,
        }

        if cfg_moe is not None:
            from nemo_automodel.components.moe.config import MoEParallelizerConfig

            if isinstance(cfg_moe, MoEParallelizerConfig):
                kwargs["moe_config"] = cfg_moe
            else:
                moe_dict = cfg_moe.to_dict() if hasattr(cfg_moe, "to_dict") else dict(cfg_moe)
                moe_dict.pop("activation_checkpointing", None)
                moe_dict.pop("_target_", None)
                kwargs["moe_config"] = MoEParallelizerConfig(**moe_dict)
            kwargs["activation_checkpointing"] = activation_checkpointing

        if cfg_fp8 is not None:
            kwargs["fp8_config"] = build_fp8_config(cfg_fp8)
        if cfg_compile is not None:
            kwargs["compile_config"] = build_compile_config(cfg_compile)

        is_nemo_auto_model = cfg_model.get("_target_", None) in (
            NeMoAutoModelForImageTextToText.from_config,
            NeMoAutoModelForImageTextToText.from_pretrained,
            NeMoAutoModelForMultimodalLM.from_config,
            NeMoAutoModelForMultimodalLM.from_pretrained,
            NeMoAutoModelForCausalLM.from_config,
            NeMoAutoModelForCausalLM.from_pretrained,
        )

        if not is_nemo_auto_model:
            raise ValueError(
                f"VLM finetuning requires NeMoAutoModelForImageTextToText. "
                f"Got model target: {cfg_model.get('_target_', None)}"
            )
        return cfg_model.instantiate(**kwargs)


__all__ = ["build_model", "build_vlm_model"]
