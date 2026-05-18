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

"""LLM / VLM model builders.

Take a ``model_factory`` callable (e.g. ``NeMoAutoModelForCausalLM.from_pretrained``)
plus its keyword arguments, and pre-built typed infrastructure configs
(:class:`FP8Config`, :class:`MoEParallelizerConfig`, ...). The recipe is
responsible for resolving YAML ``_target_`` and config dicts into these
typed objects.

When ``model_factory`` is one of the ``NeMoAutoModelFor*`` classes those
classes apply distributed / PEFT / quantization / FP8 / compile / QAT / MoE
infrastructure internally; otherwise we instantiate the bare model and
apply infrastructure via :func:`apply_model_infrastructure`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
import torch.nn as nn
from huggingface_hub import constants as hf_constants

from nemo_automodel._transformers.infrastructure import (
    apply_model_infrastructure,
    instantiate_infrastructure,
)
from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.training.rng import ScopedRNG

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def build_model(
    *,
    model_factory: Callable[..., nn.Module],
    model_kwargs: dict[str, Any],
    is_nemo_auto_model: bool,
    seed: int,
    peft_config: Any = None,
    has_packed_sequence: bool = False,
    fp8_config: Any = None,
    compile_config: Any = None,
    quantization_config: Any = None,
    device_mesh: Any = None,
    moe_mesh: Any = None,
    distributed_config: Any = None,
    pipeline_config: Any = None,
    qat_config: Any = None,
    moe_config: Any = None,
    activation_checkpointing: bool = False,
    unfreeze_modules: Optional[list[str]] = None,
    sdpa_method: Optional[list[str]] = None,
    freeze_config: Optional[dict[str, Any]] = None,
) -> nn.Module | AutoPipeline:
    """Build and initialize an LLM model.

    Args:
        model_factory: Resolved callable producing the model
            (e.g. ``NeMoAutoModelForCausalLM.from_pretrained``).
        model_kwargs: Keyword arguments forwarded to ``model_factory``
            (e.g. ``{"pretrained_model_name_or_path": "..."}``).
        is_nemo_auto_model: ``True`` when ``model_factory`` is a
            ``NeMoAutoModelFor*.from_*`` classmethod — the factory then applies
            distributed / PEFT / quantization / FP8 / compile / QAT / MoE
            infrastructure internally. ``False`` for bare-model factories
            (e.g. ``build_gpt2_model``) where this function attaches
            infrastructure via :func:`apply_model_infrastructure`.
        seed: Random seed.
        peft_config: Instantiated PEFT config object, or None.
        has_packed_sequence: Whether using packed sequences.
        fp8_config: Pre-built FP8Config or None.
        compile_config: Pre-built CompileConfig or None.
        quantization_config: Pre-built BnB quantization config or None.
        device_mesh: Device mesh for distributed training.
        moe_mesh: MoE mesh for expert parallelism.
        distributed_config: Strategy-specific distributed config (FSDP2Config, ...).
        pipeline_config: Pipeline parallelism config.
        qat_config: Pre-built QAT config object, or None. ``peft_config`` must
            be ``None`` when this is set.
        moe_config: :class:`MoEParallelizerConfig` instance, or None.
        activation_checkpointing: Whether to enable activation checkpointing
            (only consulted when ``moe_config`` is provided).
        unfreeze_modules: List of module names/substrings to unfreeze.
        sdpa_method: Explicit SDPA backend list, or None to auto-select.
    """
    if qat_config is not None and peft_config is not None:
        raise ValueError("QAT with PEFT is not currently supported")

    with ScopedRNG(seed=seed, ranked=True):
        infra_kwargs: dict[str, Any] = {
            "has_packed_sequence": has_packed_sequence,
            "peft_config": peft_config,
            "device_mesh": device_mesh,
            "moe_mesh": moe_mesh,
            "distributed_config": distributed_config,
            "pipeline_config": pipeline_config,
            "sdpa_method": sdpa_method,
        }
        if freeze_config is not None:
            infra_kwargs["freeze_config"] = freeze_config
        if qat_config is not None:
            infra_kwargs["qat_config"] = qat_config
        if moe_config is not None:
            infra_kwargs["moe_config"] = moe_config
            infra_kwargs["activation_checkpointing"] = activation_checkpointing
        if fp8_config is not None:
            infra_kwargs["fp8_config"] = fp8_config
        if compile_config is not None:
            infra_kwargs["compile_config"] = compile_config
        if quantization_config is not None:
            logger.info("Model weight quantization enabled with BitsAndBytes")
            infra_kwargs["quantization_config"] = quantization_config

        if is_nemo_auto_model:
            model = model_factory(**model_kwargs, **infra_kwargs)
        else:
            # Non-NeMoAutoModel target (e.g. build_gpt2_model): build the bare
            # model first, then attach infrastructure separately. SDPA patching
            # only runs inside NeMoAutoModel._build_model.
            if sdpa_method is not None:
                logger.warning("sdpa_method is ignored for non-NeMoAutoModel targets.")
            model = model_factory(**model_kwargs)

            mesh = MeshContext.from_meshes(device_mesh, moe_mesh)
            model_wrapper, autopipeline, parallelize_fn, qat_quantizer = instantiate_infrastructure(
                distributed_config=distributed_config,
                pipeline_config=pipeline_config,
                qat_config=qat_config,
                moe_config=moe_config,
                activation_checkpointing=activation_checkpointing,
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
                peft_config=peft_config,
                fp8_config=fp8_config,
                compile_config=compile_config,
                quantization_config=quantization_config,
                pretrained_model_name_or_path=None,
                load_base_model=False,
                cache_dir=hf_constants.HF_HUB_CACHE,
            )

    if unfreeze_modules:
        for name, param in model.named_parameters():
            if any(module_name in name for module_name in unfreeze_modules):
                param.requires_grad_(True)
        logger.info(f"Unfroze parameters matching: {unfreeze_modules}")

    return model


__all__ = ["build_model"]
