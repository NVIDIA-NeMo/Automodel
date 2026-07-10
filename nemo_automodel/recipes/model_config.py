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

"""Typed model construction shared by recipe domains."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import cast

import torch
import torch.nn as nn
from huggingface_hub import constants as hf_constants

from nemo_automodel._transformers.infrastructure import apply_model_infrastructure, instantiate_infrastructure
from nemo_automodel.components.distributed.config import DistributedSetup
from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.quantization.fp8 import FP8Config
from nemo_automodel.components.quantization.qat import QATConfig
from nemo_automodel.components.quantization.qlora import BitsAndBytesQuantizationConfig
from nemo_automodel.components.training.rng import ScopedRNG
from nemo_automodel.components.utils.compile_utils import CompileConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelTargetConfig:
    """Declarative nested ``_target_`` used while constructing a model."""

    factory: Callable[..., object]
    kwargs: Mapping[str, object] = field(default_factory=dict)

    def build(self) -> object:
        """Build the configured nested model value.

        Returns:
            Object returned by the configured target.
        """
        return self.factory(**{name: _materialize(value) for name, value in self.kwargs.items()})


def _materialize(value: object) -> object:
    """Materialize nested model targets without mutating declarative config state."""
    if isinstance(value, ModelTargetConfig):
        return value.build()
    if isinstance(value, Mapping):
        return {name: _materialize(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_materialize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize(item) for item in value)
    return value


@dataclass(frozen=True)
class ModelConfig:
    """Declarative model construction and infrastructure policy.

    Model target arguments and optional precision/quantization settings are
    serialized fields. Runtime-only objects such as PEFT and distributed setup
    are explicit :meth:`build` arguments.
    """

    model_factory: Callable[..., nn.Module | AutoPipeline]
    model_kwargs: Mapping[str, object] = field(default_factory=dict)
    factory_applies_infrastructure: bool = False
    model_name: str | None = None
    seed: int = 42
    has_packed_sequence: bool | None = None
    freeze_config: Mapping[str, object] | None = None
    fp8_config: FP8Config | None = None
    compile_config: CompileConfig | None = None
    quantization_config: BitsAndBytesQuantizationConfig | None = None
    qat_enabled: bool = False
    qat_config: ModelTargetConfig | None = None
    sdpa_method: tuple[str, ...] | None = None

    def build(
        self,
        *,
        peft_config: object | None = None,
        distributed_setup: DistributedSetup | None = None,
        unfreeze_modules: Sequence[str] | None = None,
    ) -> nn.Module | AutoPipeline:
        """Build and initialize the configured model.

        Args:
            peft_config: Runtime PEFT configuration composed by the recipe.
            distributed_setup: Resolved distributed topology and policy.
            unfreeze_modules: Module-name fragments whose parameters remain trainable.

        Returns:
            Initialized model or pipeline-parallel model wrapper.

        Raises:
            ValueError: If QAT and PEFT are enabled together.
        """
        if self.qat_enabled and peft_config is not None:
            raise ValueError("QAT with PEFT is not currently supported")

        model_kwargs = {name: _materialize(value) for name, value in self.model_kwargs.items()}
        if distributed_setup is not None and distributed_setup.mesh_context.cp_size > 1:
            backend = model_kwargs.get("backend")
            if isinstance(backend, BackendConfig) and backend.rope_fusion:
                logger.info(
                    "Disabling rope_fusion because cp_size=%d > 1",
                    distributed_setup.mesh_context.cp_size,
                )
                backend.rope_fusion = False

        runtime_kwargs: dict[str, object] = {"peft_config": peft_config}
        if self.has_packed_sequence is not None:
            runtime_kwargs["has_packed_sequence"] = self.has_packed_sequence
        if self.freeze_config is not None:
            runtime_kwargs["freeze_config"] = dict(self.freeze_config)
        if self.sdpa_method is not None:
            runtime_kwargs["sdpa_method"] = list(self.sdpa_method)
        if distributed_setup is not None:
            runtime_kwargs["distributed_setup"] = distributed_setup
        if self.qat_enabled and self.qat_config is not None:
            runtime_kwargs["qat_config"] = self.qat_config.build()
        if self.fp8_config is not None:
            runtime_kwargs["fp8_config"] = deepcopy(self.fp8_config)
        if self.compile_config is not None:
            runtime_kwargs["compile_config"] = deepcopy(self.compile_config)
        if self.quantization_config is not None:
            logger.info("Model weight quantization enabled with BitsAndBytes")
            runtime_kwargs["quantization_config"] = self.quantization_config.build()

        with ScopedRNG(seed=self.seed, ranked=True):
            if self.factory_applies_infrastructure:
                model = self.model_factory(**model_kwargs, **runtime_kwargs)
            else:
                if self.sdpa_method is not None:
                    logger.warning("sdpa_method is ignored for non-NeMoAutoModel targets.")
                model = self.model_factory(**model_kwargs)
                setup = distributed_setup or DistributedSetup(mesh_context=MeshContext())
                mesh = setup.mesh_context
                pipeline_config = setup.pipeline_config
                qat_config = cast(QATConfig | None, runtime_kwargs.get("qat_config"))
                model_wrapper, autopipeline, parallelize_fn, qat_quantizer = instantiate_infrastructure(
                    distributed_config=setup.strategy_config,
                    pipeline_config=pipeline_config,
                    qat_config=qat_config,
                    moe_parallel_config=setup.moe_parallel_config,
                    activation_checkpointing=setup.activation_checkpointing,
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
                    fp8_config=runtime_kwargs.get("fp8_config"),
                    compile_config=runtime_kwargs.get("compile_config"),
                    quantization_config=runtime_kwargs.get("quantization_config"),
                    pretrained_model_name_or_path=None,
                    load_base_model=False,
                    cache_dir=hf_constants.HF_HUB_CACHE,
                )

        if unfreeze_modules:
            model_parts = model.parts if isinstance(model, AutoPipeline) else [model]
            for model_part in model_parts:
                for name, parameter in model_part.named_parameters():
                    if any(module_name in name for module_name in unfreeze_modules):
                        parameter.requires_grad_(True)
            logger.info("Unfroze parameters matching: %s", list(unfreeze_modules))

        return model


__all__ = ["ModelConfig", "ModelTargetConfig"]
