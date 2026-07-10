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

"""Typed view over the recipe's YAML ``ConfigNode``.

The YAML→typed coercion happens **here**, at the recipe input boundary
(``recipe.__init__`` wraps the raw ``ConfigNode`` in ``RecipeConfig``), so the
recipe body only ever sees typed component configs and calls
``self.cfg.<section>.build(...)`` directly.

Known sections are exposed as cached, typed attributes that own a ``build()``:
``model``/``dataloader``/``tokenizer``/``wandb``/``mlflow``/``step_scheduler``/``lr_scheduler`` map to
component config dataclasses; the ``optimizer`` and ``loss_fn`` blocks resolve to a component
:class:`~nemo_automodel.components.optim.optimizer.OptimizerConfig` /
:class:`~nemo_automodel.components.loss.loss.LossConfig` via
``build_optimizer_config`` / ``build_loss_config`` (which own a ``build()``),
while the ``checkpoint`` block is coerced directly into a component
:class:`~nemo_automodel.components.checkpoint.config.CheckpointingConfig` (the
model-derived ``model_repo_id`` / ``model_cache_dir`` / ``is_peft`` are filled
in here from the surrounding YAML). Sections with no typed view (for example
``comet``) fall through to the raw ``ConfigNode`` via ``__getattr__``.

This is the recipe layer, so it is allowed to know the YAML schema (which keys
are runtime args, ``_target_`` resolution, etc.); the components themselves stay
YAML-free.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

from nemo_automodel.components.loggers.loggers import CometConfig, MLflowConfig, WandbConfig
from nemo_automodel.components.optim.optimizer import LRSchedulerConfig
from nemo_automodel.components.training.step_scheduler import StepSchedulerConfig

if TYPE_CHECKING:
    import torch.nn as nn

    from nemo_automodel._transformers.tokenizer_config import TokenizerConfig
    from nemo_automodel.components.checkpoint.config import CheckpointingConfig
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.components.datasets.diffusion.loader import DiffusionDataloaderConfig
    from nemo_automodel.components.datasets.loader import (
        BatchSamplerConfig,
    )
    from nemo_automodel.components.datasets.loader import (
        DataloaderConfig as ComponentDataloaderConfig,
    )
    from nemo_automodel.components.datasets.multimodal.loader import BagelDataloaderConfig
    from nemo_automodel.components.datasets.vlm.loader import VlmDataloaderConfig
    from nemo_automodel.components.distributed.pipelining import AutoPipeline
    from nemo_automodel.components.loss.loss import LossConfig
    from nemo_automodel.components.loss.mtp import MTPLossConfig
    from nemo_automodel.components.optim.optimizer import OptimizerConfig
    from nemo_automodel.recipes.llm.config import LlmInputConfig
    from nemo_automodel.recipes.model_config import ModelConfig
    from nemo_automodel.recipes.vlm.config import VlmInputConfig

# Keys present in the YAML ``step_scheduler:`` block that are runtime args passed
# to ``StepSchedulerConfig.build(...)`` separately (not config fields).
_STEP_SCHEDULER_RUNTIME_KEYS = ("local_batch_size", "dp_size", "dataloader")
_MISSING = object()


def _section_kwargs(node: Any) -> dict[str, Any]:
    """``**kwargs`` for a typed config from a ConfigNode section (drops ``_target_``)."""
    d = node.to_dict() if hasattr(node, "to_dict") else dict(node)
    d.pop("_target_", None)
    return d


def _as_dict(cfg: Any | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if isinstance(cfg, Mapping):
        return dict(cfg)
    raise TypeError(f"Expected a mapping-like config, got {type(cfg).__name__}")


def _callable_and_kwargs(cfg: Any) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Resolve a ``_target_``-style section to its factory callable plus kwargs."""
    if hasattr(cfg, "to_dict") or isinstance(cfg, Mapping):
        cfg_dict = _as_dict(cfg)
        target = cfg_dict.pop("_target_", None)
        if target is not None:
            return target, cfg_dict
    target = getattr(cfg, "_target_", None)
    if target is not None:
        return target, {}
    if callable(cfg):
        return cfg, {}
    if hasattr(cfg, "instantiate"):
        return cfg.instantiate, {}
    raise AttributeError("Config must provide _target_, be callable, or provide instantiate()")


def _target_kwargs(node: Any) -> tuple[Any, dict[str, Any]]:
    """``(target, kwargs)`` for a loader ``make_*`` resolver from a config node.

    ``None`` → ``(None, {})``; a bare string (a ``make_*`` registry key or dotted path) → ``(node, {})``;
    otherwise a ``_target_`` section resolved via :func:`_callable_and_kwargs`.
    """
    if node is None:
        return None, {}
    if isinstance(node, str):
        return node, {}
    return _callable_and_kwargs(node)


def _resolve_model_value(value: object) -> object:
    """Resolve model-section mappings into typed nested target configs."""
    from nemo_automodel.recipes.model_config import ModelTargetConfig

    if hasattr(value, "to_dict") or isinstance(value, Mapping):
        kwargs = _as_dict(value)
        target = kwargs.pop("_target_", None)
        resolved = {name: _resolve_model_value(item) for name, item in kwargs.items()}
        if target is None:
            return resolved
        if not callable(target):
            raise TypeError(f"Model _target_ must resolve to a callable, got {target!r}")
        return ModelTargetConfig(factory=target, kwargs=resolved)
    if isinstance(value, list):
        return [_resolve_model_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_model_value(item) for item in value)
    return value


def _model_name_from_cfg(cfg_model: Any) -> str | None:
    if cfg_model is None:
        return None
    pretrained = cfg_model.get("pretrained_model_name_or_path", None)
    if pretrained is not None:
        return pretrained
    model_config = cfg_model.get("config", None)
    if model_config is not None:
        if isinstance(model_config, str):
            return model_config
        return model_config.get("pretrained_model_name_or_path", None)
    return None


def _trust_remote_code_from_model(cfg_model: Any) -> bool:
    """Resolve the tokenizer's remote-code policy from the model config."""
    if cfg_model is None:
        return False

    direct = cfg_model.get("trust_remote_code", _MISSING)
    if direct is not _MISSING:
        return bool(direct)

    nested = cfg_model.get("config", None)
    if nested is not None and not isinstance(nested, str):
        nested_value = nested.get("trust_remote_code", _MISSING)
        if nested_value is not _MISSING:
            return bool(nested_value)

    from nemo_automodel.components.utils.model_utils import resolve_trust_remote_code

    return resolve_trust_remote_code(_model_name_from_cfg(cfg_model))


def _is_multimodal_model_target(target: object) -> bool:
    """Return whether a model target has a multimodal preprocessing contract."""
    from nemo_automodel._transformers import NeMoAutoModelForImageTextToText, NeMoAutoModelForMultimodalLM

    if target in {
        NeMoAutoModelForImageTextToText.from_config,
        NeMoAutoModelForImageTextToText.from_pretrained,
        NeMoAutoModelForMultimodalLM.from_config,
        NeMoAutoModelForMultimodalLM.from_pretrained,
    }:
        return True
    try:
        from nemo_automodel.components.models.gemma4_drafter.composite import Gemma4WithDrafter
    except ImportError:
        return False
    return target == Gemma4WithDrafter.from_pretrained


class RecipeConfig:
    """Typed view over the YAML config consumed by recipes.

    ``model``, ``dataloader``, ``tokenizer``, ``wandb``, ``mlflow``,
    ``step_scheduler``, ``lr_scheduler``, ``optimizer``, ``loss_fn`` and
    ``checkpoint`` are exposed as typed objects that own a ``.build(...)``
    (``optimizer`` is an
    :class:`~nemo_automodel.components.optim.optimizer.OptimizerConfig`,
    ``checkpoint`` a
    :class:`~nemo_automodel.components.checkpoint.config.CheckpointingConfig`);
    all other attributes delegate to the underlying ``ConfigNode``.
    """

    def __init__(self, raw: "ConfigNode"):
        self._raw = raw

    @cached_property
    def wandb(self) -> WandbConfig | None:
        node = self._raw.get("wandb", None)
        return WandbConfig.from_kwargs(**_section_kwargs(node)) if node else None

    @cached_property
    def mlflow(self) -> MLflowConfig | None:
        node = self._raw.get("mlflow", None)
        return MLflowConfig(**_section_kwargs(node)) if node else None

    @cached_property
    def comet(self) -> CometConfig | None:
        node = self._raw.get("comet", None)
        return CometConfig(**_section_kwargs(node)) if node else None

    @cached_property
    def step_scheduler(self) -> StepSchedulerConfig:
        node = self._raw.get("step_scheduler", None)
        if node is None:
            return StepSchedulerConfig()
        kwargs = {k: v for k, v in _section_kwargs(node).items() if k not in _STEP_SCHEDULER_RUNTIME_KEYS}
        return StepSchedulerConfig(**kwargs)

    @cached_property
    def lr_scheduler(self) -> LRSchedulerConfig | None:
        node = self._raw.get("lr_scheduler", None)
        return LRSchedulerConfig(**_section_kwargs(node)) if node else None

    @cached_property
    def optimizer(self) -> "OptimizerConfig" | None:
        from nemo_automodel.components.optim.optimizer import build_optimizer_config

        node = self._raw.get("optimizer", None)
        if node is None:
            return None
        factory, kwargs = _callable_and_kwargs(node)
        return build_optimizer_config(factory, kwargs)

    def _uses_multimodal_inputs(self) -> bool:
        """Return whether the configured model consumes a multimodal processor."""
        from transformers import AutoProcessor

        model_node = self._raw.get("model", None)
        model_target = _as_dict(model_node).get("_target_", None) if model_node is not None else None
        if _is_multimodal_model_target(model_target):
            return True
        if "processor" in self._raw.to_dict():
            return True
        tokenizer_node = self._raw.get("tokenizer", None)
        if tokenizer_node is None:
            return False
        return _as_dict(tokenizer_node).get("_target_", None) == AutoProcessor.from_pretrained

    @cached_property
    def tokenizer(self) -> "TokenizerConfig":
        """Resolve the recipe's tokenizer or processor factory configuration."""
        from transformers import AutoProcessor

        from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
        from nemo_automodel._transformers.tokenizer_config import TokenizerConfig

        dataset_node = self._raw.get("dataset", None)
        dataset_has_tokenizer = dataset_node is not None and "tokenizer" in _as_dict(dataset_node)
        raw_has_tokenizer = "tokenizer" in self._raw.to_dict()
        raw_has_processor = "processor" in self._raw.to_dict()
        uses_processor = self._uses_multimodal_inputs()

        if dataset_has_tokenizer:
            node = dataset_node.get("tokenizer", None)
            inject_model_trust = True
        elif raw_has_tokenizer:
            node = self._raw.get("tokenizer", None)
            inject_model_trust = False
        elif raw_has_processor:
            warnings.warn(
                "The top-level processor section is deprecated; rename it to tokenizer",
                DeprecationWarning,
                stacklevel=2,
            )
            node = self._raw.get("processor", None)
            inject_model_trust = False
        else:
            model_name = _model_name_from_cfg(self._raw.get("model", None))
            if model_name is None:
                return TokenizerConfig()
            return TokenizerConfig(
                factory=AutoProcessor.from_pretrained if uses_processor else NeMoAutoTokenizer.from_pretrained,
                kwargs={
                    "pretrained_model_name_or_path": model_name,
                    "trust_remote_code": _trust_remote_code_from_model(self._raw.get("model", None)),
                },
            )

        if node is None:
            return TokenizerConfig()

        kwargs = _as_dict(node)
        target = kwargs.pop("_target_", None)
        factory = (AutoProcessor if uses_processor else NeMoAutoTokenizer).from_pretrained if target is None else target
        if not callable(factory):
            raise TypeError(f"Tokenizer _target_ must resolve to a callable, got {factory!r}")
        if inject_model_trust:
            kwargs.setdefault("trust_remote_code", _trust_remote_code_from_model(self._raw.get("model", None)))
        if uses_processor:
            model_name = _model_name_from_cfg(self._raw.get("model", None))
            if model_name is not None:
                kwargs.setdefault("pretrained_model_name_or_path", model_name)
        return TokenizerConfig(factory=factory, kwargs=kwargs)

    def _packing_config(self):
        """Resolve the recipe's optional sequence-packing strategy config."""
        from nemo_automodel.components.datasets.loader import make_packing_config

        ps = _as_dict(self._raw.get("packed_sequence", None))
        if ps.get("packed_sequence_size", 0) > 0:
            return make_packing_config(ps.pop("packing_strategy", "thd"), ps)
        return None

    def _resolve_text_dataloader(self, dataset_node: Any, dataloader_node: Any) -> "ComponentDataloaderConfig":
        """Resolve text dataset/loader blocks into one typed dataloader config."""
        from torch.utils.data import DataLoader
        from torchdata.stateful_dataloader import StatefulDataLoader

        from nemo_automodel.components.datasets.llm.megatron.sampler import MegatronSamplerConfig
        from nemo_automodel.components.datasets.loader import (
            DataloaderConfig,
            DatasetBuildSchedule,
            ScheduledDatasetConfig,
            make_collate_fn,
            make_dataset_config,
        )

        target, dataset_kwargs = _callable_and_kwargs(dataset_node)
        dataset_kwargs.pop("tokenizer", None)
        dataset_config = make_dataset_config(target, dataset_kwargs)

        loader_kwargs = _as_dict(dataloader_node)
        loader_target = loader_kwargs.pop("_target_", None)
        supported_loader_targets = {
            None,
            "torch.utils.data.DataLoader",
            "torchdata.stateful_dataloader.StatefulDataLoader",
            DataLoader,
            StatefulDataLoader,
        }
        if loader_target not in supported_loader_targets:
            raise ValueError(
                f"Unsupported dataloader _target_ {loader_target!r}; typed recipes use ParallelAwareDataloader"
            )
        collate = make_collate_fn(*_target_kwargs(loader_kwargs.pop("collate_fn", None)))

        schedule = DatasetBuildSchedule(
            local_batch_size=self._raw.get("step_scheduler.local_batch_size", 1),
            global_batch_size=self._raw.get("step_scheduler.global_batch_size", 1),
            max_steps=self._raw.get("step_scheduler.max_steps", None),
            val_check_interval=self._raw.get("step_scheduler.val_every_steps", None),
        )
        dataloader_type = loader_kwargs.pop("dataloader_type", None)
        batch_sampler_config = None
        if isinstance(dataset_config, ScheduledDatasetConfig):
            if dataloader_type not in (None, "single", "cyclic"):
                raise ValueError(
                    f"Unsupported Megatron dataloader_type {dataloader_type!r}; expected 'single' or 'cyclic'"
                )
            batch_sampler_config = MegatronSamplerConfig(
                micro_batch_size=schedule.local_batch_size,
                global_batch_size=schedule.global_batch_size,
                dataloader_type=dataloader_type or "single",
            )
        elif dataloader_type is not None:
            raise ValueError("dataloader_type is only supported by Megatron dataset configs")

        config_fields = {
            "shuffle",
            "group_by_length",
            "shuffle_buffer_size",
            "batch_size",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
            "drop_last",
        }
        unknown = sorted(set(loader_kwargs) - config_fields)
        if unknown:
            raise TypeError(f"Unexpected dataloader config field(s): {', '.join(unknown)}")

        return DataloaderConfig(
            dataset_config=dataset_config,
            packing=self._packing_config(),
            batch_sampler_config=cast("BatchSamplerConfig | None", batch_sampler_config),
            dataset_build_schedule=schedule if isinstance(dataset_config, ScheduledDatasetConfig) else None,
            shuffle=loader_kwargs.pop("shuffle", None),
            group_by_length=loader_kwargs.pop("group_by_length", False),
            shuffle_buffer_size=loader_kwargs.pop("shuffle_buffer_size", 10000),
            batch_size=loader_kwargs.pop("batch_size", schedule.local_batch_size),
            seed=self._raw.get("seed", 42),
            collate_fn=collate,
            num_workers=loader_kwargs.pop("num_workers", 0),
            pin_memory=loader_kwargs.pop("pin_memory", False),
            persistent_workers=loader_kwargs.pop("persistent_workers", False),
            prefetch_factor=loader_kwargs.pop("prefetch_factor", None),
            drop_last=loader_kwargs.pop("drop_last", False),
        )

    def _resolve_text_validation_dataloaders(self) -> dict[str, "ComponentDataloaderConfig"]:
        """Resolve one text dataloader config per ``validation_dataset*`` block."""

        def _name(key: str) -> str:
            key = key.replace("validation_dataset", "")
            if len(key) > 1 and key[0] in ("_", "-", "."):
                key = key[1:]
            return key or "default"

        val_dl_node = self._raw.get("validation_dataloader", None)
        out: dict[str, "ComponentDataloaderConfig"] = {}
        for key in filter(lambda k: k.startswith("validation_dataset"), self._raw.to_dict().keys()):
            node = self._raw.get(key, None)
            if node is None:
                continue
            out[_name(key)] = self._resolve_text_dataloader(node, val_dl_node)
        return out

    @cached_property
    def dataloader(self) -> "LlmInputConfig | VlmInputConfig | None":
        """Resolve the recipe's complete train and validation input pipeline."""
        from nemo_automodel.components.models.common.packing import get_attn_implementation
        from nemo_automodel.recipes.llm.config import LlmInputConfig

        dataset_node = self._raw.get("dataset", None)
        if dataset_node is None:
            return None
        if self._uses_multimodal_inputs():
            from nemo_automodel.recipes.vlm.config import VlmInputConfig

            validation_node = self._raw.get("validation_dataset", None)
            validation = (
                {
                    "default": self._resolve_vlm_dataloader(
                        validation_node,
                        self._raw.get("validation_dataloader", None),
                    )
                }
                if validation_node is not None
                else {}
            )
            return VlmInputConfig(
                train=self._resolve_vlm_dataloader(
                    dataset_node,
                    self._raw.get("dataloader", None),
                    packed_sequence_node=self._raw.get("packed_sequence", None),
                ),
                validation=validation,
                batch_size=self._raw.get("step_scheduler.local_batch_size", 1),
                seed=self._raw.get("seed", 42),
                attn_implementation=get_attn_implementation(self._raw.get("model", None)),
            )
        return LlmInputConfig(
            train=self._resolve_text_dataloader(dataset_node, self._raw.get("dataloader", None)),
            validation=self._resolve_text_validation_dataloaders(),
            attn_implementation=get_attn_implementation(self._raw.get("model", None)),
        )

    def _resolve_model(self, model_key: str, *, include_recipe_features: bool) -> "ModelConfig | None":
        """Resolve a model section and its recipe-level construction settings."""
        from nemo_automodel._transformers import (
            NeMoAutoModelBiEncoder,
            NeMoAutoModelCrossEncoder,
            NeMoAutoModelForCausalLM,
            NeMoAutoModelForImageTextToText,
            NeMoAutoModelForMultimodalLM,
            NeMoAutoModelForSequenceClassification,
        )
        from nemo_automodel.components.quantization.fp8 import FP8Config
        from nemo_automodel.components.quantization.qlora import BitsAndBytesQuantizationConfig
        from nemo_automodel.components.utils.compile_utils import CompileConfig
        from nemo_automodel.recipes.model_config import ModelConfig, ModelTargetConfig

        node = self._raw.get(model_key, None)
        if node is None:
            return None
        target = _resolve_model_value(node)
        if not isinstance(target, ModelTargetConfig):
            raise TypeError(f"{model_key} must define a callable _target_")

        sequence_classification_targets = {
            NeMoAutoModelForSequenceClassification.from_config,
            NeMoAutoModelForSequenceClassification.from_pretrained,
        }
        is_sequence_classification = target.factory in sequence_classification_targets

        fp8_config = None
        compile_config = None
        quantization_config = None
        qat_enabled = False
        qat_config = None
        sdpa_method = None
        if include_recipe_features:
            compile_node = self._raw.get("compile", None)
            if compile_node is not None:
                compile_config = CompileConfig(**_section_kwargs(compile_node))

            quantization_node = self._raw.get("quantization", None)
            if quantization_node is not None:
                quantization_config = BitsAndBytesQuantizationConfig(**_section_kwargs(quantization_node))

            if not is_sequence_classification:
                fp8_node = self._raw.get("fp8", None)
                if fp8_node is not None:
                    fp8_config = FP8Config(**_section_kwargs(fp8_node))

                qat_node = self._raw.get("qat", None)
                if qat_node is not None:
                    qat_enabled = bool(qat_node.get("enabled", False))
                    if qat_enabled:
                        qat_target_node = qat_node.get("qat_config", qat_node.get("quantizer", None))
                        if qat_target_node is not None:
                            qat_config = _resolve_model_value(qat_target_node)
                            if not isinstance(qat_config, ModelTargetConfig):
                                raise TypeError("qat.qat_config must define a callable _target_")

                configured_sdpa = self._raw.get("sdpa_method", None)
                if configured_sdpa is not None:
                    sdpa_method = tuple(configured_sdpa)

        infrastructure_targets = {
            NeMoAutoModelForCausalLM.from_config,
            NeMoAutoModelForCausalLM.from_pretrained,
            NeMoAutoModelForImageTextToText.from_config,
            NeMoAutoModelForImageTextToText.from_pretrained,
            NeMoAutoModelForMultimodalLM.from_config,
            NeMoAutoModelForMultimodalLM.from_pretrained,
            NeMoAutoModelBiEncoder.from_pretrained,
            NeMoAutoModelCrossEncoder.from_pretrained,
            *sequence_classification_targets,
        }
        packed = _as_dict(self._raw.get("packed_sequence", None))
        has_packed_sequence = None
        packing_aware_target = target.factory in {
            NeMoAutoModelForCausalLM.from_config,
            NeMoAutoModelForCausalLM.from_pretrained,
            NeMoAutoModelForImageTextToText.from_config,
            NeMoAutoModelForImageTextToText.from_pretrained,
            NeMoAutoModelForMultimodalLM.from_config,
            NeMoAutoModelForMultimodalLM.from_pretrained,
        } or _is_multimodal_model_target(target.factory)
        if packing_aware_target and not is_sequence_classification:
            has_packed_sequence = bool(
                packed.get("packed_sequence_size", 0) > 0
                or packed.get("pack_size", 0) > 0
                or packed.get("enabled", False)
            )
        freeze_key = "freeze_config" if model_key == "model" else "teacher_freeze_config"
        freeze_node = self._raw.get(freeze_key, None)
        return ModelConfig(
            model_factory=cast("Callable[..., nn.Module | AutoPipeline]", target.factory),
            model_kwargs=target.kwargs,
            factory_applies_infrastructure=(
                target.factory in infrastructure_targets or _is_multimodal_model_target(target.factory)
            ),
            model_name=_model_name_from_cfg(node),
            seed=self._raw.get("seed", 42),
            has_packed_sequence=has_packed_sequence,
            freeze_config=_as_dict(freeze_node) if freeze_node is not None else None,
            fp8_config=fp8_config,
            compile_config=compile_config,
            quantization_config=quantization_config,
            qat_enabled=qat_enabled,
            qat_config=qat_config,
            sdpa_method=sdpa_method,
        )

    @cached_property
    def model(self) -> "ModelConfig":
        """Resolve the recipe model and its construction settings."""
        config = self._resolve_model("model", include_recipe_features=True)
        if config is None:
            raise ValueError("A recipe model configuration is required")
        return config

    @cached_property
    def teacher_model(self) -> "ModelConfig | None":
        """Resolve the optional KD teacher model without student-only features."""
        return self._resolve_model("teacher_model", include_recipe_features=False)

    def _resolve_vlm_dataloader(
        self,
        dataset_node: Any,
        dataloader_node: Any,
        *,
        packed_sequence_node: Any = None,
    ) -> "VlmDataloaderConfig":
        """Resolve VLM YAML sections into one typed input-pipeline config.

        Args:
            dataset_node: Dataset config node containing its ``_target_`` and declarative fields.
            dataloader_node: StatefulDataLoader config node.
            packed_sequence_node: Optional top-level VLM neat-packing config.

        Returns:
            Typed VLM input-pipeline config ready for runtime ``build`` arguments.
        """
        from torchdata.stateful_dataloader import StatefulDataLoader

        from nemo_automodel.components.datasets.loader import make_dataset_config
        from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapperConfig
        from nemo_automodel.components.datasets.vlm.loader import VlmCollatorConfig, VlmDataloaderConfig
        from nemo_automodel.components.datasets.vlm.neat_packing_vlm import NeatPackConfig

        target, dataset_kwargs = _callable_and_kwargs(dataset_node)
        chat_template = dataset_kwargs.pop("chat_template", None)
        legacy_packing = dataset_kwargs.pop("packing", None)
        dataset_pretokenize = dataset_kwargs.pop("pretokenize", None)
        dataset_max_length = dataset_kwargs.get("max_length", None)
        truncate = dataset_kwargs.pop("truncate", dataset_max_length is not None)
        inject_fake_images = dataset_kwargs.pop("inject_fake_images", True)
        dataset_config = make_dataset_config(target, dataset_kwargs)

        packed = _as_dict(packed_sequence_node)
        if packed_sequence_node is not None:
            packing_enabled = packed.get("pack_size", 0) > 0
            packing_node = packed if packing_enabled else None
            pretokenize = packed.get("pretokenize", packing_enabled)
            max_length = packed.get("max_length", dataset_max_length)
        else:
            legacy = _as_dict(legacy_packing)
            packing_enabled = bool(legacy.get("enabled", False))
            packing_node = legacy if packing_enabled else None
            pretokenize = dataset_pretokenize if dataset_pretokenize is not None else dataset_max_length is not None
            max_length = dataset_max_length

        pretokenization = None
        if pretokenize:
            post_tokenize_hook = packing_node.get("post_tokenize_hook_fn", None) if packing_node else None
            if post_tokenize_hook is not None and not callable(post_tokenize_hook):
                raise TypeError("packed_sequence.post_tokenize_hook_fn must resolve to a callable")
            pretokenization = PreTokenizedDatasetWrapperConfig(
                max_length=max_length,
                truncate=truncate,
                inject_fake_images=inject_fake_images,
                post_tokenize_hook=post_tokenize_hook,
            )

        packing = None
        if packing_node is not None:
            packing_fields = {
                "pack_size",
                "drop_long_samples",
                "max_packs",
                "packing_ratio",
                "balance_media_tokens",
                "collate_max_length",
                "attn_implementation",
                "enabled",
                "pretokenize",
                "max_length",
                "post_tokenize_hook_fn",
            }
            unknown = sorted(set(packing_node) - packing_fields)
            if unknown:
                raise TypeError(f"Unexpected VLM packing config field(s): {', '.join(unknown)}")
            packing = NeatPackConfig(
                pack_size=packing_node.get("pack_size", max_length or 2048),
                drop_long_samples=packing_node.get("drop_long_samples", False),
                max_packs=packing_node.get("max_packs", None),
                packing_ratio=packing_node.get("packing_ratio", 1.0),
                balance_media_tokens=packing_node.get("balance_media_tokens", True),
                collate_max_length=packing_node.get("collate_max_length", None),
                attn_implementation=packing_node.get("attn_implementation", None),
            )

        loader_kwargs = _as_dict(dataloader_node)
        loader_target = loader_kwargs.pop("_target_", None)
        if loader_target not in (None, "torchdata.stateful_dataloader.StatefulDataLoader", StatefulDataLoader):
            raise ValueError(f"Unsupported VLM dataloader _target_ {loader_target!r}; expected StatefulDataLoader")
        collate_node = loader_kwargs.pop("collate_fn", None)
        collator = None
        if collate_node is not None:
            factory, collate_kwargs = _target_kwargs(collate_node)
            if not callable(factory):
                raise TypeError(f"VLM collate_fn must resolve to a callable, got {factory!r}")
            collator = VlmCollatorConfig(factory=factory, kwargs=collate_kwargs)

        loader_fields = {
            "shuffle",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
            "drop_last",
        }
        unknown = sorted(set(loader_kwargs) - loader_fields)
        if unknown:
            raise TypeError(f"Unexpected VLM dataloader config field(s): {', '.join(unknown)}")

        return VlmDataloaderConfig(
            dataset_config=dataset_config,
            pretokenization=pretokenization,
            packing=packing,
            collator=collator,
            chat_template=chat_template,
            shuffle=loader_kwargs.pop("shuffle", True),
            num_workers=loader_kwargs.pop("num_workers", 0),
            pin_memory=loader_kwargs.pop("pin_memory", False),
            persistent_workers=loader_kwargs.pop("persistent_workers", False),
            prefetch_factor=loader_kwargs.pop("prefetch_factor", None),
            drop_last=loader_kwargs.pop("drop_last", False),
        )

    @staticmethod
    def resolve_diffusion_dataloader(node: Any) -> "DiffusionDataloaderConfig":
        """Resolve a diffusion dataloader YAML target to its typed config.

        Args:
            node: ``data.dataloader`` config node containing a supported legacy builder target.

        Returns:
            Typed dataloader config whose ``build`` accepts runtime rank and batch-size values.
        """
        from nemo_automodel.components.datasets.diffusion.collate_fns import (
            TextToImageDataloaderConfig,
            TextToVideoDataloaderConfig,
        )
        from nemo_automodel.components.datasets.diffusion.meta_files_dataset import MetaFilesDataloaderConfig
        from nemo_automodel.components.datasets.diffusion.mock_dataloader import MockWanDataloaderConfig

        target, kwargs = _callable_and_kwargs(node)
        module = getattr(target, "__module__", None)
        name = getattr(target, "__qualname__", getattr(target, "__name__", None))
        target_path = f"{module}.{name}" if module and name else None
        config_types = {
            "nemo_automodel.components.datasets.diffusion.collate_fns.build_text_to_image_multiresolution_dataloader": (
                TextToImageDataloaderConfig
            ),
            "nemo_automodel.components.datasets.diffusion.collate_fns.build_video_multiresolution_dataloader": (
                TextToVideoDataloaderConfig
            ),
            "nemo_automodel.components.datasets.diffusion.meta_files_dataset.build_dataloader": (
                MetaFilesDataloaderConfig
            ),
            "nemo_automodel.components.datasets.diffusion.mock_dataloader.build_mock_dataloader": (
                MockWanDataloaderConfig
            ),
        }
        config_type = config_types.get(target_path) if target_path is not None else None
        if config_type is None:
            config_type = target
        if not is_dataclass(config_type) or not hasattr(config_type, "build"):
            raise ValueError(f"Unsupported diffusion dataloader _target_ {target_path or target!r}")
        valid = {field.name for field in fields(config_type)}
        unknown = sorted(set(kwargs) - valid)
        if unknown:
            raise TypeError(f"Unexpected diffusion dataloader config field(s): {', '.join(unknown)}")
        if "base_resolution" in kwargs:
            kwargs["base_resolution"] = tuple(kwargs["base_resolution"])
        return config_type(**kwargs)

    @cached_property
    def diffusion_dataloader(self) -> "DiffusionDataloaderConfig" | None:
        """Typed diffusion dataloader config resolved from ``data.dataloader``."""
        node = self._raw.get("data.dataloader", None)
        return self.resolve_diffusion_dataloader(node) if node is not None else None

    @cached_property
    def bagel_dataloader(self) -> "BagelDataloaderConfig" | None:
        """Typed packed-dataset and dataloader config for BAGEL recipes."""
        from nemo_automodel.components.datasets.multimodal.datasets import BagelDatasetConfig
        from nemo_automodel.components.datasets.multimodal.loader import BagelDataloaderConfig

        dataset_node = self._raw.get("dataset", None)
        if dataset_node is None:
            return None
        dataset_kwargs = _as_dict(dataset_node)
        dataset_target = dataset_kwargs.pop("_target_", None)
        if dataset_target is not None:
            raise ValueError("BAGEL dataset config does not support _target_; use its typed dataset fields")
        valid_dataset_fields = {field.name for field in fields(BagelDatasetConfig)}
        unknown = sorted(set(dataset_kwargs) - valid_dataset_fields)
        if unknown:
            raise TypeError(f"Unexpected BAGEL dataset config field(s): {', '.join(unknown)}")
        dataset_config = BagelDatasetConfig(**dataset_kwargs)

        loader_kwargs = _as_dict(self._raw.get("dataloader", None))
        loader_target = loader_kwargs.pop("_target_", None)
        if loader_target is not None:
            raise ValueError("BAGEL dataloader config does not support _target_; use its typed loader fields")
        valid_loader_fields = {"num_workers", "pin_memory", "prefetch_factor"}
        unknown = sorted(set(loader_kwargs) - valid_loader_fields)
        if unknown:
            raise TypeError(f"Unexpected BAGEL dataloader config field(s): {', '.join(unknown)}")
        return BagelDataloaderConfig(
            dataset_config=dataset_config,
            num_workers=loader_kwargs.pop("num_workers", 1),
            pin_memory=loader_kwargs.pop("pin_memory", True),
            prefetch_factor=loader_kwargs.pop("prefetch_factor", 2),
        )

    @cached_property
    def loss_fn(self) -> "LossConfig" | None:
        from nemo_automodel.components.loss import build_loss_config

        node = self._raw.get("loss_fn", None)
        if node is None:
            return None
        factory, kwargs = _callable_and_kwargs(node)
        return build_loss_config(factory, **kwargs)

    @cached_property
    def mtp(self) -> "MTPLossConfig":
        # MTP loss params are model-driven (scaling_factor comes from the model
        # output / get_mtp_loss_scaling_factor; ignore_index is fixed) and are not
        # exposed via YAML.  This typed accessor just lets recipes build MTP through
        # the typed-config boundary like the other sections.
        from nemo_automodel.components.loss.mtp import MTPLossConfig

        return MTPLossConfig()

    @cached_property
    def checkpoint(self) -> "CheckpointingConfig":
        from nemo_automodel.components.checkpoint.config import CheckpointingConfig

        node = self._raw.get("checkpoint", None)
        kwargs = _as_dict(node) if node is not None else {}
        kwargs.pop("restore_from", None)  # consumed separately at load time, not a config field
        model = self._raw.get("model", None)
        # Model-derived values; YAML overrides win if explicitly set.
        kwargs |= {
            "model_repo_id": _model_name_from_cfg(model) if model is not None else None,
            "model_cache_dir": self._raw.get("model.cache_dir", None),
            "is_peft": bool(self._raw.get("peft", None)),
        }
        return CheckpointingConfig(**kwargs)

    # everything else delegates to the raw ConfigNode
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._raw, name)

    def __contains__(self, key: object) -> bool:
        return key in self._raw

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return self._raw.to_dict()

    def to_yaml_dict(self, **kwargs: Any) -> dict[str, Any]:
        if hasattr(self._raw, "to_yaml_dict"):
            return self._raw.to_yaml_dict(**kwargs)
        return self.to_dict()
