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

from __future__ import annotations

import warnings

# Suppress pydantic v2 UnsupportedFieldAttributeWarning before heavy imports
# (transformers, huggingface_hub) trigger schema generation.
try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning

    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except ImportError:
    pass

import inspect
import logging
import pathlib
import time
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import wandb
from huggingface_hub import constants as hf_constants
from torch.utils.data import DataLoader, IterableDataset
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from transformers import AutoConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_automodel._transformers import NeMoAutoModelForCausalLM, NeMoAutoModelForSequenceClassification
from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel._transformers.infrastructure import (
    apply_model_infrastructure,
    instantiate_infrastructure,
)
from nemo_automodel._transformers.mfu import AutoMFU
from nemo_automodel._transformers.utils import apply_cache_compatibility_patches
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.llm.megatron.sampler import create_megatron_sampler
from nemo_automodel.components.datasets.llm.megatron_dataset import MegatronPretraining
from nemo_automodel.components.datasets.llm.packed_sequence import pack_dataset
from nemo_automodel.components.distributed import build_distributed
from nemo_automodel.components.distributed.config import MegatronFSDPConfig
from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.distributed.utils import FirstRankPerNode
from nemo_automodel.components.loggers.comet_utils import build_comet
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.metric_logger import MetricsSample, build_metric_logger
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.quantization.fp8 import build_fp8_config
from nemo_automodel.components.training.rng import ScopedRNG, StatefulRNG
from nemo_automodel.components.training.utils import count_tail_padding
from nemo_automodel.components.utils.compile_utils import (
    build_compile_config,
)
from nemo_automodel.components.utils.flops_utils import calculate_mfu
from nemo_automodel.components.utils.model_utils import (
    _supports_logits_to_keep,
    _supports_seq_lens,
    resolve_trust_remote_code,
)
from nemo_automodel.recipes._component_builders import (
    build_checkpoint_config,
    build_loss_fn,
    build_lr_scheduler,
    build_mlflow,
    build_optimizer,
    build_step_scheduler,
    build_wandb,
)
from nemo_automodel.recipes._dist_setup import setup_distributed
from nemo_automodel.recipes.base_recipe import BaseRecipe
from nemo_automodel.shared.te_patches import apply_te_patches

if TYPE_CHECKING:
    from torch.optim import Optimizer

logger = logging.getLogger(__name__)


# ── Stateless helpers moved to components ──────────────────────────────
# Bodies now live under components/ and _transformers/. Re-imported here so
# external callers (sibling recipes — kd.py, train_seq_cls.py, dllm —, tests,
# RL frameworks like verl/NeMo-RL) keep working without an import-path change.

from nemo_automodel._transformers.auto_tokenizer import (  # noqa: E402, F401
    _build_tokenizer,
    _get_model_name,
    compute_trust_remote_code_from_model,
)
from nemo_automodel.components.datasets.llm.build import (  # noqa: E402, F401
    _get_num_thd_chunks,
    _uses_te_dot_product_attention,
    _uses_thd_collater,
    build_dataloader,
    build_validation_dataloader,
)
from nemo_automodel.components.loss.calculate import calculate_loss  # noqa: E402, F401
from nemo_automodel.components.loss.mtp import calculate_mtp_loss  # noqa: E402, F401
from nemo_automodel.components.training.build import build_model as _build_model_impl  # noqa: E402


_NEMO_AUTOMODEL_LLM_TARGETS = None  # lazy


def _is_nemo_automodel_llm_target(target):
    """``True`` if ``target`` is one of the ``NeMoAutoModelForCausalLM`` /
    ``NeMoAutoModelForSequenceClassification`` ``from_*`` classmethods."""
    global _NEMO_AUTOMODEL_LLM_TARGETS
    if _NEMO_AUTOMODEL_LLM_TARGETS is None:
        from nemo_automodel._transformers import (
            NeMoAutoModelForCausalLM,
            NeMoAutoModelForSequenceClassification,
        )

        _NEMO_AUTOMODEL_LLM_TARGETS = frozenset(
            {
                NeMoAutoModelForCausalLM.from_config,
                NeMoAutoModelForCausalLM.from_pretrained,
                NeMoAutoModelForSequenceClassification.from_config,
                NeMoAutoModelForSequenceClassification.from_pretrained,
            }
        )
    return target in _NEMO_AUTOMODEL_LLM_TARGETS


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
):
    """Recipe-layer wrapper around the typed :func:`components.training.build.build_model`.

    Translates recipe ``cfg_*`` ConfigNodes / configs into the typed kwargs the
    component expects. Lives here (not in components/) because the YAML-shaped
    coupling belongs at the recipe layer.
    """
    fp8_config = build_fp8_config(cfg_fp8) if cfg_fp8 is not None else None
    compile_config = build_compile_config(cfg_compile) if cfg_compile is not None else None

    quantization_config = None
    if cfg_quantization is not None:
        from nemo_automodel.components.quantization.qlora import create_bnb_config

        logger.info("Model weight quantization enabled with BitsAndBytes")
        quantization_config = create_bnb_config(cfg_quantization)

    qat_config = None
    if cfg_qat is not None and cfg_qat.get("enabled", False):
        if cfg_peft is not None:
            raise ValueError("QAT with PEFT is not currently supported")
        if (qat_attr := getattr(cfg_qat, "qat_config", None)) is not None:
            qat_config = qat_attr.instantiate()
        elif (quantizer_attr := getattr(cfg_qat, "quantizer", None)) is not None:
            qat_config = quantizer_attr.instantiate()

    return _build_model_impl(
        model_factory=cfg_model.instantiate,
        model_kwargs={},
        is_nemo_auto_model=_is_nemo_automodel_llm_target(cfg_model.get("_target_", None)),
        peft_config=cfg_peft,
        seed=seed,
        has_packed_sequence=has_packed_sequence,
        fp8_config=fp8_config,
        compile_config=compile_config,
        quantization_config=quantization_config,
        device_mesh=device_mesh,
        moe_mesh=moe_mesh,
        distributed_config=distributed_config,
        pipeline_config=pipeline_config,
        qat_config=qat_config,
        moe_config=_coerce_moe_config(cfg_moe),
        activation_checkpointing=activation_checkpointing,
        unfreeze_modules=unfreeze_modules,
        sdpa_method=sdpa_method,
    )


def _coerce_moe_config(moe_config):
    """Accept either ``MoEParallelizerConfig`` or a ConfigNode-shaped dict."""
    if moe_config is None:
        return None
    from nemo_automodel.components.moe.config import MoEParallelizerConfig

    if isinstance(moe_config, MoEParallelizerConfig):
        return moe_config
    moe_dict = moe_config.to_dict() if hasattr(moe_config, "to_dict") else dict(moe_config)
    moe_dict.pop("activation_checkpointing", None)
    moe_dict.pop("_target_", None)
    return MoEParallelizerConfig(**moe_dict)


# ---------------------------------------------------------------------------
#  Trainer class – orchestration only
# ---------------------------------------------------------------------------


class TrainFinetuneRecipeForNextTokenPrediction(BaseRecipe):
    """Recipe for fine-tuning a model for next-token prediction.

    This class orchestrates training, from setup to main training loop.
    """

    def __init__(self, cfg):
        """Initialize the recipe with configuration.

        Args:
            cfg: Configuration dictionary/object for training.
        """
        self.cfg = cfg

    # ------------------ build phase ------------------
    def setup(self):
        """Builds all components needed for training/validation/logging/checkpointing/etc.

        This is the last place where self.cfg should be referenced.

        Raises:
            NotImplemented: Raises if it tries to restore a checkpoint; will be removed.
        """
        torch.cuda.reset_peak_memory_stats()
        self.dist_env = build_distributed(self.cfg.get("dist_env", {}))
        # setups logging and adds the rankfilter to logging
        setup_logging()

        apply_cache_compatibility_patches()
        apply_te_patches()
        # Set up the stateful random number generator
        self.rng = StatefulRNG(seed=self.cfg.get("seed", 42), ranked=True)
        # Enable NVTX patching only when explicitly requested in config
        self.enable_nvtx = bool(self.cfg.get("nvtx", False))

        self.dist_setup = setup_distributed(self.cfg, world_size=self.dist_env.world_size)
        self.distributed_config = self.dist_setup.strategy_config
        self.device_mesh = self.dist_setup.device_mesh
        self.moe_mesh = self.dist_setup.moe_mesh
        self.pp_enabled = self.dist_setup.pp_enabled
        self.pipeline_config = self.dist_setup.pipeline_config

        if self.dist_env.is_main and hasattr(self.cfg, "wandb"):
            suppress_wandb_log_messages()
            run = build_wandb(self.cfg)
            logging.info("🚀 View run at {}".format(run.url))

        self.mlflow_logger = None
        if self.dist_env.is_main and hasattr(self.cfg, "mlflow"):
            self.mlflow_logger = build_mlflow(self.cfg)
            self.mlflow_logger.log_params(self.cfg.to_dict())
            logging.info("MLflow experiment tracking enabled")

        self.comet_logger = None
        if self.dist_env.is_main and hasattr(self.cfg, "comet"):
            self.comet_logger = build_comet(self.cfg)
            self.comet_logger.log_params(self.cfg.to_dict())
            logging.info("Comet experiment tracking enabled")

        # Log experiment details on main rank
        self._log_experiment_details()
        self._log_library_versions()

        # Build loss_fn (will be set on pipeline_config if PP enabled)
        self.loss_fn = build_loss_fn(self.cfg.loss_fn)

        # Pipeline runtime fields: override pp_batch_size and pp_microbatch_size
        if self.pp_enabled:
            pp_batch_size = self.cfg.step_scheduler.local_batch_size
            pp_microbatch_size = self.cfg.get("distributed.pipeline.pp_microbatch_size", 1)

            assert pp_batch_size // pp_microbatch_size >= self.dist_setup.pp_size, (
                f"pp_batch_size {pp_batch_size} // pp_microbatch_size {pp_microbatch_size} must be >= pp_size {self.dist_setup.pp_size}"
            )

            # THD override logic
            if (
                self.dist_setup.cp_size > 1
                and _uses_te_dot_product_attention(self.cfg.model)
                and _uses_thd_collater(self.cfg.dataloader)
            ):
                pp_microbatch_size = 1
                pp_batch_size = pp_batch_size // self.cfg.get("distributed.pipeline.pp_microbatch_size", 1)
                logging.info(
                    f"Overriding pp_batch_size: {pp_batch_size}, pp_microbatch_size: {pp_microbatch_size} for THD"
                )

            assert not isinstance(self.distributed_config, MegatronFSDPConfig), (
                "MegatronFSDPConfig is not supported when pipeline parallelism is enabled"
            )

            # Update pipeline_config runtime fields
            self.pipeline_config.pp_batch_size = pp_batch_size
            self.pipeline_config.pp_microbatch_size = pp_microbatch_size
            self.pipeline_config.patch_stage_backward_maybe_with_nosync = self.cfg.get(
                "model.backend.enable_fsdp_optimizations", False
            )
            self.pipeline_config.loss_fn = self.loss_fn

            # Infer pp_seq_len from dataset config if not explicitly set
            if hasattr(self.pipeline_config, "pp_seq_len") and self.pipeline_config.pp_seq_len is None:
                packed_seq_size = self.cfg.get("packed_sequence.packed_sequence_size", 0)
                if packed_seq_size > 0:
                    self.pipeline_config.pp_seq_len = packed_seq_size
                elif self.cfg.get("dataset.seq_len", None) is not None:
                    self.pipeline_config.pp_seq_len = self.cfg.dataset.seq_len

        # Build components
        self.peft_config = None
        if self.cfg.get("peft", None) is not None:
            self.peft_config = self.cfg.peft.instantiate()

        # Build checkpoint config
        checkpoint_config = build_checkpoint_config(
            self.cfg.get("checkpoint", None),
            self.cfg.get("model.cache_dir", None),
            _get_model_name(self.cfg.model),
            True if self.cfg.get("peft", None) else False,
        )

        if self.cfg.get("clip_grad_norm.max_norm", None) is not None:
            self.max_grad_norm = float(self.cfg.clip_grad_norm.max_norm)
        else:
            logging.info("No clip_grad_norm.max_norm specified in config, using default value of 1.0")
            self.max_grad_norm = 1.0

        # Create Checkpointer instance
        self.checkpointer = Checkpointer(
            config=checkpoint_config,
            dp_rank=self._get_dp_rank(include_cp=True),
            tp_rank=self._get_tp_rank(),
            pp_rank=self._get_pp_rank(),
            moe_mesh=self.moe_mesh,
        )

        # Disable fused RoPE when context parallelism is enabled (cp > 1)
        if self.dist_setup.cp_size > 1 and self.cfg.get("model.backend.rope_fusion", False):
            logging.info("Disabling rope_fusion because cp_size=%d > 1", self.dist_setup.cp_size)
            self.cfg.model.backend.rope_fusion = False

        model = build_model(
            self.cfg.model,
            self.peft_config,
            has_packed_sequence=self.cfg.get("packed_sequence.packed_sequence_size", 0) > 0,
            seed=self.cfg.get("seed", 42),
            cfg_fp8=self.cfg.get("fp8", None),
            cfg_compile=self.cfg.get("compile", None),
            cfg_quantization=self.cfg.get("quantization", None),
            device_mesh=self.device_mesh,
            moe_mesh=self.moe_mesh,
            distributed_config=self.distributed_config,
            pipeline_config=self.pipeline_config,
            cfg_qat=self.cfg.get("qat", None),
            cfg_moe=self.dist_setup.moe_config,
            activation_checkpointing=self.dist_setup.activation_checkpointing,
            sdpa_method=self.cfg.get("sdpa_method", None),
        )
        self.optimizer = build_optimizer(model, self.cfg.optimizer, self.distributed_config, self.device_mesh)

        if not _supports_logits_to_keep(model) and not isinstance(self.loss_fn, MaskedCrossEntropy):
            logger.warning("logits_to_keep not found in model.forward. Using MaskedCrossEntropy instead.")
            self.loss_fn = MaskedCrossEntropy()

        if isinstance(model, AutoPipeline):
            self.model_parts = model.parts
            self.pp = model
            if self.enable_nvtx:
                import nemo_automodel.autonvtx as autonvtx

                # Patch each pipeline stage with NVTX profiling
                for i, part in enumerate(self.model_parts):
                    autonvtx.patch(part, name=f"PipelineStage_{i}")
        else:
            if self.enable_nvtx:
                import nemo_automodel.autonvtx as autonvtx

                # Patch model with NVTX profiling
                autonvtx.patch(model, name=model.__class__.__name__)
            self.model_parts = [model]
            self.pp = None

        # Extract TE FP8 config from model backend (set after model construction)
        self.te_fp8 = self.model_parts[0].backend.te_fp8 if hasattr(self.model_parts[0], "backend") else None

        _packed_seq_size = self.cfg.get("packed_sequence.packed_sequence_size", 0)
        if self.dist_setup.cp_size > 1 and _packed_seq_size > 0:
            _m = self.model_parts[0]
            if hasattr(_m, "supports") and not _m.supports_cp_with_sequence_packing:
                raise ValueError(
                    f"Context parallelism (cp_size={self.dist_setup.cp_size}) with packed sequences "
                    f"is not supported for {type(_m).__name__}.\n"
                    f"Either disable sequence packing:\n"
                    f"  packed_sequence:\n"
                    f"    packed_sequence_size: 0\n"
                    f"or switch to the TE attention backend -- MoE models only:\n"
                    f"  model:\n"
                    f"    backend:\n"
                    f"      attn: te"
                )

        self.dataloader, self.tokenizer = build_dataloader(
            self.cfg.dataset,
            self.cfg.dataloader,
            self.cfg.model,
            self.cfg.get("packed_sequence", None),
            seed=self.cfg.get("seed", 42),
            local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
            global_batch_size=self.cfg.get("step_scheduler.global_batch_size", 1),
            max_steps=self.cfg.get("step_scheduler.max_steps", None),
            val_check_interval=self.cfg.get("step_scheduler.val_every_steps", None),
            dp_rank=self._get_dp_rank(),
            dp_world_size=self._get_dp_group_size(),
            pp_enabled=self.pp_enabled,
            cp_size=self.cfg.get("distributed.cp_size", 1),
            model=self.model_parts[0],
        )
        self.val_dataloaders = build_validation_dataloader(
            self.cfg,
            self._get_dp_group_size(),
            self._get_dp_rank(),
            self.pp_enabled,
            model=self.model_parts[0],
        )
        self.best_metric_key = self.cfg.get("checkpoint.best_metric_key", "default")
        # Scheduler
        self.step_scheduler = build_step_scheduler(
            self.cfg.get("step_scheduler", None),
            self.dataloader,
            self._get_dp_group_size(),
            local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
        )
        self._setup_garbage_collection(self.step_scheduler)

        # Build learning rate scheduler
        self.lr_scheduler = build_lr_scheduler(self.cfg.get("lr_scheduler", None), self.optimizer, self.step_scheduler)

        # Log model, parameter counts, norms, optimizer and scheduler
        self._log_model_and_optimizer_details(self.model_parts, self.optimizer, self.lr_scheduler)

        # Handle delayed fake-quant toggling for QAT if configured
        self._qat_disable_fn, self._qat_enable_fn, self._qat_enable_after = self._setup_qat(self.cfg, self.model_parts)

        # Enable MoE load balance tracking if configured
        moe_metrics_cfg = self.cfg.get("moe_metrics", None)
        if moe_metrics_cfg and moe_metrics_cfg.get("enabled", False):
            from nemo_automodel.components.moe.load_balance_metrics import enable_load_balance_tracking

            for mp in self.model_parts:
                enable_load_balance_tracking(mp)

        self.mfu_calculator = AutoMFU.from_config(self.model_parts[0])

        # NEFTune: noisy embeddings for improved instruction fine-tuning
        neftune_cfg = self.cfg.get("neftune", None)
        self.neftune = None
        if neftune_cfg is not None:
            from nemo_automodel.components.training.neftune import NEFTune

            noise_alpha = neftune_cfg.get("noise_alpha", 5.0) if hasattr(neftune_cfg, "get") else neftune_cfg
            self.neftune = NEFTune(noise_alpha=float(noise_alpha))
            self.neftune.activate(self.model_parts[0])

        restore_from = self.cfg.get("checkpoint.restore_from", None)
        # Initialize JSONL loggers
        self.metric_logger_train = build_metric_logger(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "training.jsonl"
        )
        self.metric_logger_valid = {
            name: build_metric_logger(
                pathlib.Path(self.checkpointer.config.checkpoint_dir)
                / (f"validation_{name}.jsonl" if name != "default" else "validation.jsonl")
            )
            for name in self.val_dataloaders.keys()
        }

        # Optionally resume
        self.load_checkpoint(restore_from)
        torch.cuda.empty_cache()

        # Log step scheduler details
        self._log_step_scheduler_details(self.step_scheduler)

        # ── Construct Engine, injecting recipe-built state. ──────────
        # build_model / build_optimizer / build_lr_scheduler ran above; we
        # don't call engine.build(). The Engine just gives us a stable train-
        # step surface (forward_backward / optimizer_step / lr_scheduler_step).
        from nemo_automodel.engine import Engine

        self.engine = Engine(Engine.Config(
            model=self.cfg.model,
            distributed=self.dist_setup,
            optimizer=self.cfg.optimizer,
            lr_scheduler=self.cfg.get("lr_scheduler", None),
            max_grad_norm=self.max_grad_norm,
            moe=self.dist_setup.moe_config,
            defer_fsdp_grad_sync=getattr(self.distributed_config, "defer_fsdp_grad_sync", True),
        ))
        # Inject already-built state — bypasses engine.build().
        self.engine.model = self.pp if self.pp is not None else self.model_parts[0]
        self.engine.optimizer = self.optimizer[0] if isinstance(self.optimizer, list) else self.optimizer
        self.engine.lr_scheduler = (
            self.lr_scheduler[0] if isinstance(self.lr_scheduler, list) and self.lr_scheduler else None
        )
        self.engine.mesh = self.dist_setup
        # CP/THD shaping for the recipe's model + dataloader combo.
        self.engine.cp_use_te = (
            _uses_te_dot_product_attention(self.model_parts[0]) and _uses_thd_collater(self.cfg.dataloader)
        )
        self.engine.cp_padding_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        self.engine.cp_num_chunks = _get_num_thd_chunks(self.pp_enabled, self.cfg)
        # FP8 autocast (TE) — engine wraps forward in this context when set.
        self.engine.fp8_autocast = self.te_fp8.maybe_te_autocast if self.te_fp8 is not None else None
        # MTP (Multi-Token Prediction) auxiliary loss — kicks in only when the
        # model emits mtp_per_depth_h. Closure captures self.loss_fn so MTP
        # depths use the same loss class as the main path.
        def _mtp_extra_loss(out, model, labels, num_label_tokens):
            mtp_per_depth_h = getattr(out, "mtp_per_depth_h", None)
            if mtp_per_depth_h is None:
                return None
            return calculate_mtp_loss(
                self.loss_fn,
                mtp_per_depth_h=mtp_per_depth_h,
                labels=labels,
                model=model,
                scaling_factor=out.mtp_loss_scaling_factor,
                num_label_tokens=num_label_tokens,
            )
        self.engine.extra_loss_fn = _mtp_extra_loss

    def _collect_moe_load_balance(self):
        """Collect MoE load balance metrics with DP all-reduce.

        Must be called on ALL ranks (the all-reduce is collective).
        Stores the result in ``self._moe_layer_loads`` for rank-0 logging.
        """
        moe_metrics_cfg = self.cfg.get("moe_metrics", None)
        if not (moe_metrics_cfg and moe_metrics_cfg.get("enabled", False)):
            self._moe_layer_loads = None
            return

        from nemo_automodel.components.moe.load_balance_metrics import collect_expert_loads

        dp_group = self._get_dp_group(include_cp=True)
        all_loads: dict = {}
        for mp in self.model_parts:
            all_loads.update(collect_expert_loads(mp, dp_group=dp_group))
        self._moe_layer_loads = all_loads if all_loads else None

    def _log_moe_metrics(self, step: int, wandb_log_fn) -> None:
        """Log MoE load balance metrics to wandb.

        Call after :meth:`_collect_moe_load_balance`.  Only logs when
        ``_moe_layer_loads`` is populated and a wandb log function is provided.

        Args:
            step: Current training/benchmark step for wandb x-axis.
            wandb_log_fn: Callable like ``wandb.log`` or ``wandb_run.log``.
        """
        if not getattr(self, "_moe_layer_loads", None):
            return

        from nemo_automodel.components.moe.load_balance_metrics import (
            compute_brief_metrics,
            compute_detailed_metrics,
        )

        moe_metrics_cfg = self.cfg.get("moe_metrics", None)
        mode = moe_metrics_cfg.get("mode", "brief") if moe_metrics_cfg else "brief"
        top_k = moe_metrics_cfg.get("top_k_experts", 0) if moe_metrics_cfg else 0
        if mode == "detailed":
            detailed_every = moe_metrics_cfg.get("detailed_every_steps", None) if moe_metrics_cfg else None
            if detailed_every is None or step % detailed_every == 0:
                wandb_log_fn(compute_detailed_metrics(self._moe_layer_loads, top_k=top_k), step=step)
            else:
                wandb_log_fn(compute_brief_metrics(self._moe_layer_loads, top_k=top_k), step=step)
        else:
            wandb_log_fn(compute_brief_metrics(self._moe_layer_loads, top_k=top_k), step=step)

    def _setup_qat(self, cfg, model_parts: list[nn.Module]):
        if not cfg.get("qat.enabled", False):
            return None, None, None
        from nemo_automodel.components.quantization.qat import (
            get_disable_fake_quant_fn,
            get_enable_fake_quant_fn,
        )

        qat_cfg = cfg.qat
        _qat_enable_after = qat_cfg.get("fake_quant_after_n_steps", 0)
        # Collect mode from any model part that has it
        qat_mode = getattr(model_parts[0], "_qat_mode", None)

        if qat_mode is None:
            return None, None, None

        _qat_disable_fn = get_disable_fake_quant_fn(qat_mode)
        _qat_enable_fn = get_enable_fake_quant_fn(qat_mode)
        if _qat_disable_fn is not None and _qat_enable_after is not None:
            try:
                # start with fake-quant disabled, will enable later
                for part in model_parts:
                    _qat_disable_fn(part)
                logger.info("QAT fake-quant disabled initially; will enable after %s steps", _qat_enable_after)
            except Exception as e:
                logger.warning("Failed to disable fake-quant at setup: %s", e)
        return _qat_disable_fn, _qat_enable_fn, _qat_enable_after

    def _enable_qat_if_delayed(self, step: int):
        if getattr(self, "_qat_enable_after", None) is None:
            return
        if step < self._qat_enable_after or self._qat_enable_fn is None:
            return
        try:
            for mp in self.model_parts:
                self._qat_enable_fn(mp)
            logger.info("Enabled QAT fake-quant after step %s", step)
            # Enable one
            self._qat_enable_after = None
        except Exception as e:
            logger.warning("Failed to enable fake-quant: %s", e)

    # ------------------ main loop ------------------
    def run_train_validation_loop(self):
        """Run the training loop over all epochs and batches.

        For each batch, perform a forward pass, compute loss, backpropagate,
        and update model parameters when necessary. Also prints loss every gradient step.
        """
        for mp in self.model_parts:
            mp.train()
        self.timestamp = time.perf_counter()

        pbar = self._make_progress_bar()
        try:
            for epoch in self.step_scheduler.epochs:
                self.step_scheduler.set_epoch(epoch)
                # The step scheduler yields a list of batches with the following properties:
                # 1. len(batches) == grad_acc_steps
                # 2. len(batches[0]) == batch_size
                for batches in self.step_scheduler:
                    # If QAT delayed fake-quant is configured, enable after threshold
                    self._enable_qat_if_delayed(self.step_scheduler.step)
                    train_log_data = self._run_train_optim_step(batches, self.max_grad_norm)
                    # Collect MoE load balance metrics (all ranks participate in all-reduce)
                    self._collect_moe_load_balance()
                    # log
                    self.log_train_metrics(train_log_data)
                    self._update_progress_bar(pbar, train_log_data.metrics)

                    # Run validation every val_every_steps
                    val_losses = {}
                    if self.step_scheduler.is_val_step:
                        for val_name, val_dataloader in self.val_dataloaders.items():
                            val_log_data = self._run_validation_epoch(val_dataloader)
                            val_losses[val_name] = val_log_data.metrics["val_loss"]
                            self.log_val_metrics(val_name, val_log_data, self.metric_logger_valid[val_name])
                        for mp in self.model_parts:
                            mp.train()

                    # Save the checkpoint every ckpt_every_steps
                    if self.step_scheduler.is_ckpt_step:
                        self.save_checkpoint(
                            epoch,
                            self.step_scheduler.step,
                            train_log_data.metrics["loss"],
                            val_losses,
                            best_metric_key=self.best_metric_key,
                        )
                    self._maybe_collect_garbage()
        finally:
            if pbar is not None:
                pbar.close()
        # Close JSONL loggers after training loop completes
        self.metric_logger_train.close()
        for v in self.metric_logger_valid.values():
            v.close()

        self.checkpointer.close()

    # ------------------ helpers ------------------
    def _run_train_optim_step(self, batches, max_grad_norm: Optional[float] = None):
        """Execute a single training step.

        Args:
            batches: List of batches of training data.
            max_grad_norm: Gradient clipping norm. Optional, if None will not clip gradients.
        """

        num_label_tokens = torch.tensor(
            sum((batch["labels"] != -100).sum().item() for batch in batches), dtype=torch.long
        )
        num_label_tokens = self._dp_allreduce(num_label_tokens).item()

        # number of tokens in the batch, excluding any tail padding (for TPS).
        num_tokens_in_batch = torch.tensor(
            sum(batch["labels"].numel() - count_tail_padding(batch["labels"]) for batch in batches),
            dtype=torch.long,
        )
        num_tokens_in_batch = self._dp_allreduce(num_tokens_in_batch).item()

        # Allow per-call max_grad_norm override (caller passes self.max_grad_norm
        # in practice; engine reads from its own attribute on optimizer_step).
        if max_grad_norm is not None:
            self.engine.max_grad_norm = max_grad_norm

        # Engine does: prepare_for_grad_accumulation → MoE aux-loss scale →
        # microbatch loop with prepare_after_first_microbatch / prepare_for_final_backward
        # → per-microbatch CP/PP/loss/backward (matches the inline orchestration
        # this method used to do).
        self.engine.zero_grad()
        # If multiple optimizers (PP path), zero them all.
        for opt in self.optimizer[1:] if isinstance(self.optimizer, list) and len(self.optimizer) > 1 else []:
            opt.zero_grad()

        result = self.engine.forward_backward(
            batches, loss_fn=self.loss_fn, num_label_tokens=num_label_tokens,
        )
        loss_buffer = result["losses"]

        self.checkpointer.maybe_wait_for_staging()

        ok, grad_norm = self.engine.optimizer_step(num_label_tokens=num_label_tokens)
        # Step any additional optimizers (PP path with one optimizer per stage).
        for opt in self.optimizer[1:] if isinstance(self.optimizer, list) and len(self.optimizer) > 1 else []:
            opt.step()

        if hasattr(self.model_parts[0], "update_moe_gate_bias"):
            for mp in self.model_parts:
                mp.update_moe_gate_bias()

        self.engine.lr_scheduler_step()
        # Step any additional LR schedulers.
        if isinstance(self.lr_scheduler, list) and len(self.lr_scheduler) > 1:
            for scheduler in self.lr_scheduler[1:]:
                scheduler.step(1)

        # Precompute FP8 scales
        fp8_config = self.cfg.get("fp8", None)
        if (
            fp8_config is not None
            and fp8_config.get("enabled", False)
            and fp8_config.get("precompute_float8_dynamic_scale_for_fsdp", False)
            and not self.pp_enabled
            and self.device_mesh is not None
            and self.device_mesh["dp_shard"].size() > 1
        ):
            precompute_float8_dynamic_scale_for_fsdp(self.model_parts[0])

        # Note(MegatronFSDP): Need to call these functions for MegatronFSDP if not using latest api
        # self.model_parts[0].install_optimized_model_weights()
        # self.model_parts[0].zero_grad_buffer()

        t = time.perf_counter()
        time_delta = t - self.timestamp
        self.timestamp = t
        tps = num_tokens_in_batch / time_delta

        mfu = None
        mfu_calculator = getattr(self, "mfu_calculator", None)
        if batches and mfu_calculator is not None:
            step_flops = 0.0
            flops_supported = True
            for batch in batches:
                input_ids = batch.get("input_ids")
                if input_ids is None:
                    flops_supported = False
                    break
                batch_flops = mfu_calculator.get_flops(input_ids)
                if batch_flops is None:
                    flops_supported = False
                    break
                step_flops += float(batch_flops)

            if flops_supported:
                step_flops = self._dp_allreduce(
                    torch.tensor(step_flops, dtype=torch.float64, device=self.dist_env.device), include_cp=True
                ).item()
                mfu = calculate_mfu(step_flops / 1e12, self.dist_env.world_size, time_delta)

        reporting_loss = torch.sum(torch.stack(loss_buffer))
        reporting_loss = self._dp_allreduce(reporting_loss, include_cp=True)
        if self.pp_enabled:
            reporting_loss = reporting_loss / num_label_tokens
            reporting_loss = reporting_loss.to(self.dist_env.device)
            # Send loss to first rank if pp group rank is 0
            src_rank = self.device_mesh.mesh.reshape(-1)[-1].item()
            if self.dist_env.rank == src_rank:
                torch.distributed.send(reporting_loss, dst=0)
            elif self.dist_env.is_main:
                torch.distributed.recv(reporting_loss, src=src_rank)

        reporting_loss = reporting_loss.cpu().item()
        # fix reporting_loss, tps across ranks

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "loss": reporting_loss,
                "grad_norm": grad_norm,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
                "tps": tps,
                "tps_per_gpu": tps / self._get_cp_group_size() / max(self._get_dp_group_size(), 1),
                "mfu": mfu,
                "num_tokens_per_step": num_tokens_in_batch,
                "num_label_tokens": num_label_tokens,
            },
        )

    @torch.no_grad()
    def _run_validation_epoch(self, val_dataloader):
        """Run one pass over a single validation dataloader.

        Args:
            val_name: Name of the validation dataset.
            val_dataloader: DataLoader for the validation dataset.
        """
        with ScopedRNG(seed=1, ranked=True):
            for mp in self.model_parts:
                mp.eval()

            total_loss = torch.tensor(0.0, dtype=torch.float32, device=self.dist_env.device)
            total_num_label_tokens = 0

            for batch in val_dataloader:
                num_label_tokens = (batch["labels"] != -100).sum().item()
                result = self.engine.forward_backward(
                    [batch],
                    loss_fn=self.loss_fn,
                    forward_only=True,
                    num_label_tokens=None,  # we will normalize outside.
                )
                total_loss += torch.sum(torch.stack(result["losses"])).item()
                total_num_label_tokens += num_label_tokens

        total_loss = self._dp_allreduce(total_loss, include_cp=True)
        total_num_label_tokens = self._dp_allreduce(
            torch.tensor(total_num_label_tokens, dtype=torch.long, device=self.dist_env.device)
        ).item()
        val_loss = total_loss / max(total_num_label_tokens, 1e-8)

        # For PP, send val_loss and num_label_tokens from last stage to main rank
        if self.pp_enabled:
            val_loss = val_loss.to(self.dist_env.device)
            # On non-last ranks total_num_label_tokens is 0; this tensor is just a recv buffer.
            pp_num_tokens = torch.tensor(total_num_label_tokens, dtype=torch.long, device=self.dist_env.device)
            src_rank = self.device_mesh.mesh.reshape(-1)[-1].item()
            if self.dist_env.rank == src_rank:
                torch.distributed.send(val_loss, dst=0)
                torch.distributed.send(pp_num_tokens, dst=0)
            elif self.dist_env.is_main:
                torch.distributed.recv(val_loss, src=src_rank)
                torch.distributed.recv(pp_num_tokens, src=src_rank)
                total_num_label_tokens = pp_num_tokens.item()

        val_loss = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "val_loss": val_loss,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "num_label_tokens": total_num_label_tokens,
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
            },
        )

    def log_val_metrics(self, val_name, log_data, metric_logger=None):
        """Log metrics to wandb, MLflow and other loggers
        Args:
            log_data: MetricsSample object, containing:
                step: int, the current step.
                epoch: int, the current epoch.
                metrics: Dict[str, float], containing:
                    "val_loss": Validation loss.
                    "lr": Learning rate.
                    "num_label_tokens": Number of label tokens.
                    "mem": Memory allocated.
        """

        if not self.dist_env.is_main or log_data is None:
            return

        if wandb.run is not None:
            wandb.log(log_data.to_dict() | {"val_name": val_name}, step=log_data.step)

        if self.mlflow_logger is not None:
            self.mlflow_logger.log_metrics(log_data.to_dict(), step=log_data.step)

        if self.comet_logger is not None:
            self.comet_logger.log_metrics(log_data.to_dict() | {"val_name": val_name}, step=log_data.step)

        # JSONL validation log
        if not metric_logger is None:
            metric_logger.log(log_data)

        logging.info(
            '[val] name "{}" | step {} | epoch {} | loss {:.4f} | lr {:.2e} | num_label_tokens {}'.format(
                val_name,
                log_data.step,
                log_data.epoch,
                log_data.metrics["val_loss"],
                log_data.metrics["lr"],
                log_data.metrics["num_label_tokens"],
            )
        )

    def log_train_metrics(self, log_data):
        """Log metrics to wandb and other loggers.

        Args:
            log_data: MetricsSample object, containing:
                step: int, the current step.
                epoch: int, the current epoch.
                metrics: Dict[str, float], containing:
                    "loss": Training loss.
                    "grad_norm": Grad norm from the training step.
                    "lr": Learning rate.
                    "mem": Memory allocated.
                    "tps": Tokens per second.
                    "tps_per_gpu": Tokens per second per GPU.
                    "num_label_tokens": Number of label tokens.
        """
        if not self.dist_env.is_main:
            return

        # Log to remote services (WandB, MLflow, Comet) according to step_scheduler frequency
        if self.step_scheduler.is_remote_logging_step:
            if wandb.run is not None:
                wandb.log(log_data.to_dict(), step=self.step_scheduler.step)
            if self.mlflow_logger is not None:
                self.mlflow_logger.log_metrics(log_data.to_dict(), step=log_data.step)
            if self.comet_logger is not None:
                self.comet_logger.log_metrics(log_data.to_dict(), step=log_data.step)

        # Log MoE load balance metrics (already collected/reduced on all ranks)
        if self.step_scheduler.is_remote_logging_step:
            if wandb.run is not None:
                self._log_moe_metrics(self.step_scheduler.step, wandb.log)
            if self.comet_logger is not None:
                self._log_moe_metrics(
                    self.step_scheduler.step, lambda m, step: self.comet_logger.log_metrics(m, step=step)
                )

        # JSONL training log (always log for detailed local records)
        self.metric_logger_train.log(log_data)
        logging.info(
            "step {} | epoch {} | loss {:.4f} | grad_norm {:.4f} | lr {:.2e} | mem {:.2f} GiB | tps {:.2f}({:.2f}/gpu) | num_label_tokens {}".format(
                log_data.step,
                log_data.epoch,
                log_data.metrics["loss"],
                log_data.metrics["grad_norm"],
                log_data.metrics["lr"],
                log_data.metrics["mem"],
                log_data.metrics["tps"],
                log_data.metrics["tps_per_gpu"],
                log_data.metrics["num_label_tokens"],
            )
        )
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(config_path=None):
    """Main entry point for the fine-tuning recipe.

    Loads the configuration, sets up the trainer, and initiates the training loop.
    """
    if config_path is None:
        config_path = pathlib.Path(__file__).parent.resolve() / "llama_3_2_1b_hellaswag.yaml"
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
