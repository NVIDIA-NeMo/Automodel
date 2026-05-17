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

import logging
import pathlib
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
from transformers import AutoProcessor
from transformers.processing_utils import ProcessorMixin

from nemo_automodel._transformers import (
    NeMoAutoModelForCausalLM,
    NeMoAutoModelForImageTextToText,
    NeMoAutoModelForMultimodalLM,
)
from nemo_automodel._transformers.utils import apply_cache_compatibility_patches
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.llm.formatting_utils import _resolve_chat_template
from nemo_automodel.components.datasets.vlm.collate_fns import COLLATE_FNS
from nemo_automodel.components.distributed import build_distributed
from nemo_automodel.components.distributed.config import MegatronFSDPConfig
from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.distributed.utils import FirstRankPerNode, get_sync_ctx
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.metric_logger import MetricsSample, build_metric_logger
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
from nemo_automodel.components.quantization.fp8 import build_fp8_config
from nemo_automodel.components.training.rng import ScopedRNG, StatefulRNG
from nemo_automodel.components.training.utils import (
    count_tail_padding,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from nemo_automodel.components.utils.compile_utils import build_compile_config
from nemo_automodel.components.utils.model_utils import VLM_INPUT_KEYS, _supports_logits_to_keep, filter_forward_kwargs
from nemo_automodel.recipes._component_builders import (
    build_checkpoint_config,
    build_loss_fn,
    build_lr_scheduler,
    build_optimizer,
    build_step_scheduler,
    build_wandb,
)
from nemo_automodel.recipes._dist_setup import setup_distributed
from nemo_automodel.recipes.base_recipe import BaseRecipe

if TYPE_CHECKING:
    from torch.optim import Optimizer

logger = logging.getLogger(__name__)

# ---------------------------
#  Stateless helper functions
# ---------------------------


def _get_model_name(cfg_model):
    if cfg_model.get("pretrained_model_name_or_path", None) is not None:
        return cfg_model.pretrained_model_name_or_path
    elif cfg_model.get("config", None) is not None:
        if isinstance(cfg_model.config, str):
            return cfg_model.config
        return cfg_model.config.get("pretrained_model_name_or_path", None)
    else:
        return None


def build_model(
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
) -> tuple[nn.Module | AutoPipeline, list["Optimizer"]]:  # noqa: F821
    """Build and initialize a model for VLM.

    Returns:
        The instantiated model and optimizer.
    """
    with ScopedRNG(seed=seed, ranked=True):
        # Build infrastructure kwargs
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
                # activation_checkpointing is handled separately; strip config keys
                moe_dict.pop("activation_checkpointing", None)
                moe_dict.pop("_target_", None)
                kwargs["moe_config"] = MoEParallelizerConfig(**moe_dict)
            kwargs["activation_checkpointing"] = activation_checkpointing

        if cfg_fp8 is not None:
            fp8_config = build_fp8_config(cfg_fp8)
            kwargs["fp8_config"] = fp8_config
        if cfg_compile is not None:
            kwargs["compile_config"] = build_compile_config(cfg_compile)

        # Check if using NeMoAutoModel
        is_nemo_auto_model = cfg_model.get("_target_", None) in (
            NeMoAutoModelForImageTextToText.from_config,
            NeMoAutoModelForImageTextToText.from_pretrained,
            NeMoAutoModelForMultimodalLM.from_config,
            NeMoAutoModelForMultimodalLM.from_pretrained,
            NeMoAutoModelForCausalLM.from_config,
            NeMoAutoModelForCausalLM.from_pretrained,
        )

        if is_nemo_auto_model:
            # NeMoAutoModel handles infrastructure internally
            model = cfg_model.instantiate(**kwargs)
        else:
            raise ValueError(
                f"VLM finetuning requires NeMoAutoModelForImageTextToText. "
                f"Got model target: {cfg_model.get('_target_', None)}"
            )
    return model


def _chunk_vlm_media(
    pixel_values: torch.Tensor,
    image_grid: torch.Tensor,
    batch_size: int,
    n_microbatches: int,
    n_images_per_sample: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Deprecated: use ``nemo_automodel.vlm_engine.chunk_vlm_media``.

    Re-exported here for backwards compatibility with sibling recipes and
    external callers. The body lives next to :class:`VLMEngine`.
    """
    from nemo_automodel.vlm_engine import chunk_vlm_media

    return chunk_vlm_media(pixel_values, image_grid, batch_size, n_microbatches,
                            n_images_per_sample=n_images_per_sample)


def build_dataloader(
    cfg_ds,
    cfg_dl,
    pretrained_model_name_or_path,
    cfg_processor,
    device_mesh,
    seed,
    local_batch_size,
    cfg_model=None,
    cfg_ps=None,
    get_rope_index=None,
) -> tuple[DataLoader, ProcessorMixin]:
    """Build a DataLoader for the VLM dataset.

    Args:
        cfg_ds: Dataset configuration.
        cfg_dl: DataLoader configuration.
        pretrained_model_name_or_path: Pretrained model name or path for processor loading.
        cfg_processor: Processor configuration or None.
        device_mesh: Device mesh for distributed training.
        seed: Random seed.
        local_batch_size: Local batch size.
        cfg_model: Model configuration (used to detect attention backend).
        cfg_ps: Packed sequence configuration (top-level ``packed_sequence:`` section).
            When provided, takes precedence over ``dataset.packing``.
        get_rope_index: Optional ``model.get_rope_index`` callable. When provided,
            VLM neat packing computes mRoPE 3D position IDs per sample so packed
            mRoPE-aware models (Qwen2.5-VL, Qwen3-VL, ...) preserve multimodal
            position semantics across pack boundaries instead of falling back to
            plain 1D positions.

    Returns:
        The instantiated DataLoader and processor.
    """
    dist_sampler_kwargs = {
        "shuffle": cfg_dl.get("shuffle", True),
    }
    if device_mesh is not None:
        from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh

        dp_mesh = get_flat_mesh(device_mesh, "dp")
        dist_sampler_kwargs |= {
            "num_replicas": dp_mesh.size(),
            "rank": dp_mesh.get_local_rank(),
        }

    with ScopedRNG(seed=seed, ranked=True):
        processor = None
        processor_kwargs = {}

        with FirstRankPerNode():
            # Ensure the processor has a _target_ attribute too
            if (
                cfg_processor is not None
                and hasattr(cfg_processor, "instantiate")
                and hasattr(cfg_processor, "_target_")
            ):
                processor = cfg_processor.instantiate()
            elif cfg_processor is not None:
                processor_kwargs = cfg_processor.to_dict()

            # If no processor was instantiated, try AutoProcessor
            if processor is None:
                try:
                    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, **processor_kwargs)
                except Exception as e:
                    # AutoProcessor.from_pretrained internally loads AutoConfig. Configs
                    # whose layer_types length differs from num_hidden_layers trip
                    # validate_layer_type. The processor itself doesn't depend on
                    # layer_types, so relax the validator and retry once before giving up.
                    err = str(e)
                    if "num_hidden_layers" in err and ("layer_types" in err or "layer types" in err):
                        from nemo_automodel._transformers.v4_patches.layer_types import (
                            relax_layer_types_validator,
                        )

                        relax_layer_types_validator()
                        try:
                            processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, **processor_kwargs)
                        except Exception as retry_exc:
                            processor = None
                            logging.warning(
                                f"AutoProcessor not available for {pretrained_model_name_or_path} ({retry_exc}). "
                            )
                    else:
                        # Some models do not provide an AutoProcessor
                        processor = None
                        logging.warning(f"AutoProcessor not available for {pretrained_model_name_or_path} ({e}). ")

            chat_template_raw = cfg_ds.__dict__.pop("chat_template", None)
            # Update chat_template if chat_template is given
            if chat_template_raw is not None and processor is not None:
                processor.chat_template = _resolve_chat_template(chat_template_raw)
                processor.tokenizer.chat_template = processor.chat_template

            _path_or_ds = getattr(cfg_ds, "path_or_dataset", None) or cfg_ds.get("path_or_dataset", None)
            if _path_or_ds is not None:
                ds = cfg_ds.instantiate(path_or_dataset=_path_or_ds)
            else:
                ds = cfg_ds.instantiate()

        # Resolve packing config: top-level packed_sequence (LLM-style) takes
        # precedence over legacy dataset.packing (backward compat).
        if cfg_ps is not None:
            _ps_enabled = getattr(cfg_ps, "pack_size", 0) > 0
            packing_cfg = cfg_ps if _ps_enabled else None
            pretokenize = getattr(cfg_ps, "pretokenize", _ps_enabled)
            max_length = getattr(cfg_ps, "max_length", None)
        else:
            _legacy = cfg_ds.get("packing", None)
            _ps_enabled = _legacy is not None and _legacy.get("enabled", False)
            packing_cfg = _legacy if _ps_enabled else None
            max_length = cfg_ds.get("max_length", None)
            pretokenize = cfg_ds.get("pretokenize", max_length is not None)

        if pretokenize:
            from nemo_automodel.components.datasets.vlm.collate_fns import pad_collate_fn
            from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapper

            ds_raw = ds
            truncate = cfg_ds.get("truncate", max_length is not None)

            post_tokenize_hook = cfg_ps.get("post_tokenize_hook_fn", None) if cfg_ps is not None else None

            ds = PreTokenizedDatasetWrapper(
                ds_raw,
                processor,
                max_length=max_length,
                truncate=truncate,
                post_tokenize_hook=post_tokenize_hook,
            )

            if packing_cfg:
                from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater
                from nemo_automodel.components.datasets.vlm.neat_packing_vlm import neat_pack_dataset_vlm
                from nemo_automodel.components.models.common.packing import configure_packing, get_attn_implementation

                ds = neat_pack_dataset_vlm(
                    ds,
                    pack_size=packing_cfg.get("pack_size", max_length),
                    padding_idx=getattr(processor.tokenizer, "pad_token_id", 0) or 0,
                    drop_long_samples=packing_cfg.get("drop_long_samples", True),
                    max_packs=packing_cfg.get("max_packs", None),
                    ds_raw=ds_raw,
                    packing_ratio=packing_cfg.get("packing_ratio", 1.0),
                    processor=processor,
                    balance_media_tokens=packing_cfg.get("balance_media_tokens", True),
                    get_rope_index=get_rope_index,
                )
                _pad_id = getattr(processor.tokenizer, "pad_token_id", 0) or 0
                _collate_max_length = packing_cfg.get("collate_max_length", None)
                _attn_impl = get_attn_implementation(cfg_model)

                configure_packing(attn_implementation=_attn_impl)
                logging.info(f"Configured VLM neat packing for attn_implementation={_attn_impl}")

                collate_fn = lambda examples, _pi=_pad_id, _ml=_collate_max_length, _ai=_attn_impl: (
                    neat_packed_vlm_collater(
                        examples,
                        padding_idx=_pi,
                        max_length=_ml,
                        attn_implementation=_ai,
                    )
                )
            else:
                collate_cfg = cfg_dl.get("collate_fn", None)
                if collate_cfg:
                    collate_fn = lambda examples: collate_cfg.instantiate(examples=examples, processor=processor)
                else:
                    collate_fn = lambda examples: pad_collate_fn(examples, processor)

            sampler = torch.utils.data.distributed.DistributedSampler(
                ds,
                **dist_sampler_kwargs,
            )
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds,
                **dist_sampler_kwargs,
            )
            collate_cfg = cfg_dl.get("collate_fn", None)
            if collate_cfg:
                collate_fn = lambda examples: collate_cfg.instantiate(examples=examples, processor=processor)
            else:
                processor_type = type(processor).__name__
                if processor_type not in COLLATE_FNS:
                    logging.warning(f"You are using {processor_type} with default collate function.")
                    processor_type = "default"
                collate_fn = lambda examples: COLLATE_FNS[processor_type](examples, processor)

        if hasattr(ds, "robust_collate"):
            collate_fn = ds.robust_collate(collate_fn)

        return cfg_dl.instantiate(
            dataset=ds, sampler=sampler, collate_fn=collate_fn, batch_size=local_batch_size
        ), processor


def calculate_loss(loss_fn, **kwargs) -> torch.Tensor:
    """Calculate the loss.

    Args:
        loss_fn: Loss function.
        **kwargs: Keyword arguments for the loss function.

    Returns:
        The loss.
    """
    loss_fn_kwargs = {"num_label_tokens": kwargs.pop("num_label_tokens", None)}
    if isinstance(loss_fn, FusedLinearCrossEntropy):
        model = kwargs.pop("model")
        labels = kwargs.pop("labels")

        # find the lm_head in the model
        lm_head = None
        if hasattr(model, "get_output_embeddings"):
            lm_head = model.get_output_embeddings().weight
        else:
            for n, p in model.named_parameters(remove_duplicate=False):
                if "lm_head" in n and n.endswith(".weight"):
                    lm_head = p
                    break
        if lm_head is None:
            raise ValueError("lm_head.weight not found in model")

        # unshard the possibly sharded lm_head
        lm_head = lm_head.full_tensor() if hasattr(lm_head, "full_tensor") else lm_head
        loss_fn_kwargs.update(
            {
                "hidden_states": kwargs.pop("hidden_states"),
                "labels": labels,
                "lm_weight": lm_head,
            }
        )
    else:
        loss_fn_kwargs.update(
            {
                "logits": kwargs.pop("logits"),
                "labels": kwargs.pop("labels"),
            }
        )

    return loss_fn(**loss_fn_kwargs)


# ---------------------------------------------------------------------------
#  Trainer class – orchestration only
# ---------------------------------------------------------------------------


class FinetuneRecipeForVLM(BaseRecipe):
    """Recipe for fine-tuning a VLM model."""

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
        setup_logging()

        apply_cache_compatibility_patches()

        # Set up the stateful random number generator
        self.rng = StatefulRNG(seed=self.cfg.get("seed", 42), ranked=True)

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

        # Build components with VLM-specific functions
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
            self.cfg.get("freeze_config", None),
            self.peft_config,
            seed=self.cfg.get("seed", 42),
            cfg_fp8=self.cfg.get("fp8", None),
            cfg_compile=self.cfg.get("compile", None),
            device_mesh=self.device_mesh,
            moe_mesh=self.moe_mesh,
            distributed_config=self.distributed_config,
            pipeline_config=self.pipeline_config,
            cfg_moe=self.dist_setup.moe_config,
            activation_checkpointing=self.dist_setup.activation_checkpointing,
        )
        self.optimizer = build_optimizer(model, self.cfg.optimizer, self.distributed_config, self.device_mesh)

        if not _supports_logits_to_keep(model) and not isinstance(self.loss_fn, MaskedCrossEntropy):
            logger.warning("logits_to_keep not found in model.forward. Using MaskedCrossEntropy instead.")
            self.loss_fn = MaskedCrossEntropy()

        if isinstance(model, AutoPipeline):
            self.model_parts = model.parts
            self.pp = model
        else:
            self.model_parts = [model]
            self.pp = None

        # Extract mRoPE position-id builder from the model so VLM neat packing can
        # produce 3D position_ids per sample. Without this, packed Qwen2.5-VL /
        # Qwen3-VL training silently degrades mRoPE to plain 1D positions.
        get_rope_index = getattr(self.model_parts[0], "get_rope_index", None)

        self.dataloader, self.processor = build_dataloader(
            self.cfg.dataset,
            self.cfg.dataloader,
            _get_model_name(self.cfg.model),
            self.cfg.get("processor", None),
            device_mesh=self.device_mesh,
            seed=self.cfg.get("seed", 42),
            local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
            cfg_model=self.cfg.model,
            cfg_ps=self.cfg.get("packed_sequence", None),
            get_rope_index=get_rope_index,
        )

        # Build validation dataloader if the config provides it
        self.val_dataloader = None
        if "validation_dataset" in self.cfg:
            self.val_dataloader, _ = build_dataloader(
                self.cfg.validation_dataset,
                self.cfg.validation_dataloader,
                _get_model_name(self.cfg.model),
                self.cfg.get("processor", None),
                device_mesh=self.device_mesh,
                seed=self.cfg.get("seed", 42),
                local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
                get_rope_index=get_rope_index,
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

        restore_from = self.cfg.get("checkpoint.restore_from", None)

        # Initialize JSONL loggers
        self.metric_logger_train = build_metric_logger(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "training.jsonl"
        )
        self.metric_logger_valid = build_metric_logger(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "validation.jsonl"
        )

        # Optionally resume
        self.load_checkpoint(restore_from)

        # Log step scheduler details
        self._log_step_scheduler_details(self.step_scheduler)

        # ── Construct VLMEngine, injecting recipe-built state. ───────
        # build_model / build_optimizer / build_lr_scheduler ran above; we
        # don't call engine.build(). VLMEngine adds VLM-specific intercepts
        # (CP multimodal pre-embed, PP media chunking) via subclass hooks.
        from nemo_automodel.engine import Engine
        from nemo_automodel.vlm_engine import VLMEngine

        self.engine = VLMEngine(Engine.Config(
            model=self.cfg.model,
            distributed=self.dist_setup,
            optimizer=self.cfg.optimizer,
            lr_scheduler=self.cfg.get("lr_scheduler", None),
            max_grad_norm=self.max_grad_norm,
            moe=self.dist_setup.moe_config,
            defer_fsdp_grad_sync=getattr(self.distributed_config, "defer_fsdp_grad_sync", True),
        ))
        self.engine.model = self.pp if self.pp is not None else self.model_parts[0]
        self.engine.optimizer = self.optimizer[0] if isinstance(self.optimizer, list) else self.optimizer
        self.engine.lr_scheduler = (
            self.lr_scheduler[0] if isinstance(self.lr_scheduler, list) and self.lr_scheduler else None
        )
        self.engine.mesh = self.dist_setup
        # VLM neat-packing collators are THD-style; engine reads CP knobs from
        # the recipe's own config helpers (same as train_ft.py).
        self.engine.cp_padding_token_id = (
            getattr(self.processor, "tokenizer", None) and getattr(self.processor.tokenizer, "pad_token_id", 0)
        ) or 0
        # FP8 autocast (TE) if configured.
        self.engine.fp8_autocast = getattr(self.model_parts[0], "te_fp8", None)
        if self.engine.fp8_autocast is not None:
            self.engine.fp8_autocast = self.engine.fp8_autocast.maybe_te_autocast

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
                for batch_idx, batches in enumerate(self.step_scheduler):
                    log_data = self._run_train_optim_step(batches, self.max_grad_norm)
                    # log
                    self.log_train_metrics(log_data)
                    self._update_progress_bar(pbar, log_data.metrics)

                    val_loss = {}
                    if self.step_scheduler.is_val_step and self.val_dataloader is not None:
                        if self.pp_enabled:
                            logger.warning("Validation is not supported for pipeline parallelism")
                        else:
                            val_log_data = self._run_validation_epoch(self.val_dataloader)
                            val_loss["val_loss"] = val_log_data.metrics["val_loss"]
                            self.log_val_metrics(val_log_data)
                        for mp in self.model_parts:
                            mp.train()

                    if self.step_scheduler.is_ckpt_step:
                        self.save_checkpoint(
                            epoch,
                            self.step_scheduler.step,
                            log_data.metrics["loss"],
                            val_loss,
                            best_metric_key=self.best_metric_key,
                        )
                    self._maybe_collect_garbage()
        finally:
            if pbar is not None:
                pbar.close()

        # Close JSONL loggers after training loop completes
        self.metric_logger_train.close()
        self.metric_logger_valid.close()

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

        if max_grad_norm is not None:
            self.engine.max_grad_norm = max_grad_norm

        # VLMEngine drives: prepare_for_grad_accumulation → MoE aux-loss scale →
        # microbatch loop with VLM CP pre-embed (_pre_cp_hook) and PP media
        # chunking (_pre_pp_schedule_hook) → prepare_for_final_backward.
        self.engine.zero_grad()
        for opt in self.optimizer[1:] if isinstance(self.optimizer, list) and len(self.optimizer) > 1 else []:
            opt.zero_grad(set_to_none=True)

        result = self.engine.forward_backward(
            batches, loss_fn=self.loss_fn, num_label_tokens=num_label_tokens,
        )
        loss_buffer = result["losses"]

        self.checkpointer.maybe_wait_for_staging()

        ok, grad_norm = self.engine.optimizer_step()
        for opt in self.optimizer[1:] if isinstance(self.optimizer, list) and len(self.optimizer) > 1 else []:
            opt.step()
            opt.zero_grad(set_to_none=True)

        if hasattr(self.model_parts[0], "update_moe_gate_bias"):
            for mp in self.model_parts:
                mp.update_moe_gate_bias()

        self.engine.lr_scheduler_step()
        if isinstance(self.lr_scheduler, list) and len(self.lr_scheduler) > 1:
            for scheduler in self.lr_scheduler[1:]:
                scheduler.step(1)

        # Precompute FP8 scales
        fp8_config = self.cfg.get("fp8", None)
        if (
            fp8_config is not None
            and fp8_config.get("enabled", False)
            and fp8_config.get("precompute_float8_dynamic_scale_for_fsdp", False)
            and self.device_mesh is not None
            and self.device_mesh["dp_shard"].size() > 1
        ):
            precompute_float8_dynamic_scale_for_fsdp(self.model_parts[0])

        # Note(MegatronFSDP): Need to call these functions for MegatronFSDP if not using latest api
        # self.model.install_optimized_model_weights()
        # self.model.zero_grad_buffer()

        t = time.perf_counter()
        time_delta = t - self.timestamp
        self.timestamp = t
        tps = num_tokens_in_batch / time_delta
        reporting_loss = torch.sum(torch.stack(loss_buffer))
        reporting_loss = self._dp_allreduce(reporting_loss, include_cp=True)
        if self.pp_enabled:
            # PP uses sum reduction per microbatch (no internal normalization).
            # Divide by num_label_tokens to get the mean loss, same as non-PP.
            reporting_loss = reporting_loss / num_label_tokens if num_label_tokens > 0 else reporting_loss * 0.0
            reporting_loss = reporting_loss.float().to(self.dist_env.device)
            # Send loss to first rank from the last PP stage of rank0's mesh coords.
            # This avoids picking a global-rank sender from a different EP/PP group.
            if self.device_mesh is not None and "pp" in self.device_mesh.mesh_dim_names:
                dim_names = list(self.device_mesh.mesh_dim_names)
                mesh = self.device_mesh.mesh
                idx = []
                for name in dim_names:
                    if name == "pp":
                        idx.append(-1)
                    else:
                        idx.append(0)
                src_rank = mesh[tuple(idx)].item()
            else:
                src_rank = self.device_mesh.mesh.reshape(-1)[-1].item()
            if self.dist_env.rank == src_rank:
                torch.distributed.send(reporting_loss, dst=0)
            elif self.dist_env.is_main:
                torch.distributed.recv(reporting_loss, src=src_rank)

        reporting_loss = reporting_loss.item()
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
                "num_tokens_per_step": num_tokens_in_batch,
                "num_label_tokens": num_label_tokens,
            },
        )

    @torch.no_grad()
    def _run_validation_epoch(self, val_dataloader):
        """Run one pass over `self.val_dataloader`."""
        with ScopedRNG(seed=1, ranked=True):
            for mp in self.model_parts:
                mp.eval()

            total_loss = 0.0
            total_tokens = 0
            total_num_label_tokens = 0
            for batch in val_dataloader:
                num_label_tokens = (batch["labels"] != -100).sum().item()
                # VLMEngine's _pre_cp_hook handles multimodal CP pre-embed and
                # make_cp_batch_and_ctx is applied inside engine.forward_backward.
                result = self.engine.forward_backward(
                    [batch],
                    loss_fn=self.loss_fn,
                    forward_only=True,
                    num_label_tokens=num_label_tokens,
                )
                local_loss = result["losses"][0] if result["losses"] else torch.tensor(0.0)
                total_num_label_tokens += num_label_tokens
                total_loss += float(local_loss.item()) * num_label_tokens
                total_tokens += num_label_tokens

        # Aggregate across ranks if distributed is initialized
        total_loss = self._dp_allreduce(torch.FloatTensor([total_loss]), include_cp=True).item()
        total_tokens = self._dp_allreduce(torch.LongTensor([total_tokens]), include_cp=True).item()
        total_num_label_tokens = self._dp_allreduce(torch.LongTensor([total_num_label_tokens])).item()

        val_loss = total_loss / max(total_tokens, 1e-8)

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

    def log_val_metrics(self, log_data):
        """Log metrics to wandb and other loggers
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
            wandb.log(log_data.to_dict(), step=log_data.step)

        # JSONL validation log
        self.metric_logger_valid.log(log_data)

        logging.info(
            "[val] step {} | epoch {} | loss {:.4f} | lr {:.2e} | num_label_tokens {}".format(
                log_data.step,
                log_data.epoch,
                log_data.metrics["val_loss"],
                log_data.metrics["lr"],
                log_data.metrics["num_label_tokens"],
            )
        )

    def log_train_metrics(self, log_data) -> float:
        """Log metrics to wandb.

        Args:
            train_loss: Training loss.
            grad_norm: Grad norm from the training step.
            num_tokens_in_batch: Total number of loss tokens.
            tps: Tokens per second.
        """
        if not self.dist_env.is_main:
            return

        # Log to remote services (WandB) according to step_scheduler frequency
        if self.step_scheduler.is_remote_logging_step:
            if wandb.run is not None:
                wandb.log(log_data.to_dict(), step=self.step_scheduler.step)

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
        config_path = pathlib.Path(__file__).parent.resolve() / "gemma3" / "gemma3_vl_4b_cord_v2.yaml"
    cfg = parse_args_and_load_config(config_path)
    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
