# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal, Optional, Type

import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

from nemo_lm.automodel.components.data.hf_dataset import HFDatasetBuilder
from nemo_lm.automodel.components.loss.linear_ce import HAVE_LINEAR_LOSS_CE
from nemo_lm.automodel.components.loss.masked_ce import masked_cross_entropy
from nemo_lm.automodel.components.scheduler import OptimizerParamScheduler
from nemo_lm.automodel.utils.model_utils import JitConfig, TEConfig, jit_compile_model, te_accelerate
from nemo_lm.config.common import (
    DistributedInitConfig,
    LoggerConfig,
    ProfilingConfig,
    RNGConfig,
    TrainingConfig,
)
from nemo_lm.utils.common_utils import get_rank_safe, get_world_size_safe
from nemo_lm.utils.config_utils import ConfigContainer as Container
from nemo_lm.utils.import_utils import safe_import

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class AutoModelConfig:
    model_name: str = "gpt2"
    load_pretrained_weights: bool = True
    loss_fn: Optional[Any] = partial(masked_cross_entropy, reduction="sum")  # Target callable for loss function
    model_accelerator: Optional[TEConfig] = None  # Target callable for model acceleration (e.g., TE)
    trust_remote_code: bool = False
    default_dtype: str = "bfloat16"  # e.g., "float32", "bfloat16", "float16"
    load_in_4bit: bool = False
    attn_implementation: str = "sdpa"  # e.g., "sdpa", "eager", "flash_attention_2"
    use_liger_kernel: bool = False
    enable_grad_ckpt: bool = False
    device_map: str = "cpu"
    use_linear_ce_loss: bool = True
    make_vocab_size_divisible_by: int = 128
    jit_config: Optional[JitConfig] = None
    ddp_kwargs: Optional[dict[str, Any]] = None
    calculate_per_token_loss: bool = True
    barrier_with_L1_time: bool = False

    def __post_init__(self):
        if self.use_linear_ce_loss and not HAVE_LINEAR_LOSS_CE:
            logger.warning(
                "Dependency for linear CE loss is not available. \
                    Please refer to https://github.com/apple/ml-cross-entropy."
            )
            self.use_linear_ce_loss = False
        logger.info(f"use_linear_ce_loss: {self.use_linear_ce_loss}")

    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.default_dtype, torch.bfloat16)

    def _configure_model(self, attn_implementation):
        """Helper method to initialize and configure the model."""
        # create all your layers here
        auto_cls = AutoModelForCausalLM
        if self.use_liger_kernel:
            liger_kernel_trf, HAS_LIGER_KERNEL = safe_import("liger_kernel.transformers")
            if not HAS_LIGER_KERNEL:
                logger.warning("Asked to use Liger Kernel, but could not import")
            else:
                auto_cls = liger_kernel_trf.AutoLigerKernelForCausalLM

        quantization_config = None
        torch_dtype = self.get_torch_dtype()

        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch_dtype,
            )

        if self.load_pretrained_weights:
            m = auto_cls.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=None if self.load_in_4bit else self.device_map,
                trust_remote_code=self.trust_remote_code,
                attn_implementation=attn_implementation,
                quantization_config=quantization_config,
            )
            self.hf_config = m.config
            return m
        else:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
            dtype = getattr(config, "torch_dtype", torch_dtype)
            self.hf_config = config
            return auto_cls.from_config(
                config,
                torch_dtype=dtype,
                trust_remote_code=self.trust_remote_code,
                attn_implementation=attn_implementation,
            )

    def configure_model(self):
        """
        Configure and initialize the Hugging Face model.

        This method loads a pretrained model or creates a model from configuration
        based on the config settings. It handles attention implementation fallbacks,
        Liger kernel application, model acceleration, and gradient checkpointing.

        Returns:
            The configured model instance.

        Raises:
            Exception: If model configuration fails.
        """
        try:
            model = self._configure_model(attn_implementation=self.attn_implementation)
            logger.info(f"Configuring model with attn_implementation: {self.attn_implementation}")
        except ValueError as e:
            # 'does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention'
            if "does not support an attention" in str(e):
                logger.warning("Falling back to 'eager' attention implementation.")
                model = self._configure_model(attn_implementation="eager")
            else:
                raise e

        if self.use_liger_kernel:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance

            _apply_liger_kernel_to_instance(model=model)

        if self.model_accelerator is not None:
            te_accelerate(model, self.model_accelerator.fp8_autocast)

        if self.enable_grad_ckpt:
            if getattr(model, "supports_gradient_checkpointing", False):
                model.gradient_checkpointing_enable()
            else:
                logger.warning("Asked to use gradient checkpoint, but model does not support it")

        model.train()
        return model

    def setup(
        self,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = False,
        ddp_kwargs: Optional[dict[str, Any]] = None,
    ):
        model = self.configure_model()

        # Print number of parameters.
        if get_rank_safe() == 0:
            logger.info(model)

        # GPU allocation.
        model.cuda(torch.cuda.current_device())
        if self.jit_config is not None:
            jit_compile_model(model, self.jit_config)

        if wrap_with_ddp:
            if use_torch_fsdp2:
                ...
            else:
                device_ids = [torch.cuda.current_device()]
                ctx = torch.cuda.stream(torch.cuda.Stream()) if device_ids is not None else nullcontext()
                with ctx:
                    model = DistributedDataParallel(module=model, device_ids=device_ids, **(ddp_kwargs or {}))
        return model


@dataclass(kw_only=True)
class OptimizerConfig:
    """
    Configuration for the optimizer.
    """

    optimizer_cls: Type[torch.optim.Optimizer] | Literal["te_adam"] = torch.optim.AdamW
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)

    lr: Optional[float] = None
    """Initial learning rate. Depending on decay style and initial warmup, the learning rate at each
       iteration would be different.
    """

    min_lr: Optional[float] = None
    """Minumum value for learning rate. The scheduler clip values below this threshold."""

    weight_decay: Optional[float] = None
    """Weight decay for the optimizer."""

    barrier_with_L1_time: bool = False
    """Whether to use a barrier with L1 timer."""

    clip_grad: float = 1.0
    """Gradient clipping based on global L2 norm."""

    def __post_init__(self):
        if isinstance(self.optimizer_cls, str):
            if self.optimizer_cls == "te_adam":
                from transformer_engine.pytorch.optimizers import FusedAdam as Adam

                self.optimizer_cls = Adam
            else:
                raise ValueError(
                    f"Invalid string for optimizer class: {self.optimizer_cls}. Must be one of: 'te_adam'"
                )

    def setup(self, model: torch.nn.Module):
        optimizer = self.optimizer_cls(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.optimizer_kwargs,
        )
        return optimizer


@dataclass(kw_only=True)
class CheckpointConfig:
    # ---------------- Checkpointing config. ----------------

    save: Optional[str] = None
    """Output directory to save checkpoints to."""

    save_interval: Optional[int] = None
    """Number of iterations between persistent checkpoint saves."""

    save_optim: bool = True
    """Do not save current optimizer."""

    save_rng: bool = True
    """Do not save current rng state."""

    load: Optional[str] = None
    """Directory containing a model checkpoint."""

    load_optim: bool = True
    """Do not load optimizer when loading checkpoint."""

    load_rng: bool = True
    """Do not load rng state when loading checkpoint."""

    pretrained_checkpoint: Optional[str] = None
    """Directory containing a pretrained model checkpoint for finetuning."""


@dataclass(kw_only=True)
class SchedulerConfig:
    # ---------------- Learning rate config. ----------------
    lr_decay_style: Literal["constant", "linear", "cosine", "inverse-square-root", "WSD"] = "linear"
    """Learning rate decay function."""

    lr_wsd_decay_style: Literal["exponential", "linear", "cosine"] = "exponential"
    """Decay style for the annealing phase of WSD"""

    lr_decay_iters: Optional[int] = None
    """number of iterations to decay learning rate over, If None defaults to `--train-iters`"""

    lr_wsd_decay_iters: Optional[int] = None
    """number of iterations for the annealing phase in the wsd schedule"""

    lr_warmup_fraction: Optional[float] = None
    """fraction of lr-warmup-(iters/samples) to use for warmup (as a float)"""

    lr_warmup_iters: int = 0
    """number of iterations to linearly warmup learning rate over."""

    lr_warmup_init: float = 0.0
    """Initial value for learning rate warmup. The scheduler starts warmup from this value."""

    override_opt_param_scheduler: bool = False
    """Reset the values of the scheduler (learning rate, warmup iterations, minimum learning rate, maximum number of iterations, and decay style from input arguments and ignore values from checkpoints. Note that all the above values will be reset."""

    use_checkpoint_opt_param_scheduler: bool = False
    """Use checkpoint to set the values of the scheduler (learning rate, warmup iterations, minimum learning rate, maximum number of iterations, and decay style from checkpoint and ignore input arguments."""

    # ---------------- Regularization config. ----------------

    start_weight_decay: Optional[float] = None
    """Initial weight decay coefficient for L2 regularization."""

    end_weight_decay: Optional[float] = None
    """End of run weight decay coefficient for L2 regularization."""

    weight_decay_incr_style: Literal["constant", "linear", "cosine"] = "constant"
    """Weight decay increment function."""

    lr_warmup_steps: Optional[int] = field(init=False, default=None)
    lr_decay_steps: Optional[int] = field(init=False, default=None)
    wd_incr_steps: Optional[int] = field(init=False, default=None)
    wsd_decay_steps: Optional[int] = field(init=False, default=None)

    def __post_init__(self):
        if self.start_weight_decay is not None:
            assert self.start_weight_decay >= 0.0
            assert self.end_weight_decay >= self.start_weight_decay

        if self.override_opt_param_scheduler:
            assert not self.use_checkpoint_opt_param_scheduler, "both override and use-checkpoint are set."

    def setup(self, optimizer: torch.optim.Optimizer, lr: float, min_lr: float):
        scheduler = OptimizerParamScheduler(
            optimizer,
            init_lr=self.lr_warmup_init,
            max_lr=lr,
            min_lr=min_lr,
            lr_warmup_steps=self.lr_warmup_steps,
            lr_decay_steps=self.lr_decay_steps,
            lr_decay_style=self.lr_decay_style,
            start_wd=self.start_weight_decay,
            end_wd=self.end_weight_decay,
            wd_incr_steps=self.wd_incr_steps,
            wd_incr_style=self.weight_decay_incr_style,
            use_checkpoint_opt_param_scheduler=self.use_checkpoint_opt_param_scheduler,
            override_opt_param_scheduler=self.override_opt_param_scheduler,
            wsd_decay_steps=self.wsd_decay_steps,
            lr_wsd_decay_style=self.lr_wsd_decay_style,
        )

        return scheduler


@dataclass(kw_only=True)
class ConfigContainer(Container):
    model_config: AutoModelConfig
    train_config: TrainingConfig
    optimizer_config: OptimizerConfig
    scheduler_config: SchedulerConfig
    dataset_config: HFDatasetBuilder
    logger_config: LoggerConfig
    checkpoint_config: CheckpointConfig
    dist_config: DistributedInitConfig = field(default_factory=DistributedInitConfig)
    rng_config: RNGConfig = field(default_factory=RNGConfig)
    profiling_config: Optional[ProfilingConfig] = None

    def validate(self):
        # Distributed
        world_size = get_world_size_safe()

        # TODO: Add Model Parallel support
        total_model_size = 1
        assert (
            world_size % total_model_size == 0
        ), f"world size ({world_size}) is not divisible by total_model_size ({total_model_size})"
        self.data_parallel_size = world_size // total_model_size

        if self.dist_config.lazy_init:
            # Use CPU for model initialization in lazy mode
            self.model_config.device_map = "cpu"

        # Scheduler
        if self.scheduler_config.lr_decay_iters is None:
            self.scheduler_config.lr_decay_iters = self.train_config.train_iters
        self.scheduler_config.lr_decay_steps = (
            self.scheduler_config.lr_decay_iters * self.train_config.global_batch_size
        )
        self.scheduler_config.wd_incr_steps = self.train_config.train_iters * self.train_config.global_batch_size
        self.scheduler_config.wsd_decay_steps = None
        if self.scheduler_config.lr_wsd_decay_iters is not None:
            self.scheduler_config.wsd_decay_steps = (
                self.scheduler_config.lr_wsd_decay_iters * self.train_config.global_batch_size
            )
        if self.scheduler_config.lr_warmup_fraction is not None:
            self.scheduler_config.lr_warmup_steps = (
                self.scheduler_config.lr_warmup_fraction * self.scheduler_config.lr_decay_iters
            )
        else:
            self.scheduler_config.lr_warmup_steps = (
                self.scheduler_config.lr_warmup_iters * self.train_config.global_batch_size
            )
