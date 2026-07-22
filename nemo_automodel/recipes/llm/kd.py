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

"""Knowledge Distillation recipe for next-token prediction with NeMo AutoModel.

This recipe fine-tunes a *student* model using the logits of a frozen *teacher* model. It
extends ``FinetuneRecipeForNextTokenPrediction`` adding:

1. teacher_model — an additional HF/NeMo model loaded in ``eval`` mode
2. kd_loss_fn    — KL-divergence between temperature-scaled distributions
3. kd_ratio      — linear mix between CE loss and KD loss

The training loop is copied from the parent class but the loss becomes:
    loss = (1-kd_ratio) * ce_loss + kd_ratio * kd_loss

Pipeline parallelism (PP) is supported. Teacher logits from every last-stage microbatch
are captured via a lightweight closure and injected into the corresponding student
pipeline microbatch.

The file exposes ``KnowledgeDistillationRecipeForNextTokenPrediction`` and a
``main`` entry-point so it can be launched exactly the same way as other recipes:

    python -m torch.distributed.run --nproc-per-node=8 \\
        nemo_automodel/recipes/llm/kd.py \\
        -c examples/llm_kd/llama3_2/llama3_2_1b_kd.yaml
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Dict, Optional

import torch
import wandb
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.distributed.config import DistributedSetup
from nemo_automodel.components.distributed.cp_sharder import ContextParallelSharder
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.loss.utils import calculate_loss
from nemo_automodel.components.training.rng import ScopedRNG
from nemo_automodel.components.training.signal_handler import DistributedSignalHandler
from nemo_automodel.components.training.utils import (
    ScopedModuleOffloading,
    count_tail_padding,
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from nemo_automodel.components.utils.model_utils import filter_forward_kwargs
from nemo_automodel.recipes.kd_utils import (
    RUN_TEACHER,
    STOP_TEACHER,
    KDMeshBridge,
    create_kd_distributed_setups,
    materialize_teacher_logits,
)
from nemo_automodel.recipes.llm.train_ft import (
    TrainFinetuneRecipeForNextTokenPrediction,
    _get_num_thd_chunks,
    build_model,
)

logger = logging.getLogger(__name__)


def _build_kd_loss_fn(cfg_kd):
    if cfg_kd is None:
        logger.info("No KD loss function provided, using KLDivLoss")
        return torch.nn.KLDivLoss(reduction="batchmean")
    return cfg_kd.instantiate()


def _build_teacher_model(
    cfg_teacher,
    seed,
    has_packed_sequence,
    distributed_setup: DistributedSetup | None = None,
    device=None,
):
    """Build and initialize the teacher model for knowledge distillation.

    Uses the same infrastructure as student model (NeMoAutoModelForCausalLM) but without
    PEFT, FP8, or QAT since the teacher should be frozen in full precision.

    Args:
        cfg_teacher: Configuration for teacher model instantiation.
        seed: Random seed for reproducibility.
        has_packed_sequence: Whether using packed sequences.
        distributed_setup: Resolved distributed topology and policy object.
        device: Device to place the teacher model on.

    Returns:
        The frozen teacher model ready for inference.

    Note:
        The `offload_teacher_model` config option is not supported with this approach.
        Device placement is handled internally by NeMoAutoModelForCausalLM infrastructure.
    """
    assert cfg_teacher is not None, "`teacher_model` section missing from YAML config"
    logger.info("Instantiating teacher model")

    # Build teacher model using the same infrastructure as student
    # but without PEFT/FP8/QAT (teacher should be frozen in full precision)
    with ScopedRNG(seed=seed, ranked=True):
        kwargs: Dict[str, Any] = {
            "has_packed_sequence": has_packed_sequence,
            "distributed_setup": distributed_setup,
        }

        teacher_model = cfg_teacher.instantiate(**kwargs)

        # Ensure the teacher model is on the correct device
        teacher_model = teacher_model.to(device)

        # Set teacher to eval mode and freeze parameters
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

        return teacher_model


def _build_teacher_model_with_pp(
    cfg_teacher,
    seed: int,
    has_packed_sequence: bool,
    pipeline_config: PipelineConfig,
    distributed_setup: DistributedSetup,
    activation_checkpointing: bool,
) -> Any:
    """Build a frozen teacher model with the supplied distributed setup.

    Teacher is built via build_model with pipeline_config so it becomes an AutoPipeline
    when PP is enabled. No PEFT/FP8/QAT. Teacher is frozen and set to eval mode.

    Logit capture stores every last-stage microbatch in schedule order.

    Args:
        cfg_teacher: Configuration for teacher model instantiation.
        seed: Random seed for reproducibility.
        has_packed_sequence: Whether using packed sequences.
        pipeline_config: Pipeline configuration for the teacher.
        distributed_setup: Distributed setup for the teacher.
        activation_checkpointing: Whether to enable activation checkpointing.

    Returns:
        The frozen teacher AutoPipeline with a ``_teacher_logits_capture`` attribute.
    """
    assert cfg_teacher is not None, "`teacher_model` section missing from YAML config"
    logger.info("Instantiating teacher model (parallelized with TP/EP/SP/PP)")

    # Mutable outer list so the closure and recipe can exchange a per-step list.
    teacher_logits_capture = [None]

    def _teacher_capture_loss_fn(logits, target, **kwargs):
        """Capture one teacher PP microbatch.

        Args:
            logits: Tensor of shape ``[microbatch, sequence, vocab]`` containing
                last-stage teacher logits.
            target: Tensor of shape ``[microbatch, sequence]`` containing labels
                supplied by the schedule and unused here.
            **kwargs: Schedule-owned scalar or tensor metadata ignored by the
                capture path. Tensor values may have arbitrary rank and axis
                order and are not inspected.

        Returns:
            Scalar zero tensor on the same dtype and device as ``logits``.
        """
        if teacher_logits_capture[0] is None:
            teacher_logits_capture[0] = []
        teacher_logits_capture[0].append(logits.detach().clone())
        return logits.new_tensor(0.0, dtype=logits.dtype)

    # Mirror the student pipeline config but swap in the capture loss_fn.
    teacher_pipeline_config = PipelineConfig(
        pp_schedule=pipeline_config.pp_schedule,
        pp_schedule_csv=pipeline_config.pp_schedule_csv,
        pp_microbatch_size=pipeline_config.pp_microbatch_size,
        pp_batch_size=pipeline_config.pp_batch_size,
        layers_per_stage=pipeline_config.layers_per_stage,
        round_virtual_stages_to_pp_multiple=pipeline_config.round_virtual_stages_to_pp_multiple,
        module_fqns_per_model_part=pipeline_config.module_fqns_per_model_part,
        patch_inner_model=pipeline_config.patch_inner_model,
        patch_causal_lm_model=pipeline_config.patch_causal_lm_model,
        patch_stage_backward_maybe_with_nosync=pipeline_config.patch_stage_backward_maybe_with_nosync,
        dtype=pipeline_config.dtype,
        scale_grads_in_schedule=pipeline_config.scale_grads_in_schedule,
        loss_fn=_teacher_capture_loss_fn,
    )
    teacher_distributed_setup = DistributedSetup(
        mesh_context=distributed_setup.mesh_context,
        strategy_config=distributed_setup.strategy_config,
        pipeline_config=teacher_pipeline_config,
        moe_parallel_config=distributed_setup.moe_parallel_config,
        activation_checkpointing=activation_checkpointing,
    )

    with ScopedRNG(seed=seed, ranked=True):
        teacher_model = build_model(
            cfg_teacher,
            cfg_peft=None,
            has_packed_sequence=has_packed_sequence,
            seed=seed,
            cfg_fp8=None,
            cfg_compile=None,
            cfg_quantization=None,
            distributed_setup=teacher_distributed_setup,
            cfg_qat=None,
        )

    # Freeze all teacher parameters.
    for part in getattr(teacher_model, "parts", [teacher_model]):
        part.eval()
        for p in part.parameters():
            p.requires_grad_(False)

    # Attach capture reference so the recipe can read teacher logits after eval.
    teacher_model._teacher_logits_capture = teacher_logits_capture
    return teacher_model


def _verify_tokenizer_compatibility(student_cfg, teacher_cfg, trust_remote_code=True):
    if student_cfg is None or teacher_cfg is None:
        raise ValueError("Student and teacher model configs are required")
    student_tokenizer = NeMoAutoTokenizer.from_pretrained(
        student_cfg.pretrained_model_name_or_path, trust_remote_code=trust_remote_code
    )
    teacher_tokenizer = NeMoAutoTokenizer.from_pretrained(
        teacher_cfg.pretrained_model_name_or_path, trust_remote_code=trust_remote_code
    )
    if student_tokenizer.vocab_size != teacher_tokenizer.vocab_size:
        raise ValueError(
            "Student and teacher tokenizers have different vocab sizes; Support will be added in the future"
        )
    if student_tokenizer.pad_token != teacher_tokenizer.pad_token:
        raise ValueError("Student and teacher tokenizers have different pad tokens")
    del student_tokenizer, teacher_tokenizer


class KnowledgeDistillationRecipeForNextTokenPrediction(TrainFinetuneRecipeForNextTokenPrediction):
    """Fine-tune a student model via knowledge distillation."""

    def _create_distributed_setup(self) -> DistributedSetup:
        self.kd_distributed_setups = create_kd_distributed_setups(self.cfg, world_size=self.dist_env.world_size)
        self.separate_meshes = self.kd_distributed_setups.separate
        if not self.separate_meshes:
            self.kd_mesh_bridge = None
            return self.kd_distributed_setups.student

        self.kd_mesh_bridge = KDMeshBridge(self.kd_distributed_setups, device=self.dist_env.device)
        self._training_process_group = self.kd_mesh_bridge.student_group
        if self.kd_mesh_bridge.is_student:
            setup = self.kd_distributed_setups.student
            setup.mesh_context.process_group = self.kd_mesh_bridge.student_group
        else:
            setup = self.kd_distributed_setups.teacher
            setup.mesh_context.process_group = self.kd_mesh_bridge.teacher_group
        return setup

    def _should_setup_training_components(self) -> bool:
        return not getattr(self, "separate_meshes", False) or self.kd_mesh_bridge.is_student

    def _setup_kd_state(self) -> None:
        self.kd_loss_fn = _build_kd_loss_fn(self.cfg.get("kd_loss_fn", None))
        self.kd_ratio = float(self.cfg.get("kd_ratio", 0.5))
        self._kd_loss_buffer = []
        self._ce_loss_buffer = []

    def _configure_teacher_pipeline(self) -> None:
        if not self.pp_enabled:
            return
        pp_batch_size = self.cfg.get("step_scheduler.local_batch_size", 1)
        pp_microbatch_size = self.cfg.get("teacher_distributed.pipeline.pp_microbatch_size", 1)
        if pp_batch_size % pp_microbatch_size != 0:
            raise ValueError(
                "Separate-mesh PP KD requires local_batch_size to be divisible by teacher pp_microbatch_size"
            )
        self.pipeline_config = replace(
            self.pipeline_config,
            pp_batch_size=pp_batch_size,
            pp_microbatch_size=pp_microbatch_size,
        )

    def setup(self):  # noqa: C901 – same complexity as parent
        """Build student & teacher, dataloaders, optimizers, etc."""
        # Right now, we only support tokenizer compatibility for the same tokenizer.
        # We will add support for different tokenizers in the future.
        _verify_tokenizer_compatibility(self.cfg.get("model", None), self.cfg.get("teacher_model", None))

        # Let the parent class build *everything* for the student first.
        super().setup()

        self._offload_teacher_model = self.cfg.get("offload_teacher_model", False)
        if getattr(self, "separate_meshes", False):
            if self._offload_teacher_model:
                raise ValueError("offload_teacher_model is not supported with separate_meshes=true")
            self._setup_kd_state()
            if self.kd_mesh_bridge.is_teacher:
                self._configure_teacher_pipeline()
                if self.pp_enabled:
                    self.teacher_model = _build_teacher_model_with_pp(
                        cfg_teacher=self.cfg.get("teacher_model", None),
                        seed=self.cfg.get("seed", 42),
                        has_packed_sequence=self.cfg.get("packed_sequence.packed_sequence_size", 0) > 0,
                        pipeline_config=self.pipeline_config,
                        distributed_setup=self.distributed_setup,
                        activation_checkpointing=self.activation_checkpointing,
                    )
                    self.teacher_pp = self.teacher_model
                else:
                    self.teacher_model = _build_teacher_model(
                        cfg_teacher=self.cfg.get("teacher_model", None),
                        seed=self.cfg.get("seed", 42),
                        has_packed_sequence=self.cfg.get("packed_sequence.packed_sequence_size", 0) > 0,
                        distributed_setup=self.distributed_setup,
                        device=self.dist_env.device,
                    )
                    self.teacher_pp = None
            else:
                self.teacher_model = None
                self.teacher_pp = None
                if self.pp_enabled:
                    schedule = self.pp.info.schedule
                    self._original_pp_loss_fn = getattr(schedule, "_loss_fn", None)
                    schedule._loss_fn = self._make_pp_kd_loss_wrapper()
            self.kd_mesh_bridge.synchronize()
            return

        teacher_device = self.dist_env.device if not self._offload_teacher_model else "cpu"

        if self.pp_enabled:
            # FusedLinearCrossEntropy needs hidden_states; the last PP stage only has logits.
            if isinstance(self.loss_fn, FusedLinearCrossEntropy):
                raise ValueError(
                    "Pipeline parallelism with KD requires a loss that uses only logits and labels "
                    "(e.g. MaskedCrossEntropy). FusedLinearCrossEntropy is not supported for PP KD."
                )
            self.teacher_model = _build_teacher_model_with_pp(
                cfg_teacher=self.cfg.get("teacher_model", None),
                seed=self.cfg.get("seed", 42),
                has_packed_sequence=self.cfg.get("packed_sequence.packed_sequence_size", 0) > 0,
                pipeline_config=self.pipeline_config,
                distributed_setup=self.distributed_setup,
                activation_checkpointing=self.activation_checkpointing,
            )
            self.teacher_pp = self.teacher_model
        else:
            self.teacher_model = _build_teacher_model(
                cfg_teacher=self.cfg.get("teacher_model", None),
                seed=self.cfg.get("seed", 42),
                has_packed_sequence=self.cfg.get("packed_sequence.packed_sequence_size", 0) > 0,
                distributed_setup=self.distributed_setup,
                device=teacher_device,
            )
            self.teacher_pp = None

        logger.info("Teacher Model: " + str(self.teacher_model))

        # KD
        self._setup_kd_state()
        logger.info("KD Loss config: " + str(self.cfg.get("kd_loss_fn", None)))
        temperature = getattr(self.kd_loss_fn, "temperature", "N/A")
        logger.info(f"Knowledge-distillation enabled: ratio={self.kd_ratio}, T={temperature}")

        if self.pp_enabled:
            schedule = self.pp.info.schedule
            # Schedule objects expose _loss_fn (e.g. ScheduleInterleaved1F1B uses _loss_fn).
            self._original_pp_loss_fn = getattr(schedule, "_loss_fn", None)
            schedule._loss_fn = self._make_pp_kd_loss_wrapper()

    def _make_pp_kd_loss_wrapper(self):
        """Return a student pipeline loss_fn that combines CE and KD using teacher logits.

        The wrapper reads ``self._current_teacher_logits`` which must be populated by
        the teacher eval pass before each student step in ``_forward_backward_step_pp``.
        """
        recipe_ref = self

        def pp_kd_loss_fn(logits, target, **kwargs):
            """Combine CE and KD for one student PP microbatch.

            Args:
                logits: Tensor of global shape
                    ``[microbatch, sequence, vocab]`` containing student logits.
                    Under TP, the local tensor has shape
                    ``[microbatch, sequence, local_vocab]`` with ``Shard(-1)``.
                target: Tensor of shape ``[microbatch, sequence]`` containing
                    labels with ``-100`` ignored.
                **kwargs: Pipeline schedule scalar or tensor metadata. Tensor
                    values may have arbitrary rank and axis order and are not
                    inspected.

            Returns:
                Scalar mixed CE/KD loss for the microbatch.
            """
            teacher_logits = getattr(recipe_ref, "_current_teacher_logits", None)
            if teacher_logits is None:
                raise RuntimeError(
                    "KD loss wrapper: _current_teacher_logits not set. "
                    "Teacher pipeline eval must run before student step."
                )
            if isinstance(teacher_logits, list):
                if not teacher_logits:
                    raise RuntimeError("KD loss wrapper received more student than teacher microbatches")
                teacher_logits = teacher_logits.pop(0)
            if getattr(recipe_ref, "separate_meshes", False):
                teacher_logits = recipe_ref.kd_mesh_bridge.match_student_vocab_shard(logits, teacher_logits)
            # num_label_tokens is None because
            # _run_train_optim_step_pp applies scale_grads_and_clip_grad_norm
            # (which divides grads by num_label_tokens/dp_group_size) and
            # reporting_loss / num_label_tokens (dividing the reported loss).
            if recipe_ref.kd_ratio >= 1.0:
                ce_loss = logits.new_tensor(0.0, dtype=logits.dtype)
            else:
                ce_loss = calculate_loss(
                    recipe_ref.loss_fn,
                    logits=logits,
                    labels=target,
                    num_label_tokens=None,
                )
            kd_loss = recipe_ref.kd_loss_fn(
                logits,
                teacher_logits,
                target,
                num_batch_labels=1,
            )
            recipe_ref._ce_loss_buffer.append(ce_loss.detach().clone())
            recipe_ref._kd_loss_buffer.append(kd_loss.detach().clone())
            return (1.0 - recipe_ref.kd_ratio) * ce_loss + recipe_ref.kd_ratio * kd_loss

        return pp_kd_loss_fn

    def _get_separate_teacher_logits(self, batch: dict[str, Any]) -> torch.Tensor:
        """Request teacher logits for one student batch.

        Args:
            batch: Mapping containing ``input_ids`` and ``labels`` as tensors of
                shape ``[batch, sequence]``. Other tensor leaves may have
                arbitrary rank and axis order and are transported unchanged.

        Returns:
            Replicated tensor of shape ``[batch, sequence, vocab]`` containing
            full teacher logits.
        """
        self.kd_mesh_bridge.broadcast_command(RUN_TEACHER)
        teacher_logits = None
        for wave in range(self.kd_mesh_bridge.num_waves):
            self.kd_mesh_bridge.send_batch(wave, batch)
            received = self.kd_mesh_bridge.send_logits(wave, None)
            if received is not None:
                teacher_logits = received
        if teacher_logits is None:
            raise RuntimeError("Student rank did not receive teacher logits from the separate KD mesh")
        return teacher_logits

    @torch.no_grad()
    def _teacher_forward_separate(self, batch: dict[str, Any]) -> torch.Tensor | None:
        """Run one teacher batch and materialize transport-ready logits.

        Args:
            batch: Mapping containing ``input_ids`` and ``labels`` as tensors of
                shape ``[batch, sequence]``. Other tensor leaves may have
                arbitrary rank and axis order. Tensors may initially reside on
                CPU.

        Returns:
            Detached tensor of shape ``[batch, sequence, vocab]`` on the teacher
            output rank, otherwise ``None`` for PP ranks without the final stage.
        """
        batch = self.kd_mesh_bridge.move_to_device(batch)
        sequence_length = batch["labels"].shape[1]
        cp_sharder = ContextParallelSharder(
            self.teacher_model,
            self.device_mesh,
            batch,
        )
        train_ctx, batch = cp_sharder.shard(batch)
        labels = batch.pop("labels")
        with train_ctx(), torch.no_grad():
            if self.pp_enabled:
                input_ids = batch.pop("input_ids")
                batch_filtered = {
                    key: value
                    for key, value in batch.items()
                    if value is not None and not (isinstance(value, dict) and not value)
                }
                targets = labels.clone() if self.teacher_pp.info.has_last_stage else None
                losses = [] if self.teacher_pp.info.has_last_stage else None
                if self.teacher_pp.info.has_first_stage:
                    self.teacher_pp.info.schedule.eval(
                        input_ids,
                        target=targets,
                        losses=losses,
                        **batch_filtered,
                    )
                else:
                    self.teacher_pp.info.schedule.eval(target=targets, losses=losses, **batch_filtered)
                capture = getattr(self.teacher_model, "_teacher_logits_capture", None)
                captured_logits = capture[0] if capture is not None else None
                if capture is not None:
                    capture[0] = None
                logits = None
                if captured_logits:
                    logits = torch.cat(
                        [
                            materialize_teacher_logits(
                                microbatch_logits,
                                device_mesh=self.device_mesh,
                                sequence_length=sequence_length,
                            )
                            for microbatch_logits in captured_logits
                        ],
                        dim=0,
                    )
            else:
                teacher_batch = filter_forward_kwargs(self.teacher_model, batch)
                output = self.teacher_model(**teacher_batch)
                logits = getattr(output, "logits", output).detach()
        if logits is None:
            return None
        if self.pp_enabled:
            return logits
        return materialize_teacher_logits(
            logits,
            device_mesh=self.device_mesh,
            sequence_length=sequence_length,
        )

    def _run_teacher_worker(self) -> None:
        """Serve teacher forwards until the student mesh broadcasts stop."""
        with DistributedSignalHandler(group=self.kd_mesh_bridge.teacher_group):
            while self.kd_mesh_bridge.broadcast_command() == RUN_TEACHER:
                for wave in range(self.kd_mesh_bridge.num_waves):
                    batch = self.kd_mesh_bridge.send_batch(wave, None)
                    if batch is None:
                        raise RuntimeError("Teacher rank did not receive a batch from the student mesh")
                    logits = self._teacher_forward_separate(batch)
                    self.kd_mesh_bridge.send_logits(wave, logits)

    def _forward_backward_step(
        self,
        idx,
        batch,
        *,
        num_label_tokens,
        num_batches,
        is_train: bool = True,
    ):
        """Run one non-PP student microbatch with KD.

        Args:
            idx: Zero-based accumulation microbatch index.
            batch: Mapping containing ``input_ids`` and ``labels`` as tensors of
                shape ``[batch, sequence]``. Other tensor leaves may have
                arbitrary rank and axis order.
            num_label_tokens: Valid-label count across the optimizer step.
            num_batches: Number of accumulation microbatches in the step.
            is_train: Whether to run backward.

        Returns:
            Tuple of scalar tensors containing detached mixed, KL, and CE loss.
        """
        if self.pp_enabled:
            raise RuntimeError(
                "_forward_backward_step should not be called when pp_enabled; use _forward_backward_step_pp instead."
            )
        separate_teacher_logits = (
            self._get_separate_teacher_logits(batch) if getattr(self, "separate_meshes", False) else None
        )
        batch = {k: v.to(self.dist_env.device, non_blocking=True) for k, v in batch.items()}
        if separate_teacher_logits is not None:
            batch["teacher_logits"] = separate_teacher_logits
        labels = batch.pop("labels")
        # KD has not wired model-owned CP; skip the pre-embed hook explicitly.
        # Separate-mesh teacher logits ride the batch through CP sharding.
        cp_sharder = ContextParallelSharder(
            self.model_parts[0],
            self.device_mesh,
            batch,
            loss_mask=labels,
            invoke_pre_embed=False,
            extra_seq_buffers={"teacher_logits": 1} if separate_teacher_logits is not None else None,
        )
        train_ctx, batch = cp_sharder.shard(batch)
        separate_teacher_logits = batch.pop("teacher_logits", None)

        model = self.model_parts[0]
        sync_ctx = (
            get_sync_ctx(
                model,
                idx == num_batches - 1,
                defer_fsdp_grad_sync=getattr(self.distributed_config, "defer_fsdp_grad_sync", True),
            )
            if is_train
            else nullcontext()
        )
        with train_ctx(), sync_ctx:
            # No grad for teacher forward.
            if separate_teacher_logits is None:
                with (
                    ScopedModuleOffloading(self.teacher_model, enabled=self._offload_teacher_model),
                    torch.no_grad(),
                ):
                    teacher_batch = filter_forward_kwargs(self.teacher_model, batch)
                    teacher_logits = self.teacher_model(**teacher_batch)
                    teacher_logits = getattr(teacher_logits, "logits", teacher_logits).detach().clone()
            else:
                teacher_logits = separate_teacher_logits

            # Student forward.
            student_batch = filter_forward_kwargs(model, batch)
            student_keep_last = isinstance(self.loss_fn, FusedLinearCrossEntropy)
            if student_keep_last:
                student_out = model(logits_to_keep=1, **student_batch)
            else:
                student_out = model(**student_batch)

            student_logits = getattr(student_out, "logits", student_out)  # shape (B, S, V)
            if separate_teacher_logits is not None:
                teacher_logits = self.kd_mesh_bridge.match_student_vocab_shard(student_logits, teacher_logits)

            # Cross-entropy loss against true labels (skip when kd_ratio >= 1.0).
            if self.kd_ratio >= 1.0:
                ce_loss = student_logits.new_tensor(0.0, dtype=student_logits.dtype)
            else:
                ce_loss = calculate_loss(
                    self.loss_fn,
                    logits=student_logits,
                    labels=labels,
                    model=model,
                    hidden_states=student_out.hidden_states[-1] if "hidden_states" in student_out else None,
                    num_label_tokens=num_label_tokens,
                )

            # Reminder: kd_loss is normalized by num_label_tokens, which is typically
            # larger than the number of labels in this batch alone because it covers all
            # batches in one optimizer step (grad_acc_steps = gbs / mbs).
            kd_loss = self.kd_loss_fn(
                student_logits,
                teacher_logits,
                labels,
                num_batch_labels=num_label_tokens,
            )
            local_loss = (1.0 - self.kd_ratio) * ce_loss + self.kd_ratio * kd_loss
            if is_train:
                (local_loss * self._get_dp_group_size(include_cp=True)).backward()
            detached_local = local_loss.detach().clone()
            return detached_local, kd_loss.detach().clone(), ce_loss.detach().clone()

    def _forward_backward_step_pp(
        self,
        idx,
        batch,
        *,
        loss_buffer,
        num_label_tokens,
        num_batches,
        is_train: bool = True,
    ):
        """PP path: run teacher eval to capture logits, then run student step/eval.

        Teacher logits from the last PP stage are stored in ``self._current_teacher_logits``
        before the student schedule runs, so ``pp_kd_loss_fn`` can read them.

        Args:
            idx: Zero-based accumulation microbatch index.
            batch: Mapping containing ``input_ids`` and ``labels`` as tensors of
                shape ``[batch, sequence]``. Other tensor leaves may have
                arbitrary rank and axis order.
            loss_buffer: Output list receiving one detached scalar tensor.
            num_label_tokens: Valid-label count across the optimizer step.
            num_batches: Number of accumulation microbatches in the step.
            is_train: Whether the pipeline schedule runs backward.

        Transported teacher logits have shape ``[batch, sequence, vocab]`` and
        are split into ``[microbatch, sequence, vocab]`` pipeline tensors.
        """
        separate_teacher_logits = (
            self._get_separate_teacher_logits(batch) if getattr(self, "separate_meshes", False) else None
        )
        batch = {
            k: (
                {dk: dv.to(self.dist_env.device, non_blocking=True) for dk, dv in v.items() if dv is not None}
                if isinstance(v, dict)
                else (v.to(self.dist_env.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            )
            for k, v in batch.items()
        }
        if separate_teacher_logits is not None:
            batch["teacher_logits"] = separate_teacher_logits
        # KD has not wired model-owned CP; skip the pre-embed hook explicitly.
        cp_sharder = ContextParallelSharder(
            self.model_parts[0],
            self.device_mesh,
            batch,
            padding_token_id=self.tokenizer.pad_token_id if self.tokenizer else 0,
            num_chunks=_get_num_thd_chunks(True, self.cfg),
            invoke_pre_embed=False,
            extra_seq_buffers={"teacher_logits": 1} if separate_teacher_logits is not None else None,
        )
        train_ctx, batch = cp_sharder.shard(batch)
        separate_teacher_logits = batch.pop("teacher_logits", None)
        labels = batch.pop("labels")
        input_ids = batch.pop("input_ids")
        batch_filtered = {k: v for k, v in batch.items() if v is not None and not (isinstance(v, dict) and len(v) == 0)}

        # Only the last PP stage needs targets for the loss function.
        targets = labels.clone() if self.pp.info.has_last_stage else None

        fp8_ctx = self.te_fp8.maybe_te_autocast() if self.te_fp8 is not None else nullcontext()

        with train_ctx(), fp8_ctx:
            if separate_teacher_logits is not None:
                self._current_teacher_logits = list(
                    torch.split(separate_teacher_logits, self.pipeline_config.pp_microbatch_size, dim=0)
                )
            else:
                # Run teacher under inference_mode; logits captured by _teacher_capture_loss_fn.
                with torch.inference_mode():
                    teacher_losses = [] if self.teacher_pp.info.has_last_stage else None
                    if self.teacher_pp.info.has_first_stage:
                        self.teacher_pp.info.schedule.eval(
                            input_ids, target=targets, losses=teacher_losses, **batch_filtered
                        )
                    else:
                        self.teacher_pp.info.schedule.eval(target=targets, losses=teacher_losses, **batch_filtered)
                    # Transfer captured logits into the recipe so pp_kd_loss_fn can read them.
                    capture = getattr(self.teacher_model, "_teacher_logits_capture", None)
                    if capture is not None and capture[0] is not None:
                        self._current_teacher_logits = capture[0]
                        capture[0] = None  # Reset for next call.
                    else:
                        self._current_teacher_logits = None
            self._current_num_label_tokens = num_label_tokens

            # Run student forward (+ backward if training).
            student_losses = [] if self.pp.info.has_last_stage else None
            if is_train:
                if self.pp.info.has_first_stage:
                    self.pp.info.schedule.step(input_ids, target=targets, losses=student_losses, **batch_filtered)
                else:
                    self.pp.info.schedule.step(target=targets, losses=student_losses, **batch_filtered)
            else:
                if self.pp.info.has_first_stage:
                    self.pp.info.schedule.eval(input_ids, target=targets, losses=student_losses, **batch_filtered)
                else:
                    self.pp.info.schedule.eval(target=targets, losses=student_losses, **batch_filtered)

            if self.pp.info.has_last_stage:
                loss_buffer.append(torch.sum(torch.stack(student_losses)).detach().clone())
            else:
                loss_buffer.append(torch.tensor(0.0, device=self.dist_env.device))

    def _run_train_optim_step(self, batches, max_grad_norm: Optional[float] = None):
        """Execute a single training step.

        Args:
            batches: List of batches of training data.
            max_grad_norm: Gradient clipping norm. Optional, if None will not clip gradients.
        """
        if self.pp_enabled:
            return self._run_train_optim_step_pp(batches, max_grad_norm)

        num_label_tokens = torch.tensor(
            sum((batch["labels"] != -100).sum().item() for batch in batches), dtype=torch.long
        )
        num_label_tokens = self._dp_allreduce(num_label_tokens).item()
        loss_buffer = []

        # number of tokens in the batch, excluding any tail padding.
        num_tokens_in_batch = torch.tensor(
            sum(batch["labels"].numel() - count_tail_padding(batch["labels"]) for batch in batches),
            dtype=torch.long,
        )
        num_tokens_in_batch = self._dp_allreduce(num_tokens_in_batch).item()
        num_batches = len(batches)
        for i, batch in enumerate(batches):
            local_loss, kd_loss, ce_loss = self._forward_backward_step(
                i, batch, num_label_tokens=num_label_tokens, num_batches=num_batches
            )
            loss_buffer.append(local_loss)
            self._ce_loss_buffer.append(ce_loss)
            self._kd_loss_buffer.append(kd_loss)

        grad_norm = 0
        # Clip gradients **after** any rescaling.
        # TODO(@boxiangw): Fix TP gradient clipping
        if max_grad_norm is not None:
            if not self.device_mesh or self.device_mesh["tp"].size() == 1:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model_parts[0].parameters() if p.requires_grad], max_grad_norm
                )
                if hasattr(grad_norm, "full_tensor"):
                    grad_norm = grad_norm.full_tensor()  # collect the summed grad norm across ranks

            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()

        self.checkpointer.maybe_wait_for_staging()
        for opt in self.optimizer:
            opt.step()
            opt.zero_grad()

        if self.lr_scheduler is not None:
            for scheduler in self.lr_scheduler:
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

        t = time.perf_counter()
        time_delta = t - self.timestamp
        self.timestamp = t
        tps = num_tokens_in_batch / time_delta
        reporting_loss = torch.sum(torch.stack(loss_buffer))
        reporting_loss = self._dp_allreduce(reporting_loss, include_cp=True)
        reporting_loss = reporting_loss.cpu().item()

        ce_loss = self._dp_allreduce(torch.stack(self._ce_loss_buffer).sum(), include_cp=True).item()
        kd_loss = self._dp_allreduce(torch.stack(self._kd_loss_buffer).sum(), include_cp=True).item()
        # Clear buffers for next step.
        self._ce_loss_buffer.clear()
        self._kd_loss_buffer.clear()

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "loss": reporting_loss,
                "ce_loss": ce_loss,
                "kd_loss": kd_loss,
                "grad_norm": grad_norm,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
                "tps": tps,
                "tps_per_gpu": tps / max(self._get_dp_group_size(), 1),
                "num_tokens_per_step": num_tokens_in_batch,
                "num_label_tokens": num_label_tokens,
                "kd_ratio": self.kd_ratio,
                "temperature": getattr(self.kd_loss_fn, "temperature", float("nan")),
            },
        )

    def _run_train_optim_step_pp(self, batches, max_grad_norm: Optional[float] = None):
        """Execute a single training step when pipeline parallelism is enabled."""
        num_label_tokens = torch.tensor(sum((b["labels"] != -100).sum().item() for b in batches), dtype=torch.long)
        num_label_tokens = self._dp_allreduce(num_label_tokens).item()
        loss_buffer = []

        num_tokens_in_batch = torch.tensor(
            sum(b["labels"].numel() - count_tail_padding(b["labels"]) for b in batches),
            dtype=torch.long,
        )
        num_tokens_in_batch = self._dp_allreduce(num_tokens_in_batch).item()
        num_batches = len(batches)

        prepare_for_grad_accumulation(self.model_parts, pp_enabled=True)

        for i, batch in enumerate(batches):
            if i == num_batches - 1:
                prepare_for_final_backward(self.model_parts, pp_enabled=True)
            self._forward_backward_step_pp(
                i,
                batch,
                loss_buffer=loss_buffer,
                num_label_tokens=num_label_tokens,
                num_batches=num_batches,
            )
            if i == 0:
                prepare_after_first_microbatch()

        grad_norm = scale_grads_and_clip_grad_norm(
            max_grad_norm,
            self.model_parts,
            norm_type=2.0,
            pp_enabled=True,
            device_mesh=self.device_mesh,
            moe_mesh=self.moe_mesh,
            ep_axis_name="ep" if self.moe_mesh is not None and "ep" in self.moe_mesh.mesh_dim_names else None,
            pp_axis_name="pp",
            foreach=True,
            num_label_tokens=num_label_tokens,
            dp_group_size=self._get_dp_group_size(include_cp=True),
        )

        self.checkpointer.maybe_wait_for_staging()
        for opt in self.optimizer:
            opt.step()
            opt.zero_grad()

        if hasattr(self.model_parts[0], "update_moe_gate_bias"):
            for mp in self.model_parts:
                mp.update_moe_gate_bias()

        if self.lr_scheduler is not None:
            for scheduler in self.lr_scheduler:
                scheduler.step(1)

        fp8_config = self.cfg.get("fp8", None)
        if (
            fp8_config is not None
            and fp8_config.get("enabled", False)
            and fp8_config.get("precompute_float8_dynamic_scale_for_fsdp", False)
            and self.device_mesh is not None
            and self.device_mesh["dp_shard"].size() > 1
        ):
            precompute_float8_dynamic_scale_for_fsdp(self.model_parts[0])

        t = time.perf_counter()
        time_delta = t - self.timestamp
        self.timestamp = t
        tps = num_tokens_in_batch / time_delta

        reporting_loss = torch.sum(torch.stack(loss_buffer))
        reporting_loss = self._dp_allreduce(reporting_loss, include_cp=True)
        reporting_loss = reporting_loss / num_label_tokens
        reporting_loss = reporting_loss.to(self.dist_env.device)

        # Send loss from the last PP stage to rank 0 for logging.
        src_rank = self.device_mesh.mesh.reshape(-1)[-1].item()
        if self.dist_env.rank == src_rank and not self.dist_env.is_main:
            torch.distributed.send(reporting_loss, dst=0)
        elif self.dist_env.is_main and self.dist_env.rank != src_rank:
            torch.distributed.recv(reporting_loss, src=src_rank)
        reporting_loss = reporting_loss.cpu().item()

        # CE/KD buffers are only populated on the last PP stage (in pp_kd_loss_fn).
        # Allreduce within DP group, then send from last stage to rank 0 for logging.
        ce_tensor = (
            torch.stack(self._ce_loss_buffer).sum()
            if self._ce_loss_buffer
            else torch.tensor(0.0, device=self.dist_env.device)
        )
        kd_tensor = (
            torch.stack(self._kd_loss_buffer).sum()
            if self._kd_loss_buffer
            else torch.tensor(0.0, device=self.dist_env.device)
        )
        ce_tensor = self._dp_allreduce(ce_tensor, include_cp=True)
        kd_tensor = self._dp_allreduce(kd_tensor, include_cp=True)
        # The PP wrapper buffers raw CE/KD sums; normalize them here so the logged
        # metrics match the non-PP path and stay comparable across PP settings.
        ce_tensor = ce_tensor / num_label_tokens
        kd_tensor = kd_tensor / num_label_tokens
        ce_tensor = ce_tensor.to(self.dist_env.device)
        kd_tensor = kd_tensor.to(self.dist_env.device)
        if self.dist_env.rank == src_rank and not self.dist_env.is_main:
            torch.distributed.send(ce_tensor, dst=0)
            torch.distributed.send(kd_tensor, dst=0)
        elif self.dist_env.is_main and self.dist_env.rank != src_rank:
            torch.distributed.recv(ce_tensor, src=src_rank)
            torch.distributed.recv(kd_tensor, src=src_rank)
        ce_loss = ce_tensor.cpu().item()
        kd_loss = kd_tensor.cpu().item()
        self._ce_loss_buffer.clear()
        self._kd_loss_buffer.clear()

        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item()
        grad_norm = float(grad_norm)

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "loss": reporting_loss,
                "ce_loss": ce_loss,
                "kd_loss": kd_loss,
                "grad_norm": grad_norm,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
                "tps": tps,
                "tps_per_gpu": tps / self._get_cp_group_size() / max(self._get_dp_group_size(), 1),
                "num_tokens_per_step": num_tokens_in_batch,
                "num_label_tokens": num_label_tokens,
                "kd_ratio": self.kd_ratio,
                "temperature": getattr(self.kd_loss_fn, "temperature", float("nan")),
            },
        )

    def run_train_validation_loop(self):
        """Run the student loop or serve teacher forwards on a separate mesh."""
        if getattr(self, "separate_meshes", False) and self.kd_mesh_bridge.is_teacher:
            self._run_teacher_worker()
            return
        if getattr(self, "separate_meshes", False):
            try:
                return self._run_student_train_validation_loop()
            finally:
                self.kd_mesh_bridge.broadcast_command(STOP_TEACHER)
        return self._run_student_train_validation_loop()

    def _run_student_train_validation_loop(self):
        """Run training loop; skip validation when PP is enabled (not yet supported)."""
        if not self.pp_enabled:
            return super().run_train_validation_loop()

        # PP path: same as parent but without the validation block.
        for mp in self.model_parts:
            mp.train()
        self.timestamp = time.perf_counter()
        pbar = self._make_progress_bar()
        try:
            for epoch in self.step_scheduler.epochs:
                self.step_scheduler.set_epoch(epoch)
                for batches in self.step_scheduler:
                    self._enable_qat_if_delayed(self.step_scheduler.step)
                    train_log_data = self._run_train_optim_step(batches, self.max_grad_norm)
                    self._collect_moe_load_balance()
                    self.log_train_metrics(train_log_data)
                    self._update_progress_bar(pbar, train_log_data.metrics)
                    val_losses = {}
                    if self.step_scheduler.is_val_step:
                        logger.warning("Validation is not supported for pipeline parallelism; skipping")
                    if self.step_scheduler.is_ckpt_step:
                        self.save_checkpoint(
                            epoch,
                            self.step_scheduler.step,
                            train_log_data.metrics["loss"],
                            val_losses,
                            best_metric_key=self.best_metric_key,
                        )
        finally:
            if pbar is not None:
                pbar.close()
        self.metric_logger_train.close()
        for v in self.metric_logger_valid.values():
            v.close()
        self.checkpointer.close()

    @torch.no_grad()
    def _run_validation_epoch(self, val_dataloader):
        """Run one pass over `self.val_dataloader`."""
        if self.pp_enabled:
            logger.warning("Validation is not supported for pipeline parallelism")
            return

        with ScopedRNG(seed=1, ranked=True):
            for mp in self.model_parts:
                mp.eval()

            total_loss = 0.0
            total_ce_loss = 0.0
            total_kd_loss = 0.0
            total_num_label_tokens = 0

            for batch in val_dataloader:
                num_label_tokens = (batch["labels"] != -100).sum().item()
                local_loss, _kd_loss, _ce_loss = self._forward_backward_step(
                    0,
                    batch,
                    num_label_tokens=num_label_tokens,
                    num_batches=1,
                    is_train=False,
                )
                # _forward_backward_step returns per-token-averaged losses.
                # Multiply back by num_label_tokens to get the raw sum for
                # correct weighted averaging across batches.
                total_loss += local_loss.item() * num_label_tokens
                total_ce_loss += _ce_loss.item() * num_label_tokens
                total_kd_loss += _kd_loss.item() * num_label_tokens
                total_num_label_tokens += num_label_tokens

        total_loss = self._dp_allreduce(
            torch.tensor(total_loss, dtype=torch.float32, device=self.dist_env.device), include_cp=True
        ).item()
        total_ce_loss = self._dp_allreduce(
            torch.tensor(total_ce_loss, dtype=torch.float32, device=self.dist_env.device), include_cp=True
        ).item()
        total_kd_loss = self._dp_allreduce(
            torch.tensor(total_kd_loss, dtype=torch.float32, device=self.dist_env.device), include_cp=True
        ).item()
        total_num_label_tokens = self._dp_allreduce(torch.tensor(total_num_label_tokens, dtype=torch.long)).item()

        val_loss = total_loss / max(total_num_label_tokens, 1e-8)
        val_ce_loss = total_ce_loss / max(total_num_label_tokens, 1e-8)
        val_kd_loss = total_kd_loss / max(total_num_label_tokens, 1e-8)
        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "val_loss": val_loss,
                "ce_loss": val_ce_loss,
                "kd_loss": val_kd_loss,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "num_label_tokens": total_num_label_tokens,
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
            },
        )

    def log_val_metrics(self, val_name, log_data, metric_logger=None):
        if not self.dist_env.is_main or log_data is None:
            return

        if wandb.run is not None:
            wandb.log(log_data.to_dict() | {"val_name": val_name}, step=log_data.step)

        if not metric_logger is None:
            metric_logger.log(log_data)

        if self.kd_ratio >= 1.0:
            logging.info(
                "[val] {} | step {} | epoch {} | loss {:.4f} | kd_loss {:.4f} | lr {:.2e} | num_label_tokens {}".format(
                    val_name,
                    log_data.step,
                    log_data.epoch,
                    log_data.metrics["val_loss"],
                    log_data.metrics["kd_loss"],
                    log_data.metrics["lr"],
                    log_data.metrics["num_label_tokens"],
                )
            )
        else:
            logging.info(
                "[val] {} | step {} | epoch {} | loss {:.4f} | ce_loss {:.4f} | kd_loss {:.4f} | lr {:.2e} | num_label_tokens {}".format(
                    val_name,
                    log_data.step,
                    log_data.epoch,
                    log_data.metrics["val_loss"],
                    log_data.metrics["ce_loss"],
                    log_data.metrics["kd_loss"],
                    log_data.metrics["lr"],
                    log_data.metrics["num_label_tokens"],
                )
            )

    def log_train_metrics(self, log_data) -> float:
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

        # Log to remote services (WandB) according to step_scheduler frequency.
        if self.step_scheduler.is_remote_logging_step:
            if wandb.run is not None:
                wandb.log(log_data.to_dict(), step=log_data.step)

        # JSONL training log (always log for detailed local records).
        self.metric_logger_train.log(log_data)

        if self.kd_ratio >= 1.0:
            logging.info(
                "step {} | epoch {} | "
                "loss {:.4f} | kd_loss {:.4f} | "
                "lr {:.2e} | mem {:.2f} GiB | tps {:.2f} | kd_ratio {:.2f} | temperature {:.2f}".format(
                    log_data.step,
                    log_data.epoch,
                    log_data.metrics["loss"],
                    log_data.metrics["kd_loss"],
                    log_data.metrics["lr"],
                    log_data.metrics["mem"],
                    log_data.metrics["tps"],
                    log_data.metrics["kd_ratio"],
                    log_data.metrics["temperature"],
                )
            )
        else:
            logging.info(
                "step {} | epoch {} | "
                "loss {:.4f} | ce_loss {:.4f} | kd_loss {:.4f} | "
                "lr {:.2e} | mem {:.2f} GiB | tps {:.2f} | kd_ratio {:.2f} | temperature {:.2f}".format(
                    log_data.step,
                    log_data.epoch,
                    log_data.metrics["loss"],
                    log_data.metrics["ce_loss"],
                    log_data.metrics["kd_loss"],
                    log_data.metrics["lr"],
                    log_data.metrics["mem"],
                    log_data.metrics["tps"],
                    log_data.metrics["kd_ratio"],
                    log_data.metrics["temperature"],
                )
            )
        torch.cuda.reset_peak_memory_stats()


# Entry point
def main(config_path="examples/llm_kd/llama3_2/llama3_2_1b_kd.yaml"):
    """Run the KD recipe from CLI or directly."""
    cfg = parse_args_and_load_config(config_path)
    trainer = KnowledgeDistillationRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":  # pragma: no cover
    main()
