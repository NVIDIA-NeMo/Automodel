# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Shared-trajectory assertions for checkpoint-robustness training tests."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


@dataclass(frozen=True)
class _ResumePlan:
    """Describe the shared checkpoint boundary and uninterrupted continuation."""

    checkpoint_dir: Path
    boundary_step: int
    continuation_steps: int

    @property
    def final_max_steps(self) -> int:
        """Return the total optimizer steps in the uninterrupted reference run."""
        return self.boundary_step + self.continuation_steps

    @property
    def comparison_steps(self) -> tuple[int, ...]:
        """Return zero-based post-checkpoint step indices compared after resume."""
        return tuple(range(self.boundary_step, self.final_max_steps))

    @property
    def artifact_dir(self) -> Path:
        """Return the directory shared by isolated reference and resume processes."""
        return self.checkpoint_dir / ".checkpoint_robustness" / "shared_resume"

    @property
    def resume_checkpoint_dir(self) -> Path:
        """Return a separate output root for the resumed branch."""
        return self.artifact_dir / "resumed_checkpoints"


@dataclass(frozen=True)
class _ResumeLossTolerance:
    """Resolved loss envelope for one shared-trajectory resume check."""

    profile: str
    first_step_atol: float
    first_step_rtol: float
    later_step_atol: float
    later_step_rtol: float


_RESUME_LOSS_TOLERANCE_PROFILES = {
    "strict": _ResumeLossTolerance(
        profile="strict",
        first_step_atol=1e-6,
        first_step_rtol=0.0,
        later_step_atol=1e-6,
        later_step_rtol=0.0,
    ),
    "standard": _ResumeLossTolerance(
        profile="standard",
        first_step_atol=1e-5,
        first_step_rtol=2e-3,
        later_step_atol=5e-3,
        later_step_rtol=2e-3,
    ),
    "relaxed": _ResumeLossTolerance(
        profile="relaxed",
        first_step_atol=1e-4,
        first_step_rtol=7.5e-3,
        later_step_atol=1e-2,
        later_step_rtol=7.5e-3,
    ),
}


def _resolve_resume_loss_tolerance(
    profile: str = "standard",
    *,
    first_step_override: str | float | None = None,
    later_step_override: str | float | None = None,
) -> _ResumeLossTolerance:
    """Resolve a named resume-loss profile with optional numeric overrides."""
    normalized_profile = profile.strip().lower()
    if normalized_profile not in _RESUME_LOSS_TOLERANCE_PROFILES:
        valid_profiles = ", ".join(sorted(_RESUME_LOSS_TOLERANCE_PROFILES))
        raise ValueError(f"unknown resume tolerance profile {profile!r}; expected one of: {valid_profiles}")

    selected = _RESUME_LOSS_TOLERANCE_PROFILES[normalized_profile]
    tolerance = _ResumeLossTolerance(
        profile=normalized_profile,
        first_step_atol=selected.first_step_atol if first_step_override is None else float(first_step_override),
        first_step_rtol=selected.first_step_rtol if first_step_override is None else 0.0,
        later_step_atol=selected.later_step_atol if later_step_override is None else float(later_step_override),
        later_step_rtol=selected.later_step_rtol if later_step_override is None else 0.0,
    )
    for label, threshold in (
        ("first-step absolute", tolerance.first_step_atol),
        ("first-step relative", tolerance.first_step_rtol),
        ("later-step absolute", tolerance.later_step_atol),
        ("later-step relative", tolerance.later_step_rtol),
    ):
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError(f"resume {label} loss threshold must be finite and non-negative, got {threshold}")
    return tolerance


def _loss_tolerance_for_step(
    tolerance: _ResumeLossTolerance,
    *,
    first_step: bool,
    reference_loss: float,
    resumed_loss: float,
) -> tuple[float, float, float]:
    """Return the absolute term, relative term, and effective loss allowance."""
    if first_step:
        absolute = tolerance.first_step_atol
        relative = tolerance.first_step_rtol
    else:
        absolute = tolerance.later_step_atol
        relative = tolerance.later_step_rtol
    scale = max(abs(reference_loss), abs(resumed_loss))
    return absolute, relative, absolute + relative * scale


class _TrajectoryRecorder:
    """Record checkpoint state plus the exact post-checkpoint batches and metrics."""

    def __init__(self, plan: _ResumePlan, *, capture_boundary_state: bool) -> None:
        self.plan = plan
        self.capture_boundary_state = capture_boundary_state
        self.boundary_state: dict[str, object] | None = None
        self.steps: dict[int, dict[str, object]] = {}

    def attach(self, trainer: object) -> None:
        """Attach recording hooks to one fully set-up recipe instance.

        Args:
            trainer: Recipe whose optimizer-step and checkpoint calls are recorded.
        """
        original_train_step = trainer._run_train_optim_step
        optimizers = trainer.optimizer if isinstance(trainer.optimizer, (list, tuple)) else [trainer.optimizer]
        first_comparison_step = min(self.plan.comparison_steps)
        first_step_pre_update_model_digests: dict[str, str] | None = None
        first_step_pre_update_optimizer_digests: dict[str, str] | None = None
        first_step_gradient_digests: dict[str, str] | None = None

        if optimizers:
            original_optimizer_step = optimizers[0].step

            @wraps(original_optimizer_step)
            def recorded_optimizer_step(*args, **kwargs):
                nonlocal first_step_gradient_digests
                nonlocal first_step_pre_update_model_digests
                nonlocal first_step_pre_update_optimizer_digests
                if int(trainer.step_scheduler.step) == first_comparison_step:
                    first_step_pre_update_model_digests = _model_state_digests(trainer.model_parts)
                    first_step_pre_update_optimizer_digests = _optimizer_state_digests(
                        trainer.optimizer,
                        trainer.model_parts,
                    )
                    first_step_gradient_digests = _gradient_state_digests(trainer.model_parts)
                return original_optimizer_step(*args, **kwargs)

            optimizers[0].step = recorded_optimizer_step

        @wraps(original_train_step)
        def recorded_train_step(batches, *args, **kwargs):
            step = int(trainer.step_scheduler.step)
            shared_batch_digest = getattr(trainer, "_training_reproducibility_batch_digest", None)
            batch_digest = None
            if step in self.plan.comparison_steps:
                batch_digest = (
                    shared_batch_digest[1]
                    if shared_batch_digest is not None and shared_batch_digest[0] == step
                    else _state_digest(batches)
                )
            log_data = original_train_step(batches, *args, **kwargs)
            if batch_digest is not None:
                step_record = {
                    "batch_digest": batch_digest,
                    "loss": float(log_data.metrics["loss"]),
                    "lr": float(log_data.metrics["lr"]),
                }
                if step == first_comparison_step:
                    if (
                        first_step_pre_update_model_digests is None
                        or first_step_pre_update_optimizer_digests is None
                        or first_step_gradient_digests is None
                    ):
                        raise AssertionError("first resume comparison step did not execute an optimizer update")
                    step_record["diagnostics"] = {
                        "pre_update_model_digests": first_step_pre_update_model_digests,
                        "pre_update_optimizer_digests": first_step_pre_update_optimizer_digests,
                        "gradient_digests": first_step_gradient_digests,
                        "post_step_model_digests": _model_state_digests(trainer.model_parts),
                        "post_step_optimizer_digests": _optimizer_state_digests(
                            trainer.optimizer,
                            trainer.model_parts,
                        ),
                    }
                self.steps[step] = step_record
            return log_data

        trainer._run_train_optim_step = recorded_train_step

        if not self.capture_boundary_state:
            return

        original_save_checkpoint = trainer.save_checkpoint

        @wraps(original_save_checkpoint)
        def recorded_save_checkpoint(epoch, step, *args, **kwargs):
            if int(step) == self.plan.boundary_step - 1:
                self.boundary_state = _checkpoint_state_snapshot(trainer, state_is_being_saved=True)
            return original_save_checkpoint(epoch, step, *args, **kwargs)

        trainer.save_checkpoint = recorded_save_checkpoint

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable reference trajectory."""
        if self.capture_boundary_state and self.boundary_state is None:
            raise AssertionError(
                f"Shared resume checkpoint was not captured at completed step {self.plan.boundary_step}"
            )
        missing_steps = sorted(set(self.plan.comparison_steps) - set(self.steps))
        if missing_steps:
            raise AssertionError(f"Uninterrupted resume reference is missing steps {missing_steps}")
        return {
            "boundary_step": self.plan.boundary_step,
            "continuation_steps": self.plan.continuation_steps,
            "boundary_state": self.boundary_state,
            "steps": {str(step): values for step, values in sorted(self.steps.items())},
        }


class _TrainingReproducibilityRecorder:
    """Record one independent training lifecycle for a non-blocking CI comparison."""

    def __init__(self, trainer: object) -> None:
        self.trainer = trainer
        self.fingerprint_components = _training_fingerprint_components(trainer)
        self.steps: dict[int, dict[str, object]] = {}

    def attach(self) -> None:
        """Attach a batch-and-metric recorder to the configured trainer."""
        original_train_step = self.trainer._run_train_optim_step

        @wraps(original_train_step)
        def recorded_train_step(batches, *args, **kwargs):
            step = int(self.trainer.step_scheduler.step)
            batch_digest = _state_digest(batches)
            self.trainer._training_reproducibility_batch_digest = (step, batch_digest)
            try:
                log_data = original_train_step(batches, *args, **kwargs)
            finally:
                del self.trainer._training_reproducibility_batch_digest
            self.steps[step] = {
                "batch_digest": batch_digest,
                "loss": float(log_data.metrics["loss"]),
                "lr": float(log_data.metrics["lr"]),
            }
            return log_data

        self.trainer._run_train_optim_step = recorded_train_step

    def to_dict(self) -> dict[str, object]:
        """Return the recorded lifecycle as a JSON-serializable mapping."""
        return {
            "fingerprint_components": self.fingerprint_components,
            "steps": {str(step): values for step, values in sorted(self.steps.items())},
        }


def _resume_plan_from_config(cfg: object, *, continuation_steps: int = 3) -> _ResumePlan:
    """Build a shared-trajectory plan from the original robustness config."""
    boundary_step = cfg.step_scheduler.max_steps
    if isinstance(boundary_step, bool) or not isinstance(boundary_step, int) or boundary_step < 1:
        raise ValueError(f"checkpoint robustness requires a positive integer max_steps, got {boundary_step!r}")
    if continuation_steps < 1:
        raise ValueError(f"continuation_steps must be positive, got {continuation_steps}")
    return _ResumePlan(
        checkpoint_dir=Path(cfg.checkpoint.checkpoint_dir),
        boundary_step=boundary_step,
        continuation_steps=continuation_steps,
    )


def _configure_uninterrupted_run(cfg: object, plan: _ResumePlan) -> None:
    """Extend Phase 1 while preserving its original LR schedule and checkpoint boundary."""
    cfg.step_scheduler.max_steps = plan.final_max_steps
    cfg.step_scheduler.ckpt_every_steps = plan.boundary_step
    cfg.checkpoint.save_consolidated = "final"
    if hasattr(cfg, "lr_scheduler") and cfg.lr_scheduler is not None:
        cfg.lr_scheduler.lr_decay_steps = plan.boundary_step


def _configure_resumed_run(cfg: object, plan: _ResumePlan, checkpoint_path: Path) -> None:
    """Restore the boundary checkpoint into an output directory separate from the reference branch."""
    cfg.step_scheduler.max_steps = plan.final_max_steps
    cfg.step_scheduler.ckpt_every_steps = plan.boundary_step
    if hasattr(cfg, "lr_scheduler") and cfg.lr_scheduler is not None:
        cfg.lr_scheduler.lr_decay_steps = plan.boundary_step
    cfg.checkpoint.restore_from = str(checkpoint_path)
    cfg.checkpoint.checkpoint_dir = str(plan.resume_checkpoint_dir)
    cfg.checkpoint.save_consolidated = False


def _disable_checkpoint_saves_after_restore(trainer: object) -> None:
    """Disable new checkpoint writes after the resume checkpoint has loaded."""
    trainer.checkpointer.config.enabled = False


def _checkpoint_for_completed_steps(plan: _ResumePlan, completed_steps: int) -> Path:
    """Locate the checkpoint written after exactly ``completed_steps`` optimizer steps."""
    checkpoint_step = completed_steps - 1
    matches = list(plan.checkpoint_dir.glob(f"epoch_*_step_{checkpoint_step}"))
    if not matches:
        raise AssertionError(
            f"No checkpoint for completed step {completed_steps} under {plan.checkpoint_dir}; "
            f"expected epoch_*_step_{checkpoint_step}"
        )

    def epoch_number(path: Path) -> int:
        return int(path.name.split("_", 2)[1])

    return max(matches, key=epoch_number)


def _rank() -> int:
    """Return the initialized distributed rank or the launcher-provided rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def _reference_path(plan: _ResumePlan) -> Path:
    """Return the current rank's persisted uninterrupted trajectory path."""
    return plan.artifact_dir / f"trajectory_rank_{_rank()}.json"


def _persist_reference_trajectory(recorder: _TrajectoryRecorder) -> None:
    """Atomically persist one rank's uninterrupted continuation and checkpoint state."""
    path = _reference_path(recorder.plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(recorder.to_dict(), sort_keys=True))
    temporary_path.replace(path)


def _load_reference_trajectory(plan: _ResumePlan) -> dict[str, object]:
    """Load the current rank's uninterrupted trajectory artifact."""
    path = _reference_path(plan)
    if not path.exists():
        raise AssertionError(f"Shared resume trajectory artifact not found for rank {_rank()}: {path}")
    return json.loads(path.read_text())


def _state_digest(value: object) -> str:
    """Return a deterministic digest for nested checkpoint or batch state."""
    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, DTensor):
            item = item.to_local()
        if isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().cpu()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy())
            return
        if isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                update(key)
                update(item[key])
            digest.update(b"}")
            return
        if isinstance(item, (list, tuple)):
            digest.update(f"{type(item).__name__}[".encode())
            for element in item:
                update(element)
            digest.update(b"]")
            return
        if isinstance(item, (set, frozenset)):
            digest.update(f"{type(item).__name__}[".encode())
            for element in sorted(item, key=repr):
                update(element)
            digest.update(b"]")
            return
        if hasattr(item, "dtype") and hasattr(item, "shape") and hasattr(item, "tobytes"):
            digest.update(f"array:{item.dtype}:{tuple(item.shape)}:".encode())
            digest.update(item.tobytes())
            return
        digest.update(f"{type(item).__qualname__}:{item!r};".encode())

    update(value)
    return digest.hexdigest()


def _config_section(config: dict[str, object], key: str) -> object:
    """Return one serializable config section or ``None`` when it is absent."""
    return config.get(key)


def _training_fingerprint_components(trainer: object) -> dict[str, str]:
    """Fingerprint independent-run inputs that must match before comparing metrics."""
    config = trainer.cfg.to_dict()
    step_scheduler = trainer.step_scheduler
    step_scheduler_config = config.get("step_scheduler", {})
    if not isinstance(step_scheduler_config, dict):
        raise TypeError("step_scheduler config must serialize to a mapping for reproducibility comparison")
    lr_schedulers = trainer.lr_scheduler
    if lr_schedulers is not None and not isinstance(lr_schedulers, (list, tuple)):
        lr_schedulers = [lr_schedulers]
    lr_scheduler_state = [] if lr_schedulers is None else [scheduler.state_dict() for scheduler in lr_schedulers]

    components = {
        "model_and_initialization": {
            "model": _config_section(config, "model"),
            "teacher_model": _config_section(config, "teacher_model"),
            "peft": _config_section(config, "peft"),
            "freeze_config": _config_section(config, "freeze_config"),
        },
        "seed": {
            "seed": _config_section(config, "seed"),
            "rng": _config_section(config, "rng"),
        },
        "dataset_and_ordering": {
            "dataset": _config_section(config, "dataset"),
            "dataloader": _config_section(config, "dataloader"),
            "validation_dataset": _config_section(config, "validation_dataset"),
            "validation_dataloader": _config_section(config, "validation_dataloader"),
            "packed_sequence": _config_section(config, "packed_sequence"),
            "tokenizer": _config_section(config, "tokenizer"),
        },
        "batch_and_topology": {
            "global_batch_size": int(step_scheduler_config.get("global_batch_size", 32)),
            "local_batch_size": int(step_scheduler_config.get("local_batch_size", 1)),
            "grad_acc_steps": int(step_scheduler.grad_acc_steps),
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "distributed": _config_section(config, "distributed"),
        },
        "optimizer": _config_section(config, "optimizer"),
        "lr_scheduler": {
            "config": _config_section(config, "lr_scheduler"),
            "initial_state": lr_scheduler_state,
            "initial_step": int(step_scheduler.step),
            "initial_epoch": int(step_scheduler.epoch),
            "num_epochs": int(step_scheduler.num_epochs),
            "val_every_steps": step_scheduler.val_every_steps,
        },
        "loss_and_backend": {
            "loss_fn": _config_section(config, "loss_fn"),
            "backend": _config_section(config, "backend"),
        },
    }
    return {name: _state_digest(value) for name, value in components.items()}


def _training_reproducibility_path(artifact_dir: Path, lifecycle: str) -> Path:
    """Return one rank-local lifecycle artifact path."""
    return artifact_dir / f"{lifecycle}_rank_{_rank()}.json"


def _persist_training_reproducibility(
    recorder: _TrainingReproducibilityRecorder,
    artifact_dir: Path,
    *,
    lifecycle: str,
) -> None:
    """Atomically persist one rank's independent training lifecycle."""
    path = _training_reproducibility_path(artifact_dir, lifecycle)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(recorder.to_dict(), sort_keys=True))
    temporary_path.replace(path)


def _load_training_reproducibility(artifact_dir: Path, *, lifecycle: str) -> dict[str, object]:
    """Load one rank's recorded independent training lifecycle."""
    path = _training_reproducibility_path(artifact_dir, lifecycle)
    if not path.exists():
        raise FileNotFoundError(f"training reproducibility artifact not found: {path}")
    return json.loads(path.read_text())


def _compare_training_reproducibility(
    normal_run: dict[str, object],
    checkpoint_run: dict[str, object],
    *,
    loss_threshold: float,
) -> dict[str, object]:
    """Compare independent lifecycles without making the result a resume gate."""
    if loss_threshold < 0:
        return {"status": "not_comparable", "reason": "loss threshold must be non-negative"}

    normal_fingerprint = normal_run.get("fingerprint_components", {})
    checkpoint_fingerprint = checkpoint_run.get("fingerprint_components", {})
    fingerprint_keys = set(normal_fingerprint) | set(checkpoint_fingerprint)
    mismatched_components = sorted(
        key for key in fingerprint_keys if normal_fingerprint.get(key) != checkpoint_fingerprint.get(key)
    )
    if mismatched_components:
        return {
            "status": "not_comparable",
            "reason": "configuration fingerprint mismatch",
            "mismatched_components": mismatched_components,
        }

    normal_steps = {int(step): values for step, values in normal_run.get("steps", {}).items()}
    checkpoint_steps = {int(step): values for step, values in checkpoint_run.get("steps", {}).items()}
    overlapping_steps = sorted(set(normal_steps) & set(checkpoint_steps))
    if not overlapping_steps:
        return {"status": "not_comparable", "reason": "no overlapping recorded steps"}

    for step in overlapping_steps:
        normal = normal_steps[step]
        checkpoint = checkpoint_steps[step]
        if normal["batch_digest"] != checkpoint["batch_digest"]:
            return {
                "status": "diverged",
                "reason": "batch identity mismatch",
                "step": step,
                "overlapping_steps": overlapping_steps,
            }
        if normal["lr"] != checkpoint["lr"]:
            return {
                "status": "diverged",
                "reason": "learning-rate mismatch",
                "step": step,
                "normal_lr": normal["lr"],
                "checkpoint_lr": checkpoint["lr"],
                "overlapping_steps": overlapping_steps,
            }

    loss_differences = {
        step: abs(float(normal_steps[step]["loss"]) - float(checkpoint_steps[step]["loss"]))
        for step in overlapping_steps
    }
    max_step = max(loss_differences, key=loss_differences.get)
    max_difference = loss_differences[max_step]
    status = (
        "within_tolerance"
        if math.isfinite(max_difference) and max_difference <= loss_threshold
        else "outside_tolerance"
    )
    return {
        "status": status,
        "overlapping_steps": overlapping_steps,
        "max_loss_difference": max_difference,
        "max_difference_step": max_step,
        "loss_threshold": loss_threshold,
    }


def _report_training_reproducibility(
    artifact_dir: Path,
    checkpoint_recorder: _TrainingReproducibilityRecorder,
    *,
    loss_threshold: float,
) -> None:
    """Persist and print a gathered, explicitly non-blocking independent-run comparison."""
    try:
        normal_run = _load_training_reproducibility(artifact_dir, lifecycle="normal")
        local_report = _compare_training_reproducibility(
            normal_run,
            checkpoint_recorder.to_dict(),
            loss_threshold=loss_threshold,
        )
    except (FileNotFoundError, json.JSONDecodeError) as error:
        local_report = {"status": "not_comparable", "reason": str(error)}

    reports = [local_report]
    if dist.is_initialized():
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, local_report)
    if _rank() != 0:
        return

    alert_statuses = {"diverged", "outside_tolerance"}
    statuses = [report["status"] for report in reports]
    summary_status = "alert" if any(status in alert_statuses for status in statuses) else "reported"
    report_path = artifact_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(
            {
                "blocking": False,
                "status": summary_status,
                "reports": reports,
            },
            indent=2,
            sort_keys=True,
        )
    )
    temporary_path.replace(report_path)

    for rank, report in enumerate(reports):
        print(f"[Training reproducibility][non-blocking] rank={rank} {json.dumps(report, sort_keys=True)}")
    if summary_status == "alert":
        print(
            "[Training reproducibility][non-blocking][ALERT] Independent-run reproducibility exceeded its "
            f"configured envelope; inspect {report_path}"
        )
    elif "not_comparable" in statuses:
        print(
            "[Training reproducibility][non-blocking][NOTICE] At least one rank was not comparable; "
            f"inspect {report_path}"
        )
    else:
        print(
            "[Training reproducibility][non-blocking][SUMMARY] All ranks stayed within the configured envelope; "
            f"report={report_path}"
        )


def _optimizer_step_summary(optimizers: object) -> list[dict[str, int]]:
    """Summarize per-parameter optimizer step counters without persisting optimizer tensors."""
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]
    summaries: list[dict[str, int]] = []
    for optimizer in optimizers:
        counter: Counter[str] = Counter()
        for state in optimizer.state.values():
            step = state.get("step") if isinstance(state, dict) else None
            if isinstance(step, torch.Tensor):
                step = step.item()
            if step is not None:
                counter[str(step)] += 1
        summaries.append(dict(sorted(counter.items())))
    return summaries


def _model_parameter_names(model_parts: object) -> dict[int, str]:
    """Map optimizer parameter identities to stable model-part names."""
    if not isinstance(model_parts, (list, tuple)):
        model_parts = [model_parts]
    names: dict[int, str] = {}
    for part_index, model in enumerate(model_parts):
        for name, parameter in model.named_parameters():
            names.setdefault(id(parameter), f"model_part_{part_index}.parameter.{name}")
    return names


def _model_state_digests(model_parts: object) -> dict[str, str]:
    """Fingerprint every rank-local parameter and persistent buffer without gathering full tensors."""
    if not isinstance(model_parts, (list, tuple)):
        model_parts = [model_parts]
    digests: dict[str, str] = {}
    for part_index, model in enumerate(model_parts):
        prefix = f"model_part_{part_index}"
        for name, parameter in model.named_parameters():
            digests[f"{prefix}.parameter.{name}"] = _state_digest(parameter)
        for module_name, module in model.named_modules():
            module_prefix = f"{prefix}.buffer"
            if module_name:
                module_prefix = f"{module_prefix}.{module_name}"
            for name, buffer in module._buffers.items():
                if buffer is None or name in module._non_persistent_buffers_set:
                    continue
                digests[f"{module_prefix}.{name}"] = _state_digest(buffer)
    return dict(sorted(digests.items()))


def _gradient_state_digests(model_parts: object) -> dict[str, str]:
    """Fingerprint every rank-local parameter gradient immediately before the optimizer update."""
    if not isinstance(model_parts, (list, tuple)):
        model_parts = [model_parts]
    digests: dict[str, str] = {}
    for part_index, model in enumerate(model_parts):
        for name, parameter in model.named_parameters():
            key = f"model_part_{part_index}.gradient.{name}"
            digests[key] = "none" if parameter.grad is None else _state_digest(parameter.grad)
    return dict(sorted(digests.items()))


def _optimizer_state_digests(optimizers: object, model_parts: object) -> dict[str, str]:
    """Fingerprint every rank-local optimizer state value using stable parameter names."""
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]
    parameter_names = _model_parameter_names(model_parts)
    digests: dict[str, str] = {}
    for optimizer_index, optimizer in enumerate(optimizers):
        for group_index, group in enumerate(optimizer.param_groups):
            for parameter_index, parameter in enumerate(group["params"]):
                parameter_name = parameter_names.get(
                    id(parameter),
                    f"group_{group_index}.parameter_{parameter_index}",
                )
                prefix = f"optimizer_{optimizer_index}.{parameter_name}"
                state = optimizer.state.get(parameter, {})
                if not state:
                    digests[f"{prefix}.<empty>"] = _state_digest({})
                    continue
                for name, value in sorted(state.items(), key=lambda item: str(item[0])):
                    digests[f"{prefix}.{name}"] = _state_digest(value)
    return dict(sorted(digests.items()))


def _optimizer_group_digest(optimizers: object) -> str:
    """Fingerprint complete optimizer parameter-group settings except parameter identities."""
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]
    groups = [
        [{key: value for key, value in group.items() if key != "params"} for group in optimizer.param_groups]
        for optimizer in optimizers
    ]
    return _state_digest(groups)


def _optimizer_group_state(optimizers: object) -> list[list[dict[str, float | None]]]:
    """Capture LR and weight-decay values that must resume at the checkpoint boundary."""
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]
    return [
        [
            {
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]) if "weight_decay" in group else None,
            }
            for group in optimizer.param_groups
        ]
        for optimizer in optimizers
    ]


def _checkpoint_state_snapshot(trainer: object, *, state_is_being_saved: bool) -> dict[str, object]:
    """Capture discrete optimizer, scheduler, and RNG checkpoint state."""
    step_scheduler = trainer.step_scheduler
    if state_is_being_saved:
        scheduler_state = step_scheduler.state_dict()
        scheduler_position = {"step": int(scheduler_state["step"]), "epoch": int(scheduler_state["epoch"])}
    else:
        scheduler_position = {"step": int(step_scheduler.step), "epoch": int(step_scheduler.epoch)}

    lr_schedulers = trainer.lr_scheduler
    if lr_schedulers is not None and not isinstance(lr_schedulers, (list, tuple)):
        lr_schedulers = [lr_schedulers]
    lr_scheduler_state = [] if lr_schedulers is None else [scheduler.state_dict() for scheduler in lr_schedulers]

    return {
        "step_scheduler": scheduler_position,
        "optimizer_steps": _optimizer_step_summary(trainer.optimizer),
        "optimizer_groups": _optimizer_group_state(trainer.optimizer),
        "optimizer_group_digest": _optimizer_group_digest(trainer.optimizer),
        "lr_scheduler_digest": _state_digest(lr_scheduler_state),
        "rng_digest": _state_digest(trainer.rng.state_dict()),
    }


def _digest_manifest_mismatch(reference: object, restored: object, *, label: str) -> str | None:
    """Describe missing, unexpected, and changed entries in a state-digest manifest."""
    if not isinstance(reference, dict) or not isinstance(restored, dict):
        return f"{label} must be a digest mapping"
    missing = sorted(set(reference) - set(restored))
    unexpected = sorted(set(restored) - set(reference))
    mismatched = sorted(key for key in set(reference) & set(restored) if reference[key] != restored[key])
    if not missing and not unexpected and not mismatched:
        return None
    return (
        f"{label} differs: "
        f"missing={missing[:5]}, unexpected={unexpected[:5]}, mismatched={mismatched[:5]} "
        f"(counts: {len(missing)}/{len(unexpected)}/{len(mismatched)})"
    )


def _parameter_digest_subset(model_digests: object) -> object:
    """Return only parameter entries from a combined model parameter/buffer digest manifest."""
    if not isinstance(model_digests, dict):
        return model_digests
    return {key: value for key, value in model_digests.items() if ".parameter." in key}


def _buffer_digest_subset(model_digests: object) -> object:
    """Return only buffer entries from a combined model parameter/buffer digest manifest."""
    if not isinstance(model_digests, dict):
        return model_digests
    return {key: value for key, value in model_digests.items() if ".buffer." in key}


def _restored_state_mismatch(reference: dict[str, object], restored: dict[str, object]) -> str | None:
    """Return the first missing or changed required checkpoint component."""
    component_labels = {
        "step_scheduler": "step scheduler position",
        "optimizer_steps": "optimizer step counters",
        "optimizer_groups": "learning-rate/weight-decay state",
        "optimizer_group_digest": "complete optimizer parameter-group state",
        "lr_scheduler_digest": "LR scheduler state",
        "rng_digest": "RNG state",
    }
    for key, label in component_labels.items():
        if key not in reference:
            return f"reference artifact omitted required {label} ({key})"
        if key not in restored:
            return f"restored snapshot omitted required {label} ({key})"
        if reference[key] != restored[key]:
            return f"restored {label} does not match the shared-trajectory checkpoint ({key})"
    return None


def _trajectory_mismatch(
    reference: dict[str, object],
    resumed: dict[str, object],
    *,
    tolerance: _ResumeLossTolerance,
) -> str | None:
    """Compare exact batches/LRs and bounded losses for the resumed continuation."""
    reference_steps = {int(step): values for step, values in reference["steps"].items()}
    resumed_steps = {int(step): values for step, values in resumed["steps"].items()}
    if set(reference_steps) != set(resumed_steps):
        return f"resumed step set {sorted(resumed_steps)} does not match uninterrupted steps {sorted(reference_steps)}"

    first_step = min(reference_steps)
    for step in sorted(reference_steps):
        expected = reference_steps[step]
        actual = resumed_steps[step]
        if expected["batch_digest"] != actual["batch_digest"]:
            return f"resumed batch identity differs at step {step}; stateful dataloader position was not restored"
        if expected["lr"] != actual["lr"]:
            return f"resumed learning rate differs at step {step}: {expected['lr']} != {actual['lr']}"
        if step == first_step:
            expected_diagnostics = expected.get("diagnostics", {})
            actual_diagnostics = actual.get("diagnostics", {})
            for key, label in (
                ("pre_update_model_digests", "pre-update model parameters"),
                ("pre_update_optimizer_digests", "pre-update optimizer tensor state"),
            ):
                if key not in expected_diagnostics:
                    return f"uninterrupted trajectory omitted required {label} diagnostics ({key})"
                if key not in actual_diagnostics:
                    return f"resumed trajectory omitted required {label} diagnostics ({key})"
                reference_manifest = expected_diagnostics[key]
                resumed_manifest = actual_diagnostics[key]
                if key == "pre_update_model_digests":
                    reference_manifest = _parameter_digest_subset(reference_manifest)
                    resumed_manifest = _parameter_digest_subset(resumed_manifest)
                mismatch = _digest_manifest_mismatch(reference_manifest, resumed_manifest, label=label)
                if mismatch is not None:
                    return f"shared-trajectory state mismatch at step {step}: {mismatch}"
        reference_loss = float(expected["loss"])
        resumed_loss = float(actual["loss"])
        difference = abs(reference_loss - resumed_loss)
        absolute, relative, allowed_difference = _loss_tolerance_for_step(
            tolerance,
            first_step=step == first_step,
            reference_loss=reference_loss,
            resumed_loss=resumed_loss,
        )
        if not math.isfinite(difference) or difference > allowed_difference:
            scale = max(abs(reference_loss), abs(resumed_loss))
            relative_difference = 0.0 if scale == 0 and difference == 0 else difference / scale
            return (
                f"shared-trajectory loss mismatch at step {step}: uninterrupted={expected['loss']:.6f}, "
                f"resumed={actual['loss']:.6f}, abs_diff={difference:.6e}, "
                f"relative_diff={relative_difference:.6e}, allowed_diff={allowed_difference:.6e}, "
                f"atol={absolute:.6e}, rtol={relative:.6e}"
            )
    return None


def _digest_manifest_comparison(reference: object, resumed: object) -> dict[str, object]:
    """Return a compact comparison of two state-digest manifests."""
    if not isinstance(reference, dict) or not isinstance(resumed, dict):
        return {
            "matches": False,
            "reason": "diagnostic digest manifest missing or malformed",
        }
    missing = sorted(set(reference) - set(resumed))
    unexpected = sorted(set(resumed) - set(reference))
    mismatched = sorted(key for key in set(reference) & set(resumed) if reference[key] != resumed[key])
    return {
        "matches": not missing and not unexpected and not mismatched,
        "reference_count": len(reference),
        "resumed_count": len(resumed),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "mismatched_count": len(mismatched),
        "missing": missing[:20],
        "unexpected": unexpected[:20],
        "mismatched": mismatched[:20],
    }


def _resume_comparison_report(
    reference: dict[str, object],
    resumed: dict[str, object],
    tolerance: _ResumeLossTolerance,
) -> dict[str, object]:
    """Build a structured loss and first-update diagnostic report for one rank."""
    reference_steps = {int(step): values for step, values in reference["steps"].items()}
    resumed_steps = {int(step): values for step, values in resumed["steps"].items()}
    first_step = min(reference_steps)
    step_reports = []
    for step in sorted(set(reference_steps) | set(resumed_steps)):
        if step not in reference_steps or step not in resumed_steps:
            step_reports.append(
                {
                    "step": step,
                    "status": "missing",
                    "present_in_reference": step in reference_steps,
                    "present_in_resumed": step in resumed_steps,
                }
            )
            continue
        expected = reference_steps[step]
        actual = resumed_steps[step]
        reference_loss = float(expected["loss"])
        resumed_loss = float(actual["loss"])
        difference = abs(reference_loss - resumed_loss)
        scale = max(abs(reference_loss), abs(resumed_loss))
        relative_difference = 0.0 if scale == 0 and difference == 0 else difference / scale
        absolute, relative, allowed_difference = _loss_tolerance_for_step(
            tolerance,
            first_step=step == first_step,
            reference_loss=reference_loss,
            resumed_loss=resumed_loss,
        )
        step_reports.append(
            {
                "step": step,
                "stage": "first_forward_before_resumed_update" if step == first_step else "after_resumed_update",
                "reference_loss": reference_loss,
                "resumed_loss": resumed_loss,
                "absolute_difference": difference,
                "relative_difference": relative_difference,
                "loss_atol": absolute,
                "loss_rtol": relative,
                "allowed_loss_difference": allowed_difference,
                "within_loss_tolerance": math.isfinite(difference) and difference <= allowed_difference,
                "batch_matches": expected["batch_digest"] == actual["batch_digest"],
                "lr_matches": expected["lr"] == actual["lr"],
            }
        )

    diagnostic_report: dict[str, object] = {}
    if first_step in reference_steps and first_step in resumed_steps:
        reference_diagnostics = reference_steps[first_step].get("diagnostics", {})
        resumed_diagnostics = resumed_steps[first_step].get("diagnostics", {})
        reference_model_digests = reference_diagnostics.get("pre_update_model_digests")
        resumed_model_digests = resumed_diagnostics.get("pre_update_model_digests")
        diagnostic_report["pre_update_model_parameter_digests"] = _digest_manifest_comparison(
            _parameter_digest_subset(reference_model_digests),
            _parameter_digest_subset(resumed_model_digests),
        )
        diagnostic_report["pre_update_model_buffer_digests"] = _digest_manifest_comparison(
            _buffer_digest_subset(reference_model_digests),
            _buffer_digest_subset(resumed_model_digests),
        )
        for key in (
            "pre_update_optimizer_digests",
            "gradient_digests",
            "post_step_model_digests",
            "post_step_optimizer_digests",
        ):
            diagnostic_report[key] = _digest_manifest_comparison(
                reference_diagnostics.get(key),
                resumed_diagnostics.get(key),
            )

    failure = _trajectory_mismatch(
        reference,
        resumed,
        tolerance=tolerance,
    )
    return {
        "status": "passed" if failure is None else "failed",
        "tolerance": {
            "profile": tolerance.profile,
            "first_step_atol": tolerance.first_step_atol,
            "first_step_rtol": tolerance.first_step_rtol,
            "later_step_atol": tolerance.later_step_atol,
            "later_step_rtol": tolerance.later_step_rtol,
        },
        "steps": step_reports,
        "first_step_diagnostics": diagnostic_report,
        "blocking_failure": failure,
    }


def _report_resume_comparison(
    plan: _ResumePlan,
    reference: dict[str, object],
    resumed: dict[str, object],
    tolerance: _ResumeLossTolerance,
) -> dict[str, object]:
    """Persist exact per-rank resume metrics and print a combined rank-0 summary."""
    resumed_path = plan.artifact_dir / f"resumed_trajectory_rank_{_rank()}.json"
    resumed_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = resumed_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(resumed, sort_keys=True))
    temporary_path.replace(resumed_path)

    local_report = _resume_comparison_report(reference, resumed, tolerance)
    report_path = plan.artifact_dir / f"resume_comparison_rank_{_rank()}.json"
    temporary_path = report_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(local_report, indent=2, sort_keys=True))
    temporary_path.replace(report_path)

    reports = [local_report]
    if dist.is_initialized():
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, local_report)
    if _rank() != 0:
        return local_report

    combined_path = plan.artifact_dir / "resume_comparison.json"
    temporary_path = combined_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps({"reports": reports}, indent=2, sort_keys=True))
    temporary_path.replace(combined_path)
    print(
        f"[Resume correctness] tolerance_profile={tolerance.profile} "
        f"first_step_atol={tolerance.first_step_atol:.6e} first_step_rtol={tolerance.first_step_rtol:.6e} "
        f"later_step_atol={tolerance.later_step_atol:.6e} later_step_rtol={tolerance.later_step_rtol:.6e}"
    )
    for rank, report in enumerate(reports):
        for step_report in report["steps"]:
            if step_report.get("status") == "missing":
                print(f"[Resume correctness] rank={rank} {json.dumps(step_report, sort_keys=True)}")
                continue
            print(
                f"[Resume correctness] rank={rank} step={step_report['step']} stage={step_report['stage']} "
                f"reference_loss={step_report['reference_loss']:.9f} "
                f"resumed_loss={step_report['resumed_loss']:.9f} "
                f"abs_diff={step_report['absolute_difference']:.9e} "
                f"relative_diff={step_report['relative_difference']:.9e} "
                f"allowed_diff={step_report['allowed_loss_difference']:.9e}"
            )
        for name, diagnostic in report["first_step_diagnostics"].items():
            print(
                f"[Resume correctness] rank={rank} diagnostic={name} matches={diagnostic['matches']} "
                f"mismatched={diagnostic.get('mismatched_count', 0)} "
                f"missing={diagnostic.get('missing_count', 0)} "
                f"unexpected={diagnostic.get('unexpected_count', 0)} "
                f"mismatched_names={json.dumps(diagnostic.get('mismatched', [])[:5])} "
                f"missing_names={json.dumps(diagnostic.get('missing', [])[:5])} "
                f"unexpected_names={json.dumps(diagnostic.get('unexpected', [])[:5])}"
            )
    print(f"[Resume correctness] detailed comparison report={combined_path}")
    return local_report


def _gather_rank_failures(local_failure: str | None, *, check: str) -> str | None:
    """Gather rank-local resume failures and format one rank-0 failure message."""
    failures = [local_failure]
    if dist.is_initialized():
        failures = [None] * dist.get_world_size()
        dist.all_gather_object(failures, local_failure)
    if _rank() != 0:
        return None
    formatted = [f"rank {rank}: {failure}" for rank, failure in enumerate(failures) if failure is not None]
    if not formatted:
        return None
    return f"CHECKPOINT_ROBUSTNESS_PHASE_FAILURE phase=resume check={check}\n" + "\n".join(formatted)
