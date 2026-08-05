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
                self.steps[step] = {
                    "batch_digest": batch_digest,
                    "loss": float(log_data.metrics["loss"]),
                    "lr": float(log_data.metrics["lr"]),
                }
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
            digest.update(tensor.view(torch.uint8).numpy())
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
    """Capture discrete checkpoint state without materializing model or optimizer tensors."""
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
        "lr_scheduler_digest": _state_digest(lr_scheduler_state),
        "rng_digest": _state_digest(trainer.rng.state_dict()),
        "dataloader_digest": _state_digest(trainer.dataloader.state_dict()),
    }


def _restored_state_mismatch(reference: dict[str, object], restored: dict[str, object]) -> str | None:
    """Return the first missing or changed required checkpoint component."""
    component_labels = {
        "step_scheduler": "step scheduler position",
        "optimizer_steps": "optimizer step counters",
        "optimizer_groups": "learning-rate/weight-decay state",
        "lr_scheduler_digest": "LR scheduler state",
        "rng_digest": "RNG state",
        "dataloader_digest": "stateful dataloader position",
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
    first_loss_threshold: float,
    later_loss_threshold: float,
) -> str | None:
    """Compare exact batches/LRs and bounded losses for the resumed continuation."""
    if first_loss_threshold < 0 or later_loss_threshold < 0:
        return "resume loss thresholds must be non-negative"
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
        difference = abs(float(expected["loss"]) - float(actual["loss"]))
        threshold = first_loss_threshold if step == first_step else later_loss_threshold
        if not math.isfinite(difference) or difference > threshold:
            threshold_label = "first-step" if step == first_step else "later-step"
            return (
                f"shared-trajectory loss mismatch at step {step}: uninterrupted={expected['loss']:.6f}, "
                f"resumed={actual['loss']:.6f}, diff={difference:.6e}, "
                f"{threshold_label}_threshold={threshold:.6e}"
            )
    return None


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
