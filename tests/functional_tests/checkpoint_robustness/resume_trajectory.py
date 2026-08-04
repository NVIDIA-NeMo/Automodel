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
            batch_digest = _state_digest(batches) if step in self.plan.comparison_steps else None
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
