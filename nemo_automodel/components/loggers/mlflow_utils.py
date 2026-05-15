# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, cast

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def configure_mlflow(cfg: Any) -> Optional[Any]:
    """Configure MLflow on rank 0 and start (or resume) a run.

    Also installs a `sys.excepthook` so crashed jobs report as FAILED rather
    than FINISHED. After this call the recipe logs via module-level
    `mlflow.log_params` and `mlflow.log_metrics` directly; on non-rank-0
    processes `mlflow.active_run()` is None so those calls become no-ops
    naturally.

    Returns the active run on rank 0, or None when MLflow is not configured
    or on non-rank-0 processes.
    """
    if not (dist.is_initialized() and dist.get_rank() == 0):
        return None
    mlflow_config = cfg.get("mlflow", {})
    if not mlflow_config:
        return None

    try:
        import mlflow
    except ImportError as e:
        raise ImportError("MLflow is not installed. Please install it with: uv add mlflow") from e

    if uri := mlflow_config.get("tracking_uri", None):
        mlflow.set_tracking_uri(uri)

    experiment_name = mlflow_config.get("experiment_name", "automodel-experiment")
    artifact_location = mlflow_config.get("artifact_location", None)
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else mlflow.create_experiment(name=experiment_name, artifact_location=artifact_location)
        )
    except Exception as e:
        logger.warning(f"Failed to create/get experiment: {e}")
        experiment_id = "0"

    # ConfigNode (Automodel's YAML wrapper) needs .to_dict(); plain dicts —
    # which appear as the fallback when `tags:` is absent — don't have it.
    raw_tags = mlflow_config.get("tags", {})
    tags = raw_tags.to_dict() if hasattr(raw_tags, "to_dict") else dict(raw_tags)

    # Resume the previous MLflow run on restart, instead of starting a new
    # one. The run id comes from MLFLOW_RUN_ID (an explicit user override,
    # always honoured) or from a `mlflow_run_id` sidecar in the checkpoint
    # dir. `mlflow.resume` (default true) gates the *implicit* sidecar
    # lookup only; the env var remains effective even with `resume: false`.
    # The sidecar is always written below so a future `resume: true` launch
    # (or post-hoc recovery via the env var) can still find the most recent
    # run for this checkpoint dir.
    resume_enabled = mlflow_config.get("resume", True)
    ckpt_dir = cfg.get("checkpoint.checkpoint_dir", None)
    sidecar = Path(ckpt_dir) / "mlflow_run_id" if ckpt_dir else None
    existing_run_id = os.environ.get("MLFLOW_RUN_ID") or (
        sidecar.read_text().strip() if resume_enabled and sidecar and sidecar.exists() else None
    )

    # UI "Description" field — surfaced via the `mlflow.note.content` tag.
    if description := mlflow_config.get("description", None):
        tags["mlflow.note.content"] = description

    run = mlflow.start_run(
        experiment_id=experiment_id,
        run_id=existing_run_id,  # None → new run; set → resume the same run
        run_name=mlflow_config.get("run_name", ""),  # ignored when run_id provided
        tags=tags,
    )

    # Persist the run_id so a future restart can resume this run.
    if existing_run_id is None and sidecar is not None:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(run.info.run_id)

    _install_mlflow_failure_hook()

    # `use_orig_values=True` renders callables as dotted-path strings (e.g.
    # `torch.utils.data.default_collate`) instead of `to_dict()`'s raw repr
    # (`<function ... at 0x7f...>`), which embeds an in-process memory address
    # and would log different values on every run. The same dict is used for
    # params and the artifact so they stay consistent.
    # `redact_sensitive` is left at its default (off): the substring matcher
    # treats "tokenizer" as a secret due to the "token" substring, and secrets
    # in this project come from env vars.
    config_dict = cfg.to_yaml_dict(use_orig_values=True)

    # On resume, params from the original run are already logged; re-logging
    # would raise MlflowException for any value that has legitimately changed
    # (e.g. `step_scheduler.max_steps` after a budget extension). Instead the
    # snapshot is written as a timestamped artifact so each launch's config
    # is preserved.
    if existing_run_id is None:
        # `mlflow.flatten_depth` controls nested-config flattening. Default 1
        # matches the prior implementation, keeping existing runs comparable.
        flatten_depth = mlflow_config.get("flatten_depth", 1)
        mlflow.log_params(flatten_params_for_mlflow(config_dict, max_depth=flatten_depth))
        mlflow.log_dict(config_dict, "config.yaml")
    else:
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        mlflow.log_dict(config_dict, f"config.resumed-{ts}.yaml")

    logger.info(f"MLflow run started: {run.info.run_id}")
    logger.info(f"View run at: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}/runs/{run.info.run_id}")

    return run


def flatten_params_for_mlflow(
    params: Dict[str, Any],
    max_depth: Optional[int] = 1,
    prefix: str = "",
    _depth: int = 0,
) -> Dict[str, str]:
    """Flatten nested dicts to dot-keyed strings for MLflow params.

    `max_depth` controls how many levels of dict nesting get split into
    individual keys; deeper nesting is stringified at that depth's leaf:

    * `1` (default) — split one level, e.g.
      `model.text_config: "{'output_hidden_states': True}"`.
    * `N > 1` — split up to N levels deep.
    * `None` — fully recursive: every leaf gets its own key, e.g.
      `model.text_config.output_hidden_states: 'True'`.

    Lists and tuples are always stringified; per-element keys would add
    noise without helping comparison (e.g. `betas: [0.9, 0.95]`).
    """
    out: Dict[str, str] = {}
    for k, v in params.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and (max_depth is None or _depth < max_depth):
            out.update(
                flatten_params_for_mlflow(
                    cast(Dict[str, Any], v), max_depth=max_depth, prefix=full_key, _depth=_depth + 1
                )
            )
        else:
            out[full_key] = str(v)
    return out


def to_float_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Clean a metrics dict before passing to `mlflow.log_metrics`.

    `MetricsSample.to_dict()` mixes numbers, tensors, and a string `timestamp`
    field, but `mlflow.log_metrics` only accepts numeric values. This function
    filters and coerces values so the call succeeds:

    * Non-numeric values (e.g. `timestamp`) — dropped (otherwise mlflow raises
      `TypeError: must be real number, not str`).
    * Tensors — coerced via `.item()` (multi-element tensors are reduced with
      `.mean()` first).
    * Python scalars — coerced to float.
    """
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            out[k] = float(v.item() if v.numel() == 1 else v.mean().item())
        elif isinstance(v, (int, float)):
            out[k] = float(v)
        else:
            logger.warning(f"Skipping MLflow metric {k} with unsupported type: {type(v)}")
    return out


def end_mlflow_active_run_as_killed() -> None:
    """End the active MLflow run with status=KILLED.

    Called from the SIGTERM handler so interrupted runs show as KILLED
    rather than FINISHED in the MLflow UI (mlflow's atexit handler defaults
    to FINISHED on graceful exit, making cancelled and clean runs look
    identical).

    No-op if no run is active; errors from `end_run` are suppressed so that
    signal-handler reentrancy in mlflow can't crash the SIGTERM path.
    """
    try:
        import mlflow
    except ImportError:
        return

    if mlflow.active_run() is not None:
        with contextlib.suppress(Exception):
            mlflow.end_run(status="KILLED")


def _install_mlflow_failure_hook() -> None:
    """Mark active MLflow run as FAILED on uncaught Python exceptions.

    MLflow's atexit handler ends the run with default status=FINISHED on
    process exit, making a crashed run indistinguishable from a clean one
    in the UI. We chain a `sys.excepthook` that fires before atexit and
    explicitly sets FAILED first; the previous excepthook is preserved so
    default traceback printing still happens.

    This only covers Python exceptions on the main thread. SIGKILL (OOM,
    job cancellation) and NCCL watchdog `std::terminate` paths bypass it
    and leave the run in RUNNING until a server-side janitor times it out.
    Worker-thread exceptions need `threading.excepthook` separately.
    """
    try:
        import mlflow
    except ImportError:
        return

    prev_excepthook: Callable[..., None] = sys.excepthook

    # Idempotent: avoid wrapping our own hook in chains.
    if getattr(prev_excepthook, "_mlflow_failure_hook", False):
        return

    def hook(exc_type: type[BaseException], exc_val: BaseException, exc_tb: Any) -> None:
        if mlflow.active_run() is not None:
            with contextlib.suppress(Exception):
                mlflow.end_run(status="FAILED")
        prev_excepthook(exc_type, exc_val, exc_tb)

    hook._mlflow_failure_hook = True  # type: ignore[attr-defined]
    sys.excepthook = hook
