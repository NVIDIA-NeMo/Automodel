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
import sys
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
    # Back-compat shim. All MLflow setup logic lives in MLflowConfig.build (the
    # single implementation); recipes construct MLflowConfig via RecipeConfig.mlflow
    # and call build() directly. This wrapper maps a raw cfg's `mlflow:` block onto
    # the typed config so existing callers keep working.
    from nemo_automodel.components.loggers.loggers import MLflowConfig

    if not (dist.is_initialized() and dist.get_rank() == 0):
        return None
    mlflow_config = cfg.get("mlflow", {})
    if not mlflow_config:
        return None

    # ConfigNode (Automodel's YAML wrapper) needs .to_dict(); plain dicts —
    # which appear as the fallback when `tags:` is absent — don't have it.
    raw_tags = mlflow_config.get("tags", {})
    tags = raw_tags.to_dict() if hasattr(raw_tags, "to_dict") else dict(raw_tags)

    config = MLflowConfig(
        experiment_name=mlflow_config.get("experiment_name", "automodel-experiment"),
        run_name=mlflow_config.get("run_name", ""),
        tracking_uri=mlflow_config.get("tracking_uri", None),
        artifact_location=mlflow_config.get("artifact_location", None),
        tags=tags,
        resume=mlflow_config.get("resume", True),
        description=mlflow_config.get("description", None),
        flatten_depth=mlflow_config.get("flatten_depth", 1),
    )
    return config.build(
        checkpoint_dir=cfg.get("checkpoint.checkpoint_dir", None),
        run_config=cfg.to_yaml_dict(use_orig_values=True),
    )


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
