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

"""On-policy data regeneration loop for EAGLE-3 draft training (train-with-decode).

EAGLE-3 teacher-forces the draft on a static dataset whose assistant turns were
often written by a *different* model than the target being distilled, so the
draft trains against off-distribution supervision (this is why the recipes ship
an offline ``regenerate.py`` pass). This module runs that regeneration
*online*, interleaved with training:

* On a cadence (``regen.every_steps``), rank 0 launches a detached **worker
  subprocess** pinned to a reserved GPU (``regen.cuda_visible_devices``). The
  worker boots a plain vLLM OpenAI server for the frozen target and drives the
  existing ``regenerate`` machinery over a prompt slice, writing a fresh
  directory of parquet shards. Training never blocks; at most one cycle is in
  flight.
* At the next epoch boundary the recipe checks for a completed cycle, and if one
  is ready **all data-parallel ranks** rebuild the training dataloader against
  the new shard directory in lockstep (the decision is broadcast from rank 0 so
  the ranks never diverge).

Because the EAGLE target is frozen, the regenerated distribution is stationary:
the value here is pipelining (regenerate just-in-time instead of one giant
upfront pass) and freshness (sampling new responses each cycle with
``temperature > 0``), plus the plumbing a future draft-in-the-loop mode would
reuse. The trainer process never imports vllm; the worker inherits the training
env minus the torchrun/elastic variables (see ``decode_eval._worker_env``).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass, fields
from typing import Any

from nemo_automodel.components.speculative.decode_eval import (
    _resolve_worker_port,
    _wait_for_server,
    _worker_env,
)

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = "regen_config.json"
_DONE_FILENAME = "READY"
_SHARDS_DIRNAME = "shards"
_WORKER_LOG_FILENAME = "worker.log"


@dataclass
class RegenConfig:
    """Resolved settings for the online on-policy regeneration loop.

    ``every_steps`` is the launch cadence in optimizer steps; a completed cycle
    is only swapped in at an epoch boundary (mid-epoch dataloader rebuilds would
    desync the sampler and the scheduler's step count). ``cuda_visible_devices``
    is the GPU (or comma list) reserved for the regeneration engine; training
    must not place work there. ``server_python`` is the interpreter used to
    launch the vLLM server (the training env need not have vllm installed); the
    regeneration client runs in the training env over HTTP.
    """

    every_steps: int
    cuda_visible_devices: str
    target_model: str
    input_data: str
    output_dir: str
    served_model_name: str = "target"
    server_python: str | None = None
    split: str = "train"
    messages_column: str = "messages"
    dataset_name: str | None = None
    shard_size: int = 1000
    concurrency: int = 32
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    shuffle_seed: int = 42
    reasoning: str = "none"
    port: int = 0
    gpu_memory_utilization: float = 0.85
    max_model_len: int | None = None
    trust_remote_code: bool = False
    timeout_s: float = 3600.0


def resolve_regen_config(
    recipe_cfg: Any,
    *,
    default_target: str,
    default_input_data: str | None,
    output_dir: str,
) -> RegenConfig | None:
    """Read the optional ``regen:`` block from ``recipe_args``.

    Returns ``None`` when the block is absent or ``every_steps`` is unset/0
    (feature disabled). Raises on a partially-configured block so a typo'd GPU
    reservation fails at setup rather than silently regenerating on a training
    GPU. Field defaults live on :class:`RegenConfig` only; the block just
    overrides the fields it sets.
    """
    block = recipe_cfg.get("regen", None)
    if block is None:
        return None
    every_steps = int(block.get("every_steps", 0) or 0)
    if every_steps <= 0:
        return None
    block = block.to_dict() if hasattr(block, "to_dict") else dict(block)
    unknown = set(block) - {field.name for field in fields(RegenConfig)}
    if unknown:
        raise ValueError(f"Unknown regen option(s): {', '.join(sorted(unknown))}")
    if "output_dir" in block:
        raise ValueError(
            "regen.output_dir is not configurable: shards are always written under "
            "<run output_dir>/regen. Remove it from the regen block."
        )
    cuda_visible_devices = block.get("cuda_visible_devices", None)
    if cuda_visible_devices is None or str(cuda_visible_devices) == "":
        raise ValueError(
            "regen.cuda_visible_devices is required: the regeneration engine needs a GPU the training "
            "job does not use (e.g. reserve one card and shrink the torchrun world accordingly)."
        )
    input_data = block.get("input_data", None) or default_input_data
    if not input_data:
        raise ValueError("regen.input_data is required when recipe_args.train_data_path is not set.")
    required = {"every_steps", "cuda_visible_devices", "target_model", "input_data", "output_dir"}
    overrides = {}
    for field in fields(RegenConfig):
        if field.name in required:
            continue
        value = block.get(field.name, None)
        if value is not None:
            overrides[field.name] = value
    return RegenConfig(
        every_steps=every_steps,
        cuda_visible_devices=str(cuda_visible_devices),
        target_model=str(block.get("target_model", None) or default_target),
        input_data=str(input_data),
        output_dir=os.path.join(output_dir, "regen"),
        **overrides,
    )


class RegenRunner:
    """Rank-0 trainer-side orchestrator: launch regeneration at cadence, hand ready cycles to the recipe.

    At most one regeneration subprocess is alive at a time; a launch that lands
    while the previous cycle is still running is skipped (the next cadence
    boundary picks it up). A cycle is "ready" once its worker has written the
    ``READY`` marker; :meth:`take_ready_shards` returns the newest ready cycle's
    shard directory exactly once, so the recipe can swap it into the dataloader.
    """

    def __init__(self, config: RegenConfig):
        self.config = config
        self._proc: subprocess.Popen | None = None
        self._worker_log_path: str | None = None
        self._launched_for_step = -1
        self._last_bucket = 0
        self._consumed: set[str] = set()
        # The cycle whose shards currently feed training; never deleted while active.
        self._active_cycle: str | None = None
        os.makedirs(config.output_dir, exist_ok=True)

    def _cycle_dirs(self) -> list[tuple[int, str]]:
        """Return valid cycle directories in numeric order, ignoring stray names."""
        try:
            entries = os.listdir(self.config.output_dir)
        except FileNotFoundError:
            return []
        parsed = []
        for entry in entries:
            if not entry.startswith("cycle_"):
                continue
            try:
                parsed.append((int(entry.split("_", 1)[1]), entry))
            except ValueError:
                logger.warning("regen: ignoring invalid cycle directory %s", entry)
        return sorted(parsed)

    def _reap_finished_worker(self) -> None:
        """Report a completed worker and release its Popen state."""
        if self._proc is None:
            return
        return_code = self._proc.poll()
        if return_code is None:
            return
        if return_code != 0:
            logger.error(
                "regen: worker for step %d exited with code %d; see %s",
                self._launched_for_step,
                return_code,
                self._worker_log_path,
            )
        else:
            logger.info("regen: worker for step %d completed", self._launched_for_step)
        self._proc = None
        self._worker_log_path = None

    def due(self, global_step: int) -> bool:
        """Whether a new regeneration cycle should launch at this optimizer step."""
        return global_step // self.config.every_steps > self._last_bucket

    def resume_from_step(self, global_step: int) -> None:
        """Align the launch cadence to a restored ``global_step`` after a resume.

        Without this a resumed run starts with ``_last_bucket == 0`` and
        :meth:`due` fires immediately, relaunching a redundant cycle for a cadence
        region that already ran before the checkpoint. Superseded on-disk cycles
        left by the pre-crash run are reclaimed by :meth:`take_ready_shards` on the
        next swap (it frees every ready cycle but the newest), so no cycle path
        needs to be persisted across the checkpoint.
        """
        self._last_bucket = global_step // self.config.every_steps
        self._launched_for_step = global_step

    def maybe_launch(self, global_step: int) -> bool:
        """Launch a regeneration worker if the cadence is due and none is running."""
        if not self.due(global_step):
            return False
        self._reap_finished_worker()
        if self._proc is not None and self._proc.poll() is None:
            logger.info(
                "regen: previous cycle (step %d) still running, skipping launch at step %d",
                self._launched_for_step,
                global_step,
            )
            return False
        cycle_dir = os.path.join(self.config.output_dir, f"cycle_{global_step}")
        # A half-written cycle from a crashed prior worker (leftover shards and/or a
        # stale READY marker) must be wiped, not reused: ``regenerate`` aborts on a
        # non-empty shard dir, and a stale marker could make the cycle look ready.
        # Start every launch from a clean directory.
        self._remove_cycle(cycle_dir)
        os.makedirs(cycle_dir, exist_ok=True)
        config_json = os.path.join(cycle_dir, _CONFIG_FILENAME)
        with open(config_json, "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        argv = [
            sys.executable,
            "-m",
            "nemo_automodel.components.speculative.regen_loop",
            "--config-json",
            config_json,
            "--cycle-dir",
            cycle_dir,
            "--step",
            str(global_step),
        ]
        log_path = os.path.join(cycle_dir, _WORKER_LOG_FILENAME)
        with open(log_path, "ab") as log_file:
            self._proc = subprocess.Popen(
                argv,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=_worker_env(self.config.cuda_visible_devices),
                start_new_session=True,
            )
        self._last_bucket = global_step // self.config.every_steps
        self._launched_for_step = global_step
        self._worker_log_path = log_path
        logger.info(
            "regen: launched cycle for step %d on CUDA_VISIBLE_DEVICES=%s (pid %d, log %s)",
            global_step,
            self.config.cuda_visible_devices,
            self._proc.pid,
            log_path,
        )
        return True

    def take_ready_shards(self) -> str | None:
        """Return the newest ready, not-yet-consumed cycle's shard directory, or None.

        A cycle is ready once its ``READY`` marker exists. Older ready cycles are
        marked consumed and skipped so the recipe always trains on the freshest
        regenerated data rather than replaying stale cycles.
        """
        self._reap_finished_worker()
        # Single directory scan: collect every ready cycle (sorted oldest to newest).
        ready = []
        for _, cycle_dir in self._cycle_dirs():
            cycle_path = os.path.join(self.config.output_dir, cycle_dir)
            if os.path.exists(os.path.join(cycle_path, _DONE_FILENAME)):
                ready.append(cycle_path)
        # Train on the freshest cycle not already consumed; everything ready up to
        # and including it is marked consumed so stale cycles are never replayed.
        fresh = [cycle_path for cycle_path in ready if cycle_path not in self._consumed]
        if not fresh:
            return None
        newest = fresh[-1]
        self._consumed.update(ready)
        # Free every superseded ready cycle still on disk, not just the previously
        # active one: when several cycles complete between two swaps only the newest
        # feeds training, so the intermediate ones would otherwise be stranded and
        # disk would grow one full dataset per cycle. The newest is kept.
        for cycle_path in ready:
            if cycle_path != newest:
                self._remove_cycle(cycle_path)
        self._active_cycle = newest
        return os.path.join(newest, _SHARDS_DIRNAME)

    @staticmethod
    def _remove_cycle(cycle_path: str) -> None:
        try:
            shutil.rmtree(cycle_path)
        except FileNotFoundError:
            pass
        except OSError:
            logger.exception("regen: failed to remove superseded cycle %s", cycle_path)

    def shutdown(self) -> None:
        """Terminate a still-running cycle (its vLLM child dies with the session)."""
        if self._proc is not None and self._proc.poll() is None:
            logger.info("regen: terminating in-flight cycle (pid %d)", self._proc.pid)
            try:
                process_group = os.getpgid(self._proc.pid)
                os.killpg(process_group, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process_group = None
                self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if process_group is not None:
                    try:
                        os.killpg(process_group, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        self._proc.kill()
                else:
                    self._proc.kill()
                self._proc.wait()
        elif self._proc is not None:
            self._reap_finished_worker()
        self._proc = None
        self._worker_log_path = None


# --------------------------------------------------------------------------- #
# Worker CLI: runs in its own process on the reserved GPU.
# --------------------------------------------------------------------------- #


def _target_server_argv(cfg: RegenConfig, port: int) -> list[str]:
    """Build the plain (non-speculative) vLLM OpenAI server argv for the frozen target."""
    server_python = cfg.server_python or sys.executable
    argv = [
        server_python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        cfg.target_model,
        "--served-model-name",
        cfg.served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
    ]
    if cfg.max_model_len is not None:
        argv += ["--max-model-len", str(cfg.max_model_len)]
    if cfg.trust_remote_code:
        argv += ["--trust-remote-code"]
    return argv


def _regenerate_argv(cfg: RegenConfig, server: str, shards_dir: str) -> list[str]:
    """Build the regenerate CLI args to run against the booted target server."""
    argv = [
        "--input-data",
        cfg.input_data,
        "--output-dir",
        shards_dir,
        "--target-server",
        f"{server}/v1",
        "--model",
        cfg.served_model_name,
        "--messages-column",
        cfg.messages_column,
        "--split",
        cfg.split,
        "--shuffle-seed",
        str(cfg.shuffle_seed),
        "--shard-size",
        str(cfg.shard_size),
        "--concurrency",
        str(cfg.concurrency),
        "--max-new-tokens",
        str(cfg.max_new_tokens),
        "--temperature",
        str(cfg.temperature),
        "--top-p",
        str(cfg.top_p),
        "--timeout-s",
        str(cfg.timeout_s),
        "--reasoning",
        cfg.reasoning,
    ]
    if cfg.dataset_name:
        argv += ["--dataset-name", cfg.dataset_name]
    return argv


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="regen worker: serve the frozen target and regenerate on-policy training shards"
    )
    parser.add_argument("--config-json", required=True, help="JSON dump of RegenConfig (written at launch)")
    parser.add_argument("--cycle-dir", required=True, help="output dir for this cycle (shards + READY marker)")
    parser.add_argument("--step", type=int, required=True)
    return parser


def main(argv=None) -> int:
    """Worker entry: boot a target server, regenerate a shard dir, write the READY marker."""
    from nemo_automodel.components.speculative.regenerate import main as regenerate_main

    args = _build_parser().parse_args(argv)
    with open(args.config_json) as f:
        cfg = RegenConfig(**json.load(f))
    port = _resolve_worker_port(cfg.port)
    server = f"http://127.0.0.1:{port}"
    shards_dir = os.path.join(args.cycle_dir, _SHARDS_DIRNAME)
    # The READY marker is written only after regeneration returns 0, and the
    # runner only ever hands training a cycle that has the marker, so a
    # mid-write shard set is never swapped in.
    server_argv = _target_server_argv(cfg, port)
    print(f"regen worker: starting target server: {' '.join(server_argv)}", flush=True)
    proc = subprocess.Popen(server_argv)
    try:
        _wait_for_server(server, proc, cfg.timeout_s)
        rc = regenerate_main(_regenerate_argv(cfg, server, shards_dir))
        if rc != 0:
            raise RuntimeError(f"regenerate exited with code {rc}")
        with open(os.path.join(args.cycle_dir, _DONE_FILENAME), "w") as f:
            f.write(str(args.step))
        print(f"regen worker: cycle {args.step} ready at {shards_dir}", flush=True)
        return 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
