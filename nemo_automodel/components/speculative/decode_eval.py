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

"""Periodic real-acceptance-length eval during draft training (decode eval).

Training-time draft metrics (loss, top-1 accuracy, the simulated ``tau_sim``)
are proxies: the acceptance length that matters is produced by the draft and
the target interacting inside a real speculative-decoding engine. This module
closes that gap without touching the training step:

* On a cadence (``decode_eval.every_steps``), rank 0 snapshots the current
  draft weights to disk and launches a detached **worker subprocess** pinned to
  a reserved GPU (``decode_eval.cuda_visible_devices``). Training continues
  immediately; at most one eval is in flight.
* The worker (this module's CLI) packages the snapshot through the existing
  ``serve_vllm`` conversion, boots a vLLM OpenAI server as its child, drives a
  fixed prompt set through it with the existing ``bench_vllm`` workload runner,
  and writes a JSON result (real ``accept_length`` from the engine's
  spec-decode counters) next to the snapshot.
* The trainer's logging block collects finished results and logs them as
  ``train/tau_real`` (with ``train/tau_real_step`` marking the optimizer step
  the evaluated snapshot was taken at).

The trainer process never imports vllm; the worker inherits the training env
minus the torchrun/elastic variables, so the engine initializes as a plain
single-process job on the reserved GPU.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from argparse import Namespace
from dataclasses import asdict, dataclass, fields
from typing import Any

logger = logging.getLogger(__name__)

_RESULT_FILENAME = "result.json"
_CONFIG_FILENAME = "decode_eval_config.json"
_WORKER_LOG_FILENAME = "worker.log"

# torchrun / elastic / rank state that must NOT leak into the worker: vLLM would
# otherwise try to join the training job's process group (or bind its ports).
_SCRUBBED_ENV_PREFIXES = ("TORCHELASTIC_", "PET_")
_SCRUBBED_ENV_KEYS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "ROLE_NAME",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
)


@dataclass
class DecodeEvalConfig:
    """Resolved settings for the periodic decode eval.

    ``every_steps`` is the launch cadence in optimizer steps; launches are
    checked at the recipe's logging points, so it should be a multiple of
    ``log_every_steps`` (a non-multiple fires at the first log point past the
    boundary). ``cuda_visible_devices`` is the GPU (or comma list) reserved for
    the eval engine; training must not place work there.
    """

    every_steps: int
    cuda_visible_devices: str
    target_model: str
    input_data: str
    output_dir: str
    num_speculative_tokens: int
    num_prompts: int = 32
    concurrency: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    messages_column: str = "messages"
    prompt_column: str | None = None
    split: str = "train"
    dataset_name: str | None = None
    shuffle_seed: int = 42
    port: int = 0
    gpu_memory_utilization: float = 0.8
    max_model_len: int | None = None
    trust_remote_code: bool = False
    timeout_s: float = 1800.0


def resolve_decode_eval_config(
    recipe_cfg: Any,
    *,
    default_target: str,
    default_input_data: str | None,
    default_num_speculative_tokens: int,
    output_dir: str,
) -> DecodeEvalConfig | None:
    """Read the optional ``decode_eval:`` block from ``recipe_args``.

    Returns ``None`` when the block is absent or ``every_steps`` is unset/0
    (feature disabled). Raises on a partially-configured block so a typo'd GPU
    reservation fails at setup rather than silently evaluating on a training
    GPU. Field defaults live on :class:`DecodeEvalConfig` only; the block just
    overrides the fields it sets.
    """
    block = recipe_cfg.get("decode_eval", None)
    if block is None:
        return None
    every_steps = int(block.get("every_steps", 0) or 0)
    if every_steps <= 0:
        return None
    block = block.to_dict() if hasattr(block, "to_dict") else dict(block)
    unknown = set(block) - {field.name for field in fields(DecodeEvalConfig)}
    if unknown:
        raise ValueError(f"Unknown decode_eval option(s): {', '.join(sorted(unknown))}")
    cuda_visible_devices = block.get("cuda_visible_devices", None)
    if cuda_visible_devices is None or str(cuda_visible_devices) == "":
        raise ValueError(
            "decode_eval.cuda_visible_devices is required: the eval engine needs a GPU the training "
            "job does not use (e.g. reserve one card and shrink the torchrun world accordingly)."
        )
    input_data = block.get("input_data", None) or default_input_data
    if not input_data:
        raise ValueError("decode_eval.input_data is required when recipe_args.train_data_path is not set.")
    num_speculative_tokens = int(block.get("num_speculative_tokens", None) or default_num_speculative_tokens)
    if num_speculative_tokens <= 0:
        raise ValueError("decode_eval.num_speculative_tokens must be greater than 0.")
    required = {
        "every_steps",
        "cuda_visible_devices",
        "target_model",
        "input_data",
        "output_dir",
        "num_speculative_tokens",
    }
    overrides = {}
    for field in fields(DecodeEvalConfig):
        if field.name in required:
            continue
        value = block.get(field.name, None)
        if value is not None:
            overrides[field.name] = value
    return DecodeEvalConfig(
        every_steps=every_steps,
        cuda_visible_devices=str(cuda_visible_devices),
        target_model=str(block.get("target_model", None) or default_target),
        input_data=str(input_data),
        output_dir=os.path.join(output_dir, "decode_eval"),
        num_speculative_tokens=num_speculative_tokens,
        **overrides,
    )


def export_draft_snapshot(draft_model, out_dir: str) -> str:
    """Write the draft's current weights as a serve-ready consolidated export.

    Produces ``<out_dir>/model/consolidated/{model.safetensors, config.json}``,
    the same layout the final checkpoint's consolidated export uses, so
    ``serve_vllm.resolve_draft_artifacts`` can consume it directly (the
    d2t/t2d vocab-mapping buffers ride along in the state dict).

    Assumes the draft is replicated on the calling rank (the EAGLE-3 draft
    trains under DDP, so rank 0 holds the full weights); a sharded draft would
    snapshot an incomplete state dict.

    Args:
        draft_model: the (unwrapped) draft ``nn.Module``; its parameter and
            buffer tensors are copied to CPU, so shapes/layouts are whatever the
            draft's ``state_dict()`` reports.
        out_dir: snapshot root; created if missing.

    Returns:
        The ``model/consolidated`` directory path containing the export.
    """
    from safetensors.torch import save_file

    consolidated = os.path.join(out_dir, "model", "consolidated")
    os.makedirs(consolidated, exist_ok=True)
    state = {k: v.detach().to("cpu").contiguous() for k, v in draft_model.state_dict().items()}
    save_file(state, os.path.join(consolidated, "model.safetensors"))
    draft_model.config.to_json_file(os.path.join(consolidated, "config.json"))
    return consolidated


def _worker_env(cuda_visible_devices: str) -> dict[str, str]:
    """Training env minus torchrun/elastic state, pinned to the reserved GPU."""
    env = {
        k: v for k, v in os.environ.items() if k not in _SCRUBBED_ENV_KEYS and not k.startswith(_SCRUBBED_ENV_PREFIXES)
    }
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env


class DecodeEvalRunner:
    """Rank-0 trainer-side orchestrator: launch at cadence, collect results.

    At most one eval subprocess is alive at a time; a launch that lands while
    the previous eval is still running is skipped (the next cadence boundary
    picks it up). Results are one-line JSON files the worker writes on success;
    ``collect()`` returns the not-yet-seen ones in step order.
    """

    def __init__(self, config: DecodeEvalConfig):
        self.config = config
        self._proc: subprocess.Popen | None = None
        self._worker_log_path: str | None = None
        self._launched_for_step = -1
        self._last_bucket = 0
        self._collected: set[str] = set()
        os.makedirs(config.output_dir, exist_ok=True)
        for _, step_dir in self._step_dirs():
            result_path = os.path.join(config.output_dir, step_dir, _RESULT_FILENAME)
            if os.path.exists(result_path):
                self._collected.add(result_path)
                self._cleanup_snapshot(os.path.join(config.output_dir, step_dir))

    def _step_dirs(self) -> list[tuple[int, str]]:
        """Return valid step directories in numeric order, ignoring stray names."""
        try:
            entries = os.listdir(self.config.output_dir)
        except FileNotFoundError:
            return []
        parsed = []
        for entry in entries:
            if not entry.startswith("step_"):
                continue
            try:
                parsed.append((int(entry.split("_", 1)[1]), entry))
            except ValueError:
                logger.warning("decode_eval: ignoring invalid step directory %s", entry)
        return sorted(parsed)

    @staticmethod
    def _cleanup_snapshot(step_dir: str) -> None:
        """Remove heavyweight weights after an eval no longer needs them."""
        try:
            shutil.rmtree(os.path.join(step_dir, "model"))
        except FileNotFoundError:
            pass
        except OSError:
            logger.exception("decode_eval: failed to remove snapshot under %s", step_dir)

    def _reap_finished_worker(self) -> None:
        """Report a completed worker and release its Popen state."""
        if self._proc is None:
            return
        return_code = self._proc.poll()
        if return_code is None:
            return
        if return_code != 0:
            logger.error(
                "decode_eval: worker for step %d exited with code %d; see %s",
                self._launched_for_step,
                return_code,
                self._worker_log_path,
            )
            self._cleanup_snapshot(os.path.join(self.config.output_dir, f"step_{self._launched_for_step}"))
        else:
            logger.info("decode_eval: worker for step %d completed", self._launched_for_step)
        self._proc = None
        self._worker_log_path = None

    def due(self, global_step: int) -> bool:
        """Whether a new eval should launch at this optimizer step."""
        return global_step // self.config.every_steps > self._last_bucket

    def maybe_launch(self, global_step: int, draft_model) -> bool:
        """Snapshot the draft and launch the worker if the cadence is due and no eval is running."""
        if not self.due(global_step):
            return False
        self._reap_finished_worker()
        if self._proc is not None and self._proc.poll() is None:
            logger.info(
                "decode_eval: previous eval (step %d) still running, skipping launch at step %d",
                self._launched_for_step,
                global_step,
            )
            return False
        step_dir = os.path.join(self.config.output_dir, f"step_{global_step}")
        os.makedirs(step_dir, exist_ok=True)
        result_path = os.path.join(step_dir, _RESULT_FILENAME)
        for stale_path in (result_path, result_path + ".tmp"):
            try:
                os.remove(stale_path)
            except FileNotFoundError:
                pass
        self._collected.discard(result_path)
        self._cleanup_snapshot(step_dir)
        try:
            export_draft_snapshot(draft_model, step_dir)
            # The full config crosses the subprocess boundary as a JSON sidecar next
            # to the snapshot (one declaration site, and a debugging record of what
            # the eval ran with); the argv carries only the per-launch paths.
            config_json = os.path.join(step_dir, _CONFIG_FILENAME)
            with open(config_json, "w") as f:
                json.dump(asdict(self.config), f, indent=2)
            argv = [
                sys.executable,
                "-m",
                "nemo_automodel.components.speculative.decode_eval",
                "--config-json",
                config_json,
                "--draft",
                step_dir,
                "--step",
                str(global_step),
                "--result-json",
                result_path,
            ]
            log_path = os.path.join(step_dir, _WORKER_LOG_FILENAME)
            with open(log_path, "ab") as log_file:
                self._proc = subprocess.Popen(
                    argv,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=_worker_env(self.config.cuda_visible_devices),
                    start_new_session=True,
                )
        except Exception:
            self._cleanup_snapshot(step_dir)
            raise
        self._last_bucket = global_step // self.config.every_steps
        self._launched_for_step = global_step
        self._worker_log_path = log_path
        logger.info(
            "decode_eval: launched eval for step %d on CUDA_VISIBLE_DEVICES=%s (pid %d, log %s)",
            global_step,
            self.config.cuda_visible_devices,
            self._proc.pid,
            log_path,
        )
        return True

    def collect(self) -> list[dict]:
        """Return finished, not-yet-reported eval results in snapshot-step order."""
        self._reap_finished_worker()
        results = []
        for step, step_dir in self._step_dirs():
            path = os.path.join(self.config.output_dir, step_dir, _RESULT_FILENAME)
            if path in self._collected or not os.path.exists(path):
                continue
            try:
                with open(path) as f:
                    result = json.load(f)
                if result.get("step") != step:
                    raise ValueError(f"result step {result.get('step')!r} does not match directory step {step}")
                results.append(result)
                self._collected.add(path)
                self._cleanup_snapshot(os.path.join(self.config.output_dir, step_dir))
            except (OSError, ValueError, json.JSONDecodeError):
                logger.warning("decode_eval: unreadable result %s, will retry next collect", path)
        return results

    def shutdown(self) -> None:
        """Terminate a still-running eval (its vLLM child dies with the session)."""
        if self._proc is not None and self._proc.poll() is None:
            logger.info("decode_eval: terminating in-flight eval (pid %d)", self._proc.pid)
            try:
                # The worker was started in its own session; signal the whole
                # group so the vLLM server child goes down with it.
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
        if self._launched_for_step >= 0:
            self._cleanup_snapshot(os.path.join(self.config.output_dir, f"step_{self._launched_for_step}"))
        self._proc = None
        self._worker_log_path = None


# --------------------------------------------------------------------------- #
# Worker CLI: runs in its own process on the reserved GPU.
# --------------------------------------------------------------------------- #


def _wait_for_server(server: str, proc: subprocess.Popen, timeout_s: float) -> None:
    """Poll the vLLM server's /health until it responds or dies/times out."""
    deadline = time.monotonic() + timeout_s
    url = f"{server}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited with code {proc.returncode} before becoming healthy")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except OSError:
            pass
        time.sleep(3)
    raise TimeoutError(f"vLLM server did not become healthy within {timeout_s:.0f}s")


def _resolve_worker_port(configured_port: int) -> int:
    """Choose an unused loopback port, or fail if an explicit port is occupied."""
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", configured_port))
        except OSError as exc:
            raise RuntimeError(f"decode_eval port {configured_port} is already in use") from exc
        return int(sock.getsockname()[1])


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="decode_eval worker: serve the draft snapshot and measure accept length"
    )
    parser.add_argument("--config-json", required=True, help="JSON dump of DecodeEvalConfig (written at launch)")
    parser.add_argument("--draft", required=True, help="snapshot dir containing model/consolidated")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--result-json", required=True)
    return parser


def _serve_argv(cfg: DecodeEvalConfig, draft: str) -> list[str]:
    """Build the vLLM api_server argv for the snapshot via the serve_vllm library surface."""
    from nemo_automodel.components.speculative.serve_vllm import build_vllm_argv

    serve_args = Namespace(
        target=cfg.target_model,
        draft=draft,
        method=None,
        num_speculative_tokens=cfg.num_speculative_tokens,
        dflash_causal=False,
        host="127.0.0.1",
        port=cfg.port,
        tp_size=1,
        draft_tp_size=1,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        dtype="bfloat16",
        max_model_len=cfg.max_model_len,
        trust_remote_code=cfg.trust_remote_code,
        print_only=False,
        extra=[],
    )
    return build_vllm_argv(serve_args)


def _bench_args(cfg: DecodeEvalConfig, server: str) -> Namespace:
    """Assemble the bench_vllm._run_summary argument namespace."""
    return Namespace(
        server=server,
        model=cfg.target_model,
        input_data=cfg.input_data,
        num_prompts=cfg.num_prompts,
        concurrency=cfg.concurrency,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        messages_column=cfg.messages_column,
        prompt_column=cfg.prompt_column,
        prompt_context_column=None,
        split=cfg.split,
        dataset_name=cfg.dataset_name,
        shuffle_seed=cfg.shuffle_seed,
        baseline_server=None,
        timeout_s=cfg.timeout_s,
        max_retries=2,
    )


def main(argv=None) -> int:
    """Worker entry: boot the engine on the snapshot, bench it, write the result JSON."""
    import asyncio

    from nemo_automodel.components.speculative.bench_vllm import _run_summary

    args = _build_parser().parse_args(argv)
    with open(args.config_json) as f:
        cfg = DecodeEvalConfig(**json.load(f))
    cfg.port = _resolve_worker_port(cfg.port)
    server = f"http://127.0.0.1:{cfg.port}"
    serve_argv = _serve_argv(cfg, args.draft)
    print(f"decode_eval worker: starting engine: {' '.join(serve_argv)}", flush=True)
    proc = subprocess.Popen(serve_argv)
    try:
        _wait_for_server(server, proc, cfg.timeout_s)
        summary = asyncio.run(_run_summary(_bench_args(cfg, server)))
        if summary is None:
            raise RuntimeError("bench returned no summary (no prompts loaded?)")
        result = {"step": args.step, **summary}
        tmp = args.result_json + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f)
        os.replace(tmp, args.result_json)
        print(f"decode_eval worker: wrote {args.result_json}: {json.dumps(result)}", flush=True)
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
