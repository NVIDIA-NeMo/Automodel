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
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from argparse import Namespace

logger = logging.getLogger(__name__)

_RESULT_FILENAME = "result.json"
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
    port: int = 8199
    gpu_memory_utilization: float = 0.8
    num_speculative_tokens: int | None = None
    max_model_len: int | None = None
    trust_remote_code: bool = False
    timeout_s: float = 1800.0


def resolve_decode_eval_config(recipe_cfg, *, default_target: str, default_input_data: str, output_dir: str):
    """Read the optional ``decode_eval:`` block from ``recipe_args``.

    Returns ``None`` when the block is absent or ``every_steps`` is unset/0
    (feature disabled). Raises on a partially-configured block so a typo'd GPU
    reservation fails at setup rather than silently evaluating on a training
    GPU.
    """
    block = recipe_cfg.get("decode_eval", None)
    if block is None:
        return None
    every_steps = int(block.get("every_steps", 0) or 0)
    if every_steps <= 0:
        return None
    cuda_visible_devices = block.get("cuda_visible_devices", None)
    if cuda_visible_devices is None or str(cuda_visible_devices) == "":
        raise ValueError(
            "decode_eval.cuda_visible_devices is required: the eval engine needs a GPU the training "
            "job does not use (e.g. reserve one card and shrink the torchrun world accordingly)."
        )
    return DecodeEvalConfig(
        every_steps=every_steps,
        cuda_visible_devices=str(cuda_visible_devices),
        target_model=str(block.get("target_model", None) or default_target),
        input_data=str(block.get("input_data", None) or default_input_data),
        output_dir=os.path.join(output_dir, "decode_eval"),
        num_prompts=int(block.get("num_prompts", 32)),
        concurrency=int(block.get("concurrency", 8)),
        max_new_tokens=int(block.get("max_new_tokens", 256)),
        temperature=float(block.get("temperature", 0.0)),
        top_p=float(block.get("top_p", 1.0)),
        messages_column=str(block.get("messages_column", "messages")),
        prompt_column=block.get("prompt_column", None),
        split=str(block.get("split", "train")),
        dataset_name=block.get("dataset_name", None),
        shuffle_seed=int(block.get("shuffle_seed", 42)),
        port=int(block.get("port", 8199)),
        gpu_memory_utilization=float(block.get("gpu_memory_utilization", 0.8)),
        num_speculative_tokens=block.get("num_speculative_tokens", None),
        max_model_len=block.get("max_model_len", None),
        trust_remote_code=bool(block.get("trust_remote_code", False)),
        timeout_s=float(block.get("timeout_s", 1800.0)),
    )


def export_draft_snapshot(draft_model, out_dir: str) -> str:
    """Write the draft's current weights as a serve-ready consolidated export.

    Produces ``<out_dir>/model/consolidated/{model.safetensors, config.json}``,
    the same layout the final checkpoint's consolidated export uses, so
    ``serve_vllm.resolve_draft_artifacts`` can consume it directly (the
    d2t/t2d vocab-mapping buffers ride along in the state dict).

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
        self._launched_for_step = -1
        self._last_bucket = 0
        self._collected: set[str] = set()
        os.makedirs(config.output_dir, exist_ok=True)

    def due(self, global_step: int) -> bool:
        """Whether a new eval should launch at this optimizer step."""
        bucket = global_step // self.config.every_steps
        return bucket > self._last_bucket

    def maybe_launch(self, global_step: int, draft_model) -> bool:
        """Snapshot the draft and launch the worker if the cadence is due and no eval is running."""
        if not self.due(global_step):
            return False
        if self._proc is not None and self._proc.poll() is None:
            logger.info(
                "decode_eval: previous eval (step %d) still running, skipping launch at step %d",
                self._launched_for_step,
                global_step,
            )
            return False
        self._last_bucket = global_step // self.config.every_steps
        step_dir = os.path.join(self.config.output_dir, f"step_{global_step}")
        os.makedirs(step_dir, exist_ok=True)
        export_draft_snapshot(draft_model, step_dir)
        cfg = self.config
        argv = [
            sys.executable,
            "-m",
            "nemo_automodel.components.speculative.decode_eval",
            "--draft",
            step_dir,
            "--target",
            cfg.target_model,
            "--step",
            str(global_step),
            "--result-json",
            os.path.join(step_dir, _RESULT_FILENAME),
            "--input-data",
            cfg.input_data,
            "--num-prompts",
            str(cfg.num_prompts),
            "--concurrency",
            str(cfg.concurrency),
            "--max-new-tokens",
            str(cfg.max_new_tokens),
            "--temperature",
            str(cfg.temperature),
            "--top-p",
            str(cfg.top_p),
            "--messages-column",
            cfg.messages_column,
            "--split",
            cfg.split,
            "--shuffle-seed",
            str(cfg.shuffle_seed),
            "--port",
            str(cfg.port),
            "--gpu-memory-utilization",
            str(cfg.gpu_memory_utilization),
            "--timeout-s",
            str(cfg.timeout_s),
        ]
        if cfg.prompt_column:
            argv += ["--prompt-column", str(cfg.prompt_column)]
        if cfg.dataset_name:
            argv += ["--dataset-name", str(cfg.dataset_name)]
        if cfg.num_speculative_tokens is not None:
            argv += ["--num-speculative-tokens", str(cfg.num_speculative_tokens)]
        if cfg.max_model_len is not None:
            argv += ["--max-model-len", str(cfg.max_model_len)]
        if cfg.trust_remote_code:
            argv += ["--trust-remote-code"]
        log_path = os.path.join(step_dir, _WORKER_LOG_FILENAME)
        with open(log_path, "ab") as log_file:
            self._proc = subprocess.Popen(
                argv,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=_worker_env(cfg.cuda_visible_devices),
                start_new_session=True,
            )
        self._launched_for_step = global_step
        logger.info(
            "decode_eval: launched eval for step %d on CUDA_VISIBLE_DEVICES=%s (pid %d, log %s)",
            global_step,
            cfg.cuda_visible_devices,
            self._proc.pid,
            log_path,
        )
        return True

    def collect(self) -> list[dict]:
        """Return finished, not-yet-reported eval results in snapshot-step order."""
        results = []
        try:
            step_dirs = sorted(
                (d for d in os.listdir(self.config.output_dir) if d.startswith("step_")),
                key=lambda d: int(d.split("_", 1)[1]),
            )
        except (FileNotFoundError, ValueError):
            return results
        for step_dir in step_dirs:
            path = os.path.join(self.config.output_dir, step_dir, _RESULT_FILENAME)
            if path in self._collected or not os.path.exists(path):
                continue
            try:
                with open(path) as f:
                    results.append(json.load(f))
                self._collected.add(path)
            except (OSError, json.JSONDecodeError):
                logger.warning("decode_eval: unreadable result %s, will retry next collect", path)
        return results

    def shutdown(self) -> None:
        """Terminate a still-running eval (its vLLM child dies with the session)."""
        if self._proc is not None and self._proc.poll() is None:
            logger.info("decode_eval: terminating in-flight eval (pid %d)", self._proc.pid)
            try:
                # The worker was started in its own session; signal the whole
                # group so the vLLM server child goes down with it.
                os.killpg(os.getpgid(self._proc.pid), 15)
            except (ProcessLookupError, PermissionError):
                self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


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


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="decode_eval worker: serve the draft snapshot and measure accept length"
    )
    parser.add_argument("--draft", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--input-data", required=True)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--messages-column", default="messages")
    parser.add_argument("--prompt-column", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--num-speculative-tokens", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    return parser


def _serve_argv(args) -> list[str]:
    """Build the vLLM api_server argv for the snapshot via the serve_vllm library surface."""
    from nemo_automodel.components.speculative.serve_vllm import build_vllm_argv

    serve_args = Namespace(
        target=args.target,
        draft=args.draft,
        method=None,
        num_speculative_tokens=args.num_speculative_tokens,
        dflash_causal=False,
        host="127.0.0.1",
        port=args.port,
        tp_size=1,
        draft_tp_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        print_only=False,
        extra=[],
    )
    return build_vllm_argv(serve_args)


def _bench_args(args, server: str):
    """Assemble the bench_vllm._run_summary argument namespace."""
    return Namespace(
        server=server,
        model=args.target,
        input_data=args.input_data,
        num_prompts=args.num_prompts,
        concurrency=args.concurrency,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        messages_column=args.messages_column,
        prompt_column=args.prompt_column,
        prompt_context_column=None,
        split=args.split,
        dataset_name=args.dataset_name,
        shuffle_seed=args.shuffle_seed,
        baseline_server=None,
        timeout_s=args.timeout_s,
        max_retries=2,
    )


def main(argv=None) -> int:
    """Worker entry: boot the engine on the snapshot, bench it, write the result JSON."""
    import asyncio

    from nemo_automodel.components.speculative.bench_vllm import _run_summary

    args = _build_parser().parse_args(argv)
    server = f"http://127.0.0.1:{args.port}"
    serve_argv = _serve_argv(args)
    print(f"decode_eval worker: starting engine: {' '.join(serve_argv)}", flush=True)
    proc = subprocess.Popen(serve_argv)
    try:
        _wait_for_server(server, proc, args.timeout_s)
        summary = asyncio.run(_run_summary(_bench_args(args, server)))
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
