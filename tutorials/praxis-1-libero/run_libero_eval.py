# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Evaluate a Praxis-1 checkpoint on a LIBERO task suite with the standard LIBERO protocol.

Protocol (as in OpenVLA and openpi): for every task, the benchmark's 50 recorded initial states; after
``set_init_state`` the env runs ``NUM_STEPS_WAIT`` dummy steps; the episode ends at the first success or
after ``MAX_STEPS[suite]`` control steps. The environment is the unmodified LIBERO one (OSC_POSE delta
controller at 20 Hz) in its own interpreter (:mod:`libero_sim` run as a script). The policy is the reference
closed-loop model session of ``praxis.sim.loop.ModelPolicy`` with the sampler and session settings of
``configs/sim_droid.yaml``. The bridge between the two: the policy sees one observation per 15 Hz tick
(env step ``round(4k / 3)``), its command rows are hand-pose targets per tick, and every 20 Hz step
tracks the interpolated target with an OSC delta (:mod:`libero_bridge`).

Outputs follow the reference ``run_sim_eval.py`` layout, so the reference's comparison
tools work unchanged: ``<out>/suite/<task>/episode_<idx>.json`` (+ ``.mp4``),
``<out>/suite/episodes.jsonl`` and ``<out>/summary.json``.

Run inside the policy container, with ``LIBERO_PY`` pointing at the LIBERO virtual environment and ``PRAXIS_REF``
at the reference runtime checkout (its ``praxis`` package and ``configs/sim_droid.yaml``). ``--gpus N`` runs N
shards at once, the model of shard i on GPU i and its renderer on GPU (i + 1) mod N, restarts a shard that dies
(the shards are resumable) and writes ``summary.json`` when all are done:
  python run_libero_eval.py --gpus 8 --suite libero_spatial --checkpoint checkpoints/praxis-1-droid --out eval/libero_base
A checkpoint whose config names the flash-attention backend is evaluated through a view of it
(``<checkpoint>_triton_view``: symlinked weights, ``config.json`` with the triton backend), so flash-attn is not
needed on the evaluation node; ``--keep-backend`` evaluates the checkpoint as configured.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PRAXIS_REF = os.environ.get("PRAXIS_REF")
if PRAXIS_REF:
    sys.path.insert(0, PRAXIS_REF)

if TYPE_CHECKING:
    from praxis.sim.loop import ModelPolicy

from libero_bridge import (  # noqa: E402
    LIBERO_DUMMY_ACTION,
    MAX_STEPS,
    NUM_STEPS_WAIT,
    TargetSchedule,
    checkpoint_dir,
    matrix_from_pose8,
    osc_step_action,
    policy_frame_indices,
    read_message,
    tick_time_after_step,
    write_message,
)

N_TASKS = {"libero_spatial": 10, "libero_object": 10, "libero_goal": 10, "libero_10": 10, "libero_90": 90}


def tile_size(checkpoint: Path) -> tuple[int, int]:
    """``(tile_height, tile_width)`` of the checkpoint's camera tiles: both LIBERO cameras render at this size."""
    with (checkpoint / "config.json").open(encoding="utf-8") as f:
        video = json.load(f)["video"]
    return int(video["tile_height"]), int(video["tile_width"])


def default_config() -> str:
    """``configs/sim_droid.yaml`` of the reference checkout (``$PRAXIS_REF``, else the installed ``praxis`` package)."""
    if PRAXIS_REF:
        return str(Path(PRAXIS_REF) / "configs" / "sim_droid.yaml")
    spec = importlib.util.find_spec("praxis")
    if spec is None or spec.origin is None:
        raise SystemExit("set PRAXIS_REF to the reference runtime checkout, or give --config")
    return str(Path(spec.origin).resolve().parents[1] / "configs" / "sim_droid.yaml")


PUBLIC_ENCODERS = {
    "model.vae_weights": ("Wan-AI/Wan2.2-TI2V-5B", "Wan2.2_VAE.pth"),
    "model.text_encoder": ("Qwen/Qwen3.5-2B", None),
}


def encoder_overrides(config: str, overrides: list[str]) -> list[str]:
    """``--set`` overrides that point the sim config's VAE and text encoder at the public weights in the Hub cache.

    The reference config names ``weights/Wan2.2_VAE.pth`` and ``weights/Qwen3.5-2B`` relative to the working
    directory. Each one that is neither present there nor already overridden is downloaded from its Hugging Face
    repo and its local path appended to ``overrides``.
    """
    import yaml

    with open(config, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f).get("model", {})
    out = list(overrides)
    for key, (repo, filename) in PUBLIC_ENCODERS.items():
        if any(item.startswith(key + "=") for item in out):
            continue
        local = model_cfg.get(key.split(".", 1)[1])
        if local is not None and Path(local).exists():
            continue
        from huggingface_hub import hf_hub_download, snapshot_download

        path = hf_hub_download(repo_id=repo, filename=filename) if filename else snapshot_download(repo_id=repo)
        out.append(f"{key}={path}")
    return out


def triton_view(checkpoint: Path) -> Path:
    """The checkpoint itself, or a view of it whose config names the triton attention backend.

    A config that names the flash backend needs flash-attn on the evaluation node. The view has the same
    weights (symlinks) and differs only in ``backbone.attention.backend``; it is created once next to the
    checkpoint and reused.
    """
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config["backbone"]["attention"]["backend"] != "flash":
        return checkpoint
    view = checkpoint.with_name(checkpoint.name + "_triton_view")
    if not (view / "config.json").exists():
        view.mkdir(exist_ok=True)
        for entry in checkpoint.iterdir():
            link = view / entry.name
            if entry.name != "config.json" and not link.is_symlink():
                link.symlink_to(Path("..") / checkpoint.name / entry.name)
        config["backbone"]["attention"]["backend"] = "triton"
        (view / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    return view


class EnvWorker:
    """Client of the environment worker (:mod:`libero_sim` run in the LIBERO virtual environment)."""

    def __init__(
        self, python: str, egl_device: int, suite: str, task_id: int, seed: int, height: int, width: int
    ) -> None:
        env = {
            k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")
        }
        env["MUJOCO_GL"] = "egl"
        env["PYOPENGL_PLATFORM"] = "egl"
        env["MUJOCO_EGL_DEVICE_ID"] = str(egl_device)
        env["LD_LIBRARY_PATH"] = ":".join(
            p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p and "torch" not in p and "python3" not in p
        )
        self.proc = subprocess.Popen(
            [python, str(HERE / "libero_sim.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env
        )
        info = self.request(
            "init",
            {"suite": suite, "task_id": task_id, "height": height, "width": width, "seed": seed},
        )
        self.language: str = info["language"]
        self.task_name: str = info["task"]
        self.n_init_states: int = info["n_init_states"]
        self.height, self.width = height, width

    def request(self, name: str, payload=None):
        """Send one request and return the reply payload."""
        write_message(self.proc.stdin, (name, payload))
        status, reply = read_message(self.proc.stdout)
        if status != "ok":
            raise RuntimeError(f"env worker {name}: {reply}")
        return reply

    def close(self) -> None:
        """Stop the worker."""
        try:
            self.request("close")
        finally:
            self.proc.wait(timeout=30)


@dataclass
class Obs:
    """Duck-typed observation for ``praxis.sim.loop.History`` (same attribute names as the sim's Observation)."""

    ee_pose: np.ndarray
    grip: float
    exterior: np.ndarray
    wrist: np.ndarray
    overview: object
    tick: int
    status: object = None
    ik: object = None


def to_obs(payload: dict, tick: int) -> Obs:
    """History observation from a worker reply: the hand pose is the policy's ``praxis.sim`` rig frame."""
    p8 = np.asarray(payload["hand_pose8"], dtype=np.float32)
    return Obs(
        ee_pose=p8[:7],
        grip=float(p8[7]),
        exterior=payload["exterior"],
        wrist=payload["wrist"],
        overview=None,
        tick=tick,
    )


def run_episode(worker: EnvWorker, policy: ModelPolicy, episode: int, max_steps: int, video_path: Path | None) -> dict:
    """One LIBERO episode under the standard protocol; returns the record for ``episode_<idx>.json``."""
    from praxis import single_arm
    from praxis.sim.loop import History

    wall0 = time.monotonic()
    reset = worker.request("reset")
    world_from_base = np.asarray(reset["base_pose"], dtype=np.float64)
    worker.request("set_state", worker.request("init_state", episode))
    payload = None
    for _ in range(NUM_STEPS_WAIT):
        payload = worker.request("step", LIBERO_DUMMY_ACTION)
    frame_idx = policy_frame_indices(max_steps + 64)  # env step of every tick
    history = History()
    history.push(to_obs(payload, 0))
    policy.reset(None, episode)
    frames = [np.concatenate([payload["exterior"], payload["wrist"]], axis=1)] if video_path else None
    env_steps, tick, success, success_step = 0, 0, bool(payload["success"]), None
    calls, call_s, call_rows = 0, [], []
    while not success and env_steps < max_steps:
        out = policy.act(history)
        calls += 1
        call_s.append(float(out.info.get("step_s", 0.0)))
        call_rows.append(len(out.commands))
        poses12, grips = single_arm.wire_to_arrays(out.commands)
        hand0 = matrix_from_pose8(history.latest.ee_pose)
        schedule = TargetSchedule(tick, hand0, poses12, grips)
        chunk_end = tick + len(poses12)
        while tick < chunk_end and not success and env_steps < max_steps:
            action = osc_step_action(
                schedule, tick_time_after_step(env_steps + 1), np.asarray(payload["grip_world"]), world_from_base
            )
            payload = worker.request("step", action)
            env_steps += 1
            if payload["success"] and success_step is None:
                success_step = env_steps
            success = bool(payload["success"])
            if env_steps == frame_idx[tick + 1]:
                tick += 1
                history.push(to_obs(payload, tick))
                if frames is not None:
                    frames.append(np.concatenate([payload["exterior"], payload["wrist"]], axis=1))
    record = {
        "task": worker.task_name,
        "prompt": worker.language,
        "seed": episode,
        "scene_seed": episode,
        "sampler_seed": 0,
        "success": bool(success),
        "success_env_step": success_step,
        "env_steps": env_steps,
        "ticks": tick,
        "calls": calls,
        "call_rows": call_rows,
        "call_s": call_s,
        "progress": 1.0 if success else 0.0,
        "max_steps": max_steps,
        "wall_s": time.monotonic() - wall0,
    }
    if frames is not None and video_path is not None:
        write_video(video_path, frames)
    return record


def write_video(path: Path, frames: list[np.ndarray], fps: int = 15) -> None:
    """H.264 MP4 of the policy's tick canvases."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20"}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def summarize(root: Path) -> dict:
    """Per-task success counts of all episode records under ``root``."""
    out: dict[str, dict] = {}
    for path in sorted(root.glob("suite/*/episode_*.json")):
        rec = json.loads(path.read_text())
        task = out.setdefault(rec["task"], {"episodes": 0, "successes": 0})
        task["episodes"] += 1
        task["successes"] += int(bool(rec["success"]))
    total = {
        "episodes": sum(t["episodes"] for t in out.values()),
        "successes": sum(t["successes"] for t in out.values()),
    }
    return {"tasks": out, "total": total}


def run_shards(args: argparse.Namespace, checkpoint: Path, config: str) -> int:
    """Run ``args.gpus`` shards of this script at once, each restarted up to ``args.retries`` times, then summarize."""
    out = Path(args.out).absolute()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONFAULTHANDLER="1")
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # CUDA device i is EGL device i
    env.setdefault("MUJOCO_GL", "egl")

    def shard(i: int) -> bool:
        egl = (i + 1) % args.gpus
        cmd = [sys.executable, str(Path(__file__).resolve()), "--suite", args.suite, "--tasks", args.tasks]
        cmd += ["--trials", str(args.trials), "--checkpoint", str(checkpoint), "--config", config, "--out", str(out)]
        cmd += ["--shard", f"{i}/{args.gpus}", "--seed", str(args.seed), "--worker-python", args.worker_python]
        cmd += ["--egl-device", str(egl), "--keep-backend"]
        for item in args.set:
            cmd += ["--set", item]
        if args.no_video:
            cmd.append("--no-video")
        shard_env = dict(env, CUDA_VISIBLE_DEVICES=str(i), MUJOCO_EGL_DEVICE_ID=str(egl))
        with (out / "logs" / f"shard_{i}.log").open("ab") as log:
            for attempt in range(1, args.retries + 1):
                if subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=shard_env).returncode == 0:
                    return True
                log.write(f"shard {i} attempt {attempt} failed\n".encode())
                log.flush()
                if attempt < args.retries:
                    time.sleep(20)
        return False

    with ThreadPoolExecutor(max_workers=args.gpus) as pool:
        finished = list(pool.map(shard, range(args.gpus)))
    summary = summarize(out)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for task, counts in summary["tasks"].items():
        print(f"{task:90s} {counts['successes']:3d}/{counts['episodes']}")
    total = summary["total"]
    print(
        f"TOTAL {total['successes']}/{total['episodes']} = {100 * total['successes'] / max(total['episodes'], 1):.1f}%"
    )
    failed = [i for i, ok in enumerate(finished) if not ok]
    if failed:
        print(f"shards {failed} failed after {args.retries} attempts (see {out / 'logs'})", file=sys.stderr)
    return 1 if failed else 0


def main() -> int:
    """Run the selected (task, episode) units of a suite, or all shards of it with ``--gpus``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--suite", default="libero_spatial", choices=sorted(MAX_STEPS))
    ap.add_argument("--tasks", default="all", help="comma list of task indices or 'all'")
    ap.add_argument("--trials", type=int, default=50, help="episodes per task (the benchmark has 50 init states)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="sim config (default: $PRAXIS_REF/configs/sim_droid.yaml)")
    ap.add_argument("--set", action="append", default=[], help="config overrides key.sub=value")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument(
        "--gpus", type=int, default=0, help="run this many shards at once, one per GPU (0: this process only)"
    )
    ap.add_argument("--retries", type=int, default=3, help="attempts per shard with --gpus")
    ap.add_argument(
        "--keep-backend", action="store_true", help="evaluate the checkpoint's attention backend as configured"
    )
    ap.add_argument("--seed", type=int, default=7, help="env seed (openpi uses 7)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--worker-python", default=os.environ.get("LIBERO_PY"))
    ap.add_argument("--egl-device", type=int, default=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")))
    args = ap.parse_args()
    if not args.worker_python:
        raise SystemExit("set LIBERO_PY or --worker-python to the LIBERO virtual environment's python")
    config = args.config or default_config()
    args.set = encoder_overrides(config, args.set)
    checkpoint = checkpoint_dir(args.checkpoint)
    if not args.keep_backend:
        checkpoint = triton_view(checkpoint)
    if args.gpus > 0:
        print(f"evaluating checkpoint {checkpoint} with {args.gpus} shards", flush=True)
        return run_shards(args, checkpoint, config)
    from praxis.sim.config import load_config
    from praxis.sim.loop import ModelPolicy

    cfg = load_config(config, [f"model.checkpoint={checkpoint}", "policy.decode_video=false", *args.set])
    tasks = list(range(N_TASKS[args.suite])) if args.tasks == "all" else [int(t) for t in args.tasks.split(",")]
    shard, shards = (int(x) for x in args.shard.split("/"))
    units = [(t, e) for t in tasks for e in range(args.trials)][shard::shards]
    out = Path(args.out)
    (out / "suite").mkdir(parents=True, exist_ok=True)
    max_steps = MAX_STEPS[args.suite]
    pipe_policy = None
    worker: EnvWorker | None = None
    current_task = None
    policy = None
    tile_h, tile_w = tile_size(checkpoint)
    t_start = time.time()
    done = 0
    for task_id, episode in units:
        if current_task != task_id:
            if worker is not None:
                worker.close()
            worker = EnvWorker(args.worker_python, args.egl_device, args.suite, task_id, args.seed, tile_h, tile_w)
            if pipe_policy is None:
                pipe_policy = ModelPolicy.from_config(cfg, worker.language)
                print(f"policy model ready ({time.time() - t_start:.0f} s)", flush=True)
            policy = ModelPolicy(pipe_policy.pipe, worker.language, cfg.session, decode_video=False)
            current_task = task_id
            (out / "suite" / worker.task_name).mkdir(parents=True, exist_ok=True)
        stem = out / "suite" / worker.task_name / f"episode_{episode:03d}"
        if stem.with_suffix(".json").exists():
            continue
        rec = run_episode(worker, policy, episode, max_steps, None if args.no_video else stem.with_suffix(".mp4"))
        rec["sampler_seed"] = cfg.session.seed
        rec["task_id"] = task_id
        rec["suite"] = args.suite
        rec["checkpoint"] = cfg.model.checkpoint
        rec["render"] = {"height": worker.height, "width": worker.width}
        stem.with_suffix(".json").write_text(json.dumps(rec))
        with (out / "suite" / "episodes.jsonl").open("a") as log:
            log.write(json.dumps(rec) + "\n")
        done += 1
        print(
            f"[{done}/{len(units)}] {worker.task_name} ep {episode}: success={rec['success']} steps={rec['env_steps']} "
            f"calls={rec['calls']} call {np.mean(rec['call_s']) if rec['call_s'] else 0:.2f}s wall {rec['wall_s']:.0f}s",
            flush=True,
        )
    if worker is not None:
        worker.close()
    (out / "summary.json").write_text(json.dumps(summarize(out), indent=1))
    print(json.dumps(summarize(out)["total"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
