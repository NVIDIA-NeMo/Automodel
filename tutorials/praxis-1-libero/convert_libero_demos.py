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

"""Convert LIBERO demonstrations to the policy's recorded-episode format by re-executing them in the sim.

For every demonstration of a task the script loads the recorded initial state, runs the evaluation
protocol's settle steps (``NUM_STEPS_WAIT`` dummy actions), re-executes the recorded 20 Hz actions,
and then holds the arm still until the episode is at least ``--min-steps`` long (one training window
needs 93 frames at 15 Hz, that is 124 control steps). The cameras are rendered at the policy's tile
size, so the training canvas is the two upright views side by side with no resize. The commanded pose
of a step is the goal the OSC_POSE controller sets for the recorded action; observed and commanded
hand poses are converted to the flange frame of the training files. Only re-executions that reach the
task's success condition are kept; every re-execution is recorded in the report with its tracking
error against the dataset's recorded end-effector positions and its forward-kinematics check.

Run inside the LIBERO virtual environment, one process per shard:
  MUJOCO_EGL_DEVICE_ID=0 python convert_libero_demos.py --suite libero_spatial --out data/libero_spatial --shard 0/8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from libero_bridge import (  # noqa: E402
    LIBERO_DUMMY_ACTION,
    NUM_STEPS_WAIT,
    checkpoint_dir,
    hand_to_flange_pose8,
    osc_goal,
    policy_frame_indices,
    pose8_from_matrix,
)
from libero_sim import LiberoSession  # noqa: E402

WINDOW_FRAMES = 93  # one training window at 15 Hz
MIN_ENV_STEPS = 126  # >= 94 policy frames after the 20 -> 15 Hz map
HOLD_STEPS_MIN = 10  # hold at least this long after the last recorded action


def sha256_of(path: Path) -> str:
    """Hex sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def render_check(session: LiberoSession, demo: h5py.Group, small: LiberoSession) -> dict:
    """Compare a 128x128 render of the recorded first state with the frames stored in the dataset."""
    small.reset()
    obs = small.set_state(demo["states"][0])
    out = {}
    for name, key, ours in (
        ("agentview", "agentview_rgb", obs.exterior),
        ("eye_in_hand", "eye_in_hand_rgb", obs.wrist),
    ):
        stored = np.asarray(demo["obs"][key][0], dtype=np.float64)
        ours_upright = np.asarray(ours, dtype=np.float64)
        out[name] = {
            "mae_vs_upright": float(np.abs(stored - ours_upright).mean()),
            "mae_vs_raw": float(np.abs(stored - ours_upright[::-1]).mean()),
            "mae_vs_rot180_of_raw": float(np.abs(stored - ours_upright[:, ::-1]).mean()),
        }
    return out


def convert_demo(session: LiberoSession, demo: h5py.Group, min_steps: int) -> tuple[dict, dict]:
    """Re-execute one demonstration; return the arrays of the episode and its report."""
    states = np.asarray(demo["states"])
    actions = np.asarray(demo["actions"], dtype=np.float64)
    rec_ee = np.asarray(demo["obs"]["ee_pos"], dtype=np.float64)
    session.reset()
    obs = session.set_state(states[0])
    goal_rot = session.controller_goal_rot()  # carried across steps: zero-rotation actions keep it
    ext, wrist, obs8, cmd8, fk, dev, phase = [], [], [], [], [], [], []
    success_step = None
    t_env = 0

    def record(step_obs, action, tag, rec_pos=None):
        nonlocal success_step, goal_rot
        goal_pos, goal_rot = osc_goal(step_obs.grip_world[:3, 3], step_obs.grip_world[:3, :3], action, goal_rot)
        goal_world = np.eye(4)
        goal_world[:3, :3], goal_world[:3, 3] = goal_rot, goal_pos
        hand_goal = session.hand_goal_from_grip_goal(goal_world)
        grip_cmd = 1.0 if action[6] > 0 else 0.0
        ext.append(step_obs.exterior)
        wrist.append(step_obs.wrist)
        obs8.append(step_obs.hand_pose8)
        cmd8.append(pose8_from_matrix(hand_goal, grip_cmd))
        fk.append(step_obs.fk_error_m)
        dev.append(np.nan if rec_pos is None else float(np.linalg.norm(step_obs.eef_pos_obs - rec_pos)))
        phase.append(tag)
        if step_obs.success and success_step is None:
            success_step = len(ext) - 1

    # 1. the evaluation protocol's settle steps
    for _ in range(NUM_STEPS_WAIT):
        record(obs, LIBERO_DUMMY_ACTION, 0)
        obs = session.step(LIBERO_DUMMY_ACTION)
        t_env += 1
    # 2. the recorded actions
    for t, action in enumerate(actions):
        record(obs, action, 1, rec_ee[t])
        obs = session.step(action)
        t_env += 1
    # 3. hold until the episode is long enough
    hold = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, actions[-1][6]])
    n_hold = 0
    while n_hold < HOLD_STEPS_MIN or len(ext) < min_steps:
        record(obs, hold, 2)
        obs = session.step(hold)
        n_hold += 1
    # final observation is not recorded (a row needs an action); success may still arrive now
    if obs.success and success_step is None:
        success_step = len(ext)
    idx = policy_frame_indices(len(ext))
    canvas = np.concatenate([np.stack(ext)[idx], np.stack(wrist)[idx]], axis=2)  # (T, H, 2W, 3)
    obs8_h = np.stack(obs8)[idx]
    cmd8_h = np.stack(cmd8)[idx]
    arrays = {
        "canvas": canvas,
        "obs8": hand_to_flange_pose8(obs8_h),
        "cmd8": hand_to_flange_pose8(cmd8_h),
        "obs8_hand": obs8_h.astype(np.float32),
        "cmd8_hand": cmd8_h.astype(np.float32),
        "env_step_index": idx.astype(np.int64),
        "phase": np.asarray(phase, dtype=np.int8)[idx],
        "prompt": np.array(session.language),
        "task": np.array(session.task.name),
        "success_tick": np.array(-1 if success_step is None else int(np.searchsorted(idx, success_step))),
    }
    dev_arr = np.asarray(dev, dtype=np.float64)
    report = {
        "recorded_steps": int(len(actions)),
        "env_steps": int(len(ext)),
        "policy_frames": int(len(idx)),
        "hold_steps": int(n_hold),
        "success": success_step is not None,
        "success_env_step": success_step,
        "fk_error_mm_max": float(1e3 * np.max(fk)),
        "tracking_vs_recorded_m": {
            "max": float(np.nanmax(dev_arr)),
            "mean": float(np.nanmean(dev_arr)),
            "final": float(dev_arr[NUM_STEPS_WAIT + len(actions) - 1]),
        },
    }
    return arrays, report


def main() -> int:
    """Convert the demonstrations of the selected tasks."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="all", help="comma list of task indices or 'all'")
    ap.add_argument("--demos", default="all", help="comma list of demo indices or 'all'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None, help="checkpoint directory; its config.json gives the render size")
    ap.add_argument("--height", type=int, default=None, help="render height (default: the checkpoint's tile_height)")
    ap.add_argument("--width", type=int, default=None, help="render width (default: the checkpoint's tile_width)")
    ap.add_argument("--min-steps", type=int, default=MIN_ENV_STEPS)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-render-check", action="store_true")
    args = ap.parse_args()
    height, width = args.height, args.width
    if args.checkpoint is not None:
        with (checkpoint_dir(args.checkpoint) / "config.json").open(encoding="utf-8") as f:
            video = json.load(f)["video"]
        height = int(video["tile_height"]) if height is None else height
        width = int(video["tile_width"]) if width is None else width
    if height is None or width is None:
        raise SystemExit("give --checkpoint (render at its tile size) or both --height and --width")
    from libero.libero import benchmark, get_libero_path

    bench = benchmark.get_benchmark_dict()[args.suite]()
    n_tasks = bench.get_num_tasks()
    tasks = list(range(n_tasks)) if args.tasks == "all" else [int(t) for t in args.tasks.split(",")]
    shard, shards = (int(x) for x in args.shard.split("/"))
    tasks = tasks[shard::shards]
    out = Path(args.out) / args.suite
    out.mkdir(parents=True, exist_ok=True)
    for task_id in tasks:
        task = bench.get_task(task_id)
        h5_path = Path(get_libero_path("datasets")) / bench.get_task_demonstration(task_id)
        report_path = out / f"report_task{task_id:02d}.json"
        if report_path.exists():
            print(f"task {task_id} done already ({report_path})", flush=True)
            continue
        t0 = time.time()
        session = LiberoSession(args.suite, task_id, height, width, args.seed)
        small = None if args.no_render_check else LiberoSession(args.suite, task_id, 128, 128, args.seed)
        report = {
            "suite": args.suite,
            "task_id": task_id,
            "task": task.name,
            "language": session.language,
            "bddl": session.bddl,
            "hdf5": str(h5_path),
            "hdf5_sha256": sha256_of(h5_path),
            "render": {"height": height, "width": width, "cameras": ["agentview", "robot0_eye_in_hand"]},
            "num_steps_wait": NUM_STEPS_WAIT,
            "min_env_steps": args.min_steps,
            "demos": {},
        }
        with h5py.File(h5_path, "r") as f:
            data = f["data"]
            names = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))
            if args.demos != "all":
                names = [f"demo_{int(i)}" for i in args.demos.split(",")]
            report["env_args_in_file"] = json.loads(data.attrs["env_args"])["env_kwargs"].get("controller_configs")
            if small is not None:
                report["render_check_demo0"] = render_check(session, data[names[0]], small)
                print(f"task {task_id} render check: {json.dumps(report['render_check_demo0'])}", flush=True)
            kept = 0
            for name in names:
                arrays, rep = convert_demo(session, data[name], args.min_steps)
                report["demos"][name] = rep
                if rep["success"]:
                    np.savez_compressed(out / f"{task.name}_{name}.npz", **arrays)
                    kept += 1
                print(
                    f"task {task_id} {name}: success={rep['success']} steps {rep['recorded_steps']}->{rep['env_steps']} "
                    f"frames {rep['policy_frames']} track max {rep['tracking_vs_recorded_m']['max'] * 1e3:.1f} mm "
                    f"fk {rep['fk_error_mm_max']:.2f} mm",
                    flush=True,
                )
        report["kept"] = kept
        report["total"] = len(names)
        report["seconds"] = time.time() - t0
        report_path.write_text(json.dumps(report, indent=1))
        print(f"task {task_id}: kept {kept}/{len(names)} in {time.time() - t0:.0f} s", flush=True)
        session.close()
        if small is not None:
            small.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
