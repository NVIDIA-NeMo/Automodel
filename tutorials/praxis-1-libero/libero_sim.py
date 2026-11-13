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

"""A LIBERO task environment seen through the policy's conventions, and the worker process that serves it.

``LiberoSession`` wraps ``OffScreenRenderEnv`` of one task and returns, after every control step, the
upright camera frames at the policy's tile size, the hand pose in the robot base frame (the policy's
``praxis.sim`` rig frame), the grip-site pose the OSC controller works with and the checks the
demonstration converter records. The environment itself (controller, cameras, success check, horizon)
stays the standard LIBERO one.

Run as a script, the module is the environment worker of ``run_libero_eval.py``: the policy runs in the
model container (Python 3.12, PyTorch) and LIBERO needs robosuite 1.4 on Python 3.10, so the environment
lives in its own interpreter and answers ``(name, payload)`` requests read from stdin with replies on
stdout, in the message format of :mod:`libero_bridge`. Both uses run in the LIBERO virtual environment.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from libero_bridge import (  # noqa: E402
    GRIP_TO_HAND,
    gripper_closedness,
    make_pose,
    pose8_from_matrix,
    read_message,
    upright,
    write_message,
)

FRANKA_DH: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
)


def franka_flange_position(q: np.ndarray, flange_d: float = 0.1065) -> np.ndarray:
    """Flange position in the base frame from the 7 joint angles (modified DH; robosuite's flange sits 0.1065 m past link 7)."""
    m = np.eye(4)
    for (a, d, alpha), theta in zip(FRANKA_DH, np.asarray(q, dtype=np.float64)):
        ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
        m = m @ np.array(
            [[ct, -st, 0.0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1.0]]
        )
    return m[:3, 3] + flange_d * m[:3, 2]


@dataclass
class StepObs:
    """What the policy side needs from one environment state."""

    exterior: np.ndarray  # (H, W, 3) uint8, upright agentview
    wrist: np.ndarray  # (H, W, 3) uint8, upright eye-in-hand
    hand_pose8: np.ndarray  # (8,) [xyz, quat_wxyz, grip] of robot0_right_hand in the base frame
    grip_world: np.ndarray  # (4, 4) grip_site pose in the world frame (the OSC control point)
    eef_pos_obs: np.ndarray  # (3,) robosuite's robot0_eef_pos (world), for cross-checks
    fk_error_m: float  # |hand position - Franka forward kinematics|
    success: bool


class LiberoSession:
    """One LIBERO task: the standard env plus the observation conventions of the policy."""

    def __init__(self, suite: str, task_id: int, height: int, width: int, seed: int = 7) -> None:
        # LIBERO is imported here, not at module level: the worker process redirects the simulator's
        # start-up messages away from the reply stream before it builds the first session.
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.suite = suite
        self.benchmark = benchmark.get_benchmark_dict()[suite]()
        self.task = self.benchmark.get_task(task_id)
        self.bddl = os.path.join(get_libero_path("bddl_files"), self.task.problem_folder, self.task.bddl_file)
        self.env = OffScreenRenderEnv(bddl_file_name=self.bddl, camera_heights=height, camera_widths=width)
        self.env.seed(seed)
        self.height, self.width = height, width
        self.language = str(self.task.language)
        self.init_states = self.benchmark.get_task_init_states(task_id)
        self._last_obs = None
        self.sim = None

    def reset(self) -> None:
        """Reset the env (robosuite randomisation) before an init state is applied."""
        self._last_obs = self.env.reset()
        self.sim = self.env.sim

    def set_state(self, state: np.ndarray) -> StepObs:
        """Load a flattened MuJoCo state (an init state or a recorded demo state) and observe."""
        self._last_obs = self.env.set_init_state(np.asarray(state, dtype=np.float64))
        self.sim = self.env.sim
        return self.observe(bool(self.env.check_success()))

    def step(self, action: np.ndarray) -> StepObs:
        """One 20 Hz control step with a robosuite action ``[dx, dy, dz, rx, ry, rz, grip]``."""
        self._last_obs, _reward, done, _info = self.env.step(np.asarray(action, dtype=np.float64))
        return self.observe(bool(done))

    def controller_goal_rot(self) -> np.ndarray:
        """The OSC controller's retained goal orientation (world frame, 3x3)."""
        return np.array(self.env.env.robots[0].controller.goal_ori, dtype=np.float64)

    def _pose(self, kind: str, name: str) -> np.ndarray:
        sim = self.sim
        if kind == "body":
            i = sim.model.body_name2id(name)
            return make_pose(sim.data.body_xpos[i], sim.data.body_xmat[i].reshape(3, 3))
        i = sim.model.site_name2id(name)
        return make_pose(sim.data.site_xpos[i], sim.data.site_xmat[i].reshape(3, 3))

    def observe(self, success: bool) -> StepObs:
        """Build the policy-side observation of the current state."""
        obs = self._last_obs
        base = self._pose("body", "robot0_base")
        hand = self._pose("body", "robot0_right_hand")
        grip = self._pose("site", "gripper0_grip_site")
        base_inv = np.linalg.inv(base)
        hand_base = base_inv @ hand
        q = np.asarray(obs["robot0_joint_pos"], dtype=np.float64)
        fk_err = float(np.linalg.norm(franka_flange_position(q) - hand_base[:3, 3]))
        closed = gripper_closedness(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64))
        return StepObs(
            exterior=upright(obs["agentview_image"]),
            wrist=upright(obs["robot0_eye_in_hand_image"]),
            hand_pose8=pose8_from_matrix(hand_base, closed),
            grip_world=grip,
            eef_pos_obs=np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
            fk_error_m=fk_err,
            success=success,
        )

    def base_pose(self) -> np.ndarray:
        """World pose of the robot base (constant for a fixed-base task)."""
        return self._pose("body", "robot0_base")

    def hand_goal_from_grip_goal(self, grip_goal_world: np.ndarray) -> np.ndarray:
        """Hand pose in the base frame for a grip-site goal in the world frame."""
        return np.linalg.inv(self.base_pose()) @ grip_goal_world @ GRIP_TO_HAND

    def close(self) -> None:
        """Release the renderer."""
        self.env.close()


def obs_payload(obs: StepObs) -> dict:
    """Plain dict of a :class:`StepObs` for the reply stream."""
    return {
        "exterior": obs.exterior,
        "wrist": obs.wrist,
        "hand_pose8": obs.hand_pose8,
        "grip_world": obs.grip_world,
        "eef_pos_obs": obs.eef_pos_obs,
        "fk_error_m": float(obs.fk_error_m),
        "success": bool(obs.success),
    }


def serve(stdin: BinaryIO, stdout: BinaryIO) -> int:
    """Answer requests until ``close``; an error goes back to the client as an ``error`` reply."""
    session: LiberoSession | None = None
    while True:
        name, payload = read_message(stdin)
        try:
            if name == "init":
                session = LiberoSession(
                    payload["suite"],
                    int(payload["task_id"]),
                    int(payload["height"]),
                    int(payload["width"]),
                    int(payload["seed"]),
                )
                reply = {
                    "language": session.language,
                    "task": session.task.name,
                    "bddl": session.bddl,
                    "n_init_states": int(len(session.init_states)),
                    "base_pose": session.base_pose() if session.sim is not None else None,
                }
            elif name == "reset":
                session.reset()
                reply = {"base_pose": session.base_pose()}
            elif name == "init_state":
                reply = np.asarray(session.init_states[int(payload)], dtype=np.float64)
            elif name == "set_state":
                reply = obs_payload(session.set_state(np.asarray(payload, dtype=np.float64)))
            elif name == "step":
                reply = obs_payload(session.step(np.asarray(payload, dtype=np.float64)))
            elif name == "close":
                if session is not None:
                    session.close()
                write_message(stdout, ("ok", None))
                return 0
            else:
                raise ValueError(f"unknown request {name!r}")
            write_message(stdout, ("ok", reply))
        except Exception as exc:  # report to the parent instead of dying silently
            write_message(stdout, ("error", f"{type(exc).__name__}: {exc}"))


def main() -> int:
    """Serve on stdin/stdout with Python and native simulator logging moved to stderr."""
    sys.stdout.flush()
    # Keep a dedicated descriptor for the replies. Redirect fd 1 before the first session imports
    # LIBERO so that both Python prints and native renderer messages go to stderr.
    with os.fdopen(os.dup(sys.stdout.fileno()), "wb") as replies:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        return serve(sys.stdin.buffer, replies)


if __name__ == "__main__":
    raise SystemExit(main())
