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

"""Conventions and messages shared by the LIBERO side of the tutorial and the policy side.

LIBERO (robosuite 1.4.1) runs the Franka Panda with the ``OSC_POSE`` delta controller at 20 Hz:
an action is ``[dx, dy, dz, rx, ry, rz, grip]`` in [-1, 1], scaled to at most 0.05 m and 0.5 rad
per control step, applied to the ``grip_site`` of the gripper in the world frame; ``grip`` is a
sign (-1 opens, +1 closes). The policy (Praxis-1) reasons about the Franka flange at
15 Hz in the robot base frame. This module holds the fixed transforms between the two worlds,
the 20 Hz to 15 Hz index map, the OSC goal reconstruction, and the message format of the pipe between
the policy process and the simulator worker (a JSON header plus raw array bytes, so the two interpreters
may run different NumPy versions and nothing is unpickled). It imports only NumPy so that both the LIBERO
virtual environment and the policy container can use it.

Frames (all measured in the probe of the LIBERO Panda model):

* ``robot0_right_hand`` is the flange turned by -45 degrees about its z axis; its origin sits
  0.1065 m past link 7 (the Franka datasheet says 0.107 m). This is the ``TURNED_FLANGE`` rig frame
  of the policy (its sim's ``ee`` site), so hand poses go to the policy unchanged with the
  ``praxis.sim`` descriptor, and training files hold the flange frame (hand turned back by +45 deg).
* ``gripper0_grip_site`` (the OSC control point) is 0.097 m along the hand z axis and turned by
  -90 degrees about it.
* robosuite returns offscreen renders bottom-up (OpenGL rows). A vertical flip gives the upright,
  non-mirrored image; the 180-degree rotation used by some LIBERO pipelines also mirrors left-right.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np

ENV_HZ = 20.0  # LIBERO control frequency
POLICY_HZ = 15.0  # Praxis command rate
OSC_POS_MAX = 0.05  # metres per control step at |action| = 1
OSC_ROT_MAX = 0.5  # radians per control step at |action| = 1
GRIPPER_WIDTH_MAX = 0.08  # finger_joint1 - finger_joint2 when fully open
LIBERO_DUMMY_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
NUM_STEPS_WAIT = 10  # settle steps with the dummy action before the first policy call (OpenVLA / openpi)


def rot_z(angle: float) -> np.ndarray:
    """Rotation matrix about z by ``angle`` radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# grip_site pose in the hand frame: T_hand_grip = [Rz(-90 deg) | (0, 0, 0.097)]
HAND_TO_GRIP = np.eye(4)
HAND_TO_GRIP[:3, :3] = rot_z(-np.pi / 2)
HAND_TO_GRIP[2, 3] = 0.097
GRIP_TO_HAND = np.linalg.inv(HAND_TO_GRIP)
# flange = hand turned back by +45 deg (the policy's TURNED_FLANGE rig frame has turn Rz(-45 deg))
HAND_TO_FLANGE_TURN = rot_z(np.pi / 4)


def make_pose(xyz: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Homogeneous ``(4, 4)`` pose from a position and a rotation matrix."""
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = xyz
    return pose


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a unit quaternion ``[w, x, y, z]``."""
    w, x, y, z = (float(v) for v in np.asarray(q, dtype=np.float64) / np.linalg.norm(q))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    """Unit quaternion ``[w, x, y, z]`` of a rotation matrix.

    Same operations as ``scipy.spatial.transform.Rotation.from_matrix(m).as_quat()`` (largest-component
    branch, then an exact normalization), which is what the policy's ``praxis.single_arm`` uses; the sign is
    therefore the same as in the policy's own files and is not made canonical.
    """
    c = np.asarray(m, dtype=np.float64)
    dec = [c[0, 0], c[1, 1], c[2, 2], c[0, 0] + c[1, 1] + c[2, 2]]
    choice = int(np.argmax(dec))
    q = np.empty(4)  # xyzw
    if choice != 3:
        i = choice
        j = (i + 1) % 3
        k = (j + 1) % 3
        q[i] = 1 - dec[3] + 2 * c[i, i]
        q[j] = c[j, i] + c[i, j]
        q[k] = c[k, i] + c[i, k]
        q[3] = c[k, j] - c[j, k]
    else:
        q[0] = c[2, 1] - c[1, 2]
        q[1] = c[0, 2] - c[2, 0]
        q[2] = c[1, 0] - c[0, 1]
        q[3] = 1 + dec[3]
    q = q / np.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return q[[3, 0, 1, 2]]


def axis_angle_to_matrix(v: np.ndarray) -> np.ndarray:
    """Rotation matrix of an axis-angle vector (Rodrigues)."""
    v = np.asarray(v, dtype=np.float64)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def matrix_to_axis_angle(m: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix (angle in [0, pi])."""
    m = np.asarray(m, dtype=np.float64)
    cos_angle = np.clip((np.trace(m) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-6:
        # near pi: take the axis from the symmetric part
        w, v = np.linalg.eigh(m)
        axis = v[:, np.argmin(np.abs(w - 1.0))]
        return axis / np.linalg.norm(axis) * angle
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]) / (2 * np.sin(angle))
    return axis * angle


def pose8_from_matrix(pose: np.ndarray, grip: float) -> np.ndarray:
    """``[xyz3, quat_wxyz4, grip1]`` float32 row of a homogeneous pose."""
    return np.concatenate([pose[:3, 3], matrix_to_quat_wxyz(pose[:3, :3]), [grip]]).astype(np.float32)


def matrix_from_pose8(row: np.ndarray) -> np.ndarray:
    """Homogeneous pose of a ``[xyz3, quat_wxyz4, ...]`` row."""
    return make_pose(np.asarray(row[:3], dtype=np.float64), quat_wxyz_to_matrix(row[3:7]))


def hand_to_flange_pose8(hand_rows: np.ndarray) -> np.ndarray:
    """Convert hand (turned flange) pose8 rows to the flange frame of the policy's training files.

    Same operation as ``praxis.single_arm.pose8_to_model(rows, TURNED_FLANGE)``: the position stays,
    the rotation becomes ``R_hand @ turn^T`` with ``turn = Rz(-45 deg)``, the grip is clipped to [0, 1].
    """
    rows = np.asarray(hand_rows, dtype=np.float64)
    out = rows.copy()
    for i, row in enumerate(rows):
        out[i, 3:7] = matrix_to_quat_wxyz(quat_wxyz_to_matrix(row[3:7]) @ HAND_TO_FLANGE_TURN)
    out[:, 7] = np.clip(out[:, 7], 0.0, 1.0)
    return out.astype(np.float32)


def osc_goal(
    grip_pos: np.ndarray, grip_rot: np.ndarray, action: np.ndarray, goal_rot: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The goal pose that robosuite's OSC_POSE controller holds after one delta ``action``.

    Mirrors ``OperationalSpaceController.set_goal``: the delta is clipped to [-1, 1] and scaled to
    ``OSC_POS_MAX`` / ``OSC_ROT_MAX``; the goal position is ``p_current + delta_pos``. A nonzero rotation
    delta sets the goal orientation to ``R(delta_rot) @ R_current`` (world frame); an all-zero rotation
    delta leaves the controller's retained goal orientation ``goal_rot`` in place.
    """
    delta = np.clip(np.asarray(action[:6], dtype=np.float64), -1.0, 1.0)
    goal_pos = np.asarray(grip_pos, dtype=np.float64) + delta[:3] * OSC_POS_MAX
    if not np.any(delta[3:6]):  # the controller compares each scaled component with exact zero
        return goal_pos, np.array(goal_rot, dtype=np.float64)
    return goal_pos, axis_angle_to_matrix(delta[3:6] * OSC_ROT_MAX) @ np.asarray(grip_rot, dtype=np.float64)


def osc_action_for_goal(
    grip_pos: np.ndarray, grip_rot: np.ndarray, goal_pos: np.ndarray, goal_rot: np.ndarray
) -> np.ndarray:
    """The 6-D OSC_POSE delta that makes the controller's goal equal to ``goal`` (clipped to [-1, 1])."""
    d_pos = (np.asarray(goal_pos, dtype=np.float64) - np.asarray(grip_pos, dtype=np.float64)) / OSC_POS_MAX
    d_rot = matrix_to_axis_angle(np.asarray(goal_rot) @ np.asarray(grip_rot).T) / OSC_ROT_MAX
    return np.clip(np.concatenate([d_pos, d_rot]), -1.0, 1.0)


def gripper_closedness(gripper_qpos: np.ndarray) -> float:
    """Policy grip value (0 = open, 1 = closed) from the two finger joint positions."""
    width = float(gripper_qpos[0] - gripper_qpos[1])
    return float(np.clip(1.0 - width / GRIPPER_WIDTH_MAX, 0.0, 1.0))


def gripper_action_from_grip(grip: float) -> float:
    """robosuite gripper action (-1 open, +1 close) from a policy grip value in [0, 1]."""
    return 1.0 if grip >= 0.5 else -1.0


def policy_frame_indices(n_env_steps: int) -> np.ndarray:
    """Env-step indices of the 15 Hz policy frames: ``round(k * ENV_HZ / POLICY_HZ)`` while inside the episode."""
    ratio = ENV_HZ / POLICY_HZ
    k_max = int(np.floor((n_env_steps - 1) / ratio)) + 1
    idx = np.rint(np.arange(k_max + 1) * ratio).astype(np.int64)
    return idx[idx < n_env_steps]


def slerp(r0: np.ndarray, r1: np.ndarray, t: float) -> np.ndarray:
    """Rotation matrix interpolated between ``r0`` (t = 0) and ``r1`` (t = 1) along the geodesic."""
    return axis_angle_to_matrix(matrix_to_axis_angle(r1 @ r0.T) * t) @ r0


def interpolate_pose(
    times: np.ndarray, positions: np.ndarray, rotations: np.ndarray, t: float
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear position and geodesic rotation at time ``t`` of a pose trajectory; holds the ends."""
    if t <= times[0]:
        return positions[0], rotations[0]
    if t >= times[-1]:
        return positions[-1], rotations[-1]
    i = int(np.searchsorted(times, t, side="right") - 1)
    a = (t - times[i]) / (times[i + 1] - times[i])
    return (1 - a) * positions[i] + a * positions[i + 1], slerp(rotations[i], rotations[i + 1], a)


def upright(frame: np.ndarray) -> np.ndarray:
    """Upright image of a robosuite offscreen render (rows come bottom-up)."""
    return np.ascontiguousarray(frame[::-1])


class TargetSchedule:
    """Hand-pose targets of one policy chunk as a function of time, for the 20 Hz OSC bridge.

    Node 0 is the hand pose at the tick of the policy call (time ``t0`` in ticks), nodes 1..R are the
    command rows of the chunk at ``t0 + r``. Positions interpolate linearly, rotations along the
    geodesic, and the grip of a time is the grip of the first row at or after that time (the policy's
    rows are targets to reach by their tick).
    """

    def __init__(self, t0: float, hand0: np.ndarray, rows_pose12: np.ndarray, rows_grip: np.ndarray) -> None:
        rows_pose12 = np.asarray(rows_pose12, dtype=np.float64)
        self.t0 = float(t0)
        self.rows = int(len(rows_pose12))
        self.times = self.t0 + np.arange(self.rows + 1, dtype=np.float64)
        self.positions = np.concatenate([hand0[None, :3, 3], rows_pose12[:, :3]], axis=0)
        self.rotations = np.concatenate([hand0[None, :3, :3], rows_pose12[:, 3:].reshape(-1, 3, 3)], axis=0)
        self.grips = np.asarray(rows_grip, dtype=np.float64)

    def hand_goal(self, t: float) -> np.ndarray:
        """Homogeneous hand pose (base frame) to reach at time ``t`` (ticks)."""
        pos, rot = interpolate_pose(self.times, self.positions, self.rotations, t)
        return make_pose(pos, rot)

    def grip(self, t: float) -> float:
        """Policy grip value in force at time ``t``."""
        r = int(np.ceil(t - self.t0 - 1e-9))
        return float(self.grips[min(max(r, 1), self.rows) - 1])


def osc_step_action(
    schedule: TargetSchedule, t_next: float, grip_world: np.ndarray, world_from_base: np.ndarray
) -> np.ndarray:
    """The 7-D robosuite action for one 20 Hz step: track the schedule's hand goal at ``t_next`` with the grip site."""
    goal_world = world_from_base @ schedule.hand_goal(t_next) @ HAND_TO_GRIP
    a6 = osc_action_for_goal(grip_world[:3, 3], grip_world[:3, :3], goal_world[:3, 3], goal_world[:3, :3])
    return np.concatenate([a6, [gripper_action_from_grip(schedule.grip(t_next))]])


def tick_time_after_step(env_steps_done: int) -> float:
    """Time in policy ticks at the end of env step number ``env_steps_done`` (1-based)."""
    return env_steps_done * POLICY_HZ / ENV_HZ


# ---------------------------------------------------------------------------------------------------------
# Messages between the policy process and the simulator worker: a JSON header plus raw array bytes. A
# message is any nesting of dicts, lists, tuples, strings, numbers, booleans, ``None`` and NumPy arrays.


def checkpoint_dir(name_or_path: str) -> Path:
    """Local checkpoint directory: the path itself, or the Hugging Face repo ``name_or_path`` in the Hub cache."""
    local = Path(name_or_path)
    if local.is_dir():
        return local.absolute()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=name_or_path))


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def _split(obj: object, blobs: list[bytes]) -> JsonValue:
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        blobs.append(a.tobytes())
        return {"__nd__": len(blobs) - 1, "dtype": a.dtype.str, "shape": list(a.shape), "nbytes": a.nbytes}
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _split(v, blobs) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_split(v, blobs) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"cannot send {type(obj).__name__}")


def _join(obj: JsonValue, blobs: list[bytes]) -> object:
    if isinstance(obj, dict):
        if "__nd__" in obj:
            return np.frombuffer(blobs[int(obj["__nd__"])], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()
        return {k: _join(v, blobs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_join(v, blobs) for v in obj]
    return obj


def write_message(stream: BinaryIO, message: object) -> None:
    """Write one message."""
    blobs: list[bytes] = []
    header = json.dumps(_split(message, blobs)).encode("utf-8")
    stream.write(struct.pack("<QQ", len(header), len(blobs)))
    stream.write(header)
    for blob in blobs:
        stream.write(struct.pack("<Q", len(blob)))
        stream.write(blob)
    stream.flush()


def _read_exact(stream: BinaryIO, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("worker stream closed")
        buf.extend(chunk)
    return bytes(buf)


def read_message(stream: BinaryIO) -> object:
    """Read one message written by :func:`write_message` (``EOFError`` when the stream ends)."""
    header_len, n_blobs = struct.unpack("<QQ", _read_exact(stream, 16))
    header = json.loads(_read_exact(stream, header_len).decode("utf-8"))
    blobs = [_read_exact(stream, struct.unpack("<Q", _read_exact(stream, 8))[0]) for _ in range(n_blobs)]
    return _join(header, blobs)
