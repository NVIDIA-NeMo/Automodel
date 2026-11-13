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

"""Exercise the LIBERO worker process of the Praxis-1 tutorial and its wire protocol with a noisy simulator."""

import importlib.util
import io
import subprocess
import sys
from pathlib import Path

import pytest

TUTORIAL = Path(__file__).resolve().parents[3] / "tutorials" / "praxis-1-libero"

pytestmark = pytest.mark.cpu_only


def load_bridge():
    """The tutorial's ``libero_bridge`` module (the tutorial directory is not a package)."""
    spec = importlib.util.spec_from_file_location("libero_bridge", TUTORIAL / "libero_bridge.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["libero_bridge"] = module  # dataclasses resolve the module's annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def test_simulator_logging_does_not_corrupt_binary_replies():
    bridge = load_bridge()
    requests = io.BytesIO()
    bridge.write_message(requests, ("init", {"suite": "test", "task_id": 0, "height": 16, "width": 16, "seed": 7}))
    bridge.write_message(requests, ("close", None))
    # Replace only the simulator session; exercise the worker entry point and the
    # transport in a separate, unbuffered Python process.
    script = """
import importlib.util, os, sys, types
spec = importlib.util.spec_from_file_location('libero_sim', sys.argv[1])
sim = importlib.util.module_from_spec(spec)
sys.modules['libero_sim'] = sim
spec.loader.exec_module(sim)
class Session:
    language = 'Move left'
    task = types.SimpleNamespace(name='test')
    bddl = 'test.bddl'
    init_states = [0]
    sim = None
    def __init__(self, *args):
        print('Simulator import status')
        print('Python simulator status')
        os.write(1, b'Native simulator status\\n')
    def close(self):
        print('Simulator closed')
sim.LiberoSession = Session
raise SystemExit(sim.main())
"""
    result = subprocess.run(
        [sys.executable, "-u", "-c", script, str(TUTORIAL / "libero_sim.py")],
        input=requests.getvalue(),
        capture_output=True,
        check=True,
        timeout=30,
    )
    replies = io.BytesIO(result.stdout)
    status, reply = bridge.read_message(replies)
    assert status == "ok" and reply["language"] == "Move left"
    assert bridge.read_message(replies) == ["ok", None]
    assert replies.read() == b""
    for message in (
        "Simulator import status",
        "Python simulator status",
        "Native simulator status",
        "Simulator closed",
    ):
        assert message in result.stderr.decode()


def test_osc_goal_keeps_the_retained_orientation_for_zero_rotation():
    """robosuite's OSC_POSE controller only replaces its goal orientation when the rotation delta is nonzero."""
    import numpy as np

    bridge = load_bridge()
    retained = bridge.rot_z(0.2)
    observed = bridge.rot_z(0.05)
    grip_pos = np.array([0.1, 0.2, 0.3])
    hold = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    goal_pos, goal_rot = bridge.osc_goal(grip_pos, observed, hold, retained)
    np.testing.assert_allclose(goal_pos, grip_pos)
    np.testing.assert_allclose(goal_rot, retained)
    turn = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.4, 1.0])
    goal_pos, goal_rot = bridge.osc_goal(grip_pos, observed, turn, retained)
    np.testing.assert_allclose(goal_pos, grip_pos + [0.2 * bridge.OSC_POS_MAX, 0.0, 0.0])
    np.testing.assert_allclose(goal_rot, bridge.rot_z(0.05 + 0.4 * bridge.OSC_ROT_MAX), atol=1e-12)
