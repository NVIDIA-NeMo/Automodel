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

import sys
import types

import pytest


def _build_fake_wandb_package() -> None:
    """
    Insert a minimal stub of the `wandb` package into `sys.modules`.

    The stub emulates just enough surface area for
    `suppress_wandb_log_messages` to patch:
    - wandb.sdk.internal.file_pusher
        * multiple `_footer*` functions
    - wandb.sdk.internal.run
        * `_footer_single_run_status_info` function
    """
    # Root package
    wandb = types.ModuleType("wandb")

    # Sub-packages
    sdk = types.ModuleType("wandb.sdk")
    internal = types.ModuleType("wandb.sdk.internal")
    file_pusher = types.ModuleType("wandb.sdk.internal.file_pusher")
    run_mod = types.ModuleType("wandb.sdk.internal.run")

    # Dummy footer functions that will later be patched to no-ops
    file_pusher._footer_alpha = lambda *a, **kw: "alpha"
    file_pusher._footer_bravo = lambda *a, **kw: "bravo"
    run_mod._footer_single_run_status_info = lambda *a, **kw: "charlie"

    # Expose sub-packages on their parents
    internal.file_pusher = file_pusher
    internal.run = run_mod
    sdk.internal = internal
    wandb.sdk = sdk

    # Finally, register everything in sys.modules so that normal imports work
    sys.modules.update(
        {
            "wandb": wandb,
            "wandb.sdk": sdk,
            "wandb.sdk.internal": internal,
            "wandb.sdk.internal.file_pusher": file_pusher,
            "wandb.sdk.internal.run": run_mod,
        }
    )


def test_suppress_wandb_log_messages(monkeypatch):
    """
    End-to-end test verifying that all documented side effects take place.
    """
    _build_fake_wandb_package()

    # Import *after* fake package is installed so that module under test
    # uses the stub.
    import logging

    # Pre-conditions: functions return non-None, log-level is NOT CRITICAL
    import wandb.sdk.internal.file_pusher as fp
    import wandb.sdk.internal.run as run_mod

    import nemo_automodel.loggers.wandb_utils as suppress_wandb

    assert fp._footer_alpha() == "alpha"
    assert fp._footer_bravo() == "bravo"
    assert run_mod._footer_single_run_status_info() == "charlie"

    # Explicitly set logger level to something other than CRITICAL
    logging.getLogger("wandb").setLevel(logging.INFO)

    # Suppresssss
    suppress_wandb.suppress_wandb_log_messages()

    # Assert
    # (1) Every patched footer now returns None
    assert fp._footer_alpha() is None
    assert fp._footer_bravo() is None
    assert run_mod._footer_single_run_status_info() is None

    # (2) Logger level was raised to CRITICAL
    assert logging.getLogger("wandb").level == logging.CRITICAL


@pytest.fixture(autouse=True)
def _clean_sys_modules():
    """
    Automatically clean up any injected fake wandb modules after each test.
    """
    original_modules = set(sys.modules.keys())
    yield
    for name in list(sys.modules):
        if name.startswith("wandb"):
            del sys.modules[name]
    # Remove any additional test artefacts
    for name in list(sys.modules):
        if name not in original_modules and name.startswith("wandb"):
            del sys.modules[name]
