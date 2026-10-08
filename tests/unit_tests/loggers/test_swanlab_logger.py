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

"""Tests for mirroring the W&B logger to SwanLab (``wandb.swanlab:``)."""

import sys
import types

import pytest

from nemo_automodel.components.loggers import loggers
from nemo_automodel.components.loggers.loggers import SwanLabConfig, WandbConfig, mirror_wandb_to_swanlab
from nemo_automodel.components.loggers.wandb_utils import init_wandb_run
from nemo_automodel.shared.import_utils import UnavailableError


@pytest.fixture
def fake_swanlab(monkeypatch):
    """Install a fake ``swanlab`` module and reset the once-per-process guard."""
    calls = []
    module = types.ModuleType("swanlab")
    module.sync_wandb = lambda **kw: calls.append(kw)
    module.log = lambda data, step=None: calls.append(("log", data, step))
    module.Text = lambda value: ("text", value)
    monkeypatch.setitem(sys.modules, "swanlab", module)

    class Run:
        def log(self, data=None, step=None):
            calls.append(("wandb_log", data, step))

    wandb_run_module = types.ModuleType("wandb.sdk.wandb_run")
    wandb_run_module.Run = Run
    monkeypatch.setitem(sys.modules, "wandb.sdk.wandb_run", wandb_run_module)
    monkeypatch.setattr(loggers, "_SWANLAB_MIRROR_INSTALLED", False)
    return calls


@pytest.fixture
def fake_wandb(monkeypatch):
    captured = {}
    module = types.ModuleType("wandb")
    module.init = lambda **kw: captured.update(kw) or "run"
    module.Settings = lambda **kw: None
    monkeypatch.setitem(sys.modules, "wandb", module)
    return captured


def test_builtin_scalar_unwraps_zero_dim_values():
    import numpy as np
    import torch

    assert type(loggers._builtin_scalar(np.float32(1.5))) is float
    assert loggers._builtin_scalar(torch.tensor(2.0)) == 2.0
    vector = torch.ones(2)
    assert loggers._builtin_scalar(vector) is vector
    assert loggers._builtin_scalar("text") == "text"


def test_mirror_wandb_defaults_keep_wandb_uploading(fake_swanlab):
    SwanLabConfig().mirror_wandb()
    assert fake_swanlab == [{"mode": "online", "wandb_run": True, "workspace": None, "log_dir": None}]


def test_mirror_wandb_forwards_options_and_patches_once(fake_swanlab):
    cfg = SwanLabConfig(mode="offline", workspace="team", log_dir="/tmp/swanlab", upload_to_wandb=False)
    cfg.mirror_wandb()
    cfg.mirror_wandb()
    assert fake_swanlab == [{"mode": "offline", "wandb_run": False, "workspace": "team", "log_dir": "/tmp/swanlab"}]


def test_mirror_wandb_raises_when_swanlab_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "swanlab", None)
    monkeypatch.setattr(loggers, "_SWANLAB_MIRROR_INSTALLED", False)
    with pytest.raises(UnavailableError, match="swanlab is not installed"):
        SwanLabConfig().mirror_wandb()


def test_mirror_wandb_to_swanlab_pops_block_without_mutating_input(fake_swanlab):
    raw = {"project": "p", "swanlab": {"workspace": "team"}}
    assert mirror_wandb_to_swanlab(raw) == {"project": "p"}
    assert "swanlab" in raw
    assert fake_swanlab[0]["workspace"] == "team"


def test_mirror_wandb_to_swanlab_is_noop_without_block(fake_swanlab):
    assert mirror_wandb_to_swanlab({"project": "p"}) == {"project": "p"}
    assert fake_swanlab == []


def test_wandb_config_parses_swanlab_sub_block():
    cfg = WandbConfig.from_kwargs(project="p", swanlab={"mode": "local"}, mode="offline")
    assert cfg.swanlab == SwanLabConfig(mode="local")
    assert cfg.extra == {"mode": "offline"}
    assert WandbConfig.from_kwargs(project="p").swanlab is None


def test_wandb_config_build_mirrors_before_init(fake_swanlab, fake_wandb):
    cfg = WandbConfig.from_kwargs(project="p", swanlab={"workspace": "team"})
    assert cfg.build(run_config={"lr": 0.1}, model_name="org/model") == "run"
    assert fake_swanlab[0]["workspace"] == "team"
    # The sub-block configures the mirror; it is never forwarded to wandb.init.
    assert "swanlab" not in fake_wandb
    assert fake_wandb["project"] == "p"
    assert fake_wandb["config"] == {"lr": 0.1}


def test_wandb_config_build_without_swanlab_does_not_mirror(fake_swanlab, fake_wandb):
    WandbConfig(project="p").build()
    assert fake_swanlab == []


def test_init_wandb_run_mirrors_and_strips_swanlab(fake_swanlab, fake_wandb):
    assert init_wandb_run({"project": "p", "swanlab": {"mode": "local"}}, {"lr": 0.1}, default_name="d") == "run"
    assert fake_swanlab[0]["mode"] == "local"
    assert "swanlab" not in fake_wandb
    assert fake_wandb["name"] == "d"


def test_mirror_logs_string_values_as_text(fake_swanlab):
    swanlab = sys.modules["swanlab"]
    SwanLabConfig().mirror_wandb()
    swanlab.log({"loss": 1.5, "timestamp": "2026-10-08T00:00:00Z"}, step=3)
    assert fake_swanlab[-1] == ("log", {"loss": 1.5, "timestamp": ("text", "2026-10-08T00:00:00Z")}, 3)


def test_mirror_converts_tensor_scalars_before_wandb_log(fake_swanlab):
    import numpy as np
    import torch

    run_cls = sys.modules["wandb.sdk.wandb_run"].Run
    SwanLabConfig().mirror_wandb()
    run_cls().log({"grad_norm": torch.tensor(3.0), "mfu": np.float32(0.5)}, step=7)
    run_cls().log(data={"loss": np.float64(1.0)}, step=8)
    assert fake_swanlab[-2] == ("wandb_log", {"grad_norm": 3.0, "mfu": 0.5}, 7)
    assert fake_swanlab[-1] == ("wandb_log", {"loss": 1.0}, 8)
    assert type(fake_swanlab[-2][1]["grad_norm"]) is float
