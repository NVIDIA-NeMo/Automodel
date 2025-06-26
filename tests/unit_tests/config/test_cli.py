# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import types
import builtins
import pytest

cli = __import__("nemo_automodel.config.cli", fromlist=["dummy"])



class DummyConfig:
    """
    Simple stand-in for the NeMo ConfigNode.
    It records every call to set_by_dotted so that the test can assert on them.
    """
    def __init__(self):
        self._calls = []

    def set_by_dotted(self, dotted, value):
        self._calls.append((dotted, value))

    # allow direct attribute access in case caller relies on it later
    def __getattr__(self, item):
        raise AttributeError(item)


@pytest.mark.parametrize(
    "argv, expected_cfg, expected_overrides",
    [
        (
            ["prog", "--config", "model.yaml",
             "--trainer.max_epochs", "10",
             "--precision=bf16",
             "--fast"],
            "model.yaml",
            ["trainer.max_epochs=10", "precision=bf16", "fast=True"],
        ),
        (
            ["prog", "-c", "base.yaml"],
            "base.yaml",
            [],
        ),
    ],
)
def test_parse_cli_argv_valid(monkeypatch, argv, expected_cfg, expected_overrides):
    """
    Ensure argv gets parsed into the correct (cfg_path, overrides) tuple.
    """
    monkeypatch.setattr(cli.sys, "argv", argv, raising=True)
    cfg_path, overrides = cli.parse_cli_argv()
    assert cfg_path == expected_cfg
    # Order of overrides must be preserved
    assert overrides == expected_overrides


def test_parse_cli_argv_missing_config(monkeypatch):
    """
    Omitting --config entirely should raise ValueError.
    """
    monkeypatch.setattr(cli.sys, "argv", ["prog", "--foo", "bar"], raising=True)
    with pytest.raises(ValueError, match="You must specify --config"):
        cli.parse_cli_argv()


def test_parse_cli_argv_missing_path(monkeypatch):
    """
    Passing --config with no following token should raise ValueError.
    """
    monkeypatch.setattr(cli.sys, "argv", ["prog", "--config"], raising=True)
    with pytest.raises(ValueError, match="Expected a path after --config"):
        cli.parse_cli_argv()

def test_parse_args_and_load_config(monkeypatch):
    """
    Full end-to-end check with mocked YAML loader, translate_value,
    and ConfigNode.set_by_dotted.
    """

    # 1. Fake argv
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["prog", "--config", "cfg.yaml", "--opt.lr", "1e-4", "--seed=123"],
        raising=True,
    )

    # 2. Provide dummy loader & translator implementations
    dummy_cfg = DummyConfig()
    monkeypatch.setattr(cli, "load_yaml_config", lambda path: dummy_cfg, raising=True)

    captured_pairs = []                       # (val_str)
    monkeypatch.setattr(
        cli,
        "translate_value",
        lambda s: f"<T:{s}>",
        raising=True,
    )

    # 3. Run the function under test
    returned_cfg = cli.parse_args_and_load_config()

    # 4. Assertions
    assert returned_cfg is dummy_cfg
    # function returns the same object
    # two overrides should have been applied
    assert dummy_cfg._calls == [
        ("opt.lr", "<T:1e-4>"),
        ("seed", "<T:123>"),
    ]
