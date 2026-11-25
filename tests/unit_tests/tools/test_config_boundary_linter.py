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
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tools import lint_config_boundary
from tools.lint_config_boundary import check, collect_python_files, lint_source, main

MODULE = Path("nemo_automodel/components/example.py")


def _codes(source: str) -> list[tuple[int, str]]:
    return [(error.line, error.code) for error in lint_source(textwrap.dedent(source), MODULE)]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from nemo_automodel.components.config.loader import ConfigNode", ["CB001"]),
        ("def f(cfg: ConfigNode) -> None: pass", ["CB001"]),
        ('def f(cfg: "ConfigNode | None") -> None: pass', ["CB001"]),
        ("def f(recipe: RecipeConfig) -> None: pass", ["CB001"]),
        ("x: loader.ConfigNode", ["CB001"]),
        ('def f(*args: "ConfigNode", **kwargs: "RecipeConfig"): pass', ["CB001", "CB001"]),
        ("class _ConfigNodeOptions: pass", ["CB001"]),
        ("node.to_yaml_dict(use_orig_values=True)", ["CB002"]),
        ("dataset.instantiate()", ["CB002"]),
        ("node.raw_config", ["CB002"]),
        ("def f(x=node.instantiate(), *, y=node.get_as_string('k')): pass", ["CB002", "CB002"]),
        ('block.to_dict() if hasattr(block, "to_dict") else dict(block)', ["CB003"]),
        ('getattr(optimizer, "_target_", None)', ["CB003"]),
        ('hasattr(model, "weight")', []),
        ("def f(cfg): pass", ["CB004"]),
        ("def f(model_cfg: Any): pass", ["CB004"]),
        ("def f(*, recipe_cfg: Any | None = None): pass", ["CB004"]),
        ("def f(cfg_fp8: Optional[Any]): pass", ["CB004"]),
        ("def f(cfg: 'Any'): pass", ["CB004"]),
        ("g = lambda cfg: cfg", ["CB004"]),
        ("def f(cfg: MLflowConfig): pass", []),
        ("def f(block_cfg: dict[str, int]): pass", []),
        ("def f(enable_cfg=True): pass", []),
        ("def f(config: Any): pass", []),
        ("class Source(Protocol):\n    def get(self, key: str, default: bool = ...) -> bool: ...", ["CB005"]),
        ("class Options(Protocol):\n    def to_dict(self) -> dict[str, int]: ...", ["CB005"]),
        ("class StepSchedule(Protocol):\n    @property\n    def max_steps(self) -> int | None: ...", []),
        ("class Cache:\n    def get(self, key: str) -> int: return 0", []),
    ],
)
def test_config_boundary_rules(source: str, expected: list[str]) -> None:
    assert [code for _, code in _codes(source)] == expected


def test_type_checking_import_is_still_a_reference() -> None:
    assert _codes(
        """
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            from nemo_automodel.components.config.loader import ConfigNode


        def configure(cfg: "ConfigNode") -> None:
            pass
        """
    ) == [(5, "CB001"), (8, "CB001")]


def test_rejects_dotted_yaml_paths_on_config_objects_only() -> None:
    assert _codes(
        """
        def f(cfg, state_dict, modules):
            cfg.get("model.backend.attn", "")
            self.cfg.get("model.backend.attn")
            model_cfg.get("model.backend.attn")
            state_dict.get("model.norm.weight")
            modules.get("transformer_engine.pytorch")
            cfg.get("model")
        """
    ) == [(2, "CB004"), (3, "CB006"), (4, "CB006"), (5, "CB006")]


def _write(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))
    return path


def test_findings_name_their_enclosing_function() -> None:
    (error,) = lint_source("class A:\n    def build(self, cfg):\n        pass\n", MODULE)
    assert error.function == "A.build"


def test_allowlisted_shims_pass_and_stale_entries_are_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        lint_config_boundary,
        "ALLOWED",
        {
            ("nemo_automodel/components/loggers/shim.py", "configure"): "deprecated",
            ("nemo_automodel/components/loggers/gone.py", "build"): "deprecated",
        },
    )
    _write(
        tmp_path, "nemo_automodel/components/loggers/shim.py", "def configure(cfg):\n    return cfg.to_yaml_dict()\n"
    )
    _write(tmp_path, "nemo_automodel/components/loggers/other.py", "def other(cfg):\n    pass\n")
    files = collect_python_files([tmp_path / "nemo_automodel" / "components"], tmp_path)

    lines = check(files, tmp_path, report_stale=True)

    assert lines == [
        "nemo_automodel/components/loggers/other.py:1:11: CB004 config parameter `cfg` has no type; "
        "take a typed config instead (in `other`)",
        "nemo_automodel/components/loggers/gone.py: stale ALLOWED entry for `build`; "
        "remove it from tools/lint_config_boundary.py",
    ]


def test_skips_the_config_package_and_files_outside_components(tmp_path: Path) -> None:
    _write(tmp_path, "nemo_automodel/components/config/loader.py", "class ConfigNode:\n    pass\n")
    _write(tmp_path, "nemo_automodel/recipes/recipe.py", "def f(cfg):\n    pass\n")
    kept = _write(tmp_path, "nemo_automodel/components/utils/x.py", "X = 1\n")

    files = collect_python_files([tmp_path / "nemo_automodel"], tmp_path)

    assert files == [kept]


def test_main_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lint_config_boundary, "ALLOWED", {})
    _write(tmp_path, "nemo_automodel/components/loggers/__init__.py", "")
    clean = ["--automodel-dir", str(tmp_path)]
    assert main(clean) == 0

    bad = _write(tmp_path, "nemo_automodel/components/loggers/bad.py", "def f(cfg):\n    pass\n")
    assert main(clean) == 1
    assert "CB004" in capsys.readouterr().err
    # pre-commit passes changed files; a non-component file is nothing to lint, not an error.
    outside = _write(tmp_path, "tools/script.py", "def f(cfg):\n    pass\n")
    assert main([str(outside), *clean]) == 0
    assert main([str(bad), *clean]) == 1
    assert main([str(tmp_path / "missing.py"), *clean]) == 2


def test_repository_components_are_clean() -> None:
    """The real tree passes, with every ALLOWED entry still in use."""
    root = Path(__file__).resolve().parents[3]
    files = collect_python_files([root / "nemo_automodel" / "components"], root)
    assert check(files, root, report_stale=True) == []
