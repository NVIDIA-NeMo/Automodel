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


# --------------------------------------------------------------------------- CB001 ConfigNode references


@pytest.mark.parametrize(
    "source",
    [
        "from nemo_automodel.components.config.loader import ConfigNode\n",
        "def f(cfg: ConfigNode) -> None:\n    pass\n",
        'def f(cfg: "ConfigNode | None") -> None:\n    pass\n',
        "def f(recipe: RecipeConfig) -> None:\n    pass\n",
        "x: loader.ConfigNode\n",
        "class _ConfigNodeOptions:\n    pass\n",
    ],
)
def test_rejects_config_node_references(source):
    assert "CB001" in [code for _, code in _codes(source)]


def test_type_checking_import_is_still_a_reference():
    """The PR #4947 shape: a ``TYPE_CHECKING`` import of ConfigNode to annotate a component."""
    assert _codes(
        """
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            from nemo_automodel.components.config.loader import ConfigNode


        def configure(cfg: "ConfigNode") -> None:
            pass
        """
    ) == [(5, "CB001"), (8, "CB001")]


# --------------------------------------------------------------------------- CB002 / CB003 ConfigNode API


def test_rejects_config_node_only_methods():
    assert _codes(
        """
        def f(node, dataset):
            node.to_yaml_dict(use_orig_values=True)
            dataset.instantiate()
            return node.raw_config
        """
    ) == [(3, "CB002"), (4, "CB002"), (5, "CB002")]


def test_rejects_probes_for_the_config_node_api():
    assert _codes(
        """
        def f(block, optimizer):
            block = block.to_dict() if hasattr(block, "to_dict") else dict(block)
            return getattr(optimizer, "_target_", None)
        """
    ) == [(3, "CB003"), (4, "CB003")]


def test_allows_unrelated_reflection():
    assert _codes('def f(m):\n    return hasattr(m, "weight")\n') == []


# --------------------------------------------------------------------------- CB004 duck-typed config parameters


@pytest.mark.parametrize(
    "signature",
    [
        "def f(cfg): pass",
        "def f(model_cfg: Any): pass",
        "def f(*, recipe_cfg: Any | None = None): pass",
        "def f(cfg_fp8: Optional[Any]): pass",
        "def f(cfg: 'Any'): pass",
        "g = lambda cfg: cfg",
    ],
)
def test_rejects_untyped_config_parameters(signature):
    assert [code for _, code in _codes(signature + "\n")] == ["CB004"]


@pytest.mark.parametrize(
    "signature",
    [
        "def f(cfg: MLflowConfig): pass",
        "def f(block_cfg: dict[str, int]): pass",
        "def f(enable_cfg=True): pass",  # classifier-free guidance flag, not a config
        "def f(config: Any): pass",  # HF model configs are out of scope
    ],
)
def test_allows_typed_config_parameters(signature):
    assert _codes(signature + "\n") == []


# --------------------------------------------------------------------------- CB005 / CB006


def test_rejects_protocols_that_mirror_a_config_node():
    assert _codes(
        """
        from typing import Protocol


        class _Source(Protocol):
            def get(self, key: str, default: bool = ...) -> bool: ...


        class _Options(Protocol):
            def to_dict(self) -> dict[str, int]: ...
        """
    ) == [(5, "CB005"), (9, "CB005")]


def test_allows_other_protocols_and_plain_classes_with_get():
    assert (
        _codes(
            """
        from typing import Protocol


        class StepSchedule(Protocol):
            @property
            def max_steps(self) -> int | None: ...


        class Cache:
            def get(self, key: str) -> int:
                return 0
        """
        )
        == []
    )


def test_rejects_dotted_yaml_paths_on_config_objects_only():
    assert _codes(
        """
        def f(cfg, state_dict, modules):
            cfg.get("model.backend.attn", "")
            state_dict.get("model.norm.weight")
            modules.get("transformer_engine.pytorch")
            cfg.get("model")
        """
    ) == [(2, "CB004"), (3, "CB006")]


# --------------------------------------------------------------------------- scope, allowlist and CLI


def _write(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))
    return path


def test_findings_name_their_enclosing_function():
    (error,) = lint_source("class A:\n    def build(self, cfg):\n        pass\n", MODULE)
    assert error.function == "A.build"


def test_allowlisted_shims_pass_and_stale_entries_are_reported(tmp_path, monkeypatch):
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


def test_skips_the_config_package_and_files_outside_components(tmp_path):
    _write(tmp_path, "nemo_automodel/components/config/loader.py", "class ConfigNode:\n    pass\n")
    _write(tmp_path, "nemo_automodel/recipes/recipe.py", "def f(cfg):\n    pass\n")
    kept = _write(tmp_path, "nemo_automodel/components/utils/x.py", "X = 1\n")

    files = collect_python_files([tmp_path / "nemo_automodel"], tmp_path)

    assert files == [kept]


def test_main_exit_codes(tmp_path, capsys, monkeypatch):
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


def test_repository_components_are_clean():
    """The real tree passes, with every ALLOWED entry still in use."""
    root = Path(__file__).resolve().parents[3]
    files = collect_python_files([root / "nemo_automodel" / "components"], root)
    assert check(files, root, report_stale=True) == []
