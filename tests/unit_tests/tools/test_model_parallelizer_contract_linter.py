# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the model-parallelizer architecture linter."""

from pathlib import Path

import pytest

from tools.lint_model_parallelizer_contract import lint_sidecar_exports, lint_source


def _messages(source: str, path: Path = Path("example.py")) -> list[str]:
    return [error.message for error in lint_source(source, path)]


def test_rejects_parallel_scheme_argument_and_keyword():
    messages = _messages(
        "def load(*, parallel_scheme=None):\n    return pipeline.from_pretrained(parallel_scheme=parallel_scheme)\n"
    )
    assert len(messages) == 3
    assert all("use MeshContext only" in message for message in messages)


@pytest.mark.parametrize(
    "source",
    [
        "from package import ParallelizeContext\n",
        "import package.ParallelizeContext\n",
        "class ParallelizeContext:\n    pass\n",
    ],
)
def test_rejects_parallelize_context(source):
    messages = _messages(source)
    assert messages == ["ParallelizeContext creates a second model-parallelization interface; use MeshContext only"]


def test_accepts_single_mesh_context_contract():
    assert _messages("def parallelize(model, mesh_context):\n    return model\n") == []


def test_rejects_distributed_import_of_model_implementation():
    path = Path("nemo_automodel/components/distributed/example.py")
    for source in (
        "from nemo_automodel.components.models.foo import Model\n",
        "import nemo_automodel._diffusers.auto_diffusion_pipeline\n",
        "from ..models.foo import Model\n",
    ):
        assert _messages(source, path) == ["distributed infrastructure may not import model or adapter implementations"]


def test_allows_type_only_distributed_import_of_model_implementation():
    path = Path("nemo_automodel/components/distributed/example.py")
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from nemo_automodel.components.models.foo import Model\n"
    )
    assert _messages(source, path) == []


def test_model_sidecar_may_import_only_exported_distributed_symbols(tmp_path):
    distributed = tmp_path / "nemo_automodel/components/distributed"
    distributed.mkdir(parents=True)
    (distributed / "__init__.py").write_text('__all__ = ["ModelParallelizer"]\n')
    source = (
        "from nemo_automodel.components.distributed import ModelParallelizer, InternalHelper\n"
        "class Sidecar(ModelParallelizer):\n    pass\n"
    )

    errors = lint_sidecar_exports(
        source,
        tmp_path / "nemo_automodel/components/models/example/parallelization.py",
        tmp_path,
    )

    assert [error.message for error in errors] == [
        "model sidecars may import only distributed symbols declared in __all__; InternalHelper is not exported"
    ]


def test_model_sidecar_rejects_distributed_module_import(tmp_path):
    source = (
        "import nemo_automodel.components.distributed.parallelizer as distributed\n"
        "class Sidecar(distributed.ModelParallelizer):\n    pass\n"
    )

    errors = lint_sidecar_exports(
        source,
        tmp_path / "nemo_automodel/components/models/example/parallelization.py",
        tmp_path,
    )

    assert [error.message for error in errors] == [
        "model sidecars must import named public distributed symbols, not modules"
    ]
