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

from tools.sync_model_coverage_tables import (
    HOMEPAGE_END_MARKER,
    HOMEPAGE_START_MARKER,
    REGISTRY_END_MARKER,
    REGISTRY_START_MARKER,
    _parse_registry_entries,
    _render_registry_table,
    _replace_generated_block,
    _sync_tables,
)


def test_registry_table_is_generated_from_mapping_entries():
    source = """
from collections import OrderedDict

MODEL_ARCH_MAPPING = OrderedDict(
    [
        ("ZuluForCausalLM", ("nemo_automodel.components.models.zulu.model", "ZuluForCausalLM")),
        (
            "AlphaModel",
            ("nemo_automodel.components.models.alpha.model", "AlphaModel", {"retrieval"}),
        ),
    ]
)
"""

    entries = _parse_registry_entries(source)
    generated = _render_registry_table(entries)

    assert entries == [
        ("AlphaModel", "nemo_automodel.components.models.alpha.model", "AlphaModel"),
        ("ZuluForCausalLM", "nemo_automodel.components.models.zulu.model", "ZuluForCausalLM"),
    ]
    assert "| `AlphaModel` | `nemo_automodel.components.models.alpha.model.AlphaModel` |" in generated
    assert "| `ZuluForCausalLM` | `nemo_automodel.components.models.zulu.model.ZuluForCausalLM` |" in generated


def test_generated_registry_table_replaces_the_marked_block():
    document = f"before\n{REGISTRY_START_MARKER}\nstale\n{REGISTRY_END_MARKER}\nafter\n"
    generated = _render_registry_table([("NewModel", "models.new", "NewModel")])

    updated = _replace_generated_block(document, REGISTRY_START_MARKER, REGISTRY_END_MARKER, generated)

    assert "stale" not in updated
    assert "| `NewModel` | `models.new.NewModel` |" in updated
    assert updated.startswith("before\n")
    assert updated.endswith("\nafter\n")


def test_sync_tables_writes_homepage_and_new_registry_architecture(tmp_path):
    (tmp_path / "docs" / "model-coverage").mkdir(parents=True)
    (tmp_path / "nemo_automodel" / "_transformers").mkdir(parents=True)
    (tmp_path / "examples").mkdir()

    release_rows = []
    for index in range(9):
        recipe_path = f"examples/model_{index}.yaml"
        (tmp_path / recipe_path).write_text("model: test\n", encoding="utf-8")
        release_rows.append(
            f"| 2026-07-{30 - index:02d} | Model {index} | "
            f"[`org/model-{index}`](https://huggingface.co/org/model-{index}) | LLM | "
            f"[model_{index}.yaml](https://github.com/NVIDIA-NeMo/Automodel/blob/main/{recipe_path}) | 🚧 |"
        )
    (tmp_path / "docs" / "model-coverage" / "latest-models.mdx").write_text(
        "\n".join(release_rows) + "\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "index.mdx").write_text(
        f"before\n{HOMEPAGE_START_MARKER}\nstale\n{HOMEPAGE_END_MARKER}\nafter\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "model-coverage" / "overview.mdx").write_text(
        f"before\n{REGISTRY_START_MARKER}\nstale\n{REGISTRY_END_MARKER}\nafter\n", encoding="utf-8"
    )
    (tmp_path / "nemo_automodel" / "_transformers" / "registry.py").write_text(
        'MODEL_ARCH_MAPPING = OrderedDict([("NewModel", ("models.new", "NewModel"))])\n', encoding="utf-8"
    )

    changed_paths = _sync_tables(tmp_path, check=False)

    assert changed_paths == [tmp_path / "docs" / "index.mdx", tmp_path / "docs" / "model-coverage" / "overview.mdx"]
    assert "[Model 0](https://huggingface.co/org/model-0)" in (tmp_path / "docs" / "index.mdx").read_text(
        encoding="utf-8"
    )
    assert "| `NewModel` | `models.new.NewModel` |" in (
        tmp_path / "docs" / "model-coverage" / "overview.mdx"
    ).read_text(encoding="utf-8")
