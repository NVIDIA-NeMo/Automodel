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

import json

import pytest

from tests.ci_tests.utils.sync_model_coverage_tables import (
    HOMEPAGE_END_MARKER,
    HOMEPAGE_START_MARKER,
    REGISTRY_END_MARKER,
    REGISTRY_START_MARKER,
    RELEASE_LOG_END_MARKER,
    RELEASE_LOG_START_MARKER,
    _load_model_releases,
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


def test_sync_tables_writes_release_log_homepage_and_registry(tmp_path):
    (tmp_path / "docs" / "model-coverage").mkdir(parents=True)
    (tmp_path / "nemo_automodel" / "_transformers").mkdir(parents=True)
    (tmp_path / "examples").mkdir()

    releases = [
        {
            "date": "2026-07-31",
            "model": "Documentation Model",
            "hf_model_id": "org/documentation-model",
            "modality": "VLM",
            "recipe": None,
            "brev_status": "available",
            "brev_url": "https://brev.nvidia.com/launchable/deploy/now?launchableID=test",
        }
    ]
    for index in range(9):
        recipe_path = f"examples/model_{index}.yaml"
        (tmp_path / recipe_path).write_text("model: test\n", encoding="utf-8")
        releases.append(
            {
                "date": f"2026-07-{30 - index:02d}",
                "model": f"Model {index}",
                "hf_model_id": f"org/model-{index}",
                "modality": "LLM",
                "recipe": recipe_path,
                "brev_status": "planned",
            }
        )
    (tmp_path / "docs" / "model-coverage" / "model-releases.json").write_text(json.dumps(releases), encoding="utf-8")
    (tmp_path / "docs" / "model-coverage" / "latest-models.mdx").write_text(
        f"before\n{RELEASE_LOG_START_MARKER}\nstale\n{RELEASE_LOG_END_MARKER}\nafter\n", encoding="utf-8"
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

    assert changed_paths == [
        tmp_path / "docs" / "model-coverage" / "latest-models.mdx",
        tmp_path / "docs" / "index.mdx",
        tmp_path / "docs" / "model-coverage" / "overview.mdx",
    ]
    release_log = (tmp_path / "docs" / "model-coverage" / "latest-models.mdx").read_text(encoding="utf-8")
    assert (
        "| 2026-07-31 | VLM | [`org/documentation-model`](https://huggingface.co/org/documentation-model) | "
        "Documentation only |"
    ) in release_log
    assert "Documentation Model" not in release_log
    assert "Try on Brev" not in release_log
    assert "brev.nvidia.com" not in release_log
    assert "[Model 0](https://huggingface.co/org/model-0)" in (tmp_path / "docs" / "index.mdx").read_text(
        encoding="utf-8"
    )
    assert "Documentation Model" not in (tmp_path / "docs" / "index.mdx").read_text(encoding="utf-8")
    assert "| `NewModel` | `models.new.NewModel` |" in (
        tmp_path / "docs" / "model-coverage" / "overview.mdx"
    ).read_text(encoding="utf-8")
    assert _sync_tables(tmp_path, check=True) == []


def test_model_release_catalog_rejects_non_chronological_entries(tmp_path):
    catalog = [
        {
            "date": "2026-07-29",
            "model": "Older Model",
            "hf_model_id": "org/older-model",
            "modality": "LLM",
            "recipe": None,
            "brev_status": "unavailable",
        },
        {
            "date": "2026-07-30",
            "model": "Newer Model",
            "hf_model_id": "org/newer-model",
            "modality": "LLM",
            "recipe": None,
            "brev_status": "planned",
        },
    ]
    catalog_path = tmp_path / "model-releases.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(ValueError, match="not reverse chronological"):
        _load_model_releases(catalog_path, tmp_path)


def test_model_release_catalog_requires_iso_date_format(tmp_path):
    catalog = [
        {
            "date": "20260730",
            "model": "Model",
            "hf_model_id": "org/model",
            "modality": "LLM",
            "recipe": None,
            "brev_status": "planned",
        }
    ]
    catalog_path = tmp_path / "model-releases.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        _load_model_releases(catalog_path, tmp_path)


def test_sync_tables_check_rejects_stale_generated_release_log(tmp_path):
    (tmp_path / "docs" / "model-coverage").mkdir(parents=True)
    (tmp_path / "nemo_automodel" / "_transformers").mkdir(parents=True)
    (tmp_path / "examples").mkdir()
    releases = []
    for index in range(9):
        recipe_path = f"examples/model_{index}.yaml"
        (tmp_path / recipe_path).write_text("model: test\n", encoding="utf-8")
        releases.append(
            {
                "date": f"2026-07-{30 - index:02d}",
                "model": f"Model {index}",
                "hf_model_id": f"org/model-{index}",
                "modality": "LLM",
                "recipe": recipe_path,
                "brev_status": "planned",
            }
        )
    (tmp_path / "docs" / "model-coverage" / "model-releases.json").write_text(json.dumps(releases), encoding="utf-8")
    (tmp_path / "docs" / "model-coverage" / "latest-models.mdx").write_text(
        f"{RELEASE_LOG_START_MARKER}\nstale\n{RELEASE_LOG_END_MARKER}\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "index.mdx").write_text(
        f"{HOMEPAGE_START_MARKER}\nstale\n{HOMEPAGE_END_MARKER}\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "model-coverage" / "overview.mdx").write_text(
        f"{REGISTRY_START_MARKER}\nstale\n{REGISTRY_END_MARKER}\n", encoding="utf-8"
    )
    (tmp_path / "nemo_automodel" / "_transformers" / "registry.py").write_text(
        'MODEL_ARCH_MAPPING = OrderedDict([("NewModel", ("models.new", "NewModel"))])\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="latest-models.mdx"):
        _sync_tables(tmp_path, check=True)
