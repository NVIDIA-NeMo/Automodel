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

"""Regression coverage for architecture documentation validators."""

from pathlib import Path

import pytest

from tools.docs.checks.architecture import (
    validate_architecture_aliases,
    validate_architecture_reference_tables,
    validate_registered_architectures,
)


def test_every_model_card_has_architecture_reference_table():
    validate_architecture_reference_tables(Path(__file__).resolve().parents[5])


def test_every_registered_arch_has_model_coverage_doc():
    validate_registered_architectures(Path(__file__).resolve().parents[5])


def test_doc_arch_aliases_target_strings_appear_in_docs():
    validate_architecture_aliases(Path(__file__).resolve().parents[5])


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("", "missing '## Model Reference'"),
        ("## Model Reference\n", "missing '### Model Architecture'"),
        ("## Model Reference\n### Model Architecture\n", "missing architecture Property/Value table"),
        (
            "## Model Reference\n### Model Architecture\n| Property | Value |\n### Model Summary\n",
            "legacy '### Model Summary'",
        ),
        ("## Model Reference\n### Model Architecture\n| Property | Value |\né", "non-ASCII"),
    ],
)
def test_architecture_reference_validator_reports_bad_cards(tmp_path: Path, body: str, message: str):
    card = tmp_path / "docs/model-coverage/llm/org/model.mdx"
    card.parent.mkdir(parents=True)
    card.write_text("---\nslug: model-coverage/large-language-models/org/model\n---\n" + body)

    with pytest.raises(ValueError, match=message):
        validate_architecture_reference_tables(tmp_path)


def test_architecture_reference_validator_accepts_the_shared_contract(tmp_path: Path):
    card = tmp_path / "docs/model-coverage/llm/org/model.mdx"
    card.parent.mkdir(parents=True)
    card.write_text(
        "---\nslug: model-coverage/large-language-models/org/model\n---\n"
        "## Model Reference\n### Model Architecture\n| Property | Value |\n|---|---|\n"
    )

    validate_architecture_reference_tables(tmp_path)
