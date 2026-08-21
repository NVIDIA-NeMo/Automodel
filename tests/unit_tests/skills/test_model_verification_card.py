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

import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).parents[3]
VALIDATOR = REPO_ROOT / "skills" / "nemo-automodel-model-verification-card" / "scripts" / "validate_card.py"
QWEN_CARD = REPO_ROOT / "examples" / "llm_benchmark" / "qwen" / "qwen3_moe_30b_a3b_verification_card.yaml"


def _run_validator(card: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(card)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_qwen3_model_verification_card_is_valid():
    result = _run_validator(QWEN_CARD)

    assert result.returncode == 0, result.stderr


def test_validator_rejects_index_and_item_status_mismatch(tmp_path):
    card = yaml.safe_load(QWEN_CARD.read_text(encoding="utf-8"))
    card["items"]["pretrain"]["H100"]["status"] = "not_verified"
    invalid_card = tmp_path / "qwen3_invalid_verification_card.yaml"
    invalid_card.write_text(yaml.safe_dump(card, sort_keys=False), encoding="utf-8")

    result = _run_validator(invalid_card)

    assert result.returncode == 1
    assert "index records `verified`" in result.stderr


def test_validator_requires_128k_coderforge_long_context_contract(tmp_path):
    card = yaml.safe_load(QWEN_CARD.read_text(encoding="utf-8"))
    contract = card["items"]["sft_long_context"]["H100"]["verification_contract"]
    contract["sequence_length"] = 65536
    invalid_card = tmp_path / "qwen3_invalid_verification_card.yaml"
    invalid_card.write_text(yaml.safe_dump(card, sort_keys=False), encoding="utf-8")

    result = _run_validator(invalid_card)

    assert result.returncode == 1
    assert "sequence_length must be 131072" in result.stderr
