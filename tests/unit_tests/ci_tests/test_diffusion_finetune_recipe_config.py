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

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_SCRIPT = REPO_ROOT / "tests/ci_tests/scripts/diffusion_finetune_recipe_config.sh"


def resolve_recipe(recipe_name: str) -> subprocess.CompletedProcess[str]:
    command = """
source "$1"
configure_diffusion_finetune_recipe "$2" || exit $?
printf '%s|%s|%s|%s|%s|%s' \
    "$MEDIA_TYPE" "$PROCESSOR" "$GENERATE_CONFIG" "$MODEL_NAME" \
    "${INFER_NUM_FRAMES:-}" "$PREPROCESS_EXTRA_ARGS"
"""
    return subprocess.run(
        ["bash", "-c", command, "bash", str(CONFIG_SCRIPT), recipe_name],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("recipe_name", ["ltx2_3_t2v_flow", "ltx2_3_t2v_flow_lora"])
def test_ltx2_recipe_configuration(recipe_name: str) -> None:
    result = resolve_recipe(recipe_name)

    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        "video|ltx2|examples/diffusion/generate/configs/generate_ltx2.yaml|"
        "diffusers/LTX-2.3-Diffusers|9|--num_frames 9 --output_format pt"
    )


@pytest.mark.parametrize(
    ("recipe_name", "expected_prefix"),
    [
        ("wan2_1_t2v_flow", "video|wan|examples/diffusion/generate/configs/generate_wan.yaml|"),
        ("hunyuan_t2v_flow", "video|hunyuan|examples/diffusion/generate/configs/generate_hunyuan.yaml|"),
        ("flux_t2i_flow", "image|flux|examples/diffusion/generate/configs/generate_flux.yaml|"),
        (
            "qwen_image_t2i_flow",
            "image|qwen_image|examples/diffusion/generate/configs/generate_qwen_image.yaml|",
        ),
    ],
)
def test_existing_recipe_configuration_is_preserved(recipe_name: str, expected_prefix: str) -> None:
    result = resolve_recipe(recipe_name)

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith(expected_prefix)


def test_unknown_diffusion_recipe_is_rejected() -> None:
    result = resolve_recipe("unknown_t2v_flow")

    assert result.returncode == 1
    assert "Unknown recipe 'unknown_t2v_flow'" in result.stderr
