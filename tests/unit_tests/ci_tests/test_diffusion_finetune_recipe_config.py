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

"""Exercise the Bash recipe dispatcher and launcher command boundary.

Regression failures: the single-GPU LoRA recipe inherits the full-training
pixel budget; its configured process count is ignored by training or cache
preparation; or an unset/empty process override changes existing full-training
defaults. The launcher test records its actual uv arguments and stops before
executing training.
"""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_SCRIPT = REPO_ROOT / "tests/ci_tests/scripts/diffusion_finetune_recipe_config.sh"
LAUNCHER_SCRIPT = REPO_ROOT / "tests/ci_tests/scripts/diffusion_finetune_launcher.sh"


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


@pytest.mark.parametrize(
    ("recipe_name", "resolution_option", "resolution_value"),
    [
        ("qwen_image_edit_2511_flow", "--max_pixels", "1048576"),
        ("qwen_image_edit_2511_flow_lora", "--resolution_preset", "256p"),
    ],
)
def test_qwen_edit_cache_matches_recipe_resolution(
    recipe_name: str, resolution_option: str, resolution_value: str
) -> None:
    result = resolve_recipe(recipe_name)

    assert result.returncode == 0, result.stderr
    media_type, processor, _, model_name, _, preprocess = result.stdout.split("|")
    assert (media_type, processor, model_name) == (
        "image_edit",
        "qwen_image_edit",
        "Qwen/Qwen-Image-Edit-2511",
    )
    args = shlex.split(preprocess)
    assert args[args.index("--dataset_name") + 1] == "osunlp/MagicBrush"
    assert resolution_option in args, f"wrong cache resolution: {args}"
    assert args[args.index(resolution_option) + 1] == resolution_value
    other_option = "--max_pixels" if resolution_option == "--resolution_preset" else "--resolution_preset"
    assert other_option not in args, "resolution options are mutually exclusive"


@pytest.mark.parametrize(
    ("recipe_name", "process_override", "expected_processes", "strategy"),
    [
        ("qwen_image_edit_2511_flow", None, 8, "fsdp"),
        ("qwen_image_edit_2511_flow", "", 8, "fsdp"),
        ("qwen_image_edit_2511_flow_lora", "1", 1, "ddp"),
    ],
)
def test_diffusion_launcher_uses_recipe_process_count(
    tmp_path: Path, recipe_name: str, process_override: str | None, expected_processes: int, strategy: str
) -> None:
    """Run the real launcher with only the container mount and uv boundary replaced."""
    config_path = Path("examples/diffusion/finetune") / f"{recipe_name}.yaml"
    fixture_config = tmp_path / config_path
    fixture_config.parent.mkdir(parents=True)
    fixture_config.write_text(f"{strategy}: {{}}\n")
    trace = tmp_path / "uv-arguments.bin"
    command = r"""
cd() {
    if [[ "$1" == /opt/Automodel ]]; then
        builtin cd "$CI_TEST_REPO"
    else
        builtin cd "$@"
    fi
}
grep() {
    local args=("$@")
    local index
    for index in "${!args[@]}"; do
        args[$index]="${args[$index]/#\/opt\/Automodel/$CI_TEST_REPO}"
    done
    command grep "${args[@]}"
}
uv() {
    printf '%s\0' "$@" >> "$CI_TEST_TRACE"
    printf '\n' >> "$CI_TEST_TRACE"
    if [[ " $* " == *" torchrun "* ]]; then
        return 73
    fi
}
source "$1"
"""
    env = {
        **os.environ,
        "CI_TEST_REPO": str(tmp_path),
        "CI_TEST_TRACE": str(trace),
        "CONFIG_PATH": str(config_path),
        "NPROC_PER_NODE": "8",
        "TEST_NODE_COUNT": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
        "SLURM_JOB_ID": "test",
        "PIPELINE_DIR": str(tmp_path / "pipeline"),
        "TEST_NAME": "qwen-lora-launcher",
    }
    env.pop("CONFIG_NPROC_PER_NODE", None)
    if process_override is not None:
        env["CONFIG_NPROC_PER_NODE"] = process_override
    result = subprocess.run(
        ["bash", "-c", command, "bash", str(LAUNCHER_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 73, result.stdout + result.stderr
    calls = [line.rstrip("\0").split("\0") for line in trace.read_text().splitlines()]
    train_args = next(args for args in calls if "torchrun" in args)
    assert f"--nproc-per-node={expected_processes}" in train_args
    preprocess_args = next(args for args in calls if "tools.diffusion.preprocessing_multiprocess" in args)
    assert "--num_gpus" in preprocess_args, "cache preparation must respect the configured GPU count"
    assert preprocess_args[preprocess_args.index("--num_gpus") + 1] == str(expected_processes)
    if strategy == "ddp":
        assert "--fsdp.dp_size" not in train_args
    else:
        assert train_args[train_args.index("--fsdp.dp_size") + 1] == str(expected_processes)
