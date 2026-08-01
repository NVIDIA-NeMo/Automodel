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

#!/usr/bin/env python3

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as DQ

yaml = YAML()
yaml.default_flow_style = False
yaml.preserve_quotes = True


def slurm_time_multiplier(time: str, multiplier: int):
    """
    Multiply the input time by multiplier and format in slurm time format

    Args:
        time: Slurm formatted string %H:%M:%S
        multiplier: Integer to multiply the time by

    Returns:
        updated_time: Multiplied time in slurm format
    """
    # Parse as HH:MM:SS
    t = datetime.strptime(time, "%H:%M:%S")
    delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)

    # Multiply it
    double_delta = multiplier * delta

    # Back to HH:MM:SS string
    total_seconds = int(double_delta.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    updated_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return updated_time


# Scopes that auto-discover all configs via rglob (no recipe list needed).
# Each entry maps test_folder -> subpath under examples/ (defaults to test_folder
# when the two diverge, e.g. diffusion_finetune lives at examples/diffusion/finetune).
# All other scope/folder combinations read from {scope}_recipes.yml.
AUTO_DISCOVER_SCOPES = {
    "release": {
        "llm_finetune": "llm_finetune",
        "vlm_finetune": "vlm_finetune",
        "diffusion_finetune": "diffusion/finetune",
    },
    "performance": {
        "llm_benchmark": "llm_benchmark",
        "vlm_benchmark": "vlm_benchmark",
    },
}


def _discover_via_glob(automodel_dir: str, examples_subpath: str) -> list[Path]:
    """Discover every recipe YAML under examples/<examples_subpath>/."""
    automodel_path = Path(automodel_dir)
    return sorted(
        p.relative_to(automodel_path) for p in (automodel_path / "examples" / examples_subpath).rglob("*.yaml")
    )


def _discover_via_recipe_list(automodel_dir: str, scope: str, test_folder: str) -> list[Path]:
    """Read configs/<test_folder>/<scope>_recipes.yml; return [] if the file is absent."""
    config_path = Path(automodel_dir) / "tests" / "ci_tests" / "configs" / test_folder / f"{scope}_recipes.yml"
    if not config_path.is_file():
        print(f"INFO: no recipe list at {config_path}; generating empty pipeline", file=sys.stderr)
        return []
    with open(config_path, "r", encoding="utf-8") as f:
        test_configs = yaml.load(f)
    examples_dir = test_configs.get("examples_dir", test_folder)
    return [Path(f"examples/{examples_dir}/{c}") for c in test_configs["configs"]]


def detect_yml_configurations(automodel_dir: str, scope: str, test_folder: str) -> list[Path]:
    """
    Detect recipe YAML configurations to include in the CI pipeline.

    Auto-discovery scopes (defined in AUTO_DISCOVER_SCOPES) collect all YAML files
    via rglob. All other scopes read from a scope-specific recipe list.

    Args:
        automodel_dir: Path to the Automodel directory
        scope: Scope of the testing (nightly, release, convergence, performance)
        test_folder: Name of the test folder under Automodel/examples

    Returns:
        yml_configs: List of yaml configs with path format examples/{test_folder}/
    """
    auto_folders = AUTO_DISCOVER_SCOPES.get(scope, {})
    if test_folder in auto_folders:
        return _discover_via_glob(automodel_dir, auto_folders[test_folder])
    return _discover_via_recipe_list(automodel_dir, scope, test_folder)


def _slurm_time_seconds(value: str) -> int:
    """Convert an HH:MM:SS Slurm duration to seconds."""
    try:
        hours, minutes, seconds = (int(part) for part in value.split(":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Slurm duration {value!r}; expected HH:MM:SS") from exc
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"Invalid Slurm duration {value!r}; expected HH:MM:SS")
    return hours * 3600 + minutes * 60 + seconds


def _rolling_release_weight(automodel_dir: str, config: Path) -> int:
    """Estimate node-seconds for a release recipe and its generated variants."""
    with open(Path(automodel_dir) / config, "r", encoding="utf-8") as f:
        ci_config = (yaml.load(f) or {}).get("ci") or {}

    nodes = int(ci_config.get("nodes", 1))
    if nodes <= 0:
        raise ValueError(f"Invalid ci.nodes={nodes!r} in {config}; expected a positive integer")
    weight = nodes * _slurm_time_seconds(str(ci_config.get("time", "00:10:00")))
    if ci_config.get("vllm_deploy") and not ci_config.get("vllm_deploy_known_issue_id"):
        weight += _slurm_time_seconds(str(ci_config.get("vllm_deploy_time", "00:10:00")))
    return weight


def _is_rolling_release_candidate(
    automodel_dir: str,
    config: Path,
    config_override: Dict[str, Any],
) -> bool:
    """Return whether a release-only recipe is runnable before shard assignment."""
    config_path = Path(automodel_dir) / config
    if not config_path.is_file():
        return False

    exempt_models = set(config_override.get("exempt_models") or [])
    exempt_configs = set(config_override.get("exempt_configs") or [])
    if config.parent.name in exempt_models or config.stem in exempt_configs:
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        ci_config = (yaml.load(f) or {}).get("ci") or {}
    return not (ci_config.get("known_issue_id") and not ci_config.get("allow_failure"))


def _select_rolling_release_configs(
    automodel_dir: str,
    test_folder: str,
    nightly_configs: list[Path],
    config_override: Dict[str, Any],
    shard_count: int,
    shard_index: int,
) -> list[Path]:
    """Select one deterministic, cost-balanced shard of release-only recipes."""
    if shard_count <= 0:
        raise ValueError(f"rolling release shard count must be positive, got {shard_count}")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"rolling release shard index must be in [0, {shard_count}), got {shard_index}")

    nightly_set = set(nightly_configs)
    release_configs = detect_yml_configurations(automodel_dir, "release", test_folder)
    candidates = [
        config
        for config in release_configs
        if config not in nightly_set and _is_rolling_release_candidate(automodel_dir, config, config_override)
    ]

    shard_configs: list[list[Path]] = [[] for _ in range(shard_count)]
    shard_weights = [0] * shard_count
    weighted_configs = sorted(
        ((_rolling_release_weight(automodel_dir, config), config) for config in candidates),
        key=lambda item: (-item[0], str(item[1])),
    )
    for weight, config in weighted_configs:
        target_shard = min(
            range(shard_count),
            key=lambda index: (shard_weights[index], len(shard_configs[index]), index),
        )
        shard_configs[target_shard].append(config)
        shard_weights[target_shard] += weight

    selected = sorted(shard_configs[shard_index])
    print(
        f"Selected rolling release shard {shard_index}/{shard_count}: "
        f"{len(selected)} of {len(candidates)} release-only recipes "
        f"({shard_weights[shard_index] / 60:.0f} node-minutes)",
        file=sys.stderr,
    )
    return selected


# Recipe ci: section keys mapped to the CI variables they populate on the base job.
CI_KEY_TO_VAR = {
    "time": "TIME",
    "nodes": "TEST_NODE_COUNT",
    "node_multiplier": "NODE_MULTIPLIER",
    "max_steps": "MAX_STEPS",
    "local_batch_size": "LOCAL_BATCH_SIZE",
    "ep_size": "EP_SIZE",
    "recipe_owner": "RECIPE_OWNER",
    "nproc_per_node": "CONFIG_NPROC_PER_NODE",
    "cluster_tag": "RESERVED_CLUSTER_TAG",
}


def _compute_base_stage(test_folder: str, config: Path, has_robustness: bool) -> str:
    """Pick the GitLab CI stage for a base recipe job."""
    if "benchmark" in test_folder:
        return "performance"
    if test_folder == "llm_pretrain":
        return "pretrain"
    if "benchmark" in config.stem:
        return "benchmark"
    if test_folder == "convergence":
        # Weekly train-then-eval flow; the .convergence_test template runs
        # convergence_tests_launcher.sh (train -> IFEval -> threshold gate).
        return "convergence"
    if test_folder.startswith("diffusion"):
        return "diffusion_peft" if ("lora" in config.stem or "peft" in config.stem) else "diffusion_sft"
    if test_folder.startswith("retrieval"):
        return "retrieval"
    if "peft" in config.stem or "lora" in config.stem:
        return "peft_ckpt_robustness" if has_robustness else "peft"
    return "sft_ckpt_robustness" if has_robustness else "sft"


def _build_job(
    config: Path,
    scope: str,
    *,
    extends: str,
    stage: str,
    extra_vars: Dict[str, Any] | None = None,
    allow_failure: bool = False,
    known_issue_id: str | None = None,
) -> Dict[str, Any]:
    """Build a CI job dict with the keys common to every variant (base, vllm_deploy, eval)."""
    job: Dict[str, Any] = {
        "extends": extends,
        "stage": stage,
        "variables": {
            "CONFIG_PATH": f"{config}",
            "TEST_LEVEL": f"{scope}",
            **(extra_vars or {}),
        },
    }
    if allow_failure:
        job["allow_failure"] = True
    if known_issue_id:
        job["variables"]["KNOWN_ISSUE_ID"] = known_issue_id
    return job


def _enrich_base_job(job: Dict[str, Any], ci_config: Dict[str, Any], scope: str) -> None:
    """Add base-only extras: resource overrides, env_vars, HAS_ROBUSTNESS, convergence time."""
    for ci_key, ci_var in CI_KEY_TO_VAR.items():
        if ci_key not in ci_config:
            continue
        value = ci_config[ci_key]
        if ci_var == "TIME":
            job["variables"][ci_var] = DQ(str(value))
        elif ci_var == "NODE_MULTIPLIER":
            job["variables"][ci_var] = str(value).lower()
        elif ci_var == "RESERVED_CLUSTER_TAG":
            job["variables"][ci_var] = f"/{value}/"
        else:
            job["variables"][ci_var] = value

    for key, value in ci_config.get("env_vars", {}).items():
        job["variables"][key] = str(value)

    job["variables"]["HAS_ROBUSTNESS"] = str(bool(ci_config.get("checkpoint_robustness"))).lower()

    # Convergence tests run for 2 epochs; double the slurm time allocation.
    if scope == "convergence":
        slurm_time = job["variables"].get("TIME", "00:10:00")
        job["variables"]["TIME"] = DQ(slurm_time_multiplier(slurm_time, 2))


def generate_job(
    config: Path,
    config_override: Dict[str, Any],
    scope: str,
    test_folder: str,
    automodel_dir: str,
) -> list[tuple[str, Dict[str, Any]]]:
    """
    Generate every CI job (base + opt-in variants) for a single recipe configuration.

    Args:
        config: Relative path to the recipe YAML configuration
        config_override: Override dictionary with exempt_models, exempt_configs, and known_issue
        scope: Scope test should be configured to (nightly, release, convergence)
        test_folder: Name of the test_folder under Automodel/examples
        automodel_dir: Path to the Automodel directory

    Returns:
        List of (suffix, job_dict) pairs. The base job has suffix "". An empty list
        means the recipe is fully skipped (e.g. ci.known_issue_id without ci.allow_failure).
    """
    with open(f"{automodel_dir}/{config}", "r", encoding="utf-8") as rf:
        ci_config = (yaml.load(rf) or {}).get("ci") or {}

    # allow_failure wins over known_issue_id.
    # known_issue_id alone skips the entire recipe (base + all variants).
    known_issue_id = ci_config.get("known_issue_id")
    recipe_allow_failure = bool(ci_config.get("allow_failure"))
    if known_issue_id and not recipe_allow_failure:
        return []

    has_robustness = bool(ci_config.get("checkpoint_robustness"))
    base_allow_failure = recipe_allow_failure or config.stem in (config_override.get("known_issue") or [])

    base_job = _build_job(
        config,
        scope,
        extends=".llm_benchmark_test" if "benchmark" in config.stem else f".{test_folder}_test",
        stage=_compute_base_stage(test_folder, config, has_robustness),
        allow_failure=base_allow_failure,
        known_issue_id=known_issue_id,
    )
    _enrich_base_job(base_job, ci_config, scope)
    variants: list[tuple[str, Dict[str, Any]]] = [("", base_job)]

    # vLLM deploy variant. `ci.vllm_deploy_known_issue_id` suppresses just this
    # variant (base job still runs) -- use for bugs that only manifest in vllm deploy.
    if ci_config.get("vllm_deploy") and not ci_config.get("vllm_deploy_known_issue_id"):
        vllm_deploy_vars = {}
        if "vllm_deploy_time" in ci_config:
            vllm_deploy_vars["TIME"] = DQ(str(ci_config["vllm_deploy_time"]))
        variants.append(
            (
                "_vllm_deploy",
                _build_job(
                    config,
                    scope,
                    extends=".vllm_deploy_test",
                    stage="peft_vllm_deploy" if "peft" in config.stem else "sft_vllm_deploy",
                    extra_vars=vllm_deploy_vars,
                    allow_failure=recipe_allow_failure,
                    known_issue_id=known_issue_id,
                ),
            )
        )

    # Retrieval eval variants. Bi-encoders opt into embed_eval, cross-encoders
    # into rerank_eval. Both extend the same per-folder eval template.
    for ci_key, suffix in (("embed_eval", "_embed_eval"), ("rerank_eval", "_rerank_eval")):
        if not ci_config.get(ci_key):
            continue
        variants.append(
            (
                suffix,
                _build_job(
                    config,
                    scope,
                    extends=f".{test_folder}_eval_test",
                    stage="retrieval_eval",
                    extra_vars={"FINETUNE_TEST_NAME": f"{config.stem}"},
                    allow_failure=recipe_allow_failure,
                    known_issue_id=known_issue_id,
                ),
            )
        )

    return variants


def generate_pipeline(
    automodel_dir: str,
    scope: str,
    test_folder: str,
    *,
    rolling_release_shards: int = 0,
    rolling_release_shard: int = 0,
) -> Dict[str, Any]:
    """
    Generate a complete CI test pipeline YAML for the given test folder and scope.

    Args:
        automodel_dir: Path to the Automodel directory
        scope: Scope of the testing (nightly, release, convergence)
        test_folder: Name of the test folder under Automodel/examples.
        rolling_release_shards: Number of cost-balanced release-only shards to overlay on nightly.
        rolling_release_shard: Zero-based release-only shard to overlay on nightly.

    Returns:
        pipeline: Dictionary defining the CI test pipeline
    """
    override_path = f"{automodel_dir}/tests/ci_tests/configs/{test_folder}/override_recipes.yml"
    with open(override_path, "r", encoding="utf-8") as f:
        config_override = yaml.load(f) or {}

    if not rolling_release_shards and rolling_release_shard:
        raise ValueError("rolling release shard requires a positive rolling release shard count")
    if rolling_release_shards and scope != "nightly":
        raise ValueError("rolling release shards can only be added to the nightly scope")

    # Empty yml_configs is fine -- some (folder, scope) combos have no recipes.
    yml_configs = detect_yml_configurations(automodel_dir, scope, test_folder)
    if rolling_release_shards:
        rolling_configs = _select_rolling_release_configs(
            automodel_dir,
            test_folder,
            yml_configs,
            config_override,
            rolling_release_shards,
            rolling_release_shard,
        )
        yml_configs = list(dict.fromkeys([*yml_configs, *rolling_configs]))

    exempt_models = set(config_override.get("exempt_models") or [])
    exempt_configs = set(config_override.get("exempt_configs") or [])

    pipeline: Dict[str, Any] = {"include": ["automodel/automodel_ci_template.yml"]}
    job_name_suffix = os.environ.get("AUTOMODEL_CI_JOB_NAME_SUFFIX", "")

    for config in yml_configs:
        # Skip missing recipes so one bad reference doesn't abort the whole pipeline.
        if not (Path(automodel_dir) / config).is_file():
            print(f"WARNING: recipe not found, skipping: {config}", file=sys.stderr)
            continue

        model_name = config.parent.name
        config_name = config.stem
        if model_name in exempt_models or config_name in exempt_configs:
            continue

        for suffix, job in generate_job(config, config_override, scope, test_folder, automodel_dir):
            job["variables"]["MODEL_FAMILY"] = model_name
            pipeline[f"{config_name}{suffix}{job_name_suffix}"] = job

    return pipeline


def _normalize_scope(value: str) -> str:
    """Map the GitLab CI sentinel "NONE" (unset variable) to the nightly default."""
    return "nightly" if value == "NONE" else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--automodel-dir", type=str, required=True, help="Path to Automodel directory")
    parser.add_argument("--scope", type=_normalize_scope, required=True, help="Scope of the tests (nightly, release)")
    parser.add_argument("--test-folder", type=str, required=True, help="Target folder to search")
    parser.add_argument(
        "--rolling-release-shards",
        type=int,
        default=0,
        help="Number of cost-balanced release-only shards to overlay on nightly (0 disables the overlay)",
    )
    parser.add_argument(
        "--rolling-release-shard",
        type=int,
        default=0,
        help="Zero-based release-only shard to overlay on nightly",
    )
    args = parser.parse_args()

    pipeline = generate_pipeline(
        args.automodel_dir,
        args.scope,
        args.test_folder,
        rolling_release_shards=args.rolling_release_shards,
        rolling_release_shard=args.rolling_release_shard,
    )
    with open(f"generated_automodel_{args.test_folder}_tests.yml", "w") as f:
        yaml.dump(pipeline, f)
    job_count = sum(1 for k in pipeline if k != "include")
    print(f"Generated pipeline with {job_count} jobs")


if __name__ == "__main__":
    main()
