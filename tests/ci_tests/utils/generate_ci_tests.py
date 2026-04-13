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


def detect_yml_configurations(automodel_dir: str, scope: str, test_folder: str):
    """
    Detect recipe YAML configurations to include in the CI pipeline. Nightly scope reads from
    a scope-specific recipe list. Release scope collects all YAML configurations within the test_folder.

    Args:
        automodel_dir: Path to the Automodel directory
        scope: Scope of the testing (nightly, release)
        test_folder: Name of the test folder under Automodel/examples

    Returns:
        yml_configs: List of yaml configs with path format examples/{test_folder}/
    """
    yml_configs = []
    search_path = f"{automodel_dir}/examples/{test_folder}"

    # Check scope
    if scope in ("release", "perf"):
        for f in Path(f"{search_path}").rglob("*.yaml"):
            relative_path = f.relative_to(automodel_dir)
            yml_configs.append(relative_path)
    else:
        config_path = f"{automodel_dir}/tests/ci_tests/configs/{test_folder}/{scope}_recipes.yml"
        with open(config_path, "r", encoding="utf-8") as f:
            test_configs = yaml.load(f)
            yml_configs = [Path(f"examples/{test_folder}/{c}") for c in test_configs['configs']]

    return yml_configs


def generate_job(config: str, config_override: Dict[str, Any], scope: str, test_folder: str, automodel_dir: str):
    """
    Generate a CI test job definition for a single recipe configuration.
    Resource requirements (time, nodes, etc.) are read from the recipe's `ci:` section.

    Args:
        config: Relative path to the recipe YAML configuration
        config_override: Override dictionary with exempt_models, exempt_configs, and known_issue
        scope: Scope test should be configured to (nightly, release, convergence)
        test_folder: Name of the test_folder under Automodel/examples
        automodel_dir: Path to the Automodel directory

    Returns:
        job: Dictionary defining a single CI test job
    """
    # Initialize test job
    job = {
        'variables': {
            'CONFIG_PATH': f'{config}',
            'TEST_LEVEL': f'{scope}',
        }
    }

    # Configure test template
    if 'benchmark' in config.stem:
        job['extends'] = '.llm_benchmark_test'
    else:
        job['extends'] = f'.{test_folder}_test'

    # Apply resource overrides (time, nodes, etc.) from the recipe's top-level ci: section
    recipe_path = f"{automodel_dir}/{config}"
    with open(recipe_path, "r", encoding="utf-8") as rf:
        recipe = yaml.load(rf)

    ci_config = recipe.get('ci') or {}
    ci_key_map = {
        'time': 'TIME',
        'nodes': 'TEST_NODE_COUNT',
        'node_multiplier': 'NODE_MULTIPLIER',
        'local_batch_size': 'LOCAL_BATCH_SIZE',
        'recipe_owner': 'RECIPE_OWNER',
    }
    for ci_key, ci_var in ci_key_map.items():
        if ci_key in ci_config:
            value = ci_config[ci_key]
            if ci_var == 'TIME':
                job['variables'][ci_var] = DQ(str(value))
            elif ci_var == 'NODE_MULTIPLIER':
                job['variables'][ci_var] = str(value).lower()
            else:
                job['variables'][ci_var] = value

    has_robustness = bool(ci_config.get('checkpoint_robustness'))
    job['variables']['HAS_ROBUSTNESS'] = str(has_robustness).lower()

    # Configure test stage based on recipe type and robustness config
    if 'benchmark' in test_folder:
        job['stage'] = 'performance'
    elif 'benchmark' in config.stem:
        job['stage'] = 'benchmark'
    elif 'peft' in config.stem:
        job['stage'] = 'peft_ckpt_robustness' if has_robustness else 'peft'
    else:
        job['stage'] = 'sft_ckpt_robustness' if has_robustness else 'sft'

    # Check if config has known issue
    known_issue_config_list = config_override.get('known_issue') or []
    if config.stem in known_issue_config_list:
        job['allow_failure'] = True

    # Double time allocation as tests run for 2 epoch
    if scope == "convergence":
        slurm_time = job['variables'].get('TIME', '00:10:00')
        job['variables']['TIME'] = DQ(slurm_time_multiplier(slurm_time, 2))

    # Generate vLLM deploy job if recipe opts in
    vllm_job = None
    vllm_config = ci_config.get('vllm_deploy', {})
    if vllm_config.get('enabled'):
        vllm_stage = 'peft_vllm_deploy' if 'peft' in config.stem else 'sft_vllm_deploy'
        vllm_job = {
            'extends': '.vllm_deploy_test',
            'stage': vllm_stage,
            'variables': {
                'CONFIG_PATH': f'{config}',
                'TEST_LEVEL': f'{scope}',
            }
        }
        extra_deps = vllm_config.get('extra_deps', [])
        if extra_deps:
            vllm_job['variables']['VLLM_EXTRA_DEPS'] = DQ(' '.join(extra_deps))

    return job, vllm_job


def generate_pipeline(automodel_dir: str, scope: str, test_folder: str):
    """
    Generate a complete CI test pipeline YAML for the given test folder and scope.

    Args:
        automodel_dir: Path to the Automodel directory
        scope: Scope of the testing (nightly, release, convergence)
        test_folder: Name of the test folder under Automodel/examples

    Returns:
        pipeline: Dictionary defining the CI test pipeline
    """

    # Check scope
    if scope == "NONE":
        scope = "nightly"

    override_path = f"{automodel_dir}/tests/ci_tests/configs/{test_folder}/override_recipes.yml"
    with open(override_path, "r", encoding="utf-8") as f:
        config_override = yaml.load(f)
    yml_configs = detect_yml_configurations(automodel_dir, scope, test_folder)

    if not yml_configs:
        raise Exception(f'No yml configurations were found under {automodel_dir}/examples/{test_folder}')

    pipeline = {
        'include': ['automodel/automodel_ci_template.yml']
    }

    for config in yml_configs:
        model_name = config.parent.name
        config_name = config.stem

        # Check if model is in exempt model list
        exempt_models_list = config_override.get('exempt_models') or []
        exempt_configs_list = config_override.get('exempt_configs') or []
        if model_name in exempt_models_list or config_name in exempt_configs_list:
            continue

        job, vllm_job = generate_job(config, config_override, scope, test_folder, automodel_dir)
        pipeline[f'{config_name}'] = job
        if vllm_job:
            pipeline[f'{config_name}_vllm_deploy'] = vllm_job

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--automodel-dir', type=str, required=True, help='Path to Automodel directory')
    parser.add_argument('--scope', type=str, required=True, help='Scope of the tests (nightly, release)')
    parser.add_argument('--test-folder', type=str, required=True, help='Target folder to search')

    args = parser.parse_args()

    pipeline = generate_pipeline(args.automodel_dir, args.scope, args.test_folder)

    if pipeline:
        with open(f'generated_automodel_{args.test_folder}_tests.yml', 'w') as f:
            yaml.dump(pipeline, f)
        print(f"Generated pipeline with {len([k for k in pipeline.keys() if k != 'stages'])} jobs")


if __name__ == '__main__':
    main()
