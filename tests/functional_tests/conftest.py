# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
import pytest

# Ensure the repository root is importable so functional tests can do
# `from tests.utils.test_utils import run_test_script` reliably across
# different pytest import modes / runners.
#
# (Without this, some environments do not include the repo root on sys.path,
# causing `ModuleNotFoundError: No module named 'tests.utils'` during collection.)
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# List of CLI overrides forwarded by the functional-test shell scripts.
# Registering them with pytest prevents the test discovery phase from
# aborting with "file or directory not found: --<option>" errors.
_OVERRIDES = [
    "config",
    "model.pretrained_model_name_or_path",
    "model.config.pretrained_model_name_or_path",
    "step_scheduler.max_steps",
    "step_scheduler.global_batch_size",
    "step_scheduler.local_batch_size",
    "dataset.tokenizer.pretrained_model_name_or_path",
    "validation_dataset.tokenizer.pretrained_model_name_or_path",
    "dataset.dataset_name",
    "dataset.paths",
    "dataset.splits_to_build",
    "dataset.split",
    "dataset.padding",
    "validation_dataset.dataset_name",
    "validation_dataset.padding",
    "dataset.limit_dataset_samples",
    "step_scheduler.ckpt_every_steps",
    "checkpoint.enabled",
    "checkpoint.checkpoint_dir",
    "checkpoint.model_save_format",
    "checkpoint.single_rank_consolidation",
    "checkpoint.staging_dir",
    "dataloader.batch_size",
    "checkpoint.save_consolidated",
    "peft.peft_fn",
    "peft.match_all_linear",
    "peft.dim",
    "peft.alpha",
    "peft.use_triton",
    "peft.use_dora",
    "peft._target_",
    "distributed",
    "distributed.dp_size",
    "distributed.tp_size",
    "distributed.cp_size",
    "distributed.pp_size",
    "distributed.ep_size",
    "distributed.strategy",
    "distributed.sequence_parallel",
    "distributed.activation_checkpointing",
    "dataset._target_",
    "dataset.path_or_dataset",
    "validation_dataset.path_or_dataset",
    "validation_dataset.split",
    "validation_dataset.num_samples_limit",
    "validation_dataset.limit_dataset_samples",
    "distributed.pipeline._target_",
    "distributed.pipeline.pp_schedule",
    "distributed.pipeline.pp_microbatch_size",
    "distributed.pipeline.pp_batch_size",
    "distributed.pipeline.round_virtual_stages_to_pp_multiple",
    "distributed.pipeline.scale_grads_in_schedule",
    "dataset.seq_length",
    "validation_dataset.seq_length",
    "freeze_config.freeze_language_model",
    "qat.fake_quant_after_n_steps",
    "qat.enabled",
    "qat.quantizer._target_",
    "qat.quantizer.groupsize",
]


def pytest_addoption(parser: pytest.Parser):
    """Register the NeMo-Automodel CLI overrides so that pytest accepts them.
    The functional test launchers forward these arguments after a ``--``
    separator.  If pytest is unaware of an option it treats it as a file
    path and aborts collection.  Declaring each option here is enough to
    convince pytest that they are legitimate flags while still keeping
    them intact in ``sys.argv`` for the application code to parse later.
    """
    for opt in _OVERRIDES:
        # ``dest`` must be a valid Python identifier, so replace dots.
        dest = opt.replace(".", "_")
        parser.addoption(f"--{opt}", dest=dest, action="store", help=f"(passthrough) {opt}")
