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

# List of CLI overrides forwarded by the functional-test shell scripts.
# Registering them with pytest prevents the test discovery phase from
# aborting with "file or directory not found: --<option>" errors.
_OVERRIDES = [
    "config",
    "model.pretrained_model_name_or_path",
    "step_scheduler.max_steps",
    "step_scheduler.grad_acc_steps",
    "dataset.tokenizer.pretrained_model_name_or_path",
    "validation_dataset.tokenizer.pretrained_model_name_or_path",
    "dataset.dataset_name",
    "validation_dataset.dataset_name",
    "dataset.limit_dataset_samples",
    "step_scheduler.ckpt_every_steps",
    "checkpoint.enabled",
    "checkpoint.checkpoint_dir",
    "checkpoint.model_save_format",
    "dataloader.batch_size",
    "checkpoint.save_consolidated",
    "peft.peft_fn",
    "peft.match_all_linear",
    "peft.dim",
    "peft.alpha",
    "peft.use_triton",
    "peft._target_",
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
