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
"""Ensure functional-test overrides register and retain sample-limit compatibility."""

import runpy
from pathlib import Path

from _pytest.config.argparsing import Parser


def test_functional_sample_limit_options() -> None:
    hooks = runpy.run_path(str(Path(__file__).resolve().parents[2] / "functional_tests" / "conftest.py"))
    parser = Parser()
    hooks["pytest_addoption"](parser)
    options = parser.parse(
        [
            "--dataset.limit_dataset_samples",
            "3",
            "--dataset.num_samples_limit",
            "3",
            "--validation_dataset.limit_dataset_samples",
            "2",
            "--validation_dataset.num_samples_limit",
            "2",
        ]
    )
    assert options.dataset_limit_dataset_samples == "3"
    assert options.validation_dataset_num_samples_limit == "2"
