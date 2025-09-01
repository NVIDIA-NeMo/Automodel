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

import subprocess
import os


def test_preprocess_megatron_dataset():
    jsonl_files_path = "/home/TestData/adasif/mcore_dataset_fineweb/fineweb_sample.val.part_*.jsonl"
    files_to_exist = [
        "preprocessed_data_0_text_document.bin",
        "preprocessed_data_0_text_document.idx",
        "preprocessed_data_1_text_document.bin",
        "preprocessed_data_1_text_document.idx",
    ]
    args = [
        "python",
        "tools/preprocess_megatron_dataset.py",
        "--input",
        jsonl_files_path,
        "--json-keys",
        "text",
        "--output-prefix",
        "preprocessed_data", 
        "--workers", "2", 
        "--pretrained-model-name-or-path", "openai-community/gpt2", 
        "--append-eod"
    ]
    subprocess.run(args, check=True)
    for file in files_to_exist:
        file_path = os.path.join(os.getcwd(), file)
        assert os.path.exists(file_path), f"Expected {file_path} to exist"
        assert os.path.isfile(file_path), f"Expected {file_path} to be a file"
        assert os.access(file_path, os.R_OK), f"Expected {file_path} to be readable"
        assert os.stat(file_path).st_size > 0, f"Expected {file_path} to be non-empty"
        os.remove(file_path)
