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
import tempfile
import sys
import glob

def test_preprocess_megatron_dataset():
    print("DEBUG matched:", glob.glob("/home/TestData/adasif/mcore_dataset_fineweb/fineweb_sample.val.part_*.jsonl"))
    print("DEBUG exists(TestData):", os.path.exists("/home/TestData"))
    print("DEBUG exists(mcore_dataset_fineweb):", os.path.exists("/home/TestData/adasif/mcore_dataset_fineweb/"))
    jsonl_files_path = "/home/TestData/adasif/mcore_dataset_fineweb/fineweb_sample.val.part_00.jsonl"
    files_to_exist = [
        "preprocessed_data_0_text_document.bin",
        "preprocessed_data_0_text_document.idx",
        # "preprocessed_data_1_text_document.bin",
        # "preprocessed_data_1_text_document.idx",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use a nested directory that doesn't exist yet to validate creation
        output_path = os.path.join(tmpdir, "nested", "out")
        args = [
            sys.executable,
            "tools/preprocess_megatron_dataset.py",
            "--input",
            jsonl_files_path,
            "--json-keys",
            "text",
            "--output-prefix",
            "preprocessed_data",
            "--output-path",
            output_path,
            "--workers", "2",
            "--pretrained-model-name-or-path",
            "/home/TestData/akoumparouli/hf_mixtral_2l/",
            "--append-eod",
        ]
        subprocess.run(args, check=True)
        assert os.path.isdir(output_path), f"Expected output directory {output_path} to be created"
        for file in files_to_exist:
            file_path = os.path.join(output_path, file)
            assert os.path.exists(file_path), f"Expected {file_path} to exist"
            assert os.path.isfile(file_path), f"Expected {file_path} to be a file"
            assert os.access(file_path, os.R_OK), f"Expected {file_path} to be readable"
            assert os.stat(file_path).st_size > 0, f"Expected {file_path} to be non-empty"
