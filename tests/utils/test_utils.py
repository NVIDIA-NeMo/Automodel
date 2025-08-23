# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
import subprocess

def run_test_script(folder, test_filename):
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    test_file_path = os.path.join(dir_path, 'functional_tests', folder, test_filename)
    test_cmd = ["bash", test_file_path]
    subprocess.run(test_cmd, check=True)
