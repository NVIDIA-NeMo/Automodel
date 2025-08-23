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

import pytest
from fsspec.implementations.local import LocalFileSystem

from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import (
    _write_sub_tensor_to_file_optimized,
)


@pytest.mark.run_only_on("CPU")
def test_write_scalar_tensor(tmp_path):
    """Ensure that a 0-dim (scalar) tensor shard is written to the output file.

    Regression test for a bug where `_write_sub_tensor_to_file_optimized` used to
    early-return on ``tensor_shape == []`` and therefore omitted scalar payloads,
    which produced corrupt `.safetensors` files (incomplete metadata).
    """

    fs = LocalFileSystem()

    # Prepare an empty temporary output file
    output_file = tmp_path / "scalar_tensor.bin"
    output_file.write_bytes(b"")  # create the file

    # Fake scalar tensor payload (2-byte BF16 value)
    sub_tensor_bytes = b"\x34\x12"
    element_size = len(sub_tensor_bytes)  # 2 bytes for BF16

    # Call the routine under test: scalar has empty shapes and offsets
    _write_sub_tensor_to_file_optimized(
        fs,
        sub_tensor_bytes,
        element_size,
        tensor_shape=[],  # scalar
        sub_tensor_offsets=[],
        sub_tensor_shape=[],
        output_file_path=str(output_file),
        output_start_byte=0,
    )

    # The file must now contain exactly the scalar payload
    written = output_file.read_bytes()
    assert written == sub_tensor_bytes
    assert os.path.getsize(output_file) == element_size
