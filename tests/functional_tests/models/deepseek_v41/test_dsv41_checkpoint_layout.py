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

"""Check the state-dict adapter against the released DeepSeek-V4.1-Flash checkpoint layout.

Builds the full model on the meta device from the real ``config.json``, converts
its state dict to the on-disk format with quantization placeholders and compares
every key, shape and dtype with the safetensors headers.  Only the headers are
read, so a checkpoint directory that holds just ``config.json`` plus either the
``*.safetensors`` files or a ``headers/*.json`` dump of their headers works:

    DSV41_CHECKPOINT_DIR=/path/to/DeepSeek-V4.1-Flash \\
    pytest tests/functional_tests/models/deepseek_v41/test_dsv41_checkpoint_layout.py -s
"""

from __future__ import annotations

import glob
import json
import os
import struct

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM

CHECKPOINT_DIR = os.environ.get("DSV41_CHECKPOINT_DIR")

pytestmark = pytest.mark.skipif(not CHECKPOINT_DIR, reason="set DSV41_CHECKPOINT_DIR to the released checkpoint")

_TORCH_TO_SAFETENSORS = {
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float8_e8m0fnu: "F8_E8M0",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.int8: "I8",
}
_DROPPED_PREFIXES = ("mtp.", "vision.", "aligner.", "image_")
_QUANTIZED_DTYPES = {"F8_E4M3", "F8_E8M0", "I8"}


def _read_headers(checkpoint_dir: str) -> dict[str, tuple[str, tuple[int, ...]]]:
    headers: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))):
        with open(path, "rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            headers.update(json.loads(handle.read(length)))
    if not headers:
        for path in sorted(glob.glob(os.path.join(checkpoint_dir, "headers", "*.json"))):
            with open(path) as handle:
                headers.update(json.load(handle))
    headers.pop("__metadata__", None)
    assert headers, f"no safetensors headers found under {checkpoint_dir}"
    return {key: (value["dtype"], tuple(value["shape"])) for key, value in headers.items()}


def test_adapter_matches_released_checkpoint_layout():
    on_disk = _read_headers(CHECKPOINT_DIR)
    config = DeepseekV41Config.from_pretrained(CHECKPOINT_DIR)
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch_fp32", enable_hf_state_dict_adapter=True)
    with torch.device("meta"):
        model = DeepseekV41ForCausalLM(config, backend=backend)
    cast_model_to_dtype(model, torch.bfloat16)
    hf_state = model.state_dict_adapter.to_hf(model.state_dict(), quantization=True)

    expected = {
        key: spec
        for key, spec in on_disk.items()
        if not key.startswith(_DROPPED_PREFIXES) and not key.endswith("ffn.gate.bias_vl")
    }
    missing = sorted(set(expected) - set(hf_state))
    unexpected = sorted(set(hf_state) - set(expected))
    assert not missing, f"{len(missing)} checkpoint tensors have no model counterpart, e.g. {missing[:10]}"
    assert not unexpected, f"{len(unexpected)} model tensors are absent from the checkpoint, e.g. {unexpected[:10]}"

    shape_mismatches = []
    dtype_mismatches = []
    for key, (dtype, shape) in expected.items():
        tensor = hf_state[key]
        if tuple(tensor.shape) != shape:
            shape_mismatches.append((key, tuple(tensor.shape), shape))
        model_dtype = _TORCH_TO_SAFETENSORS.get(tensor.dtype, str(tensor.dtype))
        if model_dtype != dtype and (dtype in _QUANTIZED_DTYPES or model_dtype in _QUANTIZED_DTYPES):
            dtype_mismatches.append((key, model_dtype, dtype))
    assert not shape_mismatches, f"{len(shape_mismatches)} shape mismatches, e.g. {shape_mismatches[:10]}"
    assert not dtype_mismatches, f"{len(dtype_mismatches)} quantized-dtype mismatches, e.g. {dtype_mismatches[:10]}"
    print(f"checked {len(expected)} tensors against {CHECKPOINT_DIR}")
