# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Real GPU-boundary coverage for the checkpoint harness's HF adapter reload."""

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from accelerate import dispatch_model
from peft import LoraConfig, PeftModel, get_peft_model

from tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm import (
    _assert_peft_adapter_matches_checkpoint,
    _peft_adapter_load_kwargs,
    _redispatch_peft_model,
)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Requires two CUDA devices to exercise an adapter GPU boundary"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("split_devices", [False, True])
def test_hf_peft_reload_preserves_outputs_across_device_map(
    tmp_path: Path, dtype: torch.dtype, split_devices: bool
) -> None:
    torch.manual_seed(42)
    base = torch.nn.Sequential(
        OrderedDict((name, torch.nn.Linear(16, 16, bias=False, dtype=dtype)) for name in ("first", "last"))
    )
    reload_base = deepcopy(base)
    source = get_peft_model(
        base,
        LoraConfig(target_modules=["first", "last"], r=4, lora_alpha=8, init_lora_weights=False),
    ).eval()
    source.save_pretrained(tmp_path, save_embedding_layers=False)
    inputs = torch.randn(2, 5, 16, device="cuda:0", dtype=dtype)
    source.to("cuda:0")
    with torch.no_grad():
        reference = source(inputs).cpu()

    device_map = {"first": 0, "last": 1 if split_devices else 0}
    dispatch_model(reload_base, device_map=device_map)
    loaded = PeftModel.from_pretrained(reload_base, tmp_path, **_peft_adapter_load_kwargs({}))
    _redispatch_peft_model(loaded)

    assert loaded.get_base_model().hf_device_map == device_map
    assert loaded.get_base_model().last.lora_A["default"].weight.device.index == device_map["last"]
    assert _assert_peft_adapter_matches_checkpoint(loaded, tmp_path / "adapter_model.safetensors") == (4, 0)
    with torch.no_grad():
        actual = loaded(inputs).cpu()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
