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

"""Model-owned construction contract for the frozen DSpark target."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nemo_automodel._transformers import NeMoAutoModelForCausalLM
from nemo_automodel.components.models.deepseek_v41.config import (
    DeepseekV41Config,
    DeepseekV41DSparkTargetConfig,
)


@pytest.mark.parametrize("custom_backends", [False, True])
def test_target_build_preserves_loader_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, custom_backends: bool
) -> None:
    # Read a real checkpoint config to cover the text-only override and path.
    DeepseekV41Config().save_pretrained(tmp_path)
    options = DeepseekV41DSparkTargetConfig(str(tmp_path), trust_remote_code=True)
    if custom_backends:
        options.attn_backend = "eager"
        options.dispatcher = "torch"
        options.experts = "torch"
        options.enable_fsdp_optimizations = False
    before = asdict(options)
    captured = {}
    setup = object()

    def _from_config(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(config=kwargs["config"])

    monkeypatch.setattr(NeMoAutoModelForCausalLM, "from_config", _from_config)
    model = options.build(
        device=torch.device("cuda"),
        compute_dtype=torch.bfloat16,
        distributed_setup=setup,
    )
    assert model.config is captured["config"]
    assert model.config.vision_config.num_hidden_layers == 0
    assert model.config.text_config.num_hidden_layers == 40
    assert model.config.name_or_path == str(tmp_path)
    assert captured["load_base_model"] is True
    assert captured["distributed_setup"] is setup
    assert captured["torch_dtype"] == torch.bfloat16
    assert captured["trust_remote_code"] is True
    assert captured["use_liger_kernel"] is False
    assert captured["use_sdpa_patching"] is False
    backend = captured["backend"]
    assert backend.attn == ("eager" if custom_backends else "tilelang")
    assert backend.experts == ("torch" if custom_backends else "torch_mm")
    assert backend.dispatcher == ("torch" if custom_backends else "hybridep")
    assert backend.enable_fsdp_optimizations is not custom_backends
    assert backend.linear == "torch"
    assert backend.rms_norm == "torch_fp32"
    assert backend.rope_fusion is False
    assert backend.gate_precision == torch.float32
    assert backend.enable_hf_state_dict_adapter is True
    assert asdict(options) == before


@pytest.mark.parametrize(
    "device,num_layers,error,match",
    [
        ("cpu", None, RuntimeError, "requires CUDA"),
        ("cuda", 4, ValueError, "target_num_hidden_layers"),
    ],
)
def test_target_build_rejects_unsupported_execution(
    device: str, num_layers: int | None, error: type[Exception], match: str
) -> None:
    options = DeepseekV41DSparkTargetConfig("unused", target_num_hidden_layers=num_layers)
    with pytest.raises(error, match=match):
        options.build(
            device=torch.device(device),
            compute_dtype=torch.bfloat16,
            distributed_setup=object(),
        )
