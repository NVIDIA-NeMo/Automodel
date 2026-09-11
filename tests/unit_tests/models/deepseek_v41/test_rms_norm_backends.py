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

"""CPU construction/checkpoint contracts; numerical TE execution requires CUDA."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from nemo_automodel.components.models.deepseek_v41.layers import DeepseekV41RMSNorm
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from tests.unit_tests.models.deepseek_v41.test_model import _backend, _tiny_config


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_te_norms_preserve_meta_storage_initialization_and_checkpoint_keys(dtype: torch.dtype, tmp_path: Path) -> None:
    # This checks real optional TE modules on meta/CPU, without invoking CUDA
    # kernels. Environments without TE still run the eager mathematical tests.
    te = pytest.importorskip("transformer_engine.pytorch.module.rmsnorm")
    config = _tiny_config()
    config.text_config.dtype = dtype
    config.text_config.rms_norm_eps = 1e-20
    with torch.device("meta"):
        eager = DeepseekV41ForCausalLM(config, backend=_backend())
        fused = DeepseekV41ForCausalLM(config, backend=replace(_backend(), rms_norm="te"))
    assert all(parameter.is_meta for parameter in fused.parameters())
    norm_names = {"model.norm"}
    for index in range(config.text_config.num_hidden_layers):
        prefix = f"model.layers.{index}"
        norm_names.update(f"{prefix}.{suffix}" for suffix in ("attn_norm", "ffn_norm", "attn.q_norm", "attn.kv_norm"))
        if index in config.text_config.kv_source_layer_ids:
            norm_names.update((f"{prefix}.attn.compressor.norm", f"{prefix}.attn.indexer.k_norm"))
    assert len(norm_names) == 29  # All seven norm sites across Full/Reuse/Reindex layers.
    for name in norm_names:
        baseline, actual = eager.get_submodule(name), fused.get_submodule(name)
        assert isinstance(baseline, DeepseekV41RMSNorm)
        assert isinstance(actual, te.RMSNorm)
        assert actual.weight.dtype == baseline.weight.dtype == dtype
        assert actual.weight.shape == baseline.weight.shape
        assert actual.eps == baseline.eps == 1e-20
        assert actual.zero_centered_gamma is False
        assert actual.weight.requires_grad == baseline.weight.requires_grad == (".indexer." not in name)
    for model in (eager, fused):
        model.to_empty(device="cpu")
        torch.manual_seed(913)
        model.initialize_weights(torch.device("cpu"), dtype=dtype)
        for name in norm_names:
            parameter = model.get_submodule(name).weight
            assert parameter.device.type == "cpu" and parameter.dtype == dtype
            torch.testing.assert_close(parameter, torch.ones_like(parameter), rtol=0, atol=0)
        for index in config.text_config.kv_source_layer_ids:
            compressor = model.model.layers[str(index)].attn.compressor
            assert compressor.wkv.weight.dtype == torch.float32
            if compressor.ratio > 1:
                assert compressor.wgate.weight.dtype == torch.float32
            assert compressor.norm.weight.dtype == dtype
    expected = eager.state_dict()
    actual = fused.state_dict()
    extra_state = {name: value for name, value in actual.items() if name.endswith("._extra_state")}
    assert extra_state.keys() == {f"{name}._extra_state" for name in norm_names}
    assert all(value.dtype == torch.uint8 and value.numel() == 0 for value in extra_state.values())
    assert expected.keys() == actual.keys() - extra_state.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert eager.state_dict_adapter.get_hf_state_dict_keys(expected) == fused.state_dict_adapter.get_hf_state_dict_keys(
        actual
    )
    identities = {name: id(parameter) for name, parameter in fused.named_parameters()}
    with torch.no_grad():
        for name in norm_names:
            parameter = eager.get_submodule(name).weight
            parameter.copy_(torch.linspace(-0.75, 1.25, parameter.numel(), dtype=dtype).reshape_as(parameter))
        for parameter in fused.parameters():
            parameter.fill_(-13.5)
    exported = eager.state_dict_adapter.to_hf(eager.state_dict(), exclude_key_regex=r".*_extra_state.*")
    assert not any("_extra_state" in name for name in exported)
    save_file(exported, tmp_path / "model.safetensors")
    audit = fused.state_dict_adapter.load_from_checkpoint(fused, tmp_path)
    assert set(audit.loaded_keys) == set(exported)
    assert identities == {name: id(parameter) for name, parameter in fused.named_parameters()}
    loaded = fused.state_dict()
    for name, expected in eager.state_dict().items():
        torch.testing.assert_close(loaded[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize("rms_norm", ["torch", "quack"])
def test_unvalidated_norm_backends_are_rejected(rms_norm: str) -> None:
    with torch.device("meta"), pytest.raises(ValueError, match="torch_fp32 or te RMSNorm"):
        DeepseekV41ForCausalLM(_tiny_config(), backend=replace(_backend(), rms_norm=rms_norm))
