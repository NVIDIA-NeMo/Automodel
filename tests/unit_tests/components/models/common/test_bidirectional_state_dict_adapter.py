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

import pytest
import torch

from nemo_automodel.components.models.common.bidirectional import EncoderStateDictAdapter


class TestEncoderStateDictAdapter:
    @pytest.fixture
    def adapter(self):
        return EncoderStateDictAdapter()

    def test_init(self, adapter):
        assert adapter._uses_model_prefix is True

    def test_to_hf_strips_model_prefix(self, adapter):
        encoder_state_dict = {
            "model.layer1.weight": torch.randn(10, 10),
            "model.layer2.bias": torch.randn(10),
        }

        hf_state_dict = adapter.to_hf(encoder_state_dict)

        assert "layer1.weight" in hf_state_dict
        assert "layer2.bias" in hf_state_dict
        assert len(hf_state_dict) == 2
        assert torch.equal(hf_state_dict["layer1.weight"], encoder_state_dict["model.layer1.weight"])
        assert torch.equal(hf_state_dict["layer2.bias"], encoder_state_dict["model.layer2.bias"])

    def test_to_hf_empty_state_dict(self, adapter):
        assert adapter.to_hf({}) == {}

    def test_to_hf_drops_non_model_keys(self, adapter):
        hf_state_dict = adapter.to_hf({"other.layer.weight": torch.randn(10, 10)})
        assert hf_state_dict == {}

    def test_from_hf_adds_model_prefix(self, adapter):
        hf_state_dict = {
            "layer1.weight": torch.randn(10, 10),
            "layer2.bias": torch.randn(10),
        }

        encoder_state_dict = adapter.from_hf(hf_state_dict)

        assert "model.layer1.weight" in encoder_state_dict
        assert "model.layer2.bias" in encoder_state_dict
        assert len(encoder_state_dict) == 2
        assert torch.equal(encoder_state_dict["model.layer1.weight"], hf_state_dict["layer1.weight"])
        assert torch.equal(encoder_state_dict["model.layer2.bias"], hf_state_dict["layer2.bias"])

    def test_from_hf_empty_state_dict(self, adapter):
        assert adapter.from_hf({}) == {}

    def test_convert_single_tensor_to_hf_strips_prefix(self, adapter):
        tensor = torch.randn(10, 10)
        result = adapter.convert_single_tensor_to_hf("model.layer1.weight", tensor)

        assert len(result) == 1
        assert result[0][0] == "layer1.weight"
        assert torch.equal(result[0][1], tensor)

    def test_convert_single_tensor_to_hf_no_prefix_returns_empty(self, adapter):
        tensor = torch.randn(10, 10)
        assert adapter.convert_single_tensor_to_hf("other.layer.weight", tensor) == []

    def test_roundtrip_preserves_keys_and_values(self, adapter):
        original_hf_state = {
            "embedding.weight": torch.randn(100, 768),
            "layer1.weight": torch.randn(768, 768),
            "output.bias": torch.randn(768),
        }

        recovered_hf_state = adapter.to_hf(adapter.from_hf(original_hf_state))

        assert set(recovered_hf_state.keys()) == set(original_hf_state.keys())
        for key in original_hf_state:
            assert torch.equal(recovered_hf_state[key], original_hf_state[key])

    def test_to_hf_peft_keys(self, adapter):
        tensor = torch.randn(10, 10)
        hf_state_dict = adapter.to_hf({"base_model.model.model.layer1.weight": tensor})

        assert len(hf_state_dict) == 1
        assert "base_model.model.layer1.weight" in hf_state_dict
        assert torch.equal(hf_state_dict["base_model.model.layer1.weight"], tensor)

    def test_from_hf_peft_keys(self, adapter):
        tensor = torch.randn(10, 10)
        encoder_state_dict = adapter.from_hf({"base_model.model.layer1.weight": tensor})

        assert len(encoder_state_dict) == 1
        assert "base_model.model.model.layer1.weight" in encoder_state_dict
        assert torch.equal(encoder_state_dict["base_model.model.model.layer1.weight"], tensor)

    def test_roundtrip_peft(self, adapter):
        original_hf_state = {
            "base_model.model.layer1.weight": torch.randn(10, 10),
        }

        recovered_hf_state = adapter.to_hf(adapter.from_hf(original_hf_state))

        assert set(recovered_hf_state.keys()) == set(original_hf_state.keys())
        for key in original_hf_state:
            assert torch.equal(recovered_hf_state[key], original_hf_state[key])
