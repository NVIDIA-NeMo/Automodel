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

from nemo_automodel.components.models.common.bidirectional import BiencoderStateDictAdapter


class TestBiencoderStateDictAdapter:
    """Test suite for BiencoderStateDictAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create a BiencoderStateDictAdapter instance for testing."""
        return BiencoderStateDictAdapter()

    def test_init(self, adapter):
        """Test adapter initialization."""
        assert adapter._uses_model_prefix is True

    def test_to_hf_basic(self, adapter):
        """Test basic conversion from biencoder to HuggingFace format."""
        biencoder_state_dict = {
            "lm_q.layer1.weight": torch.randn(10, 10),
            "lm_q.layer2.bias": torch.randn(10),
            "lm_p.layer1.weight": torch.randn(10, 10),
            "lm_p.layer2.bias": torch.randn(10),
        }

        hf_state_dict = adapter.to_hf(biencoder_state_dict)

        # Only lm_q keys should be converted
        assert "model.layer1.weight" in hf_state_dict
        assert "model.layer2.bias" in hf_state_dict
        # lm_p keys should not be included
        assert "model.layer1.weight" in hf_state_dict
        assert len(hf_state_dict) == 2
        # Verify tensors are the same
        assert torch.equal(hf_state_dict["model.layer1.weight"], biencoder_state_dict["lm_q.layer1.weight"])
        assert torch.equal(hf_state_dict["model.layer2.bias"], biencoder_state_dict["lm_q.layer2.bias"])

    def test_to_hf_empty_state_dict(self, adapter):
        """Test conversion with empty state dict."""
        hf_state_dict = adapter.to_hf({})
        assert hf_state_dict == {}

    def test_to_hf_no_lm_q_keys(self, adapter):
        """Test conversion when there are no lm_q keys."""
        biencoder_state_dict = {
            "lm_p.layer1.weight": torch.randn(10, 10),
            "other.layer.weight": torch.randn(10, 10),
        }

        hf_state_dict = adapter.to_hf(biencoder_state_dict)
        assert hf_state_dict == {}

    def test_to_hf_only_lm_q_keys(self, adapter):
        """Test conversion with only lm_q keys."""
        biencoder_state_dict = {
            "lm_q.embedding.weight": torch.randn(50, 768),
            "lm_q.layer1.weight": torch.randn(768, 768),
            "lm_q.layer1.bias": torch.randn(768),
        }

        hf_state_dict = adapter.to_hf(biencoder_state_dict)

        assert len(hf_state_dict) == 3
        assert "model.embedding.weight" in hf_state_dict
        assert "model.layer1.weight" in hf_state_dict
        assert "model.layer1.bias" in hf_state_dict

    def test_to_hf_includes_linear_pooler(self, adapter):
        """Pooler weights should be retained during HF conversion."""
        biencoder_state_dict = {
            "lm_q.layer.weight": torch.randn(2, 2),
            "linear_pooler.weight": torch.randn(4, 4),
            "linear_pooler.bias": torch.randn(4),
        }

        hf_state_dict = adapter.to_hf(biencoder_state_dict)

        assert "model.layer.weight" in hf_state_dict  # sanity for lm_q path
        assert "linear_pooler.weight" in hf_state_dict
        assert "linear_pooler.bias" in hf_state_dict
        assert torch.equal(hf_state_dict["linear_pooler.weight"], biencoder_state_dict["linear_pooler.weight"])
        assert torch.equal(hf_state_dict["linear_pooler.bias"], biencoder_state_dict["linear_pooler.bias"])

    def test_from_hf_basic(self, adapter):
        """Test basic conversion from HuggingFace to biencoder format."""
        hf_state_dict = {
            "model.layer1.weight": torch.randn(10, 10),
            "model.layer2.bias": torch.randn(10),
        }

        biencoder_state_dict = adapter.from_hf(hf_state_dict)

        # Both lm_q and lm_p versions should be created
        assert "lm_q.layer1.weight" in biencoder_state_dict
        assert "lm_q.layer2.bias" in biencoder_state_dict
        assert "lm_p.layer1.weight" in biencoder_state_dict
        assert "lm_p.layer2.bias" in biencoder_state_dict
        assert len(biencoder_state_dict) == 4

        # Verify tensors are the same for both encoders
        assert torch.equal(biencoder_state_dict["lm_q.layer1.weight"], hf_state_dict["model.layer1.weight"])
        assert torch.equal(biencoder_state_dict["lm_p.layer1.weight"], hf_state_dict["model.layer1.weight"])
        assert torch.equal(biencoder_state_dict["lm_q.layer2.bias"], hf_state_dict["model.layer2.bias"])
        assert torch.equal(biencoder_state_dict["lm_p.layer2.bias"], hf_state_dict["model.layer2.bias"])

    def test_from_hf_includes_linear_pooler(self, adapter):
        """Pooler weights should be retained when converting from HF."""
        hf_state_dict = {
            "model.layer.weight": torch.randn(2, 2),
            "linear_pooler.weight": torch.randn(4, 4),
            "linear_pooler.bias": torch.randn(4),
        }

        biencoder_state_dict = adapter.from_hf(hf_state_dict)

        assert "lm_q.layer.weight" in biencoder_state_dict
        assert "lm_p.layer.weight" in biencoder_state_dict
        assert "linear_pooler.weight" in biencoder_state_dict
        assert "linear_pooler.bias" in biencoder_state_dict
        assert torch.equal(biencoder_state_dict["linear_pooler.weight"], hf_state_dict["linear_pooler.weight"])
        assert torch.equal(biencoder_state_dict["linear_pooler.bias"], hf_state_dict["linear_pooler.bias"])

    def test_from_hf_empty_state_dict(self, adapter):
        """Test conversion with empty state dict."""
        biencoder_state_dict = adapter.from_hf({})
        assert biencoder_state_dict == {}

    def test_from_hf_no_model_prefix(self, adapter):
        """Test conversion when there are no model. prefix keys."""
        hf_state_dict = {
            "embedding.weight": torch.randn(10, 10),
            "other.weight": torch.randn(10, 10),
        }

        biencoder_state_dict = adapter.from_hf(hf_state_dict)
        assert biencoder_state_dict == {}

    def test_from_hf_with_device_mesh(self, adapter):
        """Test from_hf with device_mesh parameter (should be ignored)."""
        hf_state_dict = {
            "model.layer1.weight": torch.randn(10, 10),
        }

        # device_mesh parameter should be accepted but not affect the result
        biencoder_state_dict = adapter.from_hf(hf_state_dict, device_mesh=None)

        assert "lm_q.layer1.weight" in biencoder_state_dict
        assert "lm_p.layer1.weight" in biencoder_state_dict

    def test_convert_single_tensor_to_hf_lm_q(self, adapter):
        """Test converting a single lm_q tensor to HF format."""
        tensor = torch.randn(10, 10)
        result = adapter.convert_single_tensor_to_hf("lm_q.layer1.weight", tensor)

        assert len(result) == 1
        assert result[0][0] == "model.layer1.weight"
        assert torch.equal(result[0][1], tensor)

    def test_convert_single_tensor_to_hf_lm_p(self, adapter):
        """Test converting a single lm_p tensor (should return empty list)."""
        tensor = torch.randn(10, 10)
        result = adapter.convert_single_tensor_to_hf("lm_p.layer1.weight", tensor)

        assert result == []

    def test_convert_single_tensor_to_hf_other(self, adapter):
        """Test converting a non-lm_q/lm_p tensor (should return empty list)."""
        tensor = torch.randn(10, 10)
        result = adapter.convert_single_tensor_to_hf("other.layer.weight", tensor)

        assert result == []

    def test_convert_single_tensor_to_hf_linear_pooler(self, adapter):
        """Test converting linear_pooler tensor (should be passed through)."""
        tensor = torch.randn(4)
        result = adapter.convert_single_tensor_to_hf("linear_pooler.bias", tensor)

        assert len(result) == 1
        assert result[0][0] == "linear_pooler.bias"
        assert torch.equal(result[0][1], tensor)

    def test_convert_single_tensor_to_hf_with_kwargs(self, adapter):
        """Test that convert_single_tensor_to_hf accepts kwargs."""
        tensor = torch.randn(10, 10)
        result = adapter.convert_single_tensor_to_hf("lm_q.layer1.weight", tensor, some_kwarg="value")

        assert len(result) == 1
        assert result[0][0] == "model.layer1.weight"

    def test_roundtrip_conversion(self, adapter):
        """Test that converting from HF to biencoder and back preserves lm_q state."""
        original_hf_state = {
            "model.embedding.weight": torch.randn(100, 768),
            "model.layer1.weight": torch.randn(768, 768),
            "model.output.bias": torch.randn(768),
        }

        # HF -> biencoder
        biencoder_state = adapter.from_hf(original_hf_state)

        # biencoder -> HF (should recover original)
        recovered_hf_state = adapter.to_hf(biencoder_state)

        assert set(recovered_hf_state.keys()) == set(original_hf_state.keys())
        for key in original_hf_state.keys():
            assert torch.equal(recovered_hf_state[key], original_hf_state[key])

    def test_prefix_replacement_accuracy(self, adapter):
        """Test that prefix replacement is done correctly with nested names."""
        biencoder_state_dict = {
            "lm_q.model.layer.sublayer.weight": torch.randn(5, 5),
        }

        hf_state_dict = adapter.to_hf(biencoder_state_dict)

        # Should only replace the first occurrence of lm_q.
        assert "model.model.layer.sublayer.weight" in hf_state_dict
        assert "lm_q.model.layer.sublayer.weight" not in hf_state_dict
