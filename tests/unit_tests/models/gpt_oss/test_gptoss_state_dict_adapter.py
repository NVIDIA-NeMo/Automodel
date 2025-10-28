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

import torch
from unittest.mock import Mock, patch
from transformers import GptOssConfig

from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.utils import BackendConfig

from nemo_automodel.components.models.gpt_oss.state_dict_adapter import GPTOSSStateDictAdapter

class TestApplyKeyMapping:
    def _make_adapter(self):
        return GPTOSSStateDictAdapter(config=object(), moe_config=object(), backend=object(), dtype=torch.float32)

    def test_exact_suffix_replacement(self):
        adapter = self._make_adapter()
        mapping = adapter.hf_to_internal_map

        state_dict = {
            # exact suffix matches that should be replaced
            "model.layers.0.mlp.router.weight": torch.randn(1),
            "model.layers.1.mlp.router.bias": torch.randn(1),
            "model.layers.2.mlp.experts.gate_up_proj": torch.randn(4, 4),
            "model.layers.3.mlp.experts.down_proj": torch.randn(4, 4),
            # near-misses that should NOT be replaced
            "model.layers.0.mlp.experts.gate_up_proj.weight": torch.randn(4, 4),
            "prefix.mlp.experts.gate_up_proj.suffix": torch.randn(2, 2),
            "model.layers.0.mlp.experts.gate_up_projs": torch.randn(3, 3),
            # unrelated
            "some.other.weight": torch.randn(2),
        }

        original_state_dict = dict(state_dict)
        out = adapter._apply_key_mapping(state_dict, mapping)

        # Positive cases: replaced keys exist, originals do not
        assert "model.layers.0.mlp.gate.weight" in out
        assert "model.layers.0.mlp.router.weight" not in out

        assert "model.layers.1.mlp.gate.bias" in out
        assert "model.layers.1.mlp.router.bias" not in out

        assert "model.layers.2.mlp.experts.gate_and_up_projs" in out
        assert "model.layers.2.mlp.experts.gate_up_proj" not in out

        assert "model.layers.3.mlp.experts.down_projs" in out
        assert "model.layers.3.mlp.experts.down_proj" not in out

        # Negative cases: not exact suffix -> unchanged
        assert "model.layers.0.mlp.experts.gate_up_proj.weight" in out
        assert "prefix.mlp.experts.gate_and_up_projs.suffix" not in out
        assert "prefix.mlp.experts.gate_up_proj.suffix" in out
        assert "model.layers.0.mlp.experts.gate_up_projs" in out

        # Unrelated key remains
        assert "some.other.weight" in out

        # Value identity preserved for replaced entries
        torch.testing.assert_close(
            out["model.layers.0.mlp.gate.weight"], original_state_dict["model.layers.0.mlp.router.weight"]
        )
        torch.testing.assert_close(
            out["model.layers.2.mlp.experts.gate_and_up_projs"],
            original_state_dict["model.layers.2.mlp.experts.gate_up_proj"],
        )

    def test_multiple_keys_across_layers(self):
        adapter = self._make_adapter()
        mapping = adapter.hf_to_internal_map

        # Build many layered keys to ensure only endswith matches are applied
        state_dict = {}
        for layer in range(4):
            state_dict[f"model.layers.{layer}.mlp.router.weight"] = torch.randn(1)
            state_dict[f"model.layers.{layer}.mlp.router.bias"] = torch.randn(1)
            state_dict[f"model.layers.{layer}.mlp.experts.gate_up_proj"] = torch.randn(8, 8)
            state_dict[f"model.layers.{layer}.mlp.experts.down_proj"] = torch.randn(8, 8)
            # add a non-suffix variant that must not be changed
            state_dict[f"model.layers.{layer}.mlp.experts.gate_up_proj.extra"] = torch.randn(2, 2)

        out = adapter._apply_key_mapping(state_dict, mapping)

        for layer in range(4):
            assert f"model.layers.{layer}.mlp.gate.weight" in out
            assert f"model.layers.{layer}.mlp.gate.bias" in out
            assert f"model.layers.{layer}.mlp.experts.gate_and_up_projs" in out
            assert f"model.layers.{layer}.mlp.experts.down_projs" in out

            assert f"model.layers.{layer}.mlp.router.weight" not in out
            assert f"model.layers.{layer}.mlp.router.bias" not in out
            assert f"model.layers.{layer}.mlp.experts.gate_up_proj" not in out
            assert f"model.layers.{layer}.mlp.experts.down_proj" not in out

            # non-suffix remains untouched
            assert f"model.layers.{layer}.mlp.experts.gate_up_proj.extra" in out

    def test_no_accidental_partial_replacement(self):
        adapter = self._make_adapter()
        mapping = adapter.hf_to_internal_map

        # keys that contain mapping keys, but not as full suffixes
        state_dict = {
            "mlp.router.weights": torch.randn(1),  # plural, not exact
            "mlp.router.weight.extra": torch.randn(1),  # extra suffix
            "mlp.experts.down_project": torch.randn(2, 2),  # different token
            "Xmlp.router.weight": torch.randn(1),  # leading character—still endswith? yes -> should replace
        }

        out = adapter._apply_key_mapping(state_dict, mapping)

        # The first three should not be replaced
        assert "mlp.router.weights" in out
        assert "mlp.router.weight.extra" in out
        assert "mlp.experts.down_project" in out

        # This one endswith mapping key and should be replaced, preserving prefix
        assert "Xmlp.gate.weight" in out
        assert "Xmlp.router.weight" not in out

class TestGPTOSSStateDictAdapter:
    def create_mock_config(self, **overrides):
        config = Mock(spec=GptOssConfig)
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def create_mock_moe_config(self, **overrides):
        moe_config = Mock(spec=MoEConfig)
        for key, value in overrides.items():
            setattr(moe_config, key, value)
        return moe_config

    def create_mock_backend_config(self, **overrides):
        backend = Mock(spec=BackendConfig)
        for key, value in overrides.items():
            setattr(backend, key, value)
        return backend

    def test_initialization(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = GPTOSSStateDictAdapter(config=config, moe_config=moe_config, backend=backend, dtype=torch.float16)

        assert adapter.config is config
        assert adapter.moe_config is moe_config
        assert adapter.backend is backend
        assert adapter.dtype == torch.float16
        assert adapter._uses_model_prefix is True

        # Mapping structures
        assert isinstance(adapter.hf_to_internal_map, dict)
        assert isinstance(adapter.internal_to_hf_map, dict)
        # Expected four mappings
        assert len(adapter.hf_to_internal_map) == 4
        assert len(adapter.internal_to_hf_map) == 4

    def test_initialization_with_te_backend(self):
        """Test that TE backend adds sinks mapping to adapter."""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(attn="te")

        adapter = GPTOSSStateDictAdapter(config=config, moe_config=moe_config, backend=backend, dtype=torch.float16)

        # With TE backend, we should have 5 mappings (4 base + 1 sinks mapping)
        assert len(adapter.hf_to_internal_map) == 5
        assert len(adapter.internal_to_hf_map) == 5

        # Verify the sinks mapping exists
        assert "self_attn.sinks" in adapter.hf_to_internal_map
        assert adapter.hf_to_internal_map["self_attn.sinks"] == "self_attn.attn_module.softmax_offset"
        assert "self_attn.attn_module.softmax_offset" in adapter.internal_to_hf_map
        assert adapter.internal_to_hf_map["self_attn.attn_module.softmax_offset"] == "self_attn.sinks"

    def test_initialization_with_flex_backend(self):
        """Test that Flex backend does not add sinks mapping."""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(attn="flex")

        adapter = GPTOSSStateDictAdapter(config=config, moe_config=moe_config, backend=backend, dtype=torch.float16)

        # With Flex backend, we should have 4 base mappings only
        assert len(adapter.hf_to_internal_map) == 4
        assert len(adapter.internal_to_hf_map) == 4

        # Verify the sinks mapping does not exist
        assert "self_attn.sinks" not in adapter.hf_to_internal_map

    def test_to_hf_applies_mapping_and_exclude(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()
        adapter = GPTOSSStateDictAdapter(config, moe_config, backend)

        state_dict = {
            "model.layers.0.mlp.gate.weight": torch.randn(3, 3),
            "model.layers.0.mlp.gate.bias": torch.randn(3),
            "model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(2, 5),
            "model.layers.0.mlp.experts.down_projs": torch.randn(2, 5),
            "exclude_this": torch.randn(1),
        }

        out = adapter.to_hf(state_dict, exclude_key_regex=r"^exclude.*", quantization=False)

        # Mapped keys must exist
        assert "model.layers.0.mlp.router.weight" in out
        assert "model.layers.0.mlp.router.bias" in out
        assert "model.layers.0.mlp.experts.gate_up_proj" in out
        assert "model.layers.0.mlp.experts.down_proj" in out

        # Excluded key removed
        assert "exclude_this" not in out

        # No quantization artifacts when quantization=False
        assert not any(k.endswith("_blocks") or k.endswith("_scales") for k in out)

    def test_to_hf_quantization_true_calls_helper(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()
        adapter = GPTOSSStateDictAdapter(config, moe_config, backend)

        state_dict = {
            "model.layers.0.mlp.gate.weight": torch.randn(3, 3),
        }

        with patch.object(adapter, "_add_quantization_block_scale_tensors") as mock_add:
            mock_add.return_value = {"quantized": torch.randn(1)}
            out = adapter.to_hf(state_dict, quantization=True)

        mock_add.assert_called_once()
        assert "quantized" in out

    def test_add_quantization_block_scale_tensors_creates_expected_keys(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()
        adapter = GPTOSSStateDictAdapter(config, moe_config, backend)

        class FakeDTensor:
            def __init__(self, shape, placements="P", device_mesh="M"):
                self.shape = shape
                self.placements = placements
                self.device_mesh = device_mesh
                self.is_cuda = False

        # HF-style keys expected by _add_quantization_block_scale_tensors
        state_dict = {
            "model.layers.0.mlp.experts.gate_up_proj": FakeDTensor((2, 64, 128)),
            "model.layers.0.mlp.experts.down_proj": FakeDTensor((2, 32, 256)),
            # non-matching suffix should be ignored
            "model.layers.0.mlp.experts.up_proj": torch.randn(2, 2),
        }

        def fake_ones(shape, placements=None, device_mesh=None, dtype=None):
            return FakeDTensor(shape, placements=placements, device_mesh=device_mesh)

        with patch("torch.distributed.tensor.ones", create=True, side_effect=fake_ones):
            out = adapter._add_quantization_block_scale_tensors(state_dict)

        # Original keys removed
        assert "model.layers.0.mlp.experts.gate_up_proj" not in out
        assert "model.layers.0.mlp.experts.down_proj" not in out

        # New quantization keys exist with expected shapes
        g_blocks = out["model.layers.0.mlp.experts.gate_up_proj_blocks"]
        g_scales = out["model.layers.0.mlp.experts.gate_up_proj_scales"]
        d_blocks = out["model.layers.0.mlp.experts.down_proj_blocks"]
        d_scales = out["model.layers.0.mlp.experts.down_proj_scales"]

        assert g_blocks.shape == (2, 128, 90, 16)
        assert g_scales.shape == (2, 128, 90)
        assert d_blocks.shape == (2, 256, 90, 16)
        assert d_scales.shape == (2, 256, 90)

        # Irrelevant key preserved
        assert "model.layers.0.mlp.experts.up_proj" in out

    def test_dequantize_block_scale_tensors_merges_pairs(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()
        adapter = GPTOSSStateDictAdapter(config, moe_config, backend)

        # Two different layers worth of blocks/scales
        state_dict = {
            "model.layers.0.mlp.experts.gate_up_proj_blocks": torch.ones(1),
            "model.layers.0.mlp.experts.gate_up_proj_scales": torch.ones(1),
            "model.layers.0.mlp.experts.down_proj_blocks": torch.ones(1),
            "model.layers.0.mlp.experts.down_proj_scales": torch.ones(1),
            # unrelated
            "some.other": torch.randn(2, 2),
        }

        with patch.object(adapter, "_convert_moe_packed_tensors") as mock_convert:
            mock_convert.side_effect = [torch.randn(4, 4), torch.randn(3, 3)]
            out = adapter._dequantize_block_scale_tensors(state_dict)

        # New merged keys created
        assert "model.layers.0.mlp.experts.gate_up_proj" in out
        assert "model.layers.0.mlp.experts.down_proj" in out

        # Old block/scale keys removed
        assert not any(k.endswith("_blocks") or k.endswith("_scales") for k in out)

        # Irrelevant entries preserved
        assert "some.other" in out

    def test_from_hf_detects_model_prefix(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()
        adapter = GPTOSSStateDictAdapter(config, moe_config, backend)

        hf_state = {
            "model.layers.0.mlp.experts.gate_up_proj_blocks": torch.ones(1),
            "model.layers.0.mlp.experts.gate_up_proj_scales": torch.ones(1),
        }

        with patch.object(adapter, "_convert_moe_packed_tensors", return_value=torch.randn(2, 2)):
            adapter.from_hf(hf_state)

        assert adapter._uses_model_prefix is True

    def test_from_hf_end_to_end_mapping_and_dequantize(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()
        adapter = GPTOSSStateDictAdapter(config, moe_config, backend)

        hf_state = {
            # Will dequantize into gate_up_proj and then map to gate_and_up_projs
            "model.layers.0.mlp.experts.gate_up_proj_blocks": torch.ones(1),
            "model.layers.0.mlp.experts.gate_up_proj_scales": torch.ones(1),
            # Direct mapping without dequantization
            "model.layers.0.mlp.router.weight": torch.randn(2, 2),
        }

        with patch.object(adapter, "_convert_moe_packed_tensors", return_value=torch.randn(2, 2)):
            out = adapter.from_hf(hf_state)

        # Dequantized and mapped
        assert "model.layers.0.mlp.experts.gate_and_up_projs" in out

        # Router mapping applied
        assert "model.layers.0.mlp.gate.weight" in out

        # No block/scale artifacts remain
        assert not any(k.endswith("_blocks") or k.endswith("_scales") for k in out)

    def test_from_hf_only_mapping_without_quant(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()
        adapter = GPTOSSStateDictAdapter(config, moe_config, backend)

        hf_state = {
            "layers.0.mlp.router.weight": torch.randn(2, 2),
            "layers.0.mlp.router.bias": torch.randn(2),
        }

        # No 'model.' prefix present; behavior is simply mapping
        out = adapter.from_hf(hf_state)

        assert "layers.0.mlp.gate.weight" in out
        assert "layers.0.mlp.gate.bias" in out
        assert "layers.0.mlp.router.weight" not in out
        assert "layers.0.mlp.router.bias" not in out

    def test_from_hf_with_te_backend_sinks_mapping(self):
        """Test from_hf applies sinks mapping with TE backend."""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(attn="te")
        adapter = GPTOSSStateDictAdapter(config, moe_config, backend)

        hf_state = {
            "model.layers.0.self_attn.sinks": torch.randn(8),
            "model.layers.0.mlp.router.weight": torch.randn(2, 2),
        }

        out = adapter.from_hf(hf_state)

        # Sinks should be mapped to softmax_offset
        assert "model.layers.0.self_attn.attn_module.softmax_offset" in out
        assert "model.layers.0.self_attn.sinks" not in out

        # Router mapping should still work
        assert "model.layers.0.mlp.gate.weight" in out

    def test_to_hf_with_te_backend_sinks_mapping(self):
        """Test to_hf applies sinks mapping with TE backend."""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(attn="te")
        adapter = GPTOSSStateDictAdapter(config, moe_config, backend)

        state_dict = {
            "model.layers.0.self_attn.attn_module.softmax_offset": torch.randn(8),
            "model.layers.0.mlp.gate.weight": torch.randn(2, 2),
        }

        out = adapter.to_hf(state_dict, quantization=False)

        # softmax_offset should be mapped to sinks
        assert "model.layers.0.self_attn.sinks" in out
        assert "model.layers.0.self_attn.attn_module.softmax_offset" not in out

        # Gate mapping should still work
        assert "model.layers.0.mlp.router.weight" in out


class TestConvertMoePackedTensors:
    def _make_adapter(self):
        return GPTOSSStateDictAdapter(config=object(), moe_config=object(), backend=object(), dtype=torch.float32)

    def test_convert_basic_nibble_decode_and_shape(self):
        adapter = self._make_adapter()

        class FakeDTensor:
            def __init__(self, tensor, placements="P", device_mesh="M"):
                self.tensor = tensor
                self.placements = placements
                self.device_mesh = device_mesh

            @property
            def shape(self):
                return self.tensor.shape

            @property
            def device(self):
                return self.tensor.device

            @property
            def is_cuda(self):
                return self.tensor.is_cuda

            def to(self, dtype):
                return FakeDTensor(self.tensor.to(dtype), self.placements, self.device_mesh)

            def cuda(self):
                return FakeDTensor(self.tensor.cuda(), self.placements, self.device_mesh)

            def reshape(self, *shape):
                return FakeDTensor(self.tensor.reshape(*shape), self.placements, self.device_mesh)

            def view(self, *shape):
                return FakeDTensor(self.tensor.view(*shape), self.placements, self.device_mesh)

            def transpose(self, dim0, dim1):
                return FakeDTensor(self.tensor.transpose(dim0, dim1), self.placements, self.device_mesh)

            def contiguous(self):
                return FakeDTensor(self.tensor.contiguous(), self.placements, self.device_mesh)

            def to_local(self):
                return self.tensor

            def __getitem__(self, item):
                return FakeDTensor(self.tensor.__getitem__(item), self.placements, self.device_mesh)

            def __sub__(self, other):
                if isinstance(other, FakeDTensor):
                    return FakeDTensor(self.tensor - other.tensor, self.placements, self.device_mesh)
                return FakeDTensor(self.tensor - other, self.placements, self.device_mesh)

            def __rsub__(self, other):
                if isinstance(other, FakeDTensor):
                    return FakeDTensor(other.tensor - self.tensor, self.placements, self.device_mesh)
                return FakeDTensor(other - self.tensor, self.placements, self.device_mesh)

        # blocks shape: (*prefix=1,1, G=1, B=1) containing byte 0x12 -> low=2 (1.0), high=1 (0.5)
        blocks = FakeDTensor(torch.tensor([[[[18]]]], dtype=torch.uint8))
        # scales shape: (*prefix=1,1, G=1), value 127 -> exponent 0 (no scaling)
        scales = FakeDTensor(torch.tensor([[[127]]], dtype=torch.uint8))

        def fake_empty(shape, placements=None, device_mesh=None, dtype=None):
            return FakeDTensor(torch.empty(shape, dtype=dtype), placements=placements, device_mesh=device_mesh)

        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.distributed.tensor.empty", create=True, side_effect=fake_empty):
            out = adapter._convert_moe_packed_tensors(blocks, scales, dtype=torch.float32, rows_per_chunk=4)

        # Unwrap local tensor for validation
        out_local = out.to_local() if hasattr(out, "to_local") else out

        # Expect shape: (*prefix, G*B*2) = (1, 2) then transposed dims (1,2) -> (1, 2, 1)
        assert out_local.shape == (1, 2, 1)
        torch.testing.assert_close(out_local[0, 0, 0], torch.tensor(1.0, dtype=torch.float32))
        torch.testing.assert_close(out_local[0, 1, 0], torch.tensor(0.5, dtype=torch.float32))

    def test_convert_chunking_and_exponent_scaling(self):
        adapter = self._make_adapter()

        class FakeDTensor:
            def __init__(self, tensor, placements="P", device_mesh="M"):
                self.tensor = tensor
                self.placements = placements
                self.device_mesh = device_mesh

            @property
            def shape(self):
                return self.tensor.shape

            @property
            def device(self):
                return self.tensor.device

            @property
            def is_cuda(self):
                return self.tensor.is_cuda

            def to(self, dtype):
                return FakeDTensor(self.tensor.to(dtype), self.placements, self.device_mesh)

            def cuda(self):
                return FakeDTensor(self.tensor.cuda(), self.placements, self.device_mesh)

            def reshape(self, *shape):
                return FakeDTensor(self.tensor.reshape(*shape), self.placements, self.device_mesh)

            def view(self, *shape):
                return FakeDTensor(self.tensor.view(*shape), self.placements, self.device_mesh)

            def transpose(self, dim0, dim1):
                return FakeDTensor(self.tensor.transpose(dim0, dim1), self.placements, self.device_mesh)

            def contiguous(self):
                return FakeDTensor(self.tensor.contiguous(), self.placements, self.device_mesh)

            def to_local(self):
                return self.tensor

            def __getitem__(self, item):
                return FakeDTensor(self.tensor.__getitem__(item), self.placements, self.device_mesh)

            def __sub__(self, other):
                if isinstance(other, FakeDTensor):
                    return FakeDTensor(self.tensor - other.tensor, self.placements, self.device_mesh)
                return FakeDTensor(self.tensor - other, self.placements, self.device_mesh)

            def __rsub__(self, other):
                if isinstance(other, FakeDTensor):
                    return FakeDTensor(other.tensor - self.tensor, self.placements, self.device_mesh)
                return FakeDTensor(other - self.tensor, self.placements, self.device_mesh)

        # Design 3 rows (G=3), B=1, prefix=(1,1)
        # Bytes: 0x12 -> [1.0, 0.5], 0xF1 -> [0.5, -6.0], 0x80 -> [0.0, -0.0]
        blocks = FakeDTensor(torch.tensor([[[[0x12], [0xF1], [0x80]]]], dtype=torch.uint8))
        # Exponents: 127->0, 128->+1, 126->-1
        scales = FakeDTensor(torch.tensor([[[127, 128, 126]]], dtype=torch.uint8))

        def fake_empty(shape, placements=None, device_mesh=None, dtype=None):
            return FakeDTensor(torch.empty(shape, dtype=dtype), placements=placements, device_mesh=device_mesh)

        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.distributed.tensor.empty", create=True, side_effect=fake_empty):
            out = adapter._convert_moe_packed_tensors(blocks, scales, dtype=torch.float32, rows_per_chunk=1)

        out_local = out.to_local() if hasattr(out, "to_local") else out

        # Shape: (*prefix, G*B*2) -> (1, 6) then transpose -> (1, 6, 1)
        assert out_local.shape == (1, 6, 1)

        # Row 0, exponent 0 -> [1.0, 0.5]
        torch.testing.assert_close(out_local[0, 0, 0], torch.tensor(1.0))
        torch.testing.assert_close(out_local[0, 1, 0], torch.tensor(0.5))

        # Row 1, exponent +1 -> [1.0, -12.0]
        torch.testing.assert_close(out_local[0, 2, 0], torch.tensor(1.0))
        torch.testing.assert_close(out_local[0, 3, 0], torch.tensor(-12.0))

        # Row 2, exponent -1 -> [0.0, 0.0] (signed zeros tolerated)
        torch.testing.assert_close(out_local[0, 4, 0], torch.tensor(0.0))
        torch.testing.assert_close(out_local[0, 5, 0], torch.tensor(0.0))


class TestSingleGPUScenarios:
    def _make_adapter(self):
        return GPTOSSStateDictAdapter(config=object(), moe_config=object(), backend=object(), dtype=torch.float32)

    def test_convert_single_gpu_stays_on_cpu(self):
        adapter = self._make_adapter()

        # CPU tensors simulating packed nibble values and zero exponent
        blocks = torch.tensor([[[[0x12]]]], dtype=torch.uint8)
        scales = torch.tensor([[[127]]], dtype=torch.uint8)

        # CUDA available but single-process world (no distributed multi-GPU)
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=1):
            out = adapter._convert_moe_packed_tensors(blocks, scales, dtype=torch.float32, rows_per_chunk=4)

        assert not out.is_cuda
        assert out.shape == (1, 2, 1)
        torch.testing.assert_close(out[0, 0, 0], torch.tensor(1.0, dtype=torch.float32))
        torch.testing.assert_close(out[0, 1, 0], torch.tensor(0.5, dtype=torch.float32))

    def test_from_hf_single_gpu_cpu_dequantize_and_map(self):
        adapter = self._make_adapter()

        # Packed tensors (CPU) for one expert row, ensure dequantization occurs on CPU
        blocks = torch.tensor([[[[0x12]]]], dtype=torch.uint8)
        scales = torch.tensor([[[127]]], dtype=torch.uint8)
        hf_state = {
            "model.layers.0.mlp.experts.gate_up_proj_blocks": blocks,
            "model.layers.0.mlp.experts.gate_up_proj_scales": scales,
            # also exercise router mapping path
            "model.layers.0.mlp.router.weight": torch.randn(1),
        }

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=1):
            out = adapter.from_hf(hf_state)

        # Dequantized key should be mapped to internal name and remain on CPU
        assert "model.layers.0.mlp.experts.gate_and_up_projs" in out
        deq = out["model.layers.0.mlp.experts.gate_and_up_projs"]
        assert isinstance(deq, torch.Tensor)
        assert not deq.is_cuda

        # Router path mapped
        assert "model.layers.0.mlp.gate.weight" in out

        # No block/scale artifacts remain
        assert not any(k.endswith("_blocks") or k.endswith("_scales") for k in out)

    def test_add_quantization_block_scale_tensors_cpu_path_creates_expected_keys(self):
        adapter = self._make_adapter()

        state_dict = {
            "model.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 64, 128),
            "model.layers.0.mlp.experts.down_proj": torch.randn(2, 32, 256),
        }

        out = adapter._add_quantization_block_scale_tensors(state_dict)

        # Originals removed
        assert "model.layers.0.mlp.experts.gate_up_proj" not in out
        assert "model.layers.0.mlp.experts.down_proj" not in out

        # CPU path uses torch.ones producing uint8 blocks/scales with expected shapes
        g_blocks = out["model.layers.0.mlp.experts.gate_up_proj_blocks"]
        g_scales = out["model.layers.0.mlp.experts.gate_up_proj_scales"]
        d_blocks = out["model.layers.0.mlp.experts.down_proj_blocks"]
        d_scales = out["model.layers.0.mlp.experts.down_proj_scales"]

        assert g_blocks.shape == (2, 128, 90, 16)
        assert g_scales.shape == (2, 128, 90)
        assert d_blocks.shape == (2, 256, 90, 16)
        assert d_scales.shape == (2, 256, 90)
        assert g_blocks.dtype == torch.uint8 and g_scales.dtype == torch.uint8
        assert d_blocks.dtype == torch.uint8 and d_scales.dtype == torch.uint8


class TestDTensorPaths:
    def _make_adapter(self):
        return GPTOSSStateDictAdapter(config=object(), moe_config=object(), backend=object(), dtype=torch.float32)

    def test_add_quantization_block_scale_tensors_dtensor_path(self):
        adapter = self._make_adapter()

        class FakeDTensor:
            def __init__(self, shape, placements="P", device_mesh="M"):
                self.shape = shape
                self.placements = placements
                self.device_mesh = device_mesh
                self.is_cuda = False

        # HF-style keys, with values that are DTensor-like
        state_dict = {
            "model.layers.0.mlp.experts.gate_up_proj": FakeDTensor((2, 64, 128), placements="Pg", device_mesh="Mg"),
            "model.layers.0.mlp.experts.down_proj": FakeDTensor((3, 32, 256), placements="Pd", device_mesh="Md"),
        }

        def fake_ones(shape, placements=None, device_mesh=None, dtype=None):
            # Ensure dtype is the expected uint8
            assert dtype == torch.uint8
            return FakeDTensor(shape, placements=placements, device_mesh=device_mesh)

        with patch("torch.distributed.tensor.DTensor", new=FakeDTensor, create=True), \
             patch("torch.distributed.tensor.ones", create=True, side_effect=fake_ones):
            out = adapter._add_quantization_block_scale_tensors(state_dict)

        # Originals removed
        assert "model.layers.0.mlp.experts.gate_up_proj" not in out
        assert "model.layers.0.mlp.experts.down_proj" not in out

        # New DTensor-shaped blocks/scales created preserving placements and mesh
        g_blocks = out["model.layers.0.mlp.experts.gate_up_proj_blocks"]
        g_scales = out["model.layers.0.mlp.experts.gate_up_proj_scales"]
        d_blocks = out["model.layers.0.mlp.experts.down_proj_blocks"]
        d_scales = out["model.layers.0.mlp.experts.down_proj_scales"]

        assert isinstance(g_blocks, FakeDTensor)
        assert isinstance(g_scales, FakeDTensor)
        assert isinstance(d_blocks, FakeDTensor)
        assert isinstance(d_scales, FakeDTensor)

        assert g_blocks.shape == (2, 128, 90, 16)
        assert g_scales.shape == (2, 128, 90)
        assert d_blocks.shape == (3, 256, 90, 16)
        assert d_scales.shape == (3, 256, 90)

    def test_convert_moe_packed_tensors_dtensor_path_with_simulated_cuda_move(self):
        adapter = self._make_adapter()

        class FakeDTensor:
            def __init__(self, tensor, placements="P", device_mesh="M"):
                self.tensor = tensor
                self.placements = placements
                self.device_mesh = device_mesh
                self._is_cuda = False

            @property
            def shape(self):
                return self.tensor.shape

            @property
            def device(self):
                return self.tensor.device

            @property
            def is_cuda(self):
                return self._is_cuda

            def to(self, dtype):
                return FakeDTensor(self.tensor.to(dtype), self.placements, self.device_mesh)

            def cuda(self):
                # Simulate a move to CUDA without requiring a GPU
                self._is_cuda = True
                return self

            def reshape(self, *shape):
                return FakeDTensor(self.tensor.reshape(*shape), self.placements, self.device_mesh)

            def view(self, *shape):
                return FakeDTensor(self.tensor.view(*shape), self.placements, self.device_mesh)

            def transpose(self, dim0, dim1):
                return FakeDTensor(self.tensor.transpose(dim0, dim1), self.placements, self.device_mesh)

            def contiguous(self):
                return FakeDTensor(self.tensor.contiguous(), self.placements, self.device_mesh)

            def to_local(self):
                return self.tensor

            def __getitem__(self, item):
                return FakeDTensor(self.tensor.__getitem__(item), self.placements, self.device_mesh)

            def __sub__(self, other):
                if isinstance(other, FakeDTensor):
                    return FakeDTensor(self.tensor - other.tensor, self.placements, self.device_mesh)
                return FakeDTensor(self.tensor - other, self.placements, self.device_mesh)

            def __rsub__(self, other):
                if isinstance(other, FakeDTensor):
                    return FakeDTensor(other.tensor - self.tensor, self.placements, self.device_mesh)
                return FakeDTensor(other - self.tensor, self.placements, self.device_mesh)

        # blocks nibble 0x12 -> [1.0, 0.5], exponent 127 -> 0
        blocks = FakeDTensor(torch.tensor([[[[0x12]]]], dtype=torch.uint8), placements="Pb", device_mesh="Mb")
        scales = FakeDTensor(torch.tensor([[[127]]], dtype=torch.uint8), placements="Ps", device_mesh="Ms")

        def fake_empty(shape, placements=None, device_mesh=None, dtype=None):
            # Must allocate a DTensor-like container preserving placements and mesh
            return FakeDTensor(torch.empty(shape, dtype=dtype), placements=placements, device_mesh=device_mesh)

        with patch("torch.distributed.tensor.DTensor", new=FakeDTensor, create=True), \
             patch("torch.distributed.tensor.empty", create=True, side_effect=fake_empty), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=2):
            out = adapter._convert_moe_packed_tensors(blocks, scales, dtype=torch.float32, rows_per_chunk=4)

        # Simulated CUDA move occurred
        assert blocks.is_cuda and scales.is_cuda

        # DTensor path used for output
        assert isinstance(out, FakeDTensor)
        out_local = out.to_local()
        assert out_local.shape == (1, 2, 1)
        torch.testing.assert_close(out_local[0, 0, 0], torch.tensor(1.0, dtype=torch.float32))
        torch.testing.assert_close(out_local[0, 1, 0], torch.tensor(0.5, dtype=torch.float32))
