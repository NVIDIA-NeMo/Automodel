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
            out["model.layers.0.mlp.gate.weight"], state_dict["model.layers.0.mlp.router.weight"]
        )
        torch.testing.assert_close(
            out["model.layers.2.mlp.experts.gate_and_up_projs"],
            state_dict["model.layers.2.mlp.experts.gate_up_proj"],
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
