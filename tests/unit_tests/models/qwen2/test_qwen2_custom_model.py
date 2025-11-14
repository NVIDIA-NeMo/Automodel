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


import os
import tempfile

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from nemo_automodel.components.models.qwen2 import build_qwen2_model
from nemo_automodel.components.models.qwen2.state_dict_adapter import Qwen2StateDictAdapter

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# Use a small Qwen2 model for testing
# For CI/CD, you may need to update this path to point to available test data
# PRETRAINED_MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-1.5B"
PRETRAINED_MODEL_NAME_OR_PATH = "/home/TestData/automodel/hf_qwen2_5_1_5b_2l/"


class TestQwen2Model:
    def test_model_matches_hf_with_adapter_bidirectional(self):
        """Test bidirectional conversion between HF and custom models produces identical outputs.
        
        The custom Qwen2 model ALWAYS uses combined QKV and gate_up projections for efficiency.
        The state dict adapter handles conversion between HF (separate) and custom (combined) formats.
        """
        config = Qwen2Config.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH)
        adapter = Qwen2StateDictAdapter(config)

        # Load HF model
        qwen2_model_hf = (
            AutoModelForCausalLM.from_pretrained(
                PRETRAINED_MODEL_NAME_OR_PATH, attn_implementation="eager", torch_dtype=torch.bfloat16
            )
            .to("cuda")
            .to(torch.bfloat16)  # need to manual cast to bfloat16 since HF initialize weights in float32 dtype
        )

        # Build custom model
        qwen2_model_custom = build_qwen2_model(
            pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        # Verify parameter counts match
        num_params_hf = sum(p.numel() for p in qwen2_model_hf.parameters())
        num_params_custom = sum(p.numel() for p in qwen2_model_custom.parameters())
        assert num_params_hf == num_params_custom, (
            "Number of parameters in the custom model does not match the HuggingFace model"
        )

        # Test forward direction: HF → Custom
        hf_state_dict = qwen2_model_hf.state_dict()
        custom_state_dict_from_hf = adapter.from_hf(hf_state_dict)
        qwen2_model_custom.load_state_dict(custom_state_dict_from_hf, strict=True)

        # Generate test inputs
        input_ids = torch.randint(0, config.vocab_size, (1, 10)).to("cuda")
        attention_mask = torch.ones((1, 10)).to("cuda")

        # Compare HF → Custom outputs
        with torch.no_grad():
            output_hf = qwen2_model_hf(input_ids, attention_mask)
            output_custom = qwen2_model_custom(input_ids, attention_mask)

        np.testing.assert_allclose(
            output_hf.logits.float().cpu().numpy(),
            output_custom.logits.float().cpu().numpy(),
            atol=1e-5,
            rtol=1e-5,
            err_msg="HF → Custom conversion outputs don't match",
        )

        # Test reverse direction: Custom → HF
        custom_state_dict = qwen2_model_custom.state_dict()
        hf_state_dict_from_custom = adapter.to_hf(custom_state_dict)

        # Create new HF model and load converted state dict
        qwen2_model_hf_converted = (
            AutoModelForCausalLM.from_pretrained(
                PRETRAINED_MODEL_NAME_OR_PATH, attn_implementation="eager", torch_dtype=torch.bfloat16
            )
            .to("cuda")
            .to(torch.bfloat16)
        )
        qwen2_model_hf_converted.load_state_dict(hf_state_dict_from_custom, strict=True)

        # Compare Custom → HF outputs
        with torch.no_grad():
            output_hf_converted = qwen2_model_hf_converted(input_ids, attention_mask)

        np.testing.assert_allclose(
            output_custom.logits.float().cpu().numpy(),
            output_hf_converted.logits.float().cpu().numpy(),
            atol=1e-5,
            rtol=1e-5,
            err_msg="Custom → HF conversion outputs don't match",
        )

    def test_state_dict_adapter_from_hf_combined_projections(self):
        """Test converting HF state dict to custom format with combined QKV and gate_up projections.
        
        This test verifies that the adapter correctly combines HF's separate q/k/v projections
        into qkv_proj, and gate/up projections into gate_up_proj for the custom model.
        """
        config = Qwen2Config.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH)
        adapter = Qwen2StateDictAdapter(config)

        # Load HF model and get state dict
        qwen2_model_hf = AutoModelForCausalLM.from_pretrained(
            PRETRAINED_MODEL_NAME_OR_PATH, attn_implementation="eager", torch_dtype=torch.bfloat16
        )
        hf_state_dict = qwen2_model_hf.state_dict()

        # Convert to custom format
        custom_state_dict = adapter.from_hf(hf_state_dict)

        # Check that separate Q/K/V weights don't exist in custom state dict
        assert "model.layers.0.self_attn.q_proj.weight" not in custom_state_dict
        assert "model.layers.0.self_attn.k_proj.weight" not in custom_state_dict
        assert "model.layers.0.self_attn.v_proj.weight" not in custom_state_dict
        assert "model.layers.0.mlp.gate_proj.weight" not in custom_state_dict
        assert "model.layers.0.mlp.up_proj.weight" not in custom_state_dict

        # Check that combined keys exist in custom state dict
        assert "model.layers.0.self_attn.qkv_proj.weight" in custom_state_dict
        assert "model.layers.0.mlp.gate_up_proj.weight" in custom_state_dict

    def test_state_dict_adapter_to_hf(self):
        """Test converting custom model state dict back to HF format.
        
        This test verifies that the custom model (built with build_qwen2_model) has combined
        projections by default, and that these are the only projection keys present.
        """
        # Build custom model (which uses adapter internally to load from HF checkpoint)
        qwen2_model_custom = build_qwen2_model(
            pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        )
        custom_state_dict = qwen2_model_custom.state_dict()

        # Check that all original HF keys don't exist in custom state dict
        assert "model.layers.0.self_attn.q_proj.weight" not in custom_state_dict
        assert "model.layers.0.self_attn.k_proj.weight" not in custom_state_dict
        assert "model.layers.0.self_attn.v_proj.weight" not in custom_state_dict
        assert "model.layers.0.mlp.gate_proj.weight" not in custom_state_dict
        assert "model.layers.0.mlp.up_proj.weight" not in custom_state_dict

        # Check that combined keys exist in custom state dict
        assert "model.layers.0.self_attn.qkv_proj.weight" in custom_state_dict
        assert "model.layers.0.mlp.gate_up_proj.weight" in custom_state_dict

    def test_export_custom_to_hf_checkpoint(self):
        """Test exporting custom model to HF-compatible checkpoint format.
        
        This test verifies that a custom model with combined projections can be exported
        to a HuggingFace-compatible format (with separate projections) and loaded back
        with AutoModelForCausalLM, producing identical outputs.
        """
        config = Qwen2Config.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "hf_checkpoint")

            # Build custom model
            qwen2_model_custom = build_qwen2_model(
                pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
            ).to("cuda")

            # Generate test input
            input_ids = torch.randint(0, config.vocab_size, (1, 10)).to("cuda")
            attention_mask = torch.ones((1, 10)).to("cuda")

            # Get custom model output
            with torch.no_grad():
                output_custom = qwen2_model_custom(input_ids, attention_mask)

            # Save in HF-compatible format using the convenience method
            qwen2_model_custom.save_pretrained_hf_format(export_path)

            # Load from saved HF checkpoint
            qwen2_model_hf_loaded = (
                AutoModelForCausalLM.from_pretrained(
                    export_path,
                    attn_implementation="eager",
                    torch_dtype=torch.bfloat16,
                )
                .to("cuda")
                .to(torch.bfloat16)
            )

            # Compare outputs
            with torch.no_grad():
                output_hf_loaded = qwen2_model_hf_loaded(input_ids, attention_mask)

            np.testing.assert_allclose(
                output_custom.logits.float().cpu().numpy(),
                output_hf_loaded.logits.float().cpu().numpy(),
                atol=1e-5,
                rtol=1e-5,
                err_msg="HF model loaded from exported checkpoint doesn't match custom model",
            )

    def test_combined_projections_always_enabled(self):
        """Test that combined projections are always enabled in custom Qwen2 models.
        
        This test verifies the "one direction only" philosophy: the custom implementation
        ALWAYS uses combined projections, with no option to disable them.
        """
        # Build custom model
        qwen2_model_custom = build_qwen2_model(
            pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        )

        # Check that all attention layers have qkv_proj
        for layer in qwen2_model_custom.model.layers:
            assert hasattr(layer.self_attn, "qkv_proj"), "Attention layer must have qkv_proj"
            assert not hasattr(layer.self_attn, "q_proj"), "Attention layer should not have separate q_proj"
            assert not hasattr(layer.self_attn, "k_proj"), "Attention layer should not have separate k_proj"
            assert not hasattr(layer.self_attn, "v_proj"), "Attention layer should not have separate v_proj"

        # Check that all MLP layers have gate_up_proj
        for layer in qwen2_model_custom.model.layers:
            assert hasattr(layer.mlp, "gate_up_proj"), "MLP layer must have gate_up_proj"
            assert not hasattr(layer.mlp, "gate_proj"), "MLP layer should not have separate gate_proj"
            assert not hasattr(layer.mlp, "up_proj"), "MLP layer should not have separate up_proj"

    def test_qwen2_state_dict_adapter_exists(self):
        """Test that the Qwen2 model has a state dict adapter configured."""
        # Build custom model
        qwen2_model_custom = build_qwen2_model(
            pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        )

        # Check that state_dict_adapter exists and is correct type
        assert hasattr(qwen2_model_custom, "state_dict_adapter"), (
            "Custom model must have state_dict_adapter"
        )
        assert isinstance(qwen2_model_custom.state_dict_adapter, Qwen2StateDictAdapter), (
            "state_dict_adapter must be Qwen2StateDictAdapter instance"
        )

