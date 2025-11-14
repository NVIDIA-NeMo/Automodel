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


@pytest.fixture(scope="class")
def tiny_qwen2_checkpoint():
    """Create a tiny Qwen2 model with random weights in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a small config for fast testing
        config = Qwen2Config(
            vocab_size=1024,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,  # GQA
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
        )
        
        # Save config
        config.save_pretrained(tmpdir)
        
        # Create model with random weights
        model = AutoModelForCausalLM.from_config(config)
        
        # Save model
        model.save_pretrained(tmpdir)
        
        yield tmpdir


class TestQwen2Model:
    def test_model_matches_hf_with_adapter_bidirectional(self, tiny_qwen2_checkpoint):
        """Test bidirectional conversion between HF and custom models produces identical outputs.
        
        The custom Qwen2 model ALWAYS uses combined QKV and gate_up projections for efficiency.
        The state dict adapter handles conversion between HF (separate) and custom (combined) formats.
        """
        config = Qwen2Config.from_pretrained(tiny_qwen2_checkpoint)
        adapter = Qwen2StateDictAdapter(config)

        # Load HF model
        qwen2_model_hf = (
            AutoModelForCausalLM.from_pretrained(
                tiny_qwen2_checkpoint, attn_implementation="eager", torch_dtype=torch.bfloat16
            )
            .to("cuda")
            .to(torch.bfloat16)  # need to manual cast to bfloat16 since HF initialize weights in float32 dtype
        )

        # Build custom model
        qwen2_model_custom = build_qwen2_model(
            pretrained_model_name_or_path=tiny_qwen2_checkpoint,
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
                tiny_qwen2_checkpoint, attn_implementation="eager", torch_dtype=torch.bfloat16
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

    def test_state_dict_adapter_from_hf_combined_projections(self, tiny_qwen2_checkpoint):
        """Test converting HF state dict to custom format with combined QKV and gate_up projections.
        
        This test verifies that the adapter correctly combines HF's separate q/k/v projections
        into qkv_proj, and gate/up projections into gate_up_proj for the custom model.
        """
        config = Qwen2Config.from_pretrained(tiny_qwen2_checkpoint)
        adapter = Qwen2StateDictAdapter(config)

        # Load HF model and get state dict
        qwen2_model_hf = AutoModelForCausalLM.from_pretrained(
            tiny_qwen2_checkpoint, attn_implementation="eager", torch_dtype=torch.bfloat16
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

    def test_state_dict_adapter_to_hf(self, tiny_qwen2_checkpoint):
        """Test converting custom model state dict back to HF format.
        
        This test verifies that the custom model (built with build_qwen2_model) has combined
        projections by default, and that these are the only projection keys present.
        """
        # Build custom model (which uses adapter internally to load from HF checkpoint)
        qwen2_model_custom = build_qwen2_model(
            pretrained_model_name_or_path=tiny_qwen2_checkpoint,
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

    def test_export_custom_to_hf_checkpoint(self, tiny_qwen2_checkpoint):
        """Test exporting custom model to HF-compatible checkpoint format.
        
        This test verifies that a custom model with combined projections can be exported
        to a HuggingFace-compatible format (with separate projections) and loaded back
        with AutoModelForCausalLM, producing identical outputs.
        """
        config = Qwen2Config.from_pretrained(tiny_qwen2_checkpoint)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "hf_checkpoint")

            # Build custom model
            qwen2_model_custom = build_qwen2_model(
                pretrained_model_name_or_path=tiny_qwen2_checkpoint,
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