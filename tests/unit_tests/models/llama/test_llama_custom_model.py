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
from transformers import AutoModelForCausalLM, LlamaConfig	

from nemo_automodel.components.models.llama.model import build_llama_model	
from nemo_automodel.components.models.llama.state_dict_adapter import LlamaStateDictAdapter	

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")	


# PRETRAINED_MODEL_NAME_OR_PATH = "meta-llama/Llama-3.2-1B"	
PRETRAINED_MODEL_NAME_OR_PATH = "/home/TestData/automodel/hf_llama3_2_1b_2l/"	


class TestLlamaModel:	
    def test_model_matches_hf_with_adapter_bidirectional(self):	
        """Test bidirectional conversion between HF and custom models produces identical outputs."""	
        config = LlamaConfig.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH)	
        adapter = LlamaStateDictAdapter(config)	

        # Load HF model	
        llama_model_hf = (	
            AutoModelForCausalLM.from_pretrained(	
                PRETRAINED_MODEL_NAME_OR_PATH, attn_implementation="eager", torch_dtype=torch.bfloat16	
            )	
            .to("cuda")	
            .to(torch.bfloat16)  # need to manual cast to bfloat16 since HF initialize weights in float32 dtype	
        )	

        # Build custom model	
        llama_model_custom = build_llama_model(	
            pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,	
            attn_implementation="eager",	
            torch_dtype=torch.bfloat16,	
        ).to("cuda")	

        # Verify parameter counts match	
        num_params_hf = sum(p.numel() for p in llama_model_hf.parameters())	
        num_params_custom = sum(p.numel() for p in llama_model_custom.parameters())	
        assert num_params_hf == num_params_custom, (	
            "Number of parameters in the custom model does not match the HuggingFace model"	
        )	

        # Test forward direction: HF → Custom	
        hf_state_dict = llama_model_hf.state_dict()	
        custom_state_dict_from_hf = adapter.from_hf(hf_state_dict)	
        llama_model_custom.load_state_dict(custom_state_dict_from_hf, strict=True)	

        # Generate test inputs	
        input_ids = torch.randint(0, config.vocab_size, (1, 10)).to("cuda")	
        attention_mask = torch.ones((1, 10)).to("cuda")	

        # Compare HF → Custom outputs	
        with torch.no_grad():	
            output_hf = llama_model_hf(input_ids, attention_mask)	
            output_custom = llama_model_custom(input_ids, attention_mask)	

        np.testing.assert_allclose(	
            output_hf.logits.float().cpu().numpy(),	
            output_custom.logits.float().cpu().numpy(),	
            atol=1e-5,	
            rtol=1e-5,	
            err_msg="HF → Custom conversion outputs don't match",	
        )	

        # Test reverse direction: Custom → HF	
        custom_state_dict = llama_model_custom.state_dict()	
        hf_state_dict_from_custom = adapter.to_hf(custom_state_dict)	

        # Create new HF model and load converted state dict	
        llama_model_hf_converted = (	
            AutoModelForCausalLM.from_pretrained(	
                PRETRAINED_MODEL_NAME_OR_PATH, attn_implementation="eager", torch_dtype=torch.bfloat16	
            )	
            .to("cuda")	
            .to(torch.bfloat16)	
        )	
        llama_model_hf_converted.load_state_dict(hf_state_dict_from_custom, strict=True)	

        # Compare Custom → HF outputs	
        with torch.no_grad():	
            output_hf_converted = llama_model_hf_converted(input_ids, attention_mask)	

        np.testing.assert_allclose(	
            output_custom.logits.float().cpu().numpy(),	
            output_hf_converted.logits.float().cpu().numpy(),	
            atol=1e-5,	
            rtol=1e-5,	
            err_msg="Custom → HF conversion outputs don't match",	
        )	

    def test_state_dict_adapter_from_hf_combined_projections(self):	
        """Test converting HF state dict to custom format with combined QKV and gate_up projections."""	
        config = LlamaConfig.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH)	
        adapter = LlamaStateDictAdapter(config)	

        # Load HF model and get state dict	
        llama_model_hf = AutoModelForCausalLM.from_pretrained(	
            PRETRAINED_MODEL_NAME_OR_PATH, attn_implementation="eager", torch_dtype=torch.bfloat16	
        )	
        hf_state_dict = llama_model_hf.state_dict()	

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
        """Test converting custom model state dict back to HF format."""	
        # Build custom model (which uses adapter internally to load from HF checkpoint)	
        llama_model_custom = build_llama_model(	
            pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,	
            attn_implementation="eager",	
            torch_dtype=torch.bfloat16,	
        )	
        custom_state_dict = llama_model_custom.state_dict()	

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
        """Test exporting custom model to HF-compatible checkpoint format."""	
        config = LlamaConfig.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH)	

        with tempfile.TemporaryDirectory() as tmpdir:	
            export_path = os.path.join(tmpdir, "hf_checkpoint")	

            # Build custom model	
            llama_model_custom = build_llama_model(	
                pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,	
                attn_implementation="eager",	
                torch_dtype=torch.bfloat16,	
            ).to("cuda")	

            # Generate test input	
            input_ids = torch.randint(0, config.vocab_size, (1, 10)).to("cuda")	
            attention_mask = torch.ones((1, 10)).to("cuda")	

            # Get custom model output	
            with torch.no_grad():	
                output_custom = llama_model_custom(input_ids, attention_mask)	

            # Save in HF-compatible format using the convenience method	
            llama_model_custom.save_pretrained_hf_format(export_path)	

            # Load from saved HF checkpoint	
            llama_model_hf_loaded = (	
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
                output_hf_loaded = llama_model_hf_loaded(input_ids, attention_mask)	

            np.testing.assert_allclose(	
                output_custom.logits.float().cpu().numpy(),	
                output_hf_loaded.logits.float().cpu().numpy(),	
                atol=1e-5,	
                rtol=1e-5,	
                err_msg="HF model loaded from exported checkpoint doesn't match custom model",	
            )	