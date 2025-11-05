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


import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM

from nemo_automodel.components.models.llama import build_llama_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# PRETRAINED_MODEL_NAME_OR_PATH = "meta-llama/Llama-3.2-1B"
PRETRAINED_MODEL_NAME_OR_PATH = "/home/TestData/automodel/hf_llama3_2_1b_2l/"


class TestLlamaModel:
    def test_model_matches_hf(self):
        llama_model_hf = (
            AutoModelForCausalLM.from_pretrained(
                PRETRAINED_MODEL_NAME_OR_PATH, attn_implementation="eager", dtype=torch.bfloat16
            )
            .to("cuda")
            .to(torch.bfloat16)
        )
        llama_model_custom = build_llama_model(
            pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
            attn_implementation="eager",
            use_fused_qkv=False,
            use_fused_gate_up=False,
            dtype=torch.bfloat16,
        ).to("cuda")
        llama_model_custom_fused = build_llama_model(
            pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
            attn_implementation="eager",
            use_fused_qkv=True,
            use_fused_gate_up=True,
            dtype=torch.bfloat16,
        ).to("cuda")

        # calculate number of parameters in the model
        num_params_hf = sum(p.numel() for p in llama_model_hf.parameters())
        num_params_custom = sum(p.numel() for p in llama_model_custom.parameters())
        num_params_custom_fused = sum(p.numel() for p in llama_model_custom_fused.parameters())
        assert num_params_hf == num_params_custom, (
            "Number of parameters in the custom model does not match the HuggingFace model"
        )
        assert num_params_hf == num_params_custom_fused, (
            "Number of parameters in the custom model (with fused QKV and gate_up) does not match the HuggingFace model"
        )

        # how to compare the model outputs?
        llama_model_custom.load_state_dict(llama_model_hf.state_dict(), strict=True)

        with torch.no_grad():
            input_ids = torch.randint(0, llama_model_hf.config.vocab_size, (1, 10)).to("cuda")
            attention_mask = torch.ones((1, 10)).to("cuda")
            output_hf = llama_model_hf(input_ids, attention_mask)
            output_custom = llama_model_custom(input_ids, attention_mask)
        np.testing.assert_allclose(
            output_hf.logits.float().cpu().numpy(), output_custom.logits.float().cpu().numpy(), atol=1e-5, rtol=1e-5
        )

        for name, param in llama_model_custom_fused.named_parameters():
            if "qkv_proj.weight" in name:
                parts = name.split(".")
                layer_indx = int(parts[parts.index("layers") + 1])  # Find "layers" and get next element
                param.data = torch.cat(
                    [
                        llama_model_hf.model.layers[layer_indx].self_attn.q_proj.weight,
                        llama_model_hf.model.layers[layer_indx].self_attn.k_proj.weight,
                        llama_model_hf.model.layers[layer_indx].self_attn.v_proj.weight,
                    ],
                    dim=0,
                )
            elif "qkv_proj.bias" in name:
                parts = name.split(".")
                layer_indx = int(parts[parts.index("layers") + 1])
                param.data = torch.cat(
                    [
                        llama_model_hf.model.layers[layer_indx].self_attn.q_proj.bias,
                        llama_model_hf.model.layers[layer_indx].self_attn.k_proj.bias,
                        llama_model_hf.model.layers[layer_indx].self_attn.v_proj.bias,
                    ],
                    dim=0,
                )
            elif "gate_up_proj.weight" in name:
                parts = name.split(".")
                layer_indx = int(parts[parts.index("layers") + 1])
                param.data = torch.cat(
                    [
                        llama_model_hf.model.layers[layer_indx].mlp.gate_proj.weight,
                        llama_model_hf.model.layers[layer_indx].mlp.up_proj.weight,
                    ],
                    dim=0,
                )
            elif "gate_up_proj.bias" in name:
                parts = name.split(".")
                layer_indx = int(parts[parts.index("layers") + 1])
                param.data = torch.cat(
                    [
                        llama_model_hf.model.layers[layer_indx].mlp.gate_proj.bias,
                        llama_model_hf.model.layers[layer_indx].mlp.up_proj.bias,
                    ],
                    dim=0,
                )
            else:
                param.data = llama_model_hf.state_dict()[name]

        with torch.no_grad():
            input_ids = torch.randint(0, llama_model_hf.config.vocab_size, (1, 10)).to("cuda")
            attention_mask = torch.ones((1, 10)).to("cuda")
            output_hf = llama_model_hf(input_ids, attention_mask)
            output_custom_fused = llama_model_custom_fused(input_ids, attention_mask)
        np.testing.assert_allclose(
            output_hf.logits.float().cpu().numpy(),
            output_custom_fused.logits.float().cpu().numpy(),
            atol=1e-5,
            rtol=1e-5,
        )
