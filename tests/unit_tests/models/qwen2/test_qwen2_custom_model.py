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
from transformers import AutoModelForCausalLM, Qwen2Config, set_seed

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen2.state_dict_adapter import Qwen2StateDictAdapter

set_seed(42)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

# Tiny Qwen2 config for testing
TINY_DEFAULT_QWEN2_CONFIG = dict(
    vocab_size=1024,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
    rms_norm_eps=1e-5,
    tie_word_embeddings=True,
)


def _create_checkpoint(config_kwargs, tmpdir):
    """Create a tiny HF Qwen2 checkpoint in the given directory.

    Args:
        config_kwargs: Dict of Qwen2Config keyword arguments.
        tmpdir: Directory (str or Path) to save the checkpoint into.

    Returns:
        str path to the checkpoint directory.
    """
    tmpdir = str(tmpdir)
    config = Qwen2Config(**config_kwargs)
    config.save_pretrained(tmpdir)
    model = AutoModelForCausalLM.from_config(config)
    for param in model.parameters():
        # Reinitialize trivially constant parameters (e.g., norm weight=all 1s, bias=all 0s)
        if param.data.unique().numel() == 1:
            param.data.normal_(mean=0, std=0.1)
    model.save_pretrained(tmpdir)
    return tmpdir


class TestQwen2Model:
    @pytest.fixture(scope="class", autouse=True)
    def _tiny_checkpoint(self, tmp_path_factory):
        """Create a tiny HF Qwen2 checkpoint shared across tests (auto-cleaned by pytest)."""
        self.__class__.tiny_qwen2_checkpoint = _create_checkpoint(
            TINY_DEFAULT_QWEN2_CONFIG, tmp_path_factory.mktemp("qwen2_ckpt")
        )

    @pytest.mark.parametrize("rms_norm", ["torch_fp32", "te"])
    def test_model_matches_hf_with_adapter_bidirectional(self, rms_norm, tmp_path):
        """Test bidirectional conversion between HF and custom models produces identical outputs.

        Parametrized over:
          - rms_norm: "torch_fp32" | "te"

        torch_fp32: float32-upcast RMSNorm matching HF Qwen2RMSNorm exactly -> tight tol.
        te: Transformer Engine fused RMSNorm kernel -> relaxed tol.
        """
        # Set tolerances based on norm backend precision
        # te: Transformer Engine fused kernel -> relaxed tolerance
        # torch_fp32: matches HF exactly -> tight tolerance
        tolerances = {
            "te": dict(atol=1e-3, rtol=1e-3),
            "torch_fp32": dict(atol=1e-7, rtol=1e-7),
        }
        tol = tolerances[rms_norm]

        checkpoint = _create_checkpoint(TINY_DEFAULT_QWEN2_CONFIG, tmp_path)
        config = Qwen2Config.from_pretrained(checkpoint)
        adapter = Qwen2StateDictAdapter(config)

        # Load HF model
        qwen2_model_hf = (
            AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=checkpoint,
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
            )
            .to("cuda")
            .to(torch.bfloat16)
        )  # need to manual cast to bfloat16 since HF initialize weights/buffers in float32 dtype
        qwen2_model_hf.eval()

        # Build custom model with specified norm backend
        backend = BackendConfig(rms_norm=rms_norm)
        qwen2_model_custom = NeMoAutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=checkpoint,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            backend=backend,
        ).to("cuda")
        qwen2_model_custom.eval()

        # Verify parameter counts match
        num_params_hf = sum(p.numel() for p in qwen2_model_hf.parameters())
        num_params_custom = sum(p.numel() for p in qwen2_model_custom.parameters())
        assert num_params_hf == num_params_custom, (
            "Number of parameters in the custom model does not match the HuggingFace model"
        )

        # Test forward direction: HF → Custom
        hf_state_dict = qwen2_model_hf.state_dict()
        custom_state_dict_from_hf = adapter.from_hf(hf_state_dict)
        # Use nn.Module.load_state_dict directly to bypass mixin (testing adapter, not mixin)
        # Note: strict=False because HF checkpoints don't have TE's _extra_state keys
        torch.nn.Module.load_state_dict(qwen2_model_custom, custom_state_dict_from_hf, strict=False)

        # Use nn.Module.state_dict directly to get native format (testing adapter, not mixin)
        s = adapter.to_hf(torch.nn.Module.state_dict(qwen2_model_custom))

        for n1, p1 in hf_state_dict.items():
            p2 = s[n1]
            assert p1.shape == p2.shape, f"Parameter shape mismatch: {p1.shape} != {p2.shape}"
            assert p1.dtype == p2.dtype, f"Parameter dtype mismatch: {p1.dtype} != {p2.dtype}"
            assert p1.device == p2.device, f"Parameter device mismatch: {p1.device} != {p2.device}"
            assert p1.requires_grad == p2.requires_grad, (
                f"Parameter requires_grad mismatch: {p1.requires_grad} != {p2.requires_grad}"
            )
            assert torch.allclose(p1, p2, atol=1e-5, rtol=1e-5), f"Parameter mismatch: {p1} != {p2}"

        # Generate test inputs
        input_ids = torch.randint(0, config.vocab_size, (1, 10)).to("cuda")
        attention_mask = torch.ones((1, 10)).to("cuda")

        # Compare HF → Custom outputs
        with torch.no_grad():
            output_hf = qwen2_model_hf(input_ids.clone(), attention_mask.clone())
            output_custom = qwen2_model_custom(input_ids, attention_mask)

        np.testing.assert_allclose(
            output_hf.logits.float().cpu().numpy(),
            output_custom.logits.float().cpu().numpy(),
            err_msg=f"HF → Custom conversion outputs don't match with {rms_norm=}",
            **tol,
        )

        # Test reverse direction: Custom → HF
        # Use nn.Module.state_dict directly to get native format (testing adapter, not mixin)
        custom_state_dict = torch.nn.Module.state_dict(qwen2_model_custom)
        hf_state_dict_from_custom = adapter.to_hf(custom_state_dict)

        # Create new HF model and load converted state dict
        qwen2_model_hf_converted = (
            AutoModelForCausalLM.from_pretrained(checkpoint, attn_implementation="eager", torch_dtype=torch.bfloat16)
            .to("cuda")
            .to(torch.bfloat16)
        )  # need to manual cast to bfloat16 since HF initialize weights/buffers in float32 dtype
        qwen2_model_hf_converted.eval()
        # Note: strict=False because HF checkpoints don't have TE's _extra_state keys
        qwen2_model_hf_converted.load_state_dict(hf_state_dict_from_custom, strict=False)

        # Compare Custom → HF outputs
        with torch.no_grad():
            output_hf_converted = qwen2_model_hf_converted(input_ids, attention_mask)

        np.testing.assert_allclose(
            output_custom.logits.float().cpu().numpy(),
            output_hf_converted.logits.float().cpu().numpy(),
            err_msg="Custom → HF conversion outputs don't match",
            **tol,
        )

    def test_state_dict_adapter_from_hf_combined_projections(self):
        """Test converting HF state dict to custom format with combined QKV and gate_up projections.

        This test verifies that the adapter correctly combines HF's separate q/k/v projections
        into qkv_proj, and gate/up projections into gate_up_proj for the custom model.
        """
        config = Qwen2Config.from_pretrained(self.tiny_qwen2_checkpoint)
        adapter = Qwen2StateDictAdapter(config)

        # Load HF model and get state dict
        qwen2_model_hf = AutoModelForCausalLM.from_pretrained(
            self.tiny_qwen2_checkpoint, attn_implementation="eager", torch_dtype=torch.bfloat16
        ).to(torch.bfloat16)  # need to manual cast to bfloat16 since HF initialize weights/buffers in float32 dtype
        hf_state_dict = qwen2_model_hf.state_dict()

        # Convert to custom format
        custom_state_dict = adapter.from_hf(hf_state_dict)

        # Check that separate Q/K/V weights and biases don't exist in custom state dict
        assert "model.layers.0.self_attn.q_proj.weight" not in custom_state_dict
        assert "model.layers.0.self_attn.k_proj.weight" not in custom_state_dict
        assert "model.layers.0.self_attn.v_proj.weight" not in custom_state_dict
        assert "model.layers.0.self_attn.q_proj.bias" not in custom_state_dict
        assert "model.layers.0.self_attn.k_proj.bias" not in custom_state_dict
        assert "model.layers.0.self_attn.v_proj.bias" not in custom_state_dict
        assert "model.layers.0.mlp.gate_proj.weight" not in custom_state_dict
        assert "model.layers.0.mlp.up_proj.weight" not in custom_state_dict

        # Check that combined keys exist in custom state dict
        assert "model.layers.0.self_attn.qkv_proj.weight" in custom_state_dict
        assert "model.layers.0.self_attn.qkv_proj.bias" in custom_state_dict
        assert "model.layers.0.mlp.gate_up_proj.weight" in custom_state_dict

    def test_state_dict_adapter_to_hf(self):
        """Test converting custom model state dict back to HF format.

        This test verifies that the custom model (built with NeMoAutoModelForCausalLM) has combined
        projections by default, and that these are the only projection keys present.
        """
        # Build custom model (which uses adapter internally to load from HF checkpoint)
        qwen2_model_custom = NeMoAutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.tiny_qwen2_checkpoint,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        )
        # Use nn.Module.state_dict directly to get native format (testing adapter, not mixin)
        custom_state_dict = torch.nn.Module.state_dict(qwen2_model_custom)

        # Check that all original HF keys don't exist in custom state dict
        assert "model.layers.0.self_attn.q_proj.weight" not in custom_state_dict
        assert "model.layers.0.self_attn.k_proj.weight" not in custom_state_dict
        assert "model.layers.0.self_attn.v_proj.weight" not in custom_state_dict
        assert "model.layers.0.self_attn.q_proj.bias" not in custom_state_dict
        assert "model.layers.0.self_attn.k_proj.bias" not in custom_state_dict
        assert "model.layers.0.self_attn.v_proj.bias" not in custom_state_dict
        assert "model.layers.0.mlp.gate_proj.weight" not in custom_state_dict
        assert "model.layers.0.mlp.up_proj.weight" not in custom_state_dict

        # Check that combined keys exist in custom state dict
        assert "model.layers.0.self_attn.qkv_proj.weight" in custom_state_dict
        assert "model.layers.0.self_attn.qkv_proj.bias" in custom_state_dict
        assert "model.layers.0.mlp.gate_up_proj.weight" in custom_state_dict
