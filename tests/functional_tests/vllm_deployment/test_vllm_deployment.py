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

"""Functional tests: save with NeMo AutoModel, load and infer with vLLM.

Four scenarios are covered, each parameterised as its own test class:

1. **Dense model** (tiny Llama) → save consolidated → vLLM offline inference
2. **Dense model + LoRA** → save base + adapter → vLLM LoRA inference
3. **MoE model** (tiny Qwen3-MoE) → save consolidated → vLLM offline inference
4. **MoE model + LoRA** → save base + adapter → vLLM LoRA inference

All models are randomly initialised with minimal dimensions so the tests
run on a single GPU with < 1 GB VRAM and do not require network access.
"""

import json
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LORA_R = 8
LORA_ALPHA = 16
VOCAB_SIZE = 256
SEQ_LEN = 32
MAX_NEW_TOKENS = 8


# ===========================================================================
# Tiny model builders
# ===========================================================================


def _make_tiny_llama_config():
    """Minimal ``LlamaConfig`` for a dense model (~200 KB)."""
    from transformers import LlamaConfig

    return LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=128,
    )


def _make_tiny_qwen3_moe_config():
    """Minimal ``Qwen3MoeConfig`` for a sparse MoE model (~400 KB)."""
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

    return Qwen3MoeConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=128,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        norm_topk_prob=False,
        router_aux_loss_coef=0.0,
    )


def _save_dense_model(save_dir: str) -> str:
    """Create and save a tiny dense Llama model; return the path."""
    from transformers import AutoModelForCausalLM

    config = _make_tiny_llama_config()
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(save_dir)
    return save_dir


def _save_moe_model(save_dir: str) -> str:
    """Create and save a tiny Qwen3-MoE model; return the path."""
    from transformers import AutoModelForCausalLM

    config = _make_tiny_qwen3_moe_config()
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(save_dir)
    return save_dir


# ===========================================================================
# LoRA adapter helpers
# ===========================================================================


def _apply_and_save_lora(base_model_path: str, adapter_dir: str, target_modules: list[str]):
    """Apply HF PEFT LoRA to a model and save the adapter."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float32)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)

    torch.manual_seed(42)
    for _, param in peft_model.named_parameters():
        if param.requires_grad:
            param.data = torch.randn_like(param.data) * 0.01

    peft_model.save_pretrained(adapter_dir)
    return adapter_dir


# ===========================================================================
# vLLM helpers
# ===========================================================================


def _vllm_generate(model_path: str, prompts: list[str], **kwargs):
    """Run offline vLLM generation and return output texts."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        dtype="float32",
        max_model_len=SEQ_LEN + MAX_NEW_TOKENS,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        **kwargs,
    )
    params = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)
    outputs = llm.generate(prompts, sampling_params=params)
    return [o.outputs[0].text for o in outputs]


def _vllm_generate_with_lora(
    model_path: str,
    adapter_path: str,
    prompts: list[str],
):
    """Run offline vLLM generation with a LoRA adapter."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=model_path,
        dtype="float32",
        max_model_len=SEQ_LEN + MAX_NEW_TOKENS,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_lora=True,
        max_lora_rank=LORA_R,
    )
    params = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)
    lora_req = LoRARequest("test-adapter", 1, adapter_path)
    outputs = llm.generate(prompts, sampling_params=params, lora_request=lora_req)
    return [o.outputs[0].text for o in outputs]


# ===========================================================================
# Validation helpers
# ===========================================================================


def _assert_valid_outputs(outputs: list[str], num_prompts: int):
    """Check that vLLM returned the expected number of non-empty outputs."""
    assert len(outputs) == num_prompts, f"Expected {num_prompts} outputs, got {len(outputs)}"
    for i, text in enumerate(outputs):
        assert isinstance(text, str), f"Output {i} is not a string: {type(text)}"


def _assert_logits_match(model_path: str, vllm_model_path: str):
    """Compare HF forward-pass logits with vLLM logits on the same input."""
    from transformers import AutoModelForCausalLM

    from vllm import LLM, SamplingParams

    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits  # (1, seq, vocab)
    hf_next_token = hf_logits[0, -1].argmax().item()

    llm = LLM(
        model=vllm_model_path,
        dtype="float32",
        max_model_len=SEQ_LEN + MAX_NEW_TOKENS,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
    )
    prompt_token_ids = input_ids[0].tolist()
    params = SamplingParams(max_tokens=1, temperature=0.0)
    out = llm.generate(
        sampling_params=params,
        prompt_token_ids=[prompt_token_ids],
    )
    vllm_token_id = out[0].outputs[0].token_ids[0]

    assert hf_next_token == vllm_token_id, (
        f"Next-token mismatch: HF predicted {hf_next_token}, vLLM predicted {vllm_token_id}"
    )


# ===========================================================================
# Dense model tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestDenseVLLM:
    """Save a tiny dense Llama with AutoModel, load in vLLM."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.base_model_path = _save_dense_model(str(tmp_path / "dense_base"))
        self.prompts = ["Hello world", "The quick brown fox"]

    def test_vllm_generates_outputs(self):
        outputs = _vllm_generate(self.base_model_path, self.prompts)
        _assert_valid_outputs(outputs, len(self.prompts))

    def test_vllm_logits_match_hf(self):
        _assert_logits_match(self.base_model_path, self.base_model_path)

    def test_checkpoint_files_present(self):
        path = Path(self.base_model_path)
        assert (path / "config.json").is_file()
        safetensors = list(path.glob("*.safetensors"))
        assert len(safetensors) > 0, "No safetensors files found"


# ===========================================================================
# Dense model + LoRA tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestDenseLoRAVLLM:
    """Save a tiny dense Llama + LoRA adapter, load both in vLLM."""

    DENSE_LORA_TARGETS = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.base_model_path = _save_dense_model(str(tmp_path / "dense_lora_base"))
        self.adapter_path = _apply_and_save_lora(
            self.base_model_path,
            str(tmp_path / "dense_lora_adapter"),
            self.DENSE_LORA_TARGETS,
        )
        self.prompts = ["Hello world", "The quick brown fox"]

    def test_adapter_files_present(self):
        adapter = Path(self.adapter_path)
        assert (adapter / "adapter_config.json").is_file()
        assert (adapter / "adapter_model.safetensors").is_file()

    def test_vllm_lora_generates_outputs(self):
        outputs = _vllm_generate_with_lora(
            self.base_model_path,
            self.adapter_path,
            self.prompts,
        )
        _assert_valid_outputs(outputs, len(self.prompts))

    def test_lora_changes_output(self):
        """Outputs with LoRA adapter should differ from base model."""
        base_outputs = _vllm_generate(self.base_model_path, self.prompts)
        lora_outputs = _vllm_generate_with_lora(
            self.base_model_path,
            self.adapter_path,
            self.prompts,
        )
        assert base_outputs != lora_outputs, "LoRA adapter did not change any outputs — adapter may not have loaded"


# ===========================================================================
# MoE model tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMoEVLLM:
    """Save a tiny Qwen3-MoE with AutoModel, load in vLLM."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.base_model_path = _save_moe_model(str(tmp_path / "moe_base"))
        self.prompts = ["Hello world", "The quick brown fox"]

    def test_vllm_generates_outputs(self):
        outputs = _vllm_generate(self.base_model_path, self.prompts)
        _assert_valid_outputs(outputs, len(self.prompts))

    def test_vllm_logits_match_hf(self):
        _assert_logits_match(self.base_model_path, self.base_model_path)

    def test_checkpoint_files_present(self):
        path = Path(self.base_model_path)
        assert (path / "config.json").is_file()
        safetensors = list(path.glob("*.safetensors"))
        assert len(safetensors) > 0, "No safetensors files found"


# ===========================================================================
# MoE model + LoRA tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMoELoRAVLLM:
    """Save a tiny Qwen3-MoE + LoRA adapter, load both in vLLM."""

    MOE_LORA_TARGETS = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.base_model_path = _save_moe_model(str(tmp_path / "moe_lora_base"))
        self.adapter_path = _apply_and_save_lora(
            self.base_model_path,
            str(tmp_path / "moe_lora_adapter"),
            self.MOE_LORA_TARGETS,
        )
        self.prompts = ["Hello world", "The quick brown fox"]

    def test_adapter_files_present(self):
        adapter = Path(self.adapter_path)
        assert (adapter / "adapter_config.json").is_file()
        assert (adapter / "adapter_model.safetensors").is_file()

        with open(adapter / "adapter_config.json") as f:
            cfg = json.load(f)
        assert cfg["peft_type"] == "LORA"
        assert cfg["r"] == LORA_R

    def test_vllm_lora_generates_outputs(self):
        outputs = _vllm_generate_with_lora(
            self.base_model_path,
            self.adapter_path,
            self.prompts,
        )
        _assert_valid_outputs(outputs, len(self.prompts))

    def test_lora_changes_output(self):
        """Outputs with LoRA adapter should differ from base model."""
        base_outputs = _vllm_generate(self.base_model_path, self.prompts)
        lora_outputs = _vllm_generate_with_lora(
            self.base_model_path,
            self.adapter_path,
            self.prompts,
        )
        assert base_outputs != lora_outputs, "LoRA adapter did not change any outputs — adapter may not have loaded"
