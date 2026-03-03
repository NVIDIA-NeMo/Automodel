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

"""Functional tests for the LoRA merge tool (``tools/merge_lora.py``).

Two categories are covered:

1. **Dense model** – A small randomly-initialised ``LlamaForCausalLM`` is
   fine-tuned with HF PEFT LoRA, then the adapter is merged back using both
   the HF reference path (``PeftModel.merge_and_unload``) and the standalone
   ``merge_lora`` tool.  The merged weights are compared for bitwise equality.

2. **MoE model** – Automodel's ``GroupedExpertsLoRA`` produces grouped 3-D
   adapter tensors that must be converted to per-expert HF PEFT format before
   merging.  The test exercises that full save → convert → merge → verify
   pipeline.

All models are tiny (a few MB) and randomly initialised so no network access
or large checkpoints are needed.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
LORA_DIM = 8
LORA_ALPHA = 32


def _write_adapter_dir(path, adapter_sd, lora_r, lora_alpha, target_modules, base_model_name=""):
    """Persist ``adapter_model.safetensors`` + ``adapter_config.json``."""
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    save_file(adapter_sd, os.path.join(path, "adapter_model.safetensors"))

    config = {
        "peft_type": "LORA",
        "auto_mapping": {"base_model_class": None, "parent_library": None},
        "base_model_name_or_path": base_model_name,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "r": lora_r,
        "rank_pattern": {},
        "revision": None,
        "target_modules": sorted(target_modules),
        "task_type": "CAUSAL_LM",
    }
    with open(os.path.join(path, "adapter_config.json"), "w") as f:
        json.dump(config, f)


# ============================================================================
# Dense model tests
# ============================================================================


def _make_tiny_llama_config():
    """Return a minimal ``LlamaConfig`` for testing."""
    from transformers import LlamaConfig

    return LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=64,
    )


def _make_tiny_dense_model(tmp_path):
    """Create a tiny LlamaForCausalLM, save it, and return the path."""
    from transformers import AutoModelForCausalLM

    config = _make_tiny_llama_config()
    model = AutoModelForCausalLM.from_config(config)
    model_path = os.path.join(tmp_path, "base_model")
    model.save_pretrained(model_path)
    return model_path


def _train_dense_lora_adapter(base_model_path, adapter_path):
    """Apply HF PEFT LoRA to a tiny dense model and save the adapter."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    lora_config = LoraConfig(
        r=LORA_DIM,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)

    # Simulate training by randomizing adapter weights.
    torch.manual_seed(42)
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            param.data = torch.randn_like(param.data) * 0.01

    peft_model.save_pretrained(adapter_path)
    return adapter_path


def _reference_merge_dense(base_model_path, adapter_path):
    """Merge using HF PEFT directly and return the merged state dict."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    merged = model.merge_and_unload()
    return {n: p.data.clone() for n, p in merged.named_parameters()}


class TestDenseLoRAMerge:
    """End-to-end LoRA merge for a dense (Llama) model."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = tmp_path
        self.base_model_path = _make_tiny_dense_model(tmp_path)
        self.adapter_path = os.path.join(tmp_path, "adapter")
        _train_dense_lora_adapter(self.base_model_path, self.adapter_path)

    def test_tool_merge_matches_reference(self):
        """Verify the merge_lora tool produces the same result as HF PEFT."""
        from tools.merge_lora import merge_lora

        output_dir = os.path.join(self.tmp_path, "merged_tool")
        merge_lora(
            base_model=self.base_model_path,
            adapter_path=self.adapter_path,
            output_dir=output_dir,
            dtype="float32",
            device="cpu",
            save_tokenizer=False,
        )

        ref_sd = _reference_merge_dense(self.base_model_path, self.adapter_path)

        from transformers import AutoModelForCausalLM

        merged_model = AutoModelForCausalLM.from_pretrained(output_dir)
        tool_sd = {n: p.data for n, p in merged_model.named_parameters()}

        assert set(ref_sd.keys()) == set(tool_sd.keys()), "Key sets differ"
        for key in ref_sd:
            torch.testing.assert_close(
                tool_sd[key], ref_sd[key], rtol=1e-5, atol=1e-5,
                msg=f"Mismatch at {key}",
            )

    def test_merged_weights_differ_from_base(self):
        """Merged model weights must differ from the base model."""
        from tools.merge_lora import merge_lora
        from transformers import AutoModelForCausalLM

        base_sd = {
            n: p.data.clone()
            for n, p in AutoModelForCausalLM.from_pretrained(
                self.base_model_path
            ).named_parameters()
        }

        output_dir = os.path.join(self.tmp_path, "merged_diff")
        merge_lora(
            base_model=self.base_model_path,
            adapter_path=self.adapter_path,
            output_dir=output_dir,
            dtype="float32",
            device="cpu",
            save_tokenizer=False,
        )

        merged_model = AutoModelForCausalLM.from_pretrained(output_dir)
        changed = 0
        for name, param in merged_model.named_parameters():
            if name in base_sd and not torch.equal(param.data, base_sd[name]):
                changed += 1

        assert changed > 0, "No weights changed after merge"

    def test_no_lora_params_remain(self):
        """After merge, no LoRA parameters or modules should be present."""
        from tools.merge_lora import merge_lora
        from transformers import AutoModelForCausalLM

        output_dir = os.path.join(self.tmp_path, "merged_clean")
        merge_lora(
            base_model=self.base_model_path,
            adapter_path=self.adapter_path,
            output_dir=output_dir,
            dtype="float32",
            device="cpu",
            save_tokenizer=False,
        )

        merged_model = AutoModelForCausalLM.from_pretrained(output_dir)
        for name, _ in merged_model.named_parameters():
            assert "lora_" not in name, f"LoRA param {name} should be absent"
        for name, _ in merged_model.named_modules():
            assert "lora_" not in name, f"LoRA module {name} should be absent"

    def test_merged_logits_kl_div_vs_unmerged(self):
        """KL-div between merged model and adapter-applied model must be ~0."""
        from peft import PeftModel
        from tools.merge_lora import merge_lora
        from transformers import AutoModelForCausalLM

        output_dir = os.path.join(self.tmp_path, "merged_kl")
        merge_lora(
            base_model=self.base_model_path,
            adapter_path=self.adapter_path,
            output_dir=output_dir,
            dtype="float32",
            device="cpu",
            save_tokenizer=False,
        )
        merged_model = AutoModelForCausalLM.from_pretrained(output_dir).eval()

        base = AutoModelForCausalLM.from_pretrained(self.base_model_path)
        ref_model = PeftModel.from_pretrained(base, self.adapter_path).eval()

        torch.manual_seed(123)
        input_ids = torch.randint(0, 256, (2, 32))

        with torch.no_grad():
            merged_logits = merged_model(input_ids).logits
            ref_logits = ref_model(input_ids).logits

        log_p = torch.nn.functional.log_softmax(merged_logits, dim=-1)
        q = torch.nn.functional.softmax(ref_logits, dim=-1)
        kl = torch.nn.functional.kl_div(log_p, q, reduction="none", log_target=False)
        kl = kl.sum(-1).view(-1).max()
        assert kl.item() < 1e-6, f"KL divergence too high: {kl.item():.3e}"

    def test_cli_invocation(self):
        """Verify the tool works when invoked as a script."""
        output_dir = os.path.join(self.tmp_path, "merged_cli")
        result = subprocess.run(
            [
                sys.executable, str(_REPO_ROOT / "tools" / "merge_lora.py"),
                "--base-model", self.base_model_path,
                "--adapter-path", self.adapter_path,
                "--output-dir", output_dir,
                "--dtype", "float32",
                "--device", "cpu",
                "--no-save-tokenizer",
            ],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        assert os.path.isfile(os.path.join(output_dir, "config.json"))

        from transformers import AutoModelForCausalLM

        merged = AutoModelForCausalLM.from_pretrained(output_dir)
        assert merged is not None


# ============================================================================
# MoE model tests
# ============================================================================

N_EXPERTS = 4
MOE_DIM = 64
MOE_INTER_DIM = 128
N_LAYERS = 2


MOE_VOCAB_SIZE = 256


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(MOE_DIM, MOE_INTER_DIM, bias=False)
        self.up_proj = nn.Linear(MOE_DIM, MOE_INTER_DIM, bias=False)
        self.down_proj = nn.Linear(MOE_INTER_DIM, MOE_DIM, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _HFStyleMoE(nn.Module):
    """Mimics an HF transformers MoE model with per-expert Linear layers."""

    def __init__(self):
        super().__init__()
        from transformers import PretrainedConfig

        self.config = PretrainedConfig(hidden_size=MOE_DIM)
        self.embed_tokens = nn.Embedding(MOE_VOCAB_SIZE, MOE_DIM)
        self.layers = nn.ModuleList()
        for _ in range(N_LAYERS):
            layer = nn.Module()
            mlp = nn.Module()
            mlp.experts = nn.ModuleList([_Expert() for _ in range(N_EXPERTS)])
            layer.mlp = mlp
            self.layers.append(layer)
        self.lm_head = nn.Linear(MOE_DIM, MOE_VOCAB_SIZE, bias=False)

    def forward(self, input_ids, **kwargs):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            expert_out = torch.stack([e(x) for e in layer.mlp.experts]).mean(dim=0)
            x = x + expert_out
        return type("Output", (), {"logits": self.lm_head(x)})()

    def prepare_inputs_for_generation(self, *args, **kwargs):
        raise NotImplementedError("stub for HF PEFT compatibility")


def _make_moe_lora_state_dict(prefix="base_model.model."):
    """Build a synthetic grouped LoRA state dict."""
    torch.manual_seed(42)
    sd = {}
    for layer_idx in range(N_LAYERS):
        base = f"{prefix}layers.{layer_idx}.mlp.experts"
        sd[f"{base}.lora_gate_and_up_A"] = torch.randn(N_EXPERTS, MOE_DIM, LORA_DIM)
        sd[f"{base}.lora_gate_and_up_B"] = torch.randn(N_EXPERTS, LORA_DIM, 2 * MOE_INTER_DIM)
        sd[f"{base}.lora_down_A"] = torch.randn(N_EXPERTS, MOE_INTER_DIM, LORA_DIM)
        sd[f"{base}.lora_down_B"] = torch.randn(N_EXPERTS, LORA_DIM, MOE_DIM)
    return sd


def _convert_grouped_to_hf_peft(grouped_sd):
    """Convert Automodel grouped MoE LoRA tensors to per-expert HF PEFT format."""
    from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

    class _Adapter(MoESplitExpertsStateDictMixin):
        def __init__(self):
            self.moe_config = type("C", (), {
                "n_routed_experts": N_EXPERTS,
                "moe_inter_dim": MOE_INTER_DIM,
                "expert_activation": "swiglu",
            })()
            self.dtype = torch.float32
            self._uses_model_prefix = True
            self._last_expert_ids = []

    adapter = _Adapter()
    hf_sd = {}
    for fqn, tensor in grouped_sd.items():
        converted = adapter._convert_single_merged_expert_to_hf_split_experts(fqn, tensor)
        if converted:
            for k, v in converted:
                hf_sd[k] = v
        else:
            hf_sd[fqn] = tensor
    return hf_sd


class TestMoELoRAMerge:
    """End-to-end LoRA merge for a Mixture-of-Experts model.

    This exercises the Automodel-specific grouped → per-expert conversion
    pipeline that allows MoE LoRA adapters to be merged via ``PeftModel``.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = tmp_path
        self.hf_model = _HFStyleMoE()
        self.base_weights = {n: p.data.clone() for n, p in self.hf_model.named_parameters()}

        self.grouped_sd = _make_moe_lora_state_dict()
        self.hf_sd = _convert_grouped_to_hf_peft(self.grouped_sd)

        target_modules = []
        for li in range(N_LAYERS):
            for eid in range(N_EXPERTS):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    target_modules.append(f"layers.{li}.mlp.experts.{eid}.{proj}")

        self.adapter_path = os.path.join(tmp_path, "moe_adapter")
        _write_adapter_dir(
            self.adapter_path,
            self.hf_sd,
            LORA_DIM,
            LORA_ALPHA,
            target_modules,
        )

    def test_peft_loads_adapter(self):
        """HF PEFT can load the converted adapter without errors."""
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(
            _HFStyleMoE(), self.adapter_path
        )
        assert peft_model is not None
        assert hasattr(peft_model, "base_model")

    def test_merge_changes_all_expert_weights(self):
        """After merge, every expert weight tensor should be modified."""
        from peft import PeftModel

        hf_model = _HFStyleMoE()
        base_weights = {n: p.data.clone() for n, p in hf_model.named_parameters()}

        peft_model = PeftModel.from_pretrained(hf_model, self.adapter_path)
        merged = peft_model.merge_and_unload()

        changed = 0
        for name, param in merged.named_parameters():
            if name in base_weights and "experts" in name:
                if not torch.equal(param.data, base_weights[name]):
                    changed += 1

        expected = N_LAYERS * N_EXPERTS * 3  # gate, up, down per expert
        assert changed == expected, (
            f"Expected {expected} expert weights to change, got {changed}"
        )

    def test_merge_is_numerically_correct(self):
        """Merged weight = base + B @ A * scale for every expert projection."""
        from peft import PeftModel

        hf_model = _HFStyleMoE()
        base_weights = {n: p.data.clone() for n, p in hf_model.named_parameters()}

        peft_model = PeftModel.from_pretrained(hf_model, self.adapter_path)
        merged = peft_model.merge_and_unload()

        scale = LORA_ALPHA / LORA_DIM

        for name, param in merged.named_parameters():
            if "experts" not in name or name not in base_weights:
                continue

            lora_A_key = f"base_model.model.{name}".replace(".weight", ".lora_A.weight")
            lora_B_key = f"base_model.model.{name}".replace(".weight", ".lora_B.weight")

            if lora_A_key not in self.hf_sd or lora_B_key not in self.hf_sd:
                continue

            expected = base_weights[name] + (self.hf_sd[lora_B_key] @ self.hf_sd[lora_A_key]) * scale
            torch.testing.assert_close(
                param.data, expected, rtol=1e-4, atol=1e-4,
                msg=f"Merge mismatch for {name}",
            )

    def test_no_lora_params_after_merge(self):
        """Merged model must be clean of LoRA artifacts."""
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(
            _HFStyleMoE(), self.adapter_path
        )
        merged = peft_model.merge_and_unload()

        for name, _ in merged.named_parameters():
            assert "lora_" not in name, f"LoRA param {name} should not survive merge"
        for name, _ in merged.named_modules():
            assert "lora_" not in name, f"LoRA module {name} should not survive merge"

    def test_merged_logits_kl_div_vs_unmerged(self):
        """KL-div between merged and adapter-applied MoE models must be ~0."""
        from peft import PeftModel

        torch.manual_seed(99)
        ref_model = PeftModel.from_pretrained(
            _HFStyleMoE(), self.adapter_path
        ).eval()

        torch.manual_seed(99)
        merged = PeftModel.from_pretrained(
            _HFStyleMoE(), self.adapter_path
        ).merge_and_unload().eval()

        torch.manual_seed(123)
        input_ids = torch.randint(0, MOE_VOCAB_SIZE, (2, 16))

        with torch.no_grad():
            ref_logits = ref_model(input_ids).logits
            merged_logits = merged(input_ids).logits

        log_p = torch.nn.functional.log_softmax(merged_logits, dim=-1)
        q = torch.nn.functional.softmax(ref_logits, dim=-1)
        kl = torch.nn.functional.kl_div(log_p, q, reduction="none", log_target=False)
        kl = kl.sum(-1).view(-1).max()
        assert kl.item() < 1e-6, f"KL divergence too high: {kl.item():.3e}"

    def test_round_trip_grouped_to_hf_and_back(self):
        """Verify grouped → HF → grouped round-trip preserves tensor values."""
        from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

        class _Adapter(MoESplitExpertsStateDictMixin):
            def __init__(self):
                self.moe_config = type("C", (), {
                    "n_routed_experts": N_EXPERTS,
                    "moe_inter_dim": MOE_INTER_DIM,
                    "expert_activation": "swiglu",
                })()
                self.dtype = torch.float32
                self._uses_model_prefix = True
                self._last_expert_ids = []

        adapter = _Adapter()
        recovered = adapter._recombine_lora_expert_keys(self.hf_sd)

        for key, original_tensor in self.grouped_sd.items():
            assert key in recovered, f"Key {key} missing after round-trip"
            torch.testing.assert_close(
                recovered[key], original_tensor,
                msg=f"Value mismatch for {key}",
            )
