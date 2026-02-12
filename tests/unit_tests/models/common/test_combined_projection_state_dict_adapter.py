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

"""Tests for CombinedProjectionStateDictAdapter LoRA weight splitting in to_hf()."""

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from transformers import LlamaConfig, Phi3Config, Phi3ForCausalLM

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.addons import _extract_target_modules, _get_hf_peft_config
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState
from nemo_automodel.components.models.common.combined_projection.state_dict_adapter import (
    CombinedProjectionStateDictAdapter,
)
from nemo_automodel.components.models.llama.model import LlamaForCausalLM


def _make_config(
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    hidden_size=256,
):
    """Create a minimal config namespace for the adapter."""
    return SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        hidden_size=hidden_size,
    )


def _make_model_with_named_modules(module_names):
    """Build a dummy model whose ``named_modules`` yields the given names."""
    root = nn.Module()
    for name in module_names:
        parts = name.split(".")
        parent = root
        for part in parts[:-1]:
            if not hasattr(parent, part):
                setattr(parent, part, nn.Module())
            parent = getattr(parent, part)
        setattr(parent, parts[-1], nn.Identity())
    return root


class TestCombinedProjectionLoRASplitting:
    """Tests that to_hf() correctly splits LoRA adapter weights for combined projections."""

    def _adapter(self, **kwargs):
        return CombinedProjectionStateDictAdapter(_make_config(**kwargs))

    # QKV projection LoRA splitting
    def test_qkv_lora_b_weight_split(self):
        """lora_B weight (output-dimension) for qkv_proj should be split into q/k/v."""
        adapter = self._adapter(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=256,
        )
        # head_dim = 256/4 = 64; q_size = 4*64 = 256; kv_size = 2*64 = 128
        # total qkv output dim = 256 + 128 + 128 = 512
        rank = 8
        qkv_lora_b = torch.randn(512, rank)  # (out_features, rank)

        state_dict = {
            "base_model.model.model.layers.0.self_attn.qkv_proj.lora_B.default.weight": qkv_lora_b,
        }

        hf_sd = adapter.to_hf(state_dict)

        q_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
        k_key = "base_model.model.model.layers.0.self_attn.k_proj.lora_B.default.weight"
        v_key = "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight"

        assert q_key in hf_sd
        assert k_key in hf_sd
        assert v_key in hf_sd
        assert hf_sd[q_key].shape == (256, rank)
        assert hf_sd[k_key].shape == (128, rank)
        assert hf_sd[v_key].shape == (128, rank)
        # Ensure combined key is removed
        assert "base_model.model.model.layers.0.self_attn.qkv_proj.lora_B.default.weight" not in hf_sd

    def test_qkv_lora_a_weight_duplicated(self):
        """lora_A weight (input-dimension) for qkv_proj should be duplicated to q/k/v."""
        adapter = self._adapter(num_hidden_layers=1)
        rank = 8
        lora_a = torch.randn(rank, 256)  # (rank, in_features)

        state_dict = {
            "base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.default.weight": lora_a,
        }

        hf_sd = adapter.to_hf(state_dict)

        q_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        k_key = "base_model.model.model.layers.0.self_attn.k_proj.lora_A.default.weight"
        v_key = "base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight"

        assert q_key in hf_sd
        assert k_key in hf_sd
        assert v_key in hf_sd
        # lora_A is duplicated, all should be equal to original
        torch.testing.assert_close(hf_sd[q_key], lora_a)
        torch.testing.assert_close(hf_sd[k_key], lora_a)
        torch.testing.assert_close(hf_sd[v_key], lora_a)
        # Combined key must be removed
        assert "base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.default.weight" not in hf_sd

    # gate_up projection LoRA splitting
    def test_gate_up_lora_b_weight_split(self):
        """lora_B weight for gate_up_proj should be split in half into gate/up."""
        adapter = self._adapter(num_hidden_layers=1)
        rank = 8
        # intermediate_size is not in config, but gate_up weight is split 50/50
        intermediate_size = 512
        gate_up_lora_b = torch.randn(intermediate_size * 2, rank)

        state_dict = {
            "base_model.model.model.layers.0.mlp.gate_up_proj.lora_B.default.weight": gate_up_lora_b,
        }

        hf_sd = adapter.to_hf(state_dict)

        gate_key = "base_model.model.model.layers.0.mlp.gate_proj.lora_B.default.weight"
        up_key = "base_model.model.model.layers.0.mlp.up_proj.lora_B.default.weight"

        assert gate_key in hf_sd
        assert up_key in hf_sd
        assert hf_sd[gate_key].shape == (intermediate_size, rank)
        assert hf_sd[up_key].shape == (intermediate_size, rank)
        assert "base_model.model.model.layers.0.mlp.gate_up_proj.lora_B.default.weight" not in hf_sd

    def test_gate_up_lora_a_weight_duplicated(self):
        """lora_A weight for gate_up_proj should be duplicated to gate/up."""
        adapter = self._adapter(num_hidden_layers=1)
        rank = 8
        lora_a = torch.randn(rank, 256)

        state_dict = {
            "base_model.model.model.layers.0.mlp.gate_up_proj.lora_A.default.weight": lora_a,
        }

        hf_sd = adapter.to_hf(state_dict)

        gate_key = "base_model.model.model.layers.0.mlp.gate_proj.lora_A.default.weight"
        up_key = "base_model.model.model.layers.0.mlp.up_proj.lora_A.default.weight"

        assert gate_key in hf_sd
        assert up_key in hf_sd
        torch.testing.assert_close(hf_sd[gate_key], lora_a)
        torch.testing.assert_close(hf_sd[up_key], lora_a)
        # Combined key must be removed
        assert "base_model.model.model.layers.0.mlp.gate_up_proj.lora_A.default.weight" not in hf_sd

    # DoRA magnitude splitting
    def test_qkv_dora_magnitude_split(self):
        """DoRA lora_magnitude_vector for qkv_proj should be split like lora_B."""
        adapter = self._adapter(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=256,
        )
        # q_size=256, kv_size=128 => total=512
        magnitude = torch.randn(512)

        state_dict = {
            "base_model.model.model.layers.0.self_attn.qkv_proj.lora_magnitude_vector.default": magnitude,
        }

        hf_sd = adapter.to_hf(state_dict)

        q_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_magnitude_vector.default"
        k_key = "base_model.model.model.layers.0.self_attn.k_proj.lora_magnitude_vector.default"
        v_key = "base_model.model.model.layers.0.self_attn.v_proj.lora_magnitude_vector.default"

        assert q_key in hf_sd
        assert k_key in hf_sd
        assert v_key in hf_sd
        assert hf_sd[q_key].shape == (256,)
        assert hf_sd[k_key].shape == (128,)
        assert hf_sd[v_key].shape == (128,)
        # Combined key must be removed
        assert "base_model.model.model.layers.0.self_attn.qkv_proj.lora_magnitude_vector.default" not in hf_sd

    # Non-LoRA keys pass through
    def test_non_lora_keys_preserved(self):
        """Keys that don't match combined projections pass through unchanged."""
        adapter = self._adapter(num_hidden_layers=1)
        embed_weight = torch.randn(1024, 256)

        state_dict = {
            "base_model.model.model.embed_tokens.weight": embed_weight,
        }

        hf_sd = adapter.to_hf(state_dict)
        assert "base_model.model.model.embed_tokens.weight" in hf_sd
        torch.testing.assert_close(hf_sd["base_model.model.model.embed_tokens.weight"], embed_weight)

    # Mixed: base weights + LoRA weights
    def test_base_and_lora_weights_both_split(self):
        """Both base weights and LoRA weights for qkv_proj should be split."""
        adapter = self._adapter(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=256,
        )
        # q_size=256, kv_size=128 => total=512
        rank = 8
        qkv_base = torch.randn(512, 256)
        qkv_lora_a = torch.randn(rank, 256)
        qkv_lora_b = torch.randn(512, rank)

        state_dict = {
            "model.layers.0.self_attn.qkv_proj.weight": qkv_base,
            "base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.default.weight": qkv_lora_a,
            "base_model.model.model.layers.0.self_attn.qkv_proj.lora_B.default.weight": qkv_lora_b,
        }

        hf_sd = adapter.to_hf(state_dict)

        # Base weights split
        assert "model.layers.0.self_attn.q_proj.weight" in hf_sd
        assert "model.layers.0.self_attn.k_proj.weight" in hf_sd
        assert "model.layers.0.self_attn.v_proj.weight" in hf_sd
        assert hf_sd["model.layers.0.self_attn.q_proj.weight"].shape == (256, 256)
        assert hf_sd["model.layers.0.self_attn.k_proj.weight"].shape == (128, 256)

        # LoRA-A duplicated
        assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight" in hf_sd
        assert "base_model.model.model.layers.0.self_attn.k_proj.lora_A.default.weight" in hf_sd
        assert "base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight" in hf_sd

        # LoRA-B split
        assert "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight" in hf_sd
        assert "base_model.model.model.layers.0.self_attn.k_proj.lora_B.default.weight" in hf_sd
        assert "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight" in hf_sd
        assert hf_sd["base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"].shape == (256, rank)
        assert hf_sd["base_model.model.model.layers.0.self_attn.k_proj.lora_B.default.weight"].shape == (128, rank)

        # No combined keys remain
        assert not any("qkv_proj" in k for k in hf_sd)


# ---------------------------------------------------------------------------
# Functional test: 2-layer Llama model + LoRA → simulate save → verify split
# ---------------------------------------------------------------------------


class TestLlamaLoRAFunctionalSplit:
    """End-to-end test: build a tiny Llama, apply LoRA, simulate the PEFT save
    pipeline, and verify that combined projections are split correctly and that
    replicated weights (lora_A) are identical across the split projections,
    which guarantees the effective alpha is identical.
    """

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _make_tiny_llama():
        """Return a 2-layer Llama model with LoRA on qkv_proj + gate_up_proj."""
        config = LlamaConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=32,
        )
        model = LlamaForCausalLM.from_config(config)
        peft_cfg = PeftConfig(
            target_modules=["*qkv_proj", "*gate_up_proj"],
            dim=8,
            alpha=32,
        )
        n = apply_lora_to_linear_modules(model, peft_cfg)
        assert n > 0, f"Expected LoRA patches, got {n}"
        return model, peft_cfg

    @staticmethod
    def _simulate_peft_save(model):
        """Reproduce the PEFT save flow without distributed / disk I/O.

        1. Collect trainable params  (ModelState.state_dict with is_peft=True)
        2. Add ``base_model.model.`` prefix  (_add_outer_prefix)
        3. Convert via ``adapter.to_hf()``    (_maybe_adapt_state_dict_to_hf)
        """
        peft_sd = {k: v for k, v in model.named_parameters() if v.requires_grad}
        prefixed = {f"base_model.model.{k}": v for k, v in peft_sd.items()}
        return model.state_dict_adapter.to_hf(prefixed)

    # -- tests ------------------------------------------------------------

    def test_no_combined_keys_remain(self):
        """After to_hf(), no qkv_proj or gate_up_proj keys should be present."""
        model, _ = self._make_tiny_llama()
        hf_sd = self._simulate_peft_save(model)

        combined = [k for k in hf_sd if "qkv_proj" in k or "gate_up_proj" in k]
        assert combined == [], f"Combined keys should be split, found: {combined}"

    def test_no_combined_lora_keys_in_state_dict(self):
        """Explicitly verify that every combined-projection LoRA variant is absent."""
        model, _ = self._make_tiny_llama()
        hf_sd = self._simulate_peft_save(model)

        # Every combined fragment that must NOT appear in any key
        forbidden_fragments = [
            "qkv_proj.lora_A",
            "qkv_proj.lora_B",
            "qkv_proj.lora_magnitude",
            "gate_up_proj.lora_A",
            "gate_up_proj.lora_B",
            "gate_up_proj.lora_magnitude",
            # Also check bare combined projection names (covers base weight keys)
            ".qkv_proj.",
            ".gate_up_proj.",
        ]
        for fragment in forbidden_fragments:
            offending = [k for k in hf_sd if fragment in k]
            assert offending == [], (
                f"Found forbidden fragment '{fragment}' in converted state dict keys: {offending}"
            )

    def test_split_qkv_lora_a_identical(self):
        """lora_A weights for q/k/v must be identical (replicated, not split)."""
        model, _ = self._make_tiny_llama()
        hf_sd = self._simulate_peft_save(model)

        # Discover the actual lora_A suffix (may or may not include ".default")
        sample_keys = [k for k in hf_sd if "q_proj" in k and "lora_A" in k]
        assert sample_keys, f"No q_proj lora_A key found in: {list(hf_sd.keys())}"
        # Extract the suffix after "q_proj."  e.g. "lora_A.weight" or "lora_A.default.weight"
        lora_a_suffix = sample_keys[0].split("q_proj.", 1)[1]

        for layer_idx in range(2):
            pfx = f"base_model.model.model.layers.{layer_idx}.self_attn"
            q_a = hf_sd[f"{pfx}.q_proj.{lora_a_suffix}"]
            k_a = hf_sd[f"{pfx}.k_proj.{lora_a_suffix}"]
            v_a = hf_sd[f"{pfx}.v_proj.{lora_a_suffix}"]

            torch.testing.assert_close(q_a, k_a, msg=f"layer {layer_idx}: q vs k lora_A differ")
            torch.testing.assert_close(q_a, v_a, msg=f"layer {layer_idx}: q vs v lora_A differ")

    def test_alpha_identical_across_qkv(self):
        """Because lora_A is replicated, the effective alpha (= scale × rank)
        is the same for every split projection.  Verify bit-exact equality of
        the replicated weights and confirm the module-level scale is consistent.
        """
        model, peft_cfg = self._make_tiny_llama()
        hf_sd = self._simulate_peft_save(model)

        expected_scale = peft_cfg.alpha / peft_cfg.dim  # 32 / 8 = 4.0

        # Discover the actual lora_A suffix
        sample_keys = [k for k in hf_sd if "q_proj" in k and "lora_A" in k]
        lora_a_suffix = sample_keys[0].split("q_proj.", 1)[1]

        for layer_idx in range(2):
            pfx = f"base_model.model.model.layers.{layer_idx}.self_attn"
            q_a = hf_sd[f"{pfx}.q_proj.{lora_a_suffix}"]
            k_a = hf_sd[f"{pfx}.k_proj.{lora_a_suffix}"]
            v_a = hf_sd[f"{pfx}.v_proj.{lora_a_suffix}"]

            # Bit-exact equality (not just close) proves identical alpha effect
            assert torch.equal(q_a, k_a), f"layer {layer_idx}: q/k lora_A not bit-equal"
            assert torch.equal(q_a, v_a), f"layer {layer_idx}: q/v lora_A not bit-equal"

            # The original module should carry a single, consistent scale
            qkv_mod = model.model.layers[layer_idx].self_attn.qkv_proj
            assert qkv_mod.scale == expected_scale, (
                f"layer {layer_idx}: expected scale {expected_scale}, got {qkv_mod.scale}"
            )

    def test_lora_b_shapes_after_qkv_split(self):
        """lora_B output dims must match individual projection sizes."""
        model, peft_cfg = self._make_tiny_llama()
        hf_sd = self._simulate_peft_save(model)

        rank = peft_cfg.dim  # 8

        # Discover the actual lora_B suffix
        sample_keys = [k for k in hf_sd if "q_proj" in k and "lora_B" in k]
        lora_b_suffix = sample_keys[0].split("q_proj.", 1)[1]

        # head_dim=64/4=16  →  q_size=4×16=64, kv_size=2×16=32
        for layer_idx in range(2):
            pfx = f"base_model.model.model.layers.{layer_idx}.self_attn"
            assert hf_sd[f"{pfx}.q_proj.{lora_b_suffix}"].shape == (64, rank)
            assert hf_sd[f"{pfx}.k_proj.{lora_b_suffix}"].shape == (32, rank)
            assert hf_sd[f"{pfx}.v_proj.{lora_b_suffix}"].shape == (32, rank)

    def test_gate_up_split_and_lora_a_identical(self):
        """gate_up_proj should be split into gate_proj / up_proj with lora_A
        replicated and lora_B split equally."""
        model, peft_cfg = self._make_tiny_llama()
        hf_sd = self._simulate_peft_save(model)

        rank = peft_cfg.dim  # 8

        # Discover the actual lora_A / lora_B suffixes from gate_proj keys
        gate_a_keys = [k for k in hf_sd if "gate_proj" in k and "lora_A" in k]
        gate_b_keys = [k for k in hf_sd if "gate_proj" in k and "lora_B" in k]
        lora_a_suffix = gate_a_keys[0].split("gate_proj.", 1)[1]
        lora_b_suffix = gate_b_keys[0].split("gate_proj.", 1)[1]

        for layer_idx in range(2):
            pfx = f"base_model.model.model.layers.{layer_idx}.mlp"

            # lora_A duplicated
            gate_a = hf_sd[f"{pfx}.gate_proj.{lora_a_suffix}"]
            up_a = hf_sd[f"{pfx}.up_proj.{lora_a_suffix}"]
            torch.testing.assert_close(gate_a, up_a)

            # lora_B split: each half = intermediate_size = 128
            assert hf_sd[f"{pfx}.gate_proj.{lora_b_suffix}"].shape == (128, rank)
            assert hf_sd[f"{pfx}.up_proj.{lora_b_suffix}"].shape == (128, rank)

    def test_extract_target_modules_returns_split_names(self):
        """_extract_target_modules should emit HF-compatible split names."""
        model, _ = self._make_tiny_llama()
        target_modules = _extract_target_modules(model)

        # No combined names
        for m in target_modules:
            assert "qkv_proj" not in m, f"Combined name in target_modules: {m}"
            assert "gate_up_proj" not in m, f"Combined name in target_modules: {m}"

        # All split names present for both layers
        for layer_idx in range(2):
            pre = f"model.layers.{layer_idx}"
            assert f"{pre}.self_attn.q_proj" in target_modules
            assert f"{pre}.self_attn.k_proj" in target_modules
            assert f"{pre}.self_attn.v_proj" in target_modules
            assert f"{pre}.mlp.gate_proj" in target_modules
            assert f"{pre}.mlp.up_proj" in target_modules


# ---------------------------------------------------------------------------
# Identity mapping tests (Phi3/Phi4-style: fused projections kept as-is)
# ---------------------------------------------------------------------------


class _IdentityMappingAdapter(CombinedProjectionStateDictAdapter):
    """Adapter subclass with identity fused_modules_mapping (no split)."""

    _FUSED_MODULES_MAPPING = {
        "qkv_proj": ["qkv_proj"],
        "gate_up_proj": ["gate_up_proj"],
    }


class TestIdentityMappingDoesNotSplit:
    """Verify that identity-mapped adapters leave fused LoRA keys untouched."""

    @staticmethod
    def _adapter(**kwargs):
        return _IdentityMappingAdapter(_make_config(**kwargs))

    def test_qkv_lora_keys_preserved(self):
        """qkv_proj LoRA keys should pass through unchanged with identity mapping."""
        adapter = self._adapter(num_hidden_layers=1)
        rank = 8
        qkv_lora_a = torch.randn(rank, 256)
        qkv_lora_b = torch.randn(512, rank)

        state_dict = {
            "base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.default.weight": qkv_lora_a,
            "base_model.model.model.layers.0.self_attn.qkv_proj.lora_B.default.weight": qkv_lora_b,
        }

        hf_sd = adapter.to_hf(state_dict)

        # Fused keys should remain
        assert "base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.default.weight" in hf_sd
        assert "base_model.model.model.layers.0.self_attn.qkv_proj.lora_B.default.weight" in hf_sd
        # Split keys should NOT exist (use ".q_proj." to avoid matching "qkv_proj")
        assert not any(".q_proj." in k for k in hf_sd)
        assert not any(".k_proj." in k for k in hf_sd)
        assert not any(k for k in hf_sd if ".v_proj." in k and "qkv_proj" not in k)

    def test_gate_up_lora_keys_preserved(self):
        """gate_up_proj LoRA keys should pass through unchanged with identity mapping."""
        adapter = self._adapter(num_hidden_layers=1)
        rank = 8
        gate_up_lora_a = torch.randn(rank, 256)
        gate_up_lora_b = torch.randn(1024, rank)

        state_dict = {
            "base_model.model.model.layers.0.mlp.gate_up_proj.lora_A.default.weight": gate_up_lora_a,
            "base_model.model.model.layers.0.mlp.gate_up_proj.lora_B.default.weight": gate_up_lora_b,
        }

        hf_sd = adapter.to_hf(state_dict)

        # Fused keys should remain
        assert "base_model.model.model.layers.0.mlp.gate_up_proj.lora_A.default.weight" in hf_sd
        assert "base_model.model.model.layers.0.mlp.gate_up_proj.lora_B.default.weight" in hf_sd
        # Split keys should NOT exist (use ".gate_proj." / ".up_proj." to avoid matching "gate_up_proj")
        assert not any(k for k in hf_sd if ".gate_proj." in k and "gate_up_proj" not in k)
        assert not any(k for k in hf_sd if ".up_proj." in k and "gate_up_proj" not in k)

    def test_base_weights_not_split_with_identity(self):
        """Base qkv_proj / gate_up_proj weights should NOT be split with identity mapping."""
        adapter = self._adapter(num_hidden_layers=1)
        qkv_weight = torch.randn(512, 256)
        gate_up_weight = torch.randn(1024, 256)

        state_dict = {
            "model.layers.0.self_attn.qkv_proj.weight": qkv_weight,
            "model.layers.0.mlp.gate_up_proj.weight": gate_up_weight,
        }

        hf_sd = adapter.to_hf(state_dict)

        # Fused keys should remain (passed through as "other" keys)
        assert "model.layers.0.self_attn.qkv_proj.weight" in hf_sd
        assert "model.layers.0.mlp.gate_up_proj.weight" in hf_sd
        torch.testing.assert_close(hf_sd["model.layers.0.self_attn.qkv_proj.weight"], qkv_weight)
        torch.testing.assert_close(hf_sd["model.layers.0.mlp.gate_up_proj.weight"], gate_up_weight)

    def test_from_hf_does_not_combine_with_identity(self):
        """from_hf() should NOT combine q/k/v into qkv_proj when using identity mapping."""
        adapter = self._adapter(num_hidden_layers=1)
        qkv_weight = torch.randn(512, 256)

        hf_state_dict = {
            "model.layers.0.self_attn.qkv_proj.weight": qkv_weight,
        }

        native_sd = adapter.from_hf(hf_state_dict)
        # With identity mapping, qkv_proj should pass through
        assert "model.layers.0.self_attn.qkv_proj.weight" in native_sd

    def test_recombine_is_noop_with_identity(self):
        """_recombine_split_projection_keys should be no-op with identity mapping."""
        adapter = self._adapter(num_hidden_layers=1)
        rank = 8
        # Simulate a state dict that already has fused LoRA keys
        state_dict = {
            "base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.default.weight": torch.randn(rank, 256),
            "base_model.model.model.layers.0.mlp.gate_up_proj.lora_B.default.weight": torch.randn(1024, rank),
        }
        original_keys = set(state_dict.keys())

        adapter._recombine_split_projection_keys(state_dict)

        # Keys should be unchanged
        assert set(state_dict.keys()) == original_keys

    def test_fused_modules_mapping_property(self):
        """fused_modules_mapping should return the class-level mapping."""
        adapter = self._adapter(num_hidden_layers=1)
        mapping = adapter.fused_modules_mapping
        assert mapping == {"qkv_proj": ["qkv_proj"], "gate_up_proj": ["gate_up_proj"]}

    def test_default_adapter_mapping_is_llama_style(self):
        """Default CombinedProjectionStateDictAdapter should use Llama-style split."""
        adapter = CombinedProjectionStateDictAdapter(_make_config())
        mapping = adapter.fused_modules_mapping
        assert mapping == {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"],
        }


# ---------------------------------------------------------------------------
# _extract_target_modules: model without state_dict_adapter
# ---------------------------------------------------------------------------


class TestExtractTargetModulesNoAdapter:
    """Verify _extract_target_modules keeps fused names for models without a state dict adapter."""

    def test_fused_names_kept_without_adapter(self):
        """Without a state_dict_adapter, fused module names should NOT be expanded."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.qkv_proj.lora_A",
                "model.layers.0.self_attn.qkv_proj.lora_B",
                "model.layers.0.mlp.gate_up_proj.lora_A",
            ]
        )
        # No state_dict_adapter on this model
        assert not hasattr(model, "state_dict_adapter")

        result = _extract_target_modules(model)

        # Fused names should be preserved
        assert "model.layers.0.self_attn.qkv_proj" in result
        assert "model.layers.0.mlp.gate_up_proj" in result
        # Individual split names should NOT appear (exact match, not substring)
        assert "model.layers.0.self_attn.q_proj" not in result
        assert "model.layers.0.self_attn.k_proj" not in result
        assert "model.layers.0.self_attn.v_proj" not in result
        assert "model.layers.0.mlp.gate_proj" not in result
        assert "model.layers.0.mlp.up_proj" not in result

    def test_identity_adapter_keeps_fused_names(self):
        """With an identity-mapping adapter, fused names should NOT be expanded."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.qkv_proj.lora_A",
                "model.layers.0.mlp.gate_up_proj.lora_A",
            ]
        )
        # Attach an identity-mapping adapter
        model.state_dict_adapter = _IdentityMappingAdapter(_make_config(num_hidden_layers=1))

        result = _extract_target_modules(model)

        assert "model.layers.0.self_attn.qkv_proj" in result
        assert "model.layers.0.mlp.gate_up_proj" in result
        assert all("q_proj" not in m for m in result)

    def test_regular_modules_unaffected_without_adapter(self):
        """Non-fused module names pass through regardless of adapter presence."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.o_proj.lora_A",
                "model.layers.0.mlp.down_proj.lora_A",
            ]
        )
        result = _extract_target_modules(model)
        assert "model.layers.0.self_attn.o_proj" in result
        assert "model.layers.0.mlp.down_proj" in result


# ---------------------------------------------------------------------------
# Partial identity mapping (e.g. split QKV but keep gate_up_proj fused)
# ---------------------------------------------------------------------------


class _PartialIdentityAdapter(CombinedProjectionStateDictAdapter):
    """Adapter that splits QKV but keeps gate_up_proj as identity (Phi3-like)."""

    _FUSED_MODULES_MAPPING = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_up_proj"],
    }


class TestPartialIdentityMapping:
    """Verify that partial identity mappings work (split some, keep others)."""

    @staticmethod
    def _adapter(**kwargs):
        return _PartialIdentityAdapter(_make_config(**kwargs))

    def test_qkv_split_but_gate_up_preserved(self):
        """QKV should be split while gate_up_proj stays fused."""
        adapter = self._adapter(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=256,
        )
        rank = 8
        state_dict = {
            "base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.default.weight": torch.randn(rank, 256),
            "base_model.model.model.layers.0.self_attn.qkv_proj.lora_B.default.weight": torch.randn(512, rank),
            "base_model.model.model.layers.0.mlp.gate_up_proj.lora_A.default.weight": torch.randn(rank, 256),
            "base_model.model.model.layers.0.mlp.gate_up_proj.lora_B.default.weight": torch.randn(1024, rank),
        }

        hf_sd = adapter.to_hf(state_dict)

        # QKV should be split
        assert any("q_proj" in k for k in hf_sd)
        assert any("k_proj" in k for k in hf_sd)
        assert any("v_proj" in k for k in hf_sd)
        assert not any("qkv_proj" in k for k in hf_sd)

        # gate_up_proj should remain fused
        assert any("gate_up_proj" in k for k in hf_sd)
        assert not any(k for k in hf_sd if "gate_proj" in k and "gate_up_proj" not in k)
        assert not any(k for k in hf_sd if "up_proj" in k and "gate_up_proj" not in k)

    def test_extract_target_modules_partial_identity(self):
        """_extract_target_modules should split QKV but keep gate_up_proj for partial identity."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.qkv_proj.lora_A",
                "model.layers.0.mlp.gate_up_proj.lora_A",
            ]
        )
        model.state_dict_adapter = _PartialIdentityAdapter(_make_config(num_hidden_layers=1))

        result = _extract_target_modules(model)

        # QKV expanded
        assert "model.layers.0.self_attn.q_proj" in result
        assert "model.layers.0.self_attn.k_proj" in result
        assert "model.layers.0.self_attn.v_proj" in result
        # gate_up_proj kept
        assert "model.layers.0.mlp.gate_up_proj" in result


# ---------------------------------------------------------------------------
# Constructor-parameter override & comprehensive round-trip test
#
# Verifies the scenario where a future model's Automodel implementation fuses
# QKV (needing split for HF) but HF already has a fused gate_up_proj.
# Uses the *constructor* ``fused_modules_mapping`` argument (no subclass).
# ---------------------------------------------------------------------------


class TestConstructorPackedModulesMapping:
    """Verify that passing ``fused_modules_mapping`` to __init__ works
    identically to subclass overrides, and exercise the full round-trip
    (base weights + LoRA) for a partial-identity configuration:
      - QKV: split (Llama-style)
      - gate_up_proj: identity (Phi-style -- HF already fuses MLP)
    """

    PARTIAL_MAPPING = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_up_proj"],
    }

    @staticmethod
    def _cfg(**kw):
        defaults = dict(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=256,
        )
        defaults.update(kw)
        return _make_config(**defaults)

    def _adapter(self, **kw):
        return CombinedProjectionStateDictAdapter(
            self._cfg(**kw),
            fused_modules_mapping=self.PARTIAL_MAPPING,
        )

    # -- constructor precedence ------------------------------------------------

    def test_init_arg_overrides_class_default(self):
        """Constructor mapping should take precedence over _FUSED_MODULES_MAPPING."""
        adapter = self._adapter()
        assert adapter.fused_modules_mapping == self.PARTIAL_MAPPING

    def test_init_none_falls_back_to_class_default(self):
        """When init arg is None, the class-level default is used."""
        adapter = CombinedProjectionStateDictAdapter(self._cfg())
        assert adapter.fused_modules_mapping == CombinedProjectionStateDictAdapter._FUSED_MODULES_MAPPING

    # -- to_hf: base weights ---------------------------------------------------

    def test_to_hf_splits_qkv_base_weights(self):
        """Base qkv_proj.weight must be split into q/k/v."""
        adapter = self._adapter()
        hidden = 256
        q_size = 4 * 64  # num_attention_heads * head_dim
        kv_size = 2 * 64  # num_key_value_heads * head_dim
        qkv_size = q_size + 2 * kv_size

        sd = {"model.layers.0.self_attn.qkv_proj.weight": torch.randn(qkv_size, hidden)}
        hf = adapter.to_hf(sd)

        assert "model.layers.0.self_attn.q_proj.weight" in hf
        assert "model.layers.0.self_attn.k_proj.weight" in hf
        assert "model.layers.0.self_attn.v_proj.weight" in hf
        assert "model.layers.0.self_attn.qkv_proj.weight" not in hf

        # Verify shapes
        assert hf["model.layers.0.self_attn.q_proj.weight"].shape == (q_size, hidden)
        assert hf["model.layers.0.self_attn.k_proj.weight"].shape == (kv_size, hidden)
        assert hf["model.layers.0.self_attn.v_proj.weight"].shape == (kv_size, hidden)

    def test_to_hf_preserves_gate_up_base_weights(self):
        """Base gate_up_proj.weight must pass through unsplit (identity mapping)."""
        adapter = self._adapter()
        hidden = 256
        intermediate = 512  # gate_up_proj dim0 = 2 * intermediate_size

        sd = {"model.layers.0.mlp.gate_up_proj.weight": torch.randn(intermediate, hidden)}
        hf = adapter.to_hf(sd)

        assert "model.layers.0.mlp.gate_up_proj.weight" in hf
        # Split names must NOT appear
        assert "model.layers.0.mlp.gate_proj.weight" not in hf
        assert "model.layers.0.mlp.up_proj.weight" not in hf
        # Tensor should be identical (not a copy)
        assert torch.equal(hf["model.layers.0.mlp.gate_up_proj.weight"], sd["model.layers.0.mlp.gate_up_proj.weight"])

    # -- to_hf: LoRA keys -----------------------------------------------------

    def test_to_hf_splits_qkv_lora_keys(self):
        """QKV LoRA keys must be split into q/k/v."""
        adapter = self._adapter()
        rank = 8
        hidden = 256
        q_size = 4 * 64
        kv_size = 2 * 64
        qkv_size = q_size + 2 * kv_size

        sd = {
            "base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.default.weight": torch.randn(rank, hidden),
            "base_model.model.model.layers.0.self_attn.qkv_proj.lora_B.default.weight": torch.randn(qkv_size, rank),
        }
        hf = adapter.to_hf(sd)

        # lora_A duplicated to q/k/v
        for proj in ["q_proj", "k_proj", "v_proj"]:
            assert f"base_model.model.model.layers.0.self_attn.{proj}.lora_A.default.weight" in hf

        # lora_B split by size
        assert hf["base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"].shape[0] == q_size
        assert hf["base_model.model.model.layers.0.self_attn.k_proj.lora_B.default.weight"].shape[0] == kv_size

        # No fused keys left
        assert not any("qkv_proj" in k for k in hf)

    def test_to_hf_preserves_gate_up_lora_keys(self):
        """gate_up_proj LoRA keys must remain fused (identity mapping)."""
        adapter = self._adapter()
        rank = 8
        sd = {
            "base_model.model.model.layers.0.mlp.gate_up_proj.lora_A.default.weight": torch.randn(rank, 256),
            "base_model.model.model.layers.0.mlp.gate_up_proj.lora_B.default.weight": torch.randn(1024, rank),
        }
        hf = adapter.to_hf(sd)

        assert "base_model.model.model.layers.0.mlp.gate_up_proj.lora_A.default.weight" in hf
        assert "base_model.model.model.layers.0.mlp.gate_up_proj.lora_B.default.weight" in hf
        assert not any(k for k in hf if ".gate_proj." in k and "gate_up_proj" not in k)
        assert not any(k for k in hf if ".up_proj." in k and "gate_up_proj" not in k)

    # -- from_hf: base weights -------------------------------------------------

    def test_from_hf_combines_split_qkv(self):
        """Split q/k/v weights from HF must be combined back to qkv_proj."""
        adapter = self._adapter()
        hidden = 256
        q_size = 4 * 64
        kv_size = 2 * 64

        hf_sd = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(q_size, hidden),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(kv_size, hidden),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(kv_size, hidden),
        }
        combined = adapter.from_hf(hf_sd)

        assert "model.layers.0.self_attn.qkv_proj.weight" in combined
        assert combined["model.layers.0.self_attn.qkv_proj.weight"].shape == (q_size + 2 * kv_size, hidden)
        # Split keys removed
        assert "model.layers.0.self_attn.q_proj.weight" not in combined

    def test_from_hf_preserves_fused_gate_up(self):
        """gate_up_proj.weight from HF must pass through unchanged (identity)."""
        adapter = self._adapter()
        hidden = 256
        gate_up_weight = torch.randn(1024, hidden)

        hf_sd = {"model.layers.0.mlp.gate_up_proj.weight": gate_up_weight}
        combined = adapter.from_hf(hf_sd)

        assert "model.layers.0.mlp.gate_up_proj.weight" in combined
        assert torch.equal(combined["model.layers.0.mlp.gate_up_proj.weight"], gate_up_weight)
        # Must NOT look for gate_proj + up_proj to combine
        assert "model.layers.0.mlp.gate_proj.weight" not in combined
        assert "model.layers.0.mlp.up_proj.weight" not in combined

    # -- from_hf: LoRA keys ----------------------------------------------------

    def test_from_hf_recombines_split_qkv_lora(self):
        """Split q/k/v LoRA keys must be recombined into qkv_proj LoRA."""
        adapter = self._adapter()
        rank = 8
        hidden = 256
        q_size = 4 * 64
        kv_size = 2 * 64

        shared_a = torch.randn(rank, hidden)
        hf_sd = {
            "model.layers.0.self_attn.q_proj.lora_A.default.weight": shared_a,
            "model.layers.0.self_attn.k_proj.lora_A.default.weight": shared_a.clone(),
            "model.layers.0.self_attn.v_proj.lora_A.default.weight": shared_a.clone(),
            "model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.randn(q_size, rank),
            "model.layers.0.self_attn.k_proj.lora_B.default.weight": torch.randn(kv_size, rank),
            "model.layers.0.self_attn.v_proj.lora_B.default.weight": torch.randn(kv_size, rank),
        }
        combined = adapter.from_hf(hf_sd)

        assert "model.layers.0.self_attn.qkv_proj.lora_A.default.weight" in combined
        assert "model.layers.0.self_attn.qkv_proj.lora_B.default.weight" in combined
        assert combined["model.layers.0.self_attn.qkv_proj.lora_B.default.weight"].shape[0] == q_size + 2 * kv_size

    def test_from_hf_preserves_fused_gate_up_lora(self):
        """gate_up_proj LoRA keys from HF must pass through unchanged (identity)."""
        adapter = self._adapter()
        rank = 8
        lora_a = torch.randn(rank, 256)
        lora_b = torch.randn(1024, rank)

        hf_sd = {
            "model.layers.0.mlp.gate_up_proj.lora_A.default.weight": lora_a,
            "model.layers.0.mlp.gate_up_proj.lora_B.default.weight": lora_b,
        }
        combined = adapter.from_hf(hf_sd)

        assert "model.layers.0.mlp.gate_up_proj.lora_A.default.weight" in combined
        assert torch.equal(combined["model.layers.0.mlp.gate_up_proj.lora_A.default.weight"], lora_a)
        assert torch.equal(combined["model.layers.0.mlp.gate_up_proj.lora_B.default.weight"], lora_b)

    # -- full round-trip -------------------------------------------------------

    def test_round_trip_base_weights(self):
        """from_hf(to_hf(sd)) must recover the original base weights."""
        adapter = self._adapter()
        hidden = 256
        q_size = 4 * 64
        kv_size = 2 * 64
        qkv_size = q_size + 2 * kv_size

        original_qkv = torch.randn(qkv_size, hidden)
        original_gate_up = torch.randn(1024, hidden)

        sd = {
            "model.layers.0.self_attn.qkv_proj.weight": original_qkv,
            "model.layers.0.mlp.gate_up_proj.weight": original_gate_up,
            "model.layers.0.mlp.down_proj.weight": torch.randn(hidden, 512),
        }

        hf = adapter.to_hf(sd)
        recovered = adapter.from_hf(hf)

        # QKV: round-trip via split/combine must be numerically identical
        assert torch.allclose(recovered["model.layers.0.self_attn.qkv_proj.weight"], original_qkv)
        # gate_up_proj: identity pass-through, must be identical
        assert torch.equal(recovered["model.layers.0.mlp.gate_up_proj.weight"], original_gate_up)
        # Other keys pass through
        assert "model.layers.0.mlp.down_proj.weight" in recovered

    def test_round_trip_lora_keys(self):
        """from_hf(to_hf(sd)) must recover LoRA keys for partial-identity mapping."""
        adapter = self._adapter()
        rank = 8
        hidden = 256
        q_size = 4 * 64
        kv_size = 2 * 64
        qkv_size = q_size + 2 * kv_size

        original = {
            "base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.default.weight": torch.randn(rank, hidden),
            "base_model.model.model.layers.0.self_attn.qkv_proj.lora_B.default.weight": torch.randn(qkv_size, rank),
            "base_model.model.model.layers.0.mlp.gate_up_proj.lora_A.default.weight": torch.randn(rank, hidden),
            "base_model.model.model.layers.0.mlp.gate_up_proj.lora_B.default.weight": torch.randn(1024, rank),
        }

        hf = adapter.to_hf(dict(original))  # copy to avoid mutation
        recovered = adapter.from_hf(hf)

        # QKV LoRA round-trip
        assert "base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.default.weight" in recovered
        assert "base_model.model.model.layers.0.self_attn.qkv_proj.lora_B.default.weight" in recovered
        # lora_A is duplicated during split, deduplicated during recombine -- should match
        assert torch.allclose(
            recovered["base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.default.weight"],
            original["base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.default.weight"],
        )
        # lora_B is split then concatenated -- should match
        assert torch.allclose(
            recovered["base_model.model.model.layers.0.self_attn.qkv_proj.lora_B.default.weight"],
            original["base_model.model.model.layers.0.self_attn.qkv_proj.lora_B.default.weight"],
        )

        # gate_up_proj LoRA: identity pass-through both ways, must be exactly equal
        assert torch.equal(
            recovered["base_model.model.model.layers.0.mlp.gate_up_proj.lora_A.default.weight"],
            original["base_model.model.model.layers.0.mlp.gate_up_proj.lora_A.default.weight"],
        )
        assert torch.equal(
            recovered["base_model.model.model.layers.0.mlp.gate_up_proj.lora_B.default.weight"],
            original["base_model.model.model.layers.0.mlp.gate_up_proj.lora_B.default.weight"],
        )

    # -- _extract_target_modules -----------------------------------------------

    def test_extract_target_modules_via_constructor_mapping(self):
        """_extract_target_modules must respect the constructor-supplied mapping."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.qkv_proj.lora_A",
                "model.layers.0.mlp.gate_up_proj.lora_A",
            ]
        )
        model.state_dict_adapter = self._adapter()

        result = _extract_target_modules(model)

        # QKV expanded
        assert "model.layers.0.self_attn.q_proj" in result
        assert "model.layers.0.self_attn.k_proj" in result
        assert "model.layers.0.self_attn.v_proj" in result
        assert "model.layers.0.self_attn.qkv_proj" not in result
        # gate_up_proj kept (identity)
        assert "model.layers.0.mlp.gate_up_proj" in result


# ---------------------------------------------------------------------------
# Functional test: Phi3 (identity mapping) -- LoRA adapters must keep fused
# names and be loadable by HF PEFT.
#
# Phi3 uses fused qkv_proj / gate_up_proj in its HF implementation, so unlike
# Llama the adapter weights must NOT be split.  This verifies NFS-678.
# ---------------------------------------------------------------------------

TINY_PHI3_CONFIG = dict(
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    hidden_size=64,
    intermediate_size=128,
    vocab_size=256,
    max_position_embeddings=128,
    pad_token_id=0,
)


class TestPhi3LoRAFunctionalIdentity:
    """End-to-end test: build a tiny Phi3 (fused QKV/MLP in HF), apply LoRA,
    simulate the PEFT save pipeline, and verify that fused projection names
    are kept (not split) so vLLM/TensorRT-LLM with identity
    ``fused_modules_mapping`` can load them.
    """

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _make_tiny_phi3():
        """Return a 2-layer Phi3 model with LoRA on qkv_proj + gate_up_proj.

        Phi3ForCausalLM is a vanilla HF model (no custom state_dict_adapter),
        which means _extract_target_modules will NOT expand fused names.
        """
        config = Phi3Config(**TINY_PHI3_CONFIG)
        model = Phi3ForCausalLM(config)
        peft_cfg = PeftConfig(
            target_modules=["*qkv_proj", "*gate_up_proj"],
            dim=8,
            alpha=32,
        )
        n = apply_lora_to_linear_modules(model, peft_cfg)
        assert n > 0, f"Expected LoRA patches, got {n}"
        return model, peft_cfg

    @staticmethod
    def _simulate_peft_save(model):
        """Reproduce the PEFT save flow for a model WITHOUT a state_dict_adapter.

        1. Collect trainable params  (named_parameters with requires_grad)
        2. Add ``base_model.model.`` prefix
        3. No adapter.to_hf() — model has no state_dict_adapter, so
           _maybe_adapt_state_dict_to_hf is a no-op.
        """
        peft_sd = {k: v for k, v in model.named_parameters() if v.requires_grad}
        return {f"base_model.model.{k}": v for k, v in peft_sd.items()}

    # -- tests ------------------------------------------------------------

    def test_no_state_dict_adapter(self):
        """Phi3ForCausalLM should NOT have a state_dict_adapter."""
        model, _ = self._make_tiny_phi3()
        assert not hasattr(model, "state_dict_adapter"), (
            "Phi3ForCausalLM should not have a state_dict_adapter"
        )

    def test_fused_keys_preserved(self):
        """After simulated save, qkv_proj and gate_up_proj keys must remain fused."""
        model, _ = self._make_tiny_phi3()
        hf_sd = self._simulate_peft_save(model)

        # ALL LoRA keys must contain fused projection names
        lora_keys = [k for k in hf_sd if "lora" in k.lower()]
        assert len(lora_keys) > 0, "Expected LoRA keys"

        for k in lora_keys:
            assert "qkv_proj" in k or "gate_up_proj" in k, (
                f"Expected fused projection name in key: {k}"
            )

        # Individual split names must NOT appear
        split_only_keys = [
            k for k in hf_sd
            if (".q_proj." in k or ".k_proj." in k or ".v_proj." in k)
            and "qkv_proj" not in k
        ]
        assert split_only_keys == [], (
            f"Split projection keys should NOT exist for Phi3: {split_only_keys}"
        )

    def test_no_split_keys_in_gate_up(self):
        """gate_up_proj must NOT be split into gate_proj / up_proj."""
        model, _ = self._make_tiny_phi3()
        hf_sd = self._simulate_peft_save(model)

        gate_only = [k for k in hf_sd if ".gate_proj." in k and "gate_up_proj" not in k]
        up_only = [k for k in hf_sd if ".up_proj." in k and "gate_up_proj" not in k]
        assert gate_only == [], f"Unexpected gate_proj keys: {gate_only}"
        assert up_only == [], f"Unexpected up_proj keys: {up_only}"

    def test_extract_target_modules_returns_fused_names(self):
        """_extract_target_modules must return fused names for Phi3 (no adapter)."""
        model, _ = self._make_tiny_phi3()
        target_modules = _extract_target_modules(model)

        # Fused names present
        qkv_modules = [m for m in target_modules if "qkv_proj" in m]
        gate_up_modules = [m for m in target_modules if "gate_up_proj" in m]
        assert len(qkv_modules) > 0, f"Expected qkv_proj in target_modules: {target_modules}"
        assert len(gate_up_modules) > 0, f"Expected gate_up_proj in target_modules: {target_modules}"

        # Split names absent
        assert all("q_proj" not in m or "qkv_proj" in m for m in target_modules), (
            f"Unexpected split q_proj in target_modules: {target_modules}"
        )

    def test_hf_peft_can_load_phi3_adapter(self, tmp_path):
        """HF PEFT (PeftModel.from_pretrained) must load a fused-name adapter."""
        model, peft_cfg = self._make_tiny_phi3()
        hf_sd = self._simulate_peft_save(model)
        target_modules = _extract_target_modules(model)

        # --- write adapter artifacts to tmp_path ---
        adapter_config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": peft_cfg.dim,
            "lora_alpha": peft_cfg.alpha,
            "target_modules": target_modules,
            "bias": "none",
            "base_model_name_or_path": "N/A",
        }
        with open(tmp_path / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f)

        save_file(
            {k: v.contiguous() for k, v in hf_sd.items()},
            str(tmp_path / "adapter_model.safetensors"),
        )

        # --- load with HF PEFT ---
        try:
            from peft import PeftModel
        except ImportError:
            pytest.skip("peft not installed")

        base_model = Phi3ForCausalLM(Phi3Config(**TINY_PHI3_CONFIG))
        peft_model = PeftModel.from_pretrained(base_model, str(tmp_path))

        # Verify LoRA modules exist
        lora_modules = [
            name for name, mod in peft_model.named_modules()
            if "lora" in name.lower() and hasattr(mod, "weight")
        ]
        assert len(lora_modules) > 0, "Expected LoRA modules in loaded PEFT model"

        # Verify forward pass works
        peft_model.eval()
        test_input = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            output = peft_model(test_input)
            assert output.logits is not None, "Forward pass failed"
            assert output.logits.shape == (1, 16, 256), (
                f"Unexpected logits shape: {output.logits.shape}"
            )
