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

from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.addons import _extract_target_modules
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
