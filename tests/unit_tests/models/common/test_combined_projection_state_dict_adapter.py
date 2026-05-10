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

import torch

from nemo_automodel.components.models.common.combined_projection.state_dict_adapter import (
    CombinedProjectionStateDictAdapter,
)


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


def _as_dict(fqn_tensors):
    """Convert adapter tuple output to a state dict."""
    return dict(fqn_tensors)


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


class TestCombinedProjectionSingleTensorConversion:
    """Tests that single-tensor conversion matches full state-dict conversion."""

    def _adapter(self, **kwargs):
        return CombinedProjectionStateDictAdapter(_make_config(**kwargs))

    def test_qkv_weight_matches_to_hf(self):
        """qkv_proj.weight should split to q/k/v exactly like to_hf()."""
        adapter = self._adapter(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=8,
        )
        q_weight = torch.randn(8, 3)
        k_weight = torch.randn(4, 3)
        v_weight = torch.randn(4, 3)
        qkv_weight = adapter._interleave_qkv(q_weight, k_weight, v_weight)
        key = "model.layers.0.self_attn.qkv_proj.weight"

        single_tensor_sd = _as_dict(adapter.convert_single_tensor_to_hf(key, qkv_weight))
        full_sd = adapter.to_hf({key: qkv_weight})

        assert single_tensor_sd.keys() == full_sd.keys()
        for converted_key, converted_tensor in single_tensor_sd.items():
            torch.testing.assert_close(converted_tensor, full_sd[converted_key])

    def test_qkv_bias_splits(self):
        """qkv_proj.bias should split using the grouped QKV layout."""
        adapter = self._adapter(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=8,
        )
        q_bias = torch.arange(8, dtype=torch.float32)
        k_bias = torch.arange(4, dtype=torch.float32) + 10
        v_bias = torch.arange(4, dtype=torch.float32) + 20
        qkv_bias = adapter._interleave_qkv(q_bias, k_bias, v_bias)

        result = _as_dict(
            adapter.convert_single_tensor_to_hf(
                "model.layers.0.self_attn.qkv_proj.bias",
                qkv_bias,
            )
        )

        torch.testing.assert_close(result["model.layers.0.self_attn.q_proj.bias"], q_bias)
        torch.testing.assert_close(result["model.layers.0.self_attn.k_proj.bias"], k_bias)
        torch.testing.assert_close(result["model.layers.0.self_attn.v_proj.bias"], v_bias)

    def test_gate_up_weight_and_bias_split(self):
        """gate_up_proj tensors should split to gate/up projections."""
        adapter = self._adapter(num_hidden_layers=1)
        gate_weight = torch.randn(6, 3)
        up_weight = torch.randn(6, 3)
        gate_up_weight = adapter._interleave_gate_up(gate_weight, up_weight)
        gate_bias = torch.arange(6, dtype=torch.float32)
        up_bias = torch.arange(6, dtype=torch.float32) + 10
        gate_up_bias = adapter._interleave_gate_up(gate_bias, up_bias)

        weight_result = _as_dict(
            adapter.convert_single_tensor_to_hf(
                "model.layers.0.mlp.gate_up_proj.weight",
                gate_up_weight,
            )
        )
        bias_result = _as_dict(
            adapter.convert_single_tensor_to_hf(
                "model.layers.0.mlp.gate_up_proj.bias",
                gate_up_bias,
            )
        )

        torch.testing.assert_close(weight_result["model.layers.0.mlp.gate_proj.weight"], gate_weight)
        torch.testing.assert_close(weight_result["model.layers.0.mlp.up_proj.weight"], up_weight)
        torch.testing.assert_close(bias_result["model.layers.0.mlp.gate_proj.bias"], gate_bias)
        torch.testing.assert_close(bias_result["model.layers.0.mlp.up_proj.bias"], up_bias)

    def test_qkv_lora_a_duplicated_and_lora_b_split(self):
        """Single-tensor conversion should preserve PEFT-prefixed LoRA semantics."""
        adapter = self._adapter(
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=8,
        )
        lora_a = torch.randn(2, 3)
        q_lora_b = torch.randn(8, 2)
        k_lora_b = torch.randn(4, 2)
        v_lora_b = torch.randn(4, 2)
        qkv_lora_b = adapter._interleave_qkv(q_lora_b, k_lora_b, v_lora_b)

        lora_a_result = _as_dict(
            adapter.convert_single_tensor_to_hf(
                "base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.default.weight",
                lora_a,
            )
        )
        lora_b_result = _as_dict(
            adapter.convert_single_tensor_to_hf(
                "base_model.model.model.layers.0.self_attn.qkv_proj.lora_B.default.weight",
                qkv_lora_b,
            )
        )

        torch.testing.assert_close(
            lora_a_result["base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"],
            lora_a,
        )
        torch.testing.assert_close(
            lora_a_result["base_model.model.model.layers.0.self_attn.k_proj.lora_A.default.weight"],
            lora_a,
        )
        torch.testing.assert_close(
            lora_a_result["base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight"],
            lora_a,
        )
        torch.testing.assert_close(
            lora_b_result["base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"],
            q_lora_b,
        )
        torch.testing.assert_close(
            lora_b_result["base_model.model.model.layers.0.self_attn.k_proj.lora_B.default.weight"],
            k_lora_b,
        )
        torch.testing.assert_close(
            lora_b_result["base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight"],
            v_lora_b,
        )

    def test_exclude_regex_filters_input_key(self):
        """Excluded tensors should return no converted entries."""
        adapter = self._adapter(num_hidden_layers=1)

        result = adapter.convert_single_tensor_to_hf(
            "model.layers.0.self_attn.qkv_proj._extra_state",
            torch.randn(1),
            exclude_key_regex=r".*_extra_state.*",
        )

        assert result == []

    def test_passthrough_key(self):
        """Non-combined tensors should pass through unchanged."""
        adapter = self._adapter(num_hidden_layers=1)
        tensor = torch.randn(5, 7)

        result = adapter.convert_single_tensor_to_hf("model.embed_tokens.weight", tensor)

        assert result == [("model.embed_tokens.weight", tensor)]


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
        from transformers import LlamaConfig

        from nemo_automodel.components._peft.lora import (
            PeftConfig,
            apply_lora_to_linear_modules,
        )
        from nemo_automodel.components.models.llama.model import LlamaForCausalLM

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
            assert offending == [], f"Found forbidden fragment '{fragment}' in converted state dict keys: {offending}"

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
        from nemo_automodel.components.checkpoint.addons import _extract_target_modules

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


class TestCombinedProjectionBiasDTensorHelpers:
    def test_gather_1d_bias_uses_full_tensor_for_sharded_dtensor(self, monkeypatch):
        import torch.distributed.tensor as dtensor_mod
        import torch.distributed.tensor.placement_types as placement_types

        class FakeShard:
            def __init__(self, dim):
                self.dim = dim

        class FakeReplicate:
            pass

        class FakeDTensor:
            def __init__(self, tensor, device_mesh, placements):
                self._tensor = tensor
                self.device_mesh = device_mesh
                self.placements = tuple(placements)
                self.ndim = tensor.ndim
                self.full_tensor_calls = 0
                self.redistribute_calls = []

            def full_tensor(self):
                self.full_tensor_calls += 1
                return self._tensor.clone()

            def redistribute(self, device_mesh=None, placements=None):
                self.redistribute_calls.append((device_mesh, placements))
                return FakeDTensor(
                    self._tensor.clone(),
                    device_mesh if device_mesh is not None else self.device_mesh,
                    placements if placements is not None else self.placements,
                )

        monkeypatch.setattr(dtensor_mod, "DTensor", FakeDTensor)
        monkeypatch.setattr(placement_types, "Replicate", FakeReplicate)
        monkeypatch.setattr(placement_types, "Shard", FakeShard)

        mesh = object()
        dtensor = FakeDTensor(torch.arange(8.0), mesh, (FakeShard(0),))

        gathered, orig = CombinedProjectionStateDictAdapter._gather_1d_bias(dtensor)

        assert isinstance(gathered, torch.Tensor)
        torch.testing.assert_close(gathered, torch.arange(8.0))
        assert orig == (mesh, dtensor.placements)
        assert dtensor.full_tensor_calls == 1
        assert dtensor.redistribute_calls == []

    def test_restore_1d_bias_rebuilds_replicated_dtensor_before_resharding(self, monkeypatch):
        import torch.distributed.tensor as dtensor_mod
        import torch.distributed.tensor.placement_types as placement_types

        class FakeShard:
            def __init__(self, dim):
                self.dim = dim

        class FakeReplicate:
            pass

        class FakeDTensor:
            from_local_calls = []
            redistribute_calls = []

            def __init__(self, tensor, device_mesh, placements):
                self._tensor = tensor
                self.device_mesh = device_mesh
                self.placements = tuple(placements)
                self.ndim = tensor.ndim

            @classmethod
            def from_local(cls, local_tensor, device_mesh=None, placements=None, **kwargs):
                cls.from_local_calls.append((local_tensor.clone(), device_mesh, tuple(placements), kwargs))
                return cls(local_tensor.clone(), device_mesh, placements)

            def redistribute(self, device_mesh=None, placements=None):
                FakeDTensor.redistribute_calls.append((device_mesh, tuple(placements)))
                return FakeDTensor(
                    self._tensor.clone(),
                    device_mesh if device_mesh is not None else self.device_mesh,
                    placements if placements is not None else self.placements,
                )

        monkeypatch.setattr(dtensor_mod, "DTensor", FakeDTensor)
        monkeypatch.setattr(placement_types, "Replicate", FakeReplicate)
        monkeypatch.setattr(placement_types, "Shard", FakeShard)

        mesh = object()
        shard = FakeShard(0)
        full_bias = torch.arange(8.0)

        restored = CombinedProjectionStateDictAdapter._restore_1d_bias(full_bias, (mesh, (shard,)))

        assert isinstance(restored, FakeDTensor)
        assert restored.device_mesh is mesh
        assert restored.placements == (shard,)
        assert len(FakeDTensor.from_local_calls) == 1
        gathered_bias, gathered_mesh, gathered_placements, kwargs = FakeDTensor.from_local_calls[0]
        torch.testing.assert_close(gathered_bias, full_bias)
        assert gathered_mesh is mesh
        assert len(gathered_placements) == 1
        assert isinstance(gathered_placements[0], FakeReplicate)
        assert kwargs == {"run_check": False}
        assert FakeDTensor.redistribute_calls == [(mesh, (shard,))]
