# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Qwen3.5 dense state-dict adapter (fp32-wrapper key rename)."""

from __future__ import annotations

import torch

from nemo_automodel.components.models.qwen3_5.state_dict_adapter import (
    Qwen3_5DenseStateDictAdapter,
    _route_to_fp32_holder,
    _strip_fp32_prefix,
)


class TestStripFp32Prefix:
    def test_strips_linear_attn_fp32_prefix(self):
        assert (
            _strip_fp32_prefix(
                "model.language_model.layers.0.linear_attn._fp32_params.A_log"
            )
            == "model.language_model.layers.0.linear_attn.A_log"
        )

    def test_passes_through_keys_without_fp32(self):
        for key in (
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.embed_tokens.weight",
            "model.visual.merger.linear_fc1.weight",
        ):
            assert _strip_fp32_prefix(key) == key

    def test_only_strips_under_linear_attn(self):
        # _fp32_params under any non-linear_attn parent must NOT be stripped.
        key = "model.language_model.layers.0.self_attn._fp32_params.A_log"
        assert _strip_fp32_prefix(key) == key


class TestRouteToFp32Holder:
    def test_routes_bare_a_log_to_holder(self):
        assert (
            _route_to_fp32_holder("model.language_model.layers.0.linear_attn.A_log")
            == "model.language_model.layers.0.linear_attn._fp32_params.A_log"
        )

    def test_passes_through_already_routed_keys(self):
        key = "model.language_model.layers.0.linear_attn._fp32_params.A_log"
        assert _route_to_fp32_holder(key) == key

    def test_does_not_route_non_a_log_keys(self):
        for key in (
            "model.language_model.layers.0.linear_attn.dt_bias",
            "model.language_model.layers.0.linear_attn.norm.weight",
        ):
            assert _route_to_fp32_holder(key) == key

    def test_does_not_route_a_log_outside_linear_attn(self):
        # Defensive: only linear_attn.A_log should be routed.
        key = "model.some.other.path.A_log"
        assert _route_to_fp32_holder(key) == key


class TestAdapter:
    def setup_method(self):
        self.adapter = Qwen3_5DenseStateDictAdapter()

    def _sample_state_dict(self):
        return {
            "model.language_model.layers.0.linear_attn._fp32_params.A_log": torch.zeros(4),
            "model.language_model.layers.1.linear_attn._fp32_params.A_log": torch.ones(4),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
            "model.language_model.embed_tokens.weight": torch.zeros(8, 2),
        }

    def test_to_hf_renames_fp32_params(self):
        sd = self._sample_state_dict()
        out = self.adapter.to_hf(sd)
        assert (
            "model.language_model.layers.0.linear_attn.A_log" in out
            and "model.language_model.layers.0.linear_attn._fp32_params.A_log" not in out
        )
        # Tensors are passed through by reference (no copy).
        assert out["model.language_model.layers.0.linear_attn.A_log"] is sd[
            "model.language_model.layers.0.linear_attn._fp32_params.A_log"
        ]
        # Non-fp32 keys are unchanged.
        assert out["model.language_model.embed_tokens.weight"] is sd[
            "model.language_model.embed_tokens.weight"
        ]
        # Number of keys preserved.
        assert len(out) == len(sd)

    def test_to_hf_accepts_kwargs(self):
        # Save callsites pass exclude_key_regex, quantization, device_mesh, etc.
        out = self.adapter.to_hf(
            {"x.linear_attn._fp32_params.A_log": torch.zeros(2)},
            exclude_key_regex=r".*_extra_state.*",
            quantization=False,
            v4_compatible=False,
        )
        assert list(out.keys()) == ["x.linear_attn.A_log"]

    def test_from_hf_routes_a_log_to_holder(self):
        hf_sd = {
            "model.language_model.layers.0.linear_attn.A_log": torch.zeros(4),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
        }
        out = self.adapter.from_hf(hf_sd)
        assert (
            "model.language_model.layers.0.linear_attn._fp32_params.A_log" in out
            and "model.language_model.layers.0.linear_attn.A_log" not in out
        )
        assert "model.language_model.layers.0.self_attn.q_proj.weight" in out

    def test_round_trip_is_identity(self):
        sd = self._sample_state_dict()
        round_tripped = self.adapter.from_hf(self.adapter.to_hf(sd))
        assert set(round_tripped.keys()) == set(sd.keys())
        for k, v in sd.items():
            assert round_tripped[k] is v

    def test_convert_single_tensor_to_hf(self):
        t = torch.ones(4)
        out = self.adapter.convert_single_tensor_to_hf(
            "model.language_model.layers.0.linear_attn._fp32_params.A_log", t
        )
        assert out == [("model.language_model.layers.0.linear_attn.A_log", t)]

    def test_convert_single_tensor_passthrough(self):
        t = torch.zeros(2, 2)
        out = self.adapter.convert_single_tensor_to_hf(
            "model.language_model.embed_tokens.weight", t
        )
        assert out == [("model.language_model.embed_tokens.weight", t)]
