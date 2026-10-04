# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Qwen3.5 dense state-dict adapter (identity round trip, MTP key mapping)."""

from __future__ import annotations

import torch

from nemo_automodel.components.models.qwen3_5.state_dict_adapter import (
    Qwen3_5DenseStateDictAdapter,
    map_qwen3_5_mtp_from_hf_key,
    map_qwen3_5_mtp_to_hf_key,
)

_A_LOG = "model.language_model.layers.0.linear_attn.A_log"
_DT_BIAS = "model.language_model.layers.0.linear_attn.dt_bias"
_Q_PROJ = "model.language_model.layers.0.self_attn.q_proj.weight"


class TestAdapter:
    def setup_method(self):
        self.adapter = Qwen3_5DenseStateDictAdapter()

    def _sample_state_dict(self):
        return {
            _A_LOG: torch.zeros(4),
            _DT_BIAS: torch.zeros(4),
            "model.language_model.layers.1.linear_attn.A_log": torch.ones(4),
            "model.language_model.layers.1.linear_attn.dt_bias": torch.ones(4),
            _Q_PROJ: torch.zeros(2, 2),
            "model.language_model.embed_tokens.weight": torch.zeros(8, 2),
        }

    def test_to_hf_accepts_kwargs(self):
        # Save callsites pass exclude_key_regex, quantization, device_mesh, etc.
        out = self.adapter.to_hf(
            {"x.linear_attn.A_log": torch.zeros(2)},
            exclude_key_regex=r".*_extra_state.*",
            quantization=False,
            v4_compatible=False,
        )
        assert list(out.keys()) == ["x.linear_attn.A_log"]

    def test_round_trip_is_identity(self):
        sd = self._sample_state_dict()
        round_tripped = self.adapter.from_hf(self.adapter.to_hf(sd))
        assert set(round_tripped.keys()) == set(sd.keys())
        for k, v in sd.items():
            assert round_tripped[k] is v

    def test_convert_single_tensor_passthrough(self):
        t = torch.zeros(2, 2)
        out = self.adapter.convert_single_tensor_to_hf("model.language_model.embed_tokens.weight", t)
        assert out == [("model.language_model.embed_tokens.weight", t)]


class TestMTPKeyMapping:
    def test_maps_hf_mtp_fusion_keys_to_native(self):
        assert map_qwen3_5_mtp_from_hf_key("mtp.fc.weight") == "mtp.layers.0.eh_proj.weight"
        assert map_qwen3_5_mtp_from_hf_key("mtp.pre_fc_norm_embedding.weight") == "mtp.layers.0.enorm.weight"
        assert map_qwen3_5_mtp_from_hf_key("mtp.pre_fc_norm_hidden.weight") == "mtp.layers.0.hnorm.weight"
        assert map_qwen3_5_mtp_from_hf_key("mtp.norm.weight") == "mtp.layers.0.final_layernorm.weight"

    def test_maps_native_mtp_fusion_keys_to_hf(self):
        assert map_qwen3_5_mtp_to_hf_key("mtp.layers.0.eh_proj.weight") == "mtp.fc.weight"
        assert map_qwen3_5_mtp_to_hf_key("mtp.layers.0.enorm.weight") == "mtp.pre_fc_norm_embedding.weight"
        assert map_qwen3_5_mtp_to_hf_key("mtp.layers.0.hnorm.weight") == "mtp.pre_fc_norm_hidden.weight"
        assert map_qwen3_5_mtp_to_hf_key("mtp.layers.0.final_layernorm.weight") == "mtp.norm.weight"

    def test_adapter_round_trips_mtp_keys(self):
        adapter = Qwen3_5DenseStateDictAdapter()
        native = {
            "mtp.layers.0.eh_proj.weight": torch.randn(8, 16),
            "mtp.layers.0.self_attn.q_proj.weight": torch.randn(16, 8),
        }

        hf = adapter.to_hf(native)
        assert "mtp.fc.weight" in hf
        assert "mtp.layers.0.self_attn.q_proj.weight" in hf

        roundtrip = adapter.from_hf(hf)
        assert set(roundtrip) == set(native)
        for key, tensor in native.items():
            assert roundtrip[key] is tensor
