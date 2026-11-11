# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""CPU unit tests of the Praxis model package: config parsing, weight mapping, masks and shapes."""

from __future__ import annotations

import copy

import pytest
import torch

from nemo_automodel.components.models.common.utils import BackendConfig
from nemo_automodel.components.models.praxis.config import PraxisConfig, PraxisConfigError
from nemo_automodel.components.models.praxis.masks import (
    KIND_CLEAN,
    KIND_GLOBAL,
    KIND_NOISED,
    Slot,
    chunk_causal_rule,
    dense_mask,
    expert_cross_rule,
    expert_self_rule,
    window_tags,
)
from nemo_automodel.components.models.praxis.model import PraxisPolicy

TINY_CONFIG = {
    "format_version": 2,
    "model_type": "praxis",
    "embodiment": "single_arm",
    "torch_dtype": "float32",
    "backbone": {
        "dim": 64,
        "timestep_dim": 16,
        "num_registers": 4,
        "modulation_dtype": "mixed",
        "dense_ffn_hidden": 96,
        "attention": {"num_heads": 4, "num_kv_heads": 2, "head_dim": 16, "flex_block_size": 128},
        "rope": {"effective_dim": 8, "base": 26_000_000, "efficient_cache_max": 1_000_000},
        "moe": {"num_routed_experts": 4, "top_k_routed": 2, "expert_hidden": 32, "routed_scaling_factor": 2.5},
        "encoder": {
            "depth": 3,
            "dense_blocks": [0],
            "tap_blocks": [0, 2],
            "input_projections": {"TextEncodingTensor": {"c_in": 32}, "VideoTensor": {"c_in": 192}},
            "positive_zero_residual_add": True,
        },
        "decoder": {
            "depth": 2,
            "dense_blocks": [0],
            "modulation_dim": 64,
            "unpatch_factor": [1, 2, 2],
            "input_projections": {"VideoTensor": {"c_in": 48}},
            "positive_zero_residual_add": True,
        },
    },
    "action_expert": {
        "dim": 64,
        "num_layers": 4,
        "num_heads": 1,
        "head_dim": 64,
        "trunk_dim": 64,
        "action_patch_dim": 40,
        "proprio_patch_dim": 10,
        "num_future_per_chunk": 2,
        "max_chunks": 16,
        "max_action_tokens": 64,
        "rope": {
            "kind": "trunk",
            "dim": 64,
            "effective_dim": 64,
            "base": 26_000_000,
            "table_size": 1_000_000,
            "order": "before_norm",
        },
        "global_order": ["register", "anchor", "text"],
        "attention": "sdpa",
        "head_plus_zero": True,
    },
    "chunking": {
        "n_chunks": 4,
        "action_chunk": 24,
        "action_frames_per_token": 4,
        "latents_per_chunk": 6,
        "past_frames": 15,
        "future_frames": 24,
        "window_frames": 93,
        "chunk_geometry": [[0, 21], [21, 24], [45, 24], [69, 24]],
        "grid_hz": 15.0,
    },
    "video": {
        "height": 256,
        "width": 832,
        "views": ["exterior", "wrist"],
        "layout": "hstack",
        "tile_bounds_rows_cols": {"exterior": [0, 256, 0, 416], "wrist": [0, 256, 416, 832]},
        "vae": "Wan2.2",
        "vae_encode_chunk_frames": 4,
        "latent_channels": 48,
        "vae_stride": [4, 16, 16],
        "patch_size": [1, 2, 2],
    },
    "text_encoder": {
        "model": "Qwen3.5-2B",
        "hidden_layer": -3,
        "pad_to_multiple_of": 32,
        "trim_padding": True,
        "hidden_dim": 32,
    },
    "action_space": {"dim": 10, "layout": "[xyz(3) rot6d(6) grip(1)]", "pose_layout": "[xyz(3) quat_wxyz(4) grip(1)]"},
    "normalization": {
        "domain": "droid",
        "action_q01": [0.0] * 10,
        "action_q99": [1.0] * 10,
        "q99_normed_dims": [0, 1, 2, 9],
        "proprio_mean": [0.0] * 10,
        "proprio_std": [1.0] * 10,
        "action_formula": "",
        "proprio_formula": "",
    },
}


def cpu_backend() -> BackendConfig:
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch_fp32",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        gate_precision=torch.float32,
        enable_hf_state_dict_adapter=True,
    )


def test_config_parses_and_derives_shared_expert_counts() -> None:
    cfg = PraxisConfig.from_json(TINY_CONFIG)
    assert cfg.backbone.moe.num_experts == 5 and cfg.backbone.moe.top_k == 3
    assert cfg.backbone.encoder.tap_blocks == (0, 2)
    assert cfg.action_expert.global_order == ("register", "anchor", "text")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.__setitem__("embodiment", "bimanual"),
        lambda c: c["backbone"]["moe"].__setitem__("mystery", 1),
        lambda c: c["action_expert"].__setitem__("absolute_positions", False),
        lambda c: c["backbone"].__setitem__("layout", "dense"),
    ],
)
def test_config_rejects_other_variants(mutate) -> None:
    c = copy.deepcopy(TINY_CONFIG)
    mutate(c)
    with pytest.raises((PraxisConfigError, ValueError)):
        PraxisConfig.from_json(c)


def test_state_dict_adapter_round_trip_and_shared_expert_split() -> None:
    cfg = PraxisConfig.from_json(TINY_CONFIG)
    model = PraxisPolicy(cfg, cpu_backend())
    # Seeded values instead of the module's uninitialized parameters: a NaN bit pattern there fails torch.equal.
    gen = torch.Generator().manual_seed(0)
    native = {k: torch.randn(v.shape, generator=gen).to(v.dtype) for k, v in model.state_dict().items()}
    hf = model.state_dict_adapter.to_hf(native)
    assert "backbone.encoder.core.blocks.1.ffn.shared.gate_up" in hf
    assert "backbone.encoder.core.blocks.1.ffn.router.weight" in hf
    assert "backbone.encoder.core.blocks.1.ffn.expert_bias" in hf
    assert "backbone.decoder.core.modulators.0.weight" in hf
    assert not any(".shared_experts." in k or ".gate.weight" in k or ".modulator." in k for k in hf)
    gate_up = hf["backbone.encoder.core.blocks.1.ffn.shared.gate_up"]
    assert gate_up.shape == (64, 2 * 32)
    back = model.state_dict_adapter.from_hf(hf)
    assert set(back) == set(native)
    for k in native:
        assert torch.equal(back[k], native[k]), k


def test_chunk_causal_rule_matches_the_window_contract() -> None:
    slots = (Slot(KIND_GLOBAL, 1, 2), Slot(KIND_NOISED, 4, 1), Slot(KIND_CLEAN, 2, 1))
    tags = window_tags(slots, chunk_size=2, device=torch.device("cpu"))
    mask = dense_mask(chunk_causal_rule(tags), tags.length, tags.length, torch.device("cpu"))
    # Tokens: 0-1 global, 2-5 noised frames 0..3 (chunks 0,0,1,1), 6-7 clean frames 0..1 (chunk 0).
    assert mask[:, :2].all()  # everyone reads the globals
    assert mask[0].tolist() == [True, True, False, False, False, False, False, False]
    assert mask[2].tolist() == [True, True, True, True, False, False, False, False]  # noised chunk 0: own chunk only
    assert mask[4].tolist() == [
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        True,
    ]  # noised chunk 1: own chunk + clean chunk 0
    assert mask[6].tolist() == [True, True, False, False, False, False, True, True]  # clean reads clean only


def test_expert_masks() -> None:
    cs, n_clean, n_noised, tpf, n_global = 2, 2, 4, 1, 3
    device = torch.device("cpu")
    q_chunk = torch.tensor([0, 1, 0, 0, 1, 1, 0, 0, 1, 1])  # proprio(2) | future(2x2) | actions(4)
    from nemo_automodel.components.models.praxis.masks import TokenTags

    q_tags = TokenTags(
        kind=torch.full_like(q_chunk, KIND_NOISED, dtype=torch.int32), chunk=q_chunk.int(), frame=(q_chunk * cs).int()
    )
    kv_tags = window_tags(
        (Slot(KIND_GLOBAL, 1, n_global), Slot(KIND_CLEAN, n_clean, tpf), Slot(KIND_NOISED, n_noised, tpf)), cs, device
    )
    cross = dense_mask(expert_cross_rule(q_tags, kv_tags, n_clean + n_noised), q_tags.length, kv_tags.length, device)
    # KV: 0-2 global, 3-4 clean frames 0,1, 5-8 noised frames 0..3.
    assert cross[0].tolist() == [True, True, True, False, False, True, True, False, False]  # chunk 0 query
    assert cross[1].tolist() == [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
    ]  # chunk 1 query reads clean 0,1
    self_mask = dense_mask(expert_self_rule(q_tags), q_tags.length, q_tags.length, device)
    assert self_mask[0, 2] and not self_mask[0, 1]


def test_forward_shapes_cpu() -> None:
    torch.manual_seed(0)
    cfg = PraxisConfig.from_json(TINY_CONFIG)
    model = PraxisPolicy(cfg, cpu_backend()).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.05)
    kk, lat, h, w = 2, 6, 2, 4
    out = model(
        torch.randn(1, 5, 32),
        torch.randn(1, 192, 1, h, w),
        torch.randn(1, 192, (kk - 1) * lat, h, w),
        torch.randn(1, 192, kk * lat, h, w),
        torch.rand(1, kk),
        torch.randn(1, kk * lat, 40),
        torch.rand(1, kk),
        torch.randn(1, kk, 10),
    )
    assert out.velocity is not None and out.velocity.shape == (1, 192, kk * lat, h, w)
    assert out.action_velocity.shape == (1, kk * lat, 40)
    assert len(out.features.taps) == 2
    assert torch.isfinite(out.velocity).all() and torch.isfinite(out.action_velocity).all()


def test_place_rotary_tables_moves_the_constructor_frequencies() -> None:
    """The frequency tables built on the CPU are moved as they are, not recomputed on the target device."""
    model = PraxisPolicy(PraxisConfig.from_json(TINY_CONFIG), cpu_backend())
    ropes = (model.action_expert.rope, model.backbone.encoder.core.rope, model.backbone.decoder.core.rope)
    before = [rope.theta.clone() for rope in ropes]
    for rope in ropes:
        rope.make_theta = None  # the move must not call the builder again
    model.place_rotary_tables(torch.device("cpu"))
    for rope, table in zip(ropes, before):
        assert rope.theta.device.type == "cpu"
        assert torch.equal(rope.theta, table)
