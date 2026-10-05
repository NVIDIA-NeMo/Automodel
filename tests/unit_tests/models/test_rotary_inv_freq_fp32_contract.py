# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Rotary ``inv_freq`` buffers stay bitwise fp32 through every whole-model cast path.

The HF rotary classes derive cos/sin from the stored ``inv_freq`` buffer, so a bf16
round-trip (``0.5994842 -> 0.59765625``) skews every RoPE phase and the error grows with
the position id. The keep-in-fp32 name lists (``_keep_in_fp32_modules``) are the single
mechanism that protects those buffers: ``cast_model_to_dtype`` (``initialize_weights``,
the DDP / Megatron-FSDP single-rank casts, retrieval backbone construction) snapshots and
restores the matched buffers around its bulk ``nn.Module.to``, and
``cast_frozen_modules_to_compute_dtype`` (frozen vision tower recipes) skips them.

Each family builds a tiny fp32 model, records the exact fp32 tables, runs each cast path
and asserts the tables are unchanged, then checks that the rotary forward of the cast
model equals the fp32 reference bit for bit. Runs on CPU except where the model class
itself requires CUDA to build.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from nemo_automodel._transformers.retrieval import _move_to_extracted_dtype
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import (
    cast_frozen_modules_to_compute_dtype,
    cast_model_to_dtype,
)
from nemo_automodel.components.moe.layers import MoEConfig


def _backend(**overrides) -> BackendConfig:
    options = dict(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )
    options.update(overrides)
    return BackendConfig(**options)


@dataclass(frozen=True)
class Family:
    """One model family: how to build it and which rotary buffers its fp32 lists protect."""

    build: Callable[[], nn.Module]
    # Exact FQNs of the fp32 rotary tables the ``_keep_in_fp32_modules`` tokens must select.
    rotary_buffers: frozenset[str]
    # Submodule frozen by the frozen-tower recipes (vision tower, or the decoder for text-only).
    frozen_tower: str
    text_rotary: str
    vision_rotary: str | None


# --------------------------------------------------------------------------- families
def _qwen3_5() -> nn.Module:
    pytest.importorskip("transformers.models.qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )

    from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForConditionalGeneration

    text_config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        layer_types=["linear_attention", "full_attention"],
        attn_implementation="eager",
        torch_dtype="float32",
        tie_word_embeddings=False,
    )
    vision_config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=16,
    )
    config = Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
        tie_word_embeddings=False,
    )
    return Qwen3_5ForConditionalGeneration(config, backend=_backend(enable_hf_state_dict_adapter=True))


def _qwen3_5_moe() -> nn.Module:
    pytest.importorskip("transformers.models.qwen3_5_moe")
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig

    from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForConditionalGeneration

    text_config = Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        router_aux_loss_coef=0.01,
        pad_token_id=0,
        tie_word_embeddings=False,
        layer_types=["linear_attention", "full_attention"],
    )
    moe_config = MoEConfig(
        dim=text_config.hidden_size,
        inter_dim=text_config.hidden_size,
        moe_inter_dim=text_config.moe_intermediate_size,
        n_routed_experts=text_config.num_experts,
        n_shared_experts=1,
        n_activated_experts=text_config.num_experts_per_tok,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=text_config.router_aux_loss_coef,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
        shared_expert_gate=True,
        shared_expert_inter_dim=text_config.shared_expert_intermediate_size,
    )
    vision_config = dict(
        depth=2,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=32,
        num_position_embeddings=8,
    )
    config = Qwen3_5MoeConfig(text_config=text_config.to_dict(), vision_config=vision_config)
    return Qwen3_5MoeForConditionalGeneration(config, backend=_backend(), moe_config=moe_config)


def _qwen3_vl_moe() -> nn.Module:
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeConfig, Qwen3VLMoeTextConfig

    from nemo_automodel.components.models.qwen3_vl_moe.model import Qwen3VLMoeForConditionalGeneration

    text_config = Qwen3VLMoeTextConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        router_aux_loss_coef=0.01,
        norm_topk_prob=False,
        pad_token_id=0,
        rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 1.0},
    )
    moe_config = MoEConfig(
        dim=text_config.hidden_size,
        inter_dim=text_config.intermediate_size,
        moe_inter_dim=text_config.moe_intermediate_size,
        n_routed_experts=text_config.num_experts,
        n_shared_experts=0,
        n_activated_experts=text_config.num_experts_per_tok,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=text_config.router_aux_loss_coef,
        norm_topk_prob=text_config.norm_topk_prob,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        activation_alpha=1.702,
        activation_limit=7.0,
        softmax_before_topk=True,
    )
    vision_config = dict(
        depth=2,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=32,
        num_position_embeddings=8,
        deepstack_visual_indexes=[0, 1],
    )
    config = Qwen3VLMoeConfig(text_config=text_config.to_dict(), vision_config=vision_config)
    return Qwen3VLMoeForConditionalGeneration(config, backend=_backend(), moe_config=moe_config)


def _qwen3_8_flash_next() -> nn.Module:
    from nemo_automodel.components.models.qwen3_8_flash_next.config import (
        Qwen3_8_FlashNextConfig,
        Qwen3_8_FlashNextTextConfig,
    )
    from nemo_automodel.components.models.qwen3_8_flash_next.model import Qwen3_8_FlashNextForConditionalGeneration

    text = Qwen3_8_FlashNextTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["linear_attention"],
        full_attention_interval=1,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=1,
        hc_count=2,
        hc_lowrank=4,
        ple_layer_ids=[],
        indexer_budget=8,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        max_position_embeddings=16,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default", "partial_rotary_factor": 1.0},
        partial_rotary_factor=1.0,
        dtype="float32",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
        tie_word_embeddings=False,
    )
    config = Qwen3_8_FlashNextConfig(text_config=text, language_model_only=True, tie_word_embeddings=False)
    moe_config = MoEConfig(
        dim=text.hidden_size,
        inter_dim=text.hidden_size,
        moe_inter_dim=text.moe_intermediate_size,
        n_routed_experts=text.num_experts,
        n_shared_experts=1,
        n_activated_experts=text.num_experts_per_tok,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=text.router_aux_loss_coef,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
        shared_expert_gate=True,
        shared_expert_inter_dim=text.shared_expert_intermediate_size,
        dtype=torch.float32,
    )
    return Qwen3_8_FlashNextForConditionalGeneration.from_config(config, moe_config=moe_config, backend=_backend())


_TEXT_ROTARY_BUFFERS = (
    "model.language_model.rotary_emb.inv_freq",
    "model.language_model.rotary_emb.original_inv_freq",
)
_VISION_ROTARY_BUFFER = "model.visual.rotary_pos_emb.inv_freq"
_VLM_BUFFERS = frozenset((*_TEXT_ROTARY_BUFFERS, _VISION_ROTARY_BUFFER))

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Qwen3VLMoeForConditionalGeneration builds on the current CUDA device"
)
FAMILIES = [
    pytest.param(
        Family(
            _qwen3_5, _VLM_BUFFERS, "model.visual", "model.language_model.rotary_emb", "model.visual.rotary_pos_emb"
        ),
        id="qwen3_5",
    ),
    pytest.param(
        Family(
            _qwen3_5_moe, _VLM_BUFFERS, "model.visual", "model.language_model.rotary_emb", "model.visual.rotary_pos_emb"
        ),
        id="qwen3_5_moe",
    ),
    pytest.param(
        Family(
            _qwen3_vl_moe,
            _VLM_BUFFERS,
            "model.visual",
            "model.language_model.rotary_emb",
            "model.visual.rotary_pos_emb",
        ),
        id="qwen3_vl_moe",
        marks=needs_cuda,
    ),
    pytest.param(
        Family(
            _qwen3_8_flash_next,
            frozenset(_TEXT_ROTARY_BUFFERS),
            "model.language_model",
            "model.language_model.rotary_emb",
            None,
        ),
        id="qwen3_8_flash_next",
    ),
]


# --------------------------------------------------------------------------- cast paths
def _cast_model_to_dtype(model: nn.Module, family: Family) -> None:
    # The DDP and Megatron-FSDP single-rank paths cast through this helper.
    cast_model_to_dtype(model, torch.bfloat16)


def _retrieval_move_to_extracted_dtype(model: nn.Module, family: Family) -> None:
    # Retrieval rebuilds a backbone and matches the dtype of the extracted submodel.
    _move_to_extracted_dtype(model, nn.Linear(2, 2).to(torch.bfloat16))


def _initialize_weights(model: nn.Module, family: Family) -> None:
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)


def _frozen_tower_cast(model: nn.Module, family: Family) -> None:
    # fp32 master weights with a frozen tower: the infrastructure casts every frozen
    # tensor to the compute dtype, except the names pinned by the fp32 lists.
    for param in model.get_submodule(family.frozen_tower).parameters():
        param.requires_grad_(False)
    assert any(param.requires_grad for param in model.parameters())
    cast_frozen_modules_to_compute_dtype(model, torch.bfloat16)


CAST_PATHS = [
    pytest.param(_cast_model_to_dtype, id="cast_model_to_dtype"),
    pytest.param(_retrieval_move_to_extracted_dtype, id="retrieval_move_to_extracted_dtype"),
    pytest.param(_initialize_weights, id="initialize_weights"),
    pytest.param(_frozen_tower_cast, id="frozen_tower_cast"),
]


# --------------------------------------------------------------------------- helpers
def _build(family: Family) -> nn.Module:
    torch.manual_seed(0)
    return family.build().eval()


def _rotary_tables(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: buf.detach().cpu().clone()
        for name, buf in model.named_buffers(remove_duplicate=False)
        if name.endswith("inv_freq")
    }


def _assert_tables_exact_fp32(model: nn.Module, family: Family, reference: dict[str, torch.Tensor]) -> None:
    tables = _rotary_tables(model)
    assert set(tables) == family.rotary_buffers
    for name, table in tables.items():
        assert table.dtype is torch.float32, name
        assert torch.equal(table, reference[name]), name


# --------------------------------------------------------------------------- tests
@pytest.mark.parametrize("family", FAMILIES)
def test_fp32_tokens_select_exactly_the_rotary_buffers(family: Family):
    model = _build(family)
    tokens = list(model._keep_in_fp32_modules)
    assert tokens

    matched_buffers = {
        name for name, _ in model.named_buffers(remove_duplicate=False) if any(t in name for t in tokens)
    }
    matched_parameters = {name for name, _ in model.named_parameters() if any(t in name for t in tokens)}

    assert matched_buffers == family.rotary_buffers
    assert matched_parameters == set()
    assert set(_rotary_tables(model)) == family.rotary_buffers
    # Buffer preservation is the non-strict list's job; strict only matters for parameters.
    assert not any(t in set(model._keep_in_fp32_modules_strict) for t in tokens)


@pytest.mark.parametrize("cast", CAST_PATHS)
@pytest.mark.parametrize("family", FAMILIES)
def test_cast_path_keeps_rotary_inv_freq_exact_fp32(family: Family, cast: Callable[[nn.Module, Family], None]):
    model = _build(family)
    reference = _rotary_tables(model)
    assert all(table.dtype is torch.float32 for table in reference.values())

    cast(model, family)

    _assert_tables_exact_fp32(model, family, reference)
    # The cast itself did happen for ordinary weights.
    tower = model.get_submodule(family.frozen_tower)
    assert next(p for p in tower.parameters() if p.dim() > 1).dtype is torch.bfloat16


@pytest.mark.parametrize("family", FAMILIES)
def test_rotary_forward_of_bf16_model_equals_fp32_reference(family: Family):
    reference = _build(family)
    model = _build(family)
    # Full bf16 initialization followed by the frozen-tower infrastructure cast.
    _initialize_weights(model, family)
    _frozen_tower_cast(model, family)

    text_rotary = model.get_submodule(family.text_rotary)
    reference_text_rotary = reference.get_submodule(family.text_rotary)
    device = text_rotary.inv_freq.device
    reference_text_rotary.to(device)
    # Position ids reach far enough that a bf16-rounded inv_freq changes the phases.
    position_ids = torch.arange(2048, device=device).unsqueeze(0)
    hidden_states = torch.zeros(1, position_ids.shape[1], 8, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        cos, sin = text_rotary(hidden_states, position_ids)
        reference_cos, reference_sin = reference_text_rotary(hidden_states, position_ids)
    assert torch.equal(cos, reference_cos)
    assert torch.equal(sin, reference_sin)

    if family.vision_rotary is None:
        return
    vision_rotary = model.get_submodule(family.vision_rotary)
    reference_vision_rotary = reference.get_submodule(family.vision_rotary)
    reference_vision_rotary.to(vision_rotary.inv_freq.device)
    # The HF vision rotary multiplies ``inv_freq`` straight into the grid positions: older
    # releases take the max grid extent, newer ones the per-patch (t, h, w) position ids.
    grid = _vision_rotary_input(vision_rotary, vision_rotary.inv_freq.device)
    with torch.no_grad():
        freqs = vision_rotary(grid)
        reference_freqs = reference_vision_rotary(grid)
    assert freqs.dtype is torch.float32
    assert torch.equal(freqs, reference_freqs)


def _vision_rotary_input(vision_rotary: nn.Module, device: torch.device) -> int | torch.Tensor:
    parameters = [name for name in inspect.signature(type(vision_rotary).forward).parameters if name != "self"]
    if parameters[0] == "position_ids":
        return torch.tensor(
            [[[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [63, 63, 63]]], device=device
        ).float()
    return 64
