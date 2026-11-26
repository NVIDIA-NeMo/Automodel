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

"""Pins the tiny models to the Transformers reference (PR 50903 at ``eb1d5db4``, 5.24.0.dev0) and to the released keys.

Every weight is a closed form of its *released* key name (``released_weight``) and enters through the state-dict
adapter. The expected values are what the reference computes, in FP32, for the same weights assigned under its own
key names, so a wrong key mapping and a changed forward both move them. The key names are those of
``model.safetensors.index.json`` at revision ``eccd3123``.
"""

import pytest
import torch

from nemo_automodel.components.models.hy_v4_flash.model import HYV4FlashForCausalLM, compute_mrope_position_ids
from nemo_automodel.components.models.hy_v4_flash.processing import num_image_tokens
from nemo_automodel.components.models.hy_v4_flash.state_dict_adapter import HYV4FlashStateDictAdapter
from tests.unit_tests.models.hy_v4_flash.conftest import (
    CPU_BACKEND,
    PAD_ID,
    checksum,
    released_weight,
    tiny_config,
    uniform,
)

IMAGE_TOKEN_ID = 250
HC = [f"hc_{site}_layer.hc_pre.hc_{part}" for site in ("attn", "mlp") for part in ("base", "fn", "scale")]
NORMS = ["input_layernorm.weight", "post_attention_layernorm.weight"]
ATTENTION = [
    "kv_a_layernorm.weight",
    "kv_a_proj_with_mqa.weight",
    "kv_b_proj.weight",
    "learnable_sink_param",
    "linear_gate.weight",
    "o_proj.weight",
    "post_proj_norm.weight",
    "q_a_layernorm.weight",
    "q_a_proj.weight",
    "q_b_proj.weight",
]
INDEXER = [
    "head_weight_projection.weight",
    "intra_group_ape",
    "key_gate_projection.weight",
    "key_norm.weight",
    "key_value_projection.weight",
    "query_projection.weight",
    "token_key_norm.weight",
    "token_key_projection.weight",
]
DENSE_MLP = ["down_proj.weight", "gate_proj.weight", "post_fc2_norm.weight", "up_proj.weight"]
MOE_MLP = [
    "combine_output_norm.weight",
    "experts.down_proj",
    "experts.gate_up_proj",
    "gate.e_score_correction_bias",
    "gate.weight",
    "routed_expert_down_proj.weight",
    "routed_expert_up_proj.weight",
    *(f"shared_experts.{name}_proj.weight" for name in ("down", "gate", "up")),
]
ENGRAM = [
    "lookup_table",
    "multi_head_embedding.embedding.weight",
    "unified_pre_gamma",
    "value_proj.bias",
    "value_proj.weight",
    *(f"key_projs.{stream}.{part}" for stream in range(4) for part in ("bias", "weight")),
    *(f"{name}.{stream}.weight" for name in ("lookup_per_head_norms", "norm_key", "norm_query") for stream in range(4)),
    *(f"unified_per_hc_scales.{stream}" for stream in range(4)),
]
VISION_BLOCK = [
    *(f"{norm}.{part}" for norm in ("input_layernorm", "post_attention_layernorm") for part in ("bias", "weight")),
    *(f"mlp.{name}.{part}" for name in ("dense_4h_to_h", "dense_h_to_4h") for part in ("bias", "weight")),
    *(f"self_attn.{name}_proj.{part}" for name in "koqv" for part in ("bias", "weight")),
]
TOP_LEVEL = [
    "lm_head.weight",
    "model.embed_tokens.weight",
    "model.norm.weight",
    *(f"model.hc_head.hc_head_{part}" for part in ("base", "fn", "scale")),
    *(
        f"vit.embeddings.{name}"
        for name in ("patch_embedding.bias", "patch_embedding.weight", "position_embedding.weight")
    ),
    *(f"vit.perceive.{name}" for name in ("image_newline", "norm.weight")),
    *(f"vit.perceive.proj.{index}.{part}" for index in (0, 2) for part in ("bias", "weight")),
]

# The tensors the release stores in FP32 (safetensors headers); everything else is BF16 and ``lookup_table`` int32.
FP32_INDEXER = ["intra_group_ape", "key_gate_projection.weight", "key_norm.weight", "key_value_projection.weight"]
FP32_TOP_LEVEL = [f"model.hc_head.hc_head_{part}" for part in ("base", "fn", "scale")]


def _released_keys() -> set[str]:
    """Released names of the tiny model: sparse layers 0, 1, 5; dense MLP in layer 0; Engram in layers 1 and 3."""
    keys = set(TOP_LEVEL) | {f"vit.layers.{block}.{name}" for block in range(2) for name in VISION_BLOCK}
    for layer in range(6):
        names = [*HC, *NORMS, *(f"self_attn.{name}" for name in ATTENTION)]
        names += [f"mlp.{name}" for name in (DENSE_MLP if layer == 0 else MOE_MLP)]
        if layer in (0, 1, 5):
            names += [f"self_attn.compressed_indexer.{name}" for name in INDEXER]
        if layer in (1, 3):
            names += [f"engram_layer.{name}" for name in ENGRAM]
        keys |= {f"model.layers.{layer}.{name}" for name in names}
    return keys


def test_export_keeps_the_released_fp32_tensors_in_fp32() -> None:
    model = HYV4FlashForCausalLM(tiny_config(), backend=CPU_BACKEND)
    adapter = HYV4FlashStateDictAdapter(model.config, model.moe_config, CPU_BACKEND, dtype=torch.bfloat16)
    expected = {*FP32_TOP_LEVEL, "lm_head.weight"}  # the head is BF16 in the release and trained in FP32 here
    for layer in range(6):
        names = [*HC, "self_attn.learnable_sink_param"]
        names += [] if layer == 0 else ["mlp.gate.e_score_correction_bias"]
        names += [f"self_attn.compressed_indexer.{name}" for name in FP32_INDEXER] if layer in (0, 1, 5) else []
        expected |= {f"model.layers.{layer}.{name}" for name in names}
    mapping = adapter.forced_hf_dtype_mapping(model.state_dict())
    assert set(mapping) == expected and set(mapping.values()) == {"float32"}


# Reference values: ``checksum`` of every layer's output, of the Engram outputs and of the logits; for the sparse
# layers the number of causal keys the indexer keeps over all queries and the sum of their indices.
TEXT = dict(
    stages={
        "0": (1.102671, -146.3713),
        "0.selected": (703, 19663),
        "1": (1.620185, 862.6643),
        "1.engram_layer": (0.04112817, 3.822461),
        "1.selected": (703, 21255),
        "2": (1.947128, 581.9415),
        "3": (2.26536, -291.7462),
        "3.engram_layer": (0.03978138, 18.80031),
        "4": (2.548363, -491.9839),
        "5": (2.750068, -519.3273),
        "5.selected": (703, 20455),
        "logits": (0.7154446, -138.5421),
    },
    logits=[  # positions 0, 48 and 96, first four vocabulary entries
        [-0.685872, 0.136391, 0.342911, 0.023891],
        [0.452040, 0.341398, -0.837456, -0.471342],
        [-0.478336, 0.512612, -1.204353, 0.417377],
    ],
)
IMAGE_TEXT = dict(
    stages={
        "0": (1.086473, -171.582),
        "0.selected": (156, 1116),
        "1": (1.614086, 247.9287),
        "1.engram_layer": (0.02783681, -0.4776347),
        "1.selected": (156, 1136),
        "2": (1.96769, 346.3191),
        "3": (2.240186, -358.8477),
        "3.engram_layer": (0.02457694, 3.853204),
        "4": (2.54636, -579.3707),
        "5": (2.803804, -684.3079),
        "5.selected": (156, 1188),
        "logits": (0.7228421, -58.70998),
        "vision": (0.2898406, 1.073672),
    },
    logits=[  # positions 0, 10 and 23
        [-0.685872, 0.136391, 0.342911, 0.023891],
        [-0.729968, -0.220921, 1.297997, 0.438887],
        [0.441232, -0.894164, -1.297873, 1.266826],
    ],
)
# Time, height and width positions of 7 text tokens, a 2 x 4 image block (three columns and a newline) and 9 text tokens.
POSITIONS = [
    [*range(7), *[7] * 8, *range(11, 20)],
    [*range(7), *[7] * 4, *[8] * 4, *range(11, 20)],
    [*range(7), *range(7, 11), *range(7, 11), *range(11, 20)],
]


def _token_ids(length: int) -> torch.Tensor:
    """Ids of shape [1, length] above the special ids; tokens 5-7 repeat 2-4 so Engram rows collide."""
    ids = (3 + (uniform("ids", (length,)) * 2**20).long() % 237)[None]
    ids[0, 5:8] = ids[0, 2:5]
    return ids


@pytest.fixture(scope="module")
def model() -> HYV4FlashForCausalLM:
    torch.manual_seed(0)
    model = HYV4FlashForCausalLM(tiny_config(image_token_id=IMAGE_TOKEN_ID), backend=CPU_BACKEND).float().eval()
    model.initialize_weights(dtype=torch.float32)
    adapter = HYV4FlashStateDictAdapter(model.config, model.moe_config, CPU_BACKEND, dtype=torch.float32)
    shapes = {name: tuple(value.shape) for name, value in adapter.to_hf(model.state_dict()).items()}
    assert set(shapes) == _released_keys()
    released = {name: released_weight(name, shape) for name, shape in shapes.items() if "lookup_table" not in name}
    missing, unexpected = model.load_state_dict(adapter.from_hf(released), strict=False)
    assert not unexpected and all("lookup_table" in name for name in missing)
    return model


def _stages(model: HYV4FlashForCausalLM, **inputs: torch.Tensor) -> tuple[dict[str, tuple[float, float]], torch.Tensor]:
    """Run ``model`` and return the stage values in the layout of ``TEXT["stages"]`` and the logits [tokens, vocab]."""
    stages, hooks, attends = {}, [], []
    for index, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(lambda m, a, o, k=str(index): stages.__setitem__(k, checksum(o))))
        if layer.engram_layer is not None:
            key = f"{index}.engram_layer"
            hooks.append(
                layer.engram_layer.register_forward_hook(lambda m, a, o, k=key: stages.__setitem__(k, checksum(o)))
            )
        if layer.self_attn.compressed_indexer is not None:
            attention, original = layer.self_attn, layer.self_attn._attend

            def spy(query, kv, plan, sinks, layout, key=f"{index}.selected", original=original):
                indices = plan.indices.long()
                kept = (indices >= 0) & (indices <= torch.arange(indices.shape[0])[:, None])
                stages[key] = (int(kept.sum()), int((indices * kept).sum()))
                return original(query, kv, plan, sinks, layout)

            attention._attend = spy
            attends.append((attention, original))
    hooks.append(model.vit.register_forward_hook(lambda m, a, o: stages.__setitem__("vision", checksum(o))))
    try:
        with torch.no_grad():
            logits = model(**inputs).logits[0]
    finally:
        for hook in hooks:
            hook.remove()
        for attention, original in attends:
            attention._attend = original
    stages["logits"] = checksum(logits)
    return stages, logits


def _assert_matches(stages: dict[str, tuple[float, float]], expected: dict[str, tuple[float, float]]) -> None:
    for name, values in expected.items():
        assert stages[name] == pytest.approx(values, rel=2e-4, abs=2e-4), name


def test_text_forward_matches_the_reference(model: HYV4FlashForCausalLM) -> None:
    stages, logits = _stages(model, input_ids=_token_ids(97))
    _assert_matches(stages, TEXT["stages"])
    torch.testing.assert_close(logits[[0, 48, 96], :4], torch.tensor(TEXT["logits"]), rtol=2e-4, atol=2e-4)
    padded = torch.cat([_token_ids(97), torch.full((1, 3), PAD_ID)], dim=1)
    with torch.no_grad():
        logits = model(padded, padding_mask=padded == PAD_ID).logits[0]
    torch.testing.assert_close(logits[[0, 48, 96], :4], torch.tensor(TEXT["logits"]), rtol=2e-4, atol=2e-4)


def test_image_text_forward_and_positions_match_the_reference(model: HYV4FlashForCausalLM) -> None:
    text = _token_ids(16)[0]
    ids = torch.cat([text[:7], torch.full((8,), IMAGE_TOKEN_ID), text[7:]])[None]
    types, grid = (ids == IMAGE_TOKEN_ID).long(), torch.tensor([[1, 4, 6]])
    assert num_image_tokens(4, 6, 2) == 8
    assert compute_mrope_position_ids(types, grid, None, 2)[:, 0].tolist() == POSITIONS
    stages, logits = _stages(
        model,
        input_ids=ids,
        pixel_values=uniform("pixels", (24, 48)).float(),
        image_grid_thw=grid,
        mm_token_type_ids=types,
    )
    _assert_matches(stages, IMAGE_TEXT["stages"])
    torch.testing.assert_close(logits[[0, 10, 23], :4], torch.tensor(IMAGE_TEXT["logits"]), rtol=2e-4, atol=2e-4)


def test_two_frame_positions_match_the_reference() -> None:
    # Two text tokens, two frames of 2 x (2 + newline) tokens, two text tokens: a frame takes max(h, w + 1) positions.
    types = torch.tensor([[0, 0, *[1] * 12, 0, 0]])
    positions = compute_mrope_position_ids(types, torch.tensor([[2, 4, 4]]), None, 2)[:, 0].tolist()
    assert positions == [
        [0, 1, *[2] * 6, *[5] * 6, 8, 9],
        [0, 1, 2, 2, 2, 3, 3, 3, 5, 5, 5, 6, 6, 6, 8, 9],
        [0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7, 5, 6, 7, 8, 9],
    ]
