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

"""Parity against the transformers reference (PR 50903) on a tiny random model.

The reference needs a newer transformers than AutoModel pins, so it runs in a separate process
(``ref_dump.py``) that writes released-format weights, inputs and logits to ``HY4_REF_DUMP``.
"""

import os

import pytest
import torch

from nemo_automodel.components.models.hy_v4_flash.config import HY4FlashVisionConfig
from nemo_automodel.components.models.hy_v4_flash.model import HY4FlashForCausalLM
from nemo_automodel.components.models.hy_v4_flash.state_dict_adapter import HY4FlashStateDictAdapter
from nemo_automodel.components.models.hy_v4_flash.vision import HY4FlashVisionTower
from tests.unit_tests.models.hy_v4_flash.tiny import CPU_BACKEND, tiny_config

DUMP = os.environ.get("HY4_REF_DUMP")
needs_text_dump = pytest.mark.skipif(
    DUMP is None or not os.path.exists(DUMP), reason="HY4_REF_DUMP reference dump not provided"
)


def _load() -> tuple[dict[str, torch.Tensor], HY4FlashForCausalLM, list[str], list[str]]:
    dump = torch.load(DUMP, weights_only=False)
    model = HY4FlashForCausalLM(tiny_config(vision={"num_hidden_layers": 0}), backend=CPU_BACKEND).float().eval()
    adapter = HY4FlashStateDictAdapter(model.config, model.moe_config, CPU_BACKEND, dtype=torch.float32)
    native = adapter.from_hf({k: v.clone() for k, v in dump["state"].items()})
    missing, unexpected = model.load_state_dict(native, strict=False)
    return dump, model, missing, unexpected


@needs_text_dump
def test_released_state_loads_without_missing_or_unexpected_keys() -> None:
    _, _, missing, unexpected = _load()
    assert not unexpected, unexpected
    assert not missing, missing


@needs_text_dump
def test_every_layer_matches_reference() -> None:
    dump, model, _, _ = _load()
    outputs = []
    hooks = [
        layer.register_forward_hook(lambda mod, args, out: outputs.append(out.detach())) for layer in model.model.layers
    ]
    with torch.no_grad():
        model(dump["ids"])
    for hook in hooks:
        hook.remove()
    for index, (got, ref) in enumerate(zip(outputs, dump["layer_out"])):
        torch.testing.assert_close(got, ref, rtol=2e-4, atol=2e-4, msg=lambda m, i=index: f"layer {i}: {m}")


@needs_text_dump
def test_submodules_match_reference() -> None:
    dump, model, _, _ = _load()
    got = {}
    for index, layer in enumerate(model.model.layers):
        for name, module in (("engram_layer", layer.engram_layer), ("mlp", layer.mlp), ("self_attn", layer.self_attn)):
            if module is not None:
                module.register_forward_hook(lambda m, a, o, key=f"{index}.{name}": got.__setitem__(key, o.detach()))
    with torch.no_grad():
        model(dump["ids"])
    for key, ref in dump["sub_out"].items():
        if key.rsplit(".", 1)[1] not in ("engram_layer", "mlp", "self_attn"):
            continue
        if key.endswith("mlp") and key.split(".")[0] != "0":
            # the reference MoE block includes the combine norm; ours applies it in the block
            continue
        value = got[key]
        if key.endswith("self_attn"):
            value = value.view_as(ref) if value.numel() == ref.numel() else value
        torch.testing.assert_close(value, ref, rtol=2e-4, atol=2e-4, msg=lambda m, key=key: f"{key}: {m}")


@needs_text_dump
def test_logits_match_reference() -> None:
    dump, model, missing, _ = _load()
    assert not missing, missing
    with torch.no_grad():
        out = model(dump["ids"], output_hidden_states=True)
    torch.testing.assert_close(out.hidden_states, dump["hidden"], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(out.logits, dump["logits"], rtol=2e-4, atol=2e-4)


VLM_DUMP = os.environ.get("HY4_REF_VLM_DUMP")


@pytest.mark.skipif(
    VLM_DUMP is None or not os.path.exists(VLM_DUMP), reason="HY4_REF_VLM_DUMP reference dump not provided"
)
def test_image_text_logits_and_positions_match_reference() -> None:
    from nemo_automodel.components.models.hy_v4_flash.model import compute_mrope_position_ids

    dump = torch.load(VLM_DUMP, weights_only=False)
    config = tiny_config(vision=dump["vision_kwargs"], image_token_id=dump["image_token_id"])
    model = HY4FlashForCausalLM(config, backend=CPU_BACKEND).float().eval()
    adapter = HY4FlashStateDictAdapter(config, model.moe_config, CPU_BACKEND, dtype=torch.float32)
    missing, unexpected = model.load_state_dict(
        adapter.from_hf({k: v.clone() for k, v in dump["state"].items()}), strict=False
    )
    assert not missing and not unexpected, (missing, unexpected)
    positions = compute_mrope_position_ids(dump["types"], dump["grid"], None, 2)
    assert torch.equal(positions, dump["position_ids"])
    with torch.no_grad():
        out = model(
            dump["ids"], pixel_values=dump["pixels"], image_grid_thw=dump["grid"], mm_token_type_ids=dump["types"]
        )
    torch.testing.assert_close(out.logits, dump["logits"], rtol=2e-4, atol=2e-4)


@needs_text_dump
def test_selected_key_sets_match_reference_and_selection_is_sparse() -> None:
    """Per-query key sets of every sparse layer agree with the reference; report how many queries dropped valid keys."""
    dump, model, _, _ = _load()
    captured = {}
    for index, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if attn.compressed_indexer is not None:
            original = attn._attend

            def spy(query, kv, plan, sinks, layout, index=index, original=original):
                captured[index] = plan.indices.clone()
                return original(query, kv, plan, sinks, layout)

            attn._attend = spy
    with torch.no_grad():
        model(dump["ids"])
    total_dropped = 0
    for key, selected in dump["sub_out"].items():
        if not key.endswith(".selected"):
            continue
        layer = int(key.split(".")[0])
        indices = captured[layer].long()
        mine = torch.zeros_like(selected)
        for query, row in enumerate(indices):
            mine[query, row[row >= 0]] = True
        causal = torch.tril(torch.ones_like(selected))
        # the reference only recalls groups once the sequence is longer than index_topk; compare causal-visible keys
        assert torch.equal(mine & causal, selected & causal), f"selected key sets differ in layer {layer}"
        total_dropped += int(((selected & causal).sum(-1) < causal.sum(-1)).sum())
    assert total_dropped > 0, "selection never dropped a valid key: sparse behavior is untested"


@needs_text_dump
@pytest.mark.parametrize("control", ["rope_theta_swapped", "engram_layer_id_shifted"])
def test_parity_fails_for_deliberately_wrong_variants(control: str) -> None:
    dump, model, _, _ = _load()
    if control == "rope_theta_swapped":
        rope = model.model.config.layer_rope_parameters
        rope["sliding_attention"]["rope_theta"], rope["indexed_attention"]["rope_theta"] = (
            rope["indexed_attention"]["rope_theta"],
            rope["sliding_attention"]["rope_theta"],
        )
    else:
        for layer in model.model.layers:
            if layer.engram_layer is not None:
                layer.engram_layer.layer_id += 1
                layer.engram_layer.refresh_hash_state()
    with torch.no_grad():
        logits = model(dump["ids"]).logits
    assert not torch.allclose(logits, dump["logits"], rtol=2e-4, atol=2e-4)


VISION_DUMP = os.environ.get("HY4_REF_VISION_DUMP")


@pytest.mark.skipif(
    VISION_DUMP is None or not os.path.exists(VISION_DUMP), reason="HY4_REF_VISION_DUMP reference dump not provided"
)
def test_vision_tower_matches_reference() -> None:
    dump = torch.load(VISION_DUMP, weights_only=False)
    tower = HY4FlashVisionTower(HY4FlashVisionConfig(**dump["kwargs"])).float().eval()
    state = {k.removeprefix("vit."): v for k, v in dump["state"].items()}
    missing, unexpected = tower.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    with torch.no_grad():
        features = tower(dump["pixels"], dump["grid"])
    torch.testing.assert_close(features, dump["features"], rtol=1e-4, atol=1e-4)
