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

"""Engram layer: configuration checks, hash state after loading, and the gate."""

import pytest
import torch

from nemo_automodel.components.models.hy_v4_flash.engram import (
    HYV4FlashEngramLayer,
    HYV4FlashWeightOnlyNorm,
    weighted_rms_norm,
)
from nemo_automodel.components.models.hy_v4_flash.model import HYV4FlashForCausalLM
from tests.unit_tests.models.hy_v4_flash.tiny import build_model, pack, random_documents, tiny_text_config


def test_engram_fp8_lookup_is_rejected_instead_of_ignored() -> None:
    with pytest.raises(NotImplementedError, match="engram_fp8_embedding"):
        HYV4FlashEngramLayer(tiny_text_config(engram_fp8_embedding=True), 1, dtype=torch.float32)


def test_wrong_engram_layer_offset_is_detected_by_the_layer_id_check() -> None:
    # a zero-based reading of the ids [2, 4] would put Engram on layers 2 and 4, which the config lookup rejects
    cfg = tiny_text_config()
    assert cfg.engram_layer_indices == [1, 3]
    with pytest.raises(ValueError, match="has no Engram block"):
        HYV4FlashEngramLayer(cfg, 2, dtype=torch.float32)  # one-based id 3 is not an Engram layer


def test_engram_hash_multipliers_follow_a_lookup_table_loaded_after_initialization() -> None:
    """Recipe order: weights are initialized first, the checkpoint (with its frozen vocabulary map) is loaded second.

    The hash multipliers are bounded by the number of distinct compressed ids of the loaded map. Multipliers drawn
    at initialization, for the placeholder identity map, send every n-gram to a different table row than the
    checkpoint was trained with; they must be redrawn when the map arrives, whichever way it is loaded.
    """
    from nemo_automodel.components.models.hy_v4_flash.engram import draw_hash_multipliers

    def engram_layers(model: HYV4FlashForCausalLM) -> list[HYV4FlashEngramLayer]:
        return [layer.engram_layer for layer in model.model.layers if layer.engram_layer is not None]

    def run(model: HYV4FlashForCausalLM) -> torch.Tensor:
        with torch.no_grad():
            return model(**pack(random_documents((9, 12)))).logits

    source = build_model()
    for engram in engram_layers(source):  # a many-to-few map, like the released one (fewer compressed ids than tokens)
        engram.lookup_table.copy_(torch.arange(engram.lookup_table.numel(), dtype=torch.int32) % 97)
    expected_logits = run(source)
    state = source.state_dict()
    for in_place in (True, False):
        target = build_model()  # ``initialize_weights`` has drawn multipliers for the identity map
        run(target)
        target.load_state_dict(state, assign=not in_place)
        logits = run(target)
        for engram in engram_layers(target):
            compressed = int(engram.lookup_table.max()) + 1
            assert compressed == 97
            expected = draw_hash_multipliers(engram.seed, engram.layer_id, engram.max_ngram_size, compressed)
            torch.testing.assert_close(engram.hash_multipliers, expected)
        torch.testing.assert_close(logits, expected_logits)


def test_engram_gate_is_one_half_at_a_logit_of_exactly_zero() -> None:
    """The reference gates with ``sqrt(clamp(|x|, 1e-6)) * sign(x)``; ``sign(0) = 0`` gives ``sigmoid(0) = 0.5``."""
    import torch.nn.functional as F

    from nemo_automodel.components.models.hy_v4_flash.engram import build_ngram_tokens

    model = build_model()
    engram = next(layer.engram_layer for layer in model.model.layers if layer.engram_layer is not None)
    with torch.no_grad():
        for projection in engram.key_projs:  # zero keys: every gate logit is exactly 0
            projection.weight.zero_()
            projection.bias.zero_()
        ids = random_documents((11,))[0][None]
        ngram = build_ngram_tokens(ids, torch.arange(ids.shape[1])[None], engram.max_ngram_size, engram.pad_id)
        streams = torch.randn(1, ids.shape[1], engram.hc_mult, engram.hidden_size)
        out = engram(streams, ngram, torch.ones_like(ids, dtype=torch.bool))
        # the gated signal, computed independently of the gate
        emb = engram.multi_head_embedding.embedding(engram.hash_ngrams(ngram))
        emb = F.rms_norm(emb, (engram.head_dim,)) * torch.stack([n.weight for n in engram.lookup_per_head_norms])
        main = engram.unified_pre_gamma * F.rms_norm(engram.value_proj(emb.flatten(-2)), (engram.hidden_size,))
        scales = torch.stack(list(engram.unified_per_hc_scales))
        expected = 0.5 * main.unsqueeze(2) + main.unsqueeze(2) * scales
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scale", [1.0, 1e-3, 40.0])
def test_weighted_rms_norm_equals_torch_rms_norm_bitwise(dtype: torch.dtype, scale: float) -> None:
    """One ``nn.RMSNorm`` per slice with a non-unit weight, on inputs of ordinary, small and large magnitude.

    In BF16, normalizing and then multiplying a BF16 weight rounds twice and differs from ``nn.RMSNorm`` on about a
    quarter of the elements; ``nn.RMSNorm`` multiplies the weight in fp32 and casts once.
    """
    torch.manual_seed(0)
    slices, features = 4, 96
    x = (scale * torch.randn(2, 37, slices, features)).to(dtype)
    weights = [(1 + 0.1 * torch.randn(features)).to(dtype) for _ in range(slices)]
    ours = torch.nn.ModuleList(HYV4FlashWeightOnlyNorm(features, dtype) for _ in range(slices))
    reference = [torch.nn.RMSNorm(features).to(dtype) for _ in range(slices)]
    with torch.no_grad():
        for norm, module, weight in zip(ours, reference, weights):
            norm.weight.copy_(weight)
            module.weight.copy_(weight)
    expected = torch.stack([module(x[..., index, :]) for index, module in enumerate(reference)], dim=-2)
    got = weighted_rms_norm(x, ours)
    assert got.dtype == dtype
    assert torch.equal(got, expected)
