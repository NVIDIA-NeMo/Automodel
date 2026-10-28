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

"""Real Zephon batches exercise gathered-token and recurrent boundary handoffs."""

from contextlib import closing
from dataclasses import replace

import pytest
import torch

from nemo_automodel.components.models.common.packing import (
    PackingCapabilities,
    configure_packing_for_models,
    flatten_packed_sequence_metadata,
)
from tests.utils.zephon_text_fixtures import config as config
from tests.utils.zephon_text_fixtures import tokenizer as tokenizer


@pytest.mark.parametrize("xdoc", [False, True])
def test_gathered_metadata_follows_selected_boundaries_after_shift(config, tokenizer, xdoc):
    config = replace(
        config,
        zephon=replace(
            config.zephon,
            packing=replace(config.zephon.packing, seq_len=10, pack_algorithm="first_fit"),
            training=replace(config.zephon.training, allow_cross_document_attention=xdoc),
        ),
    )
    contract = PackingCapabilities("document_ids", return_packed_sequence_metadata=True)
    with closing(config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1, packing_contract=contract)) as loader:
        batch = next(loader)
    selected = batch["_packed_seq_ids"]
    assert selected[0].tolist() == ([1] * 8 + [0, 0] if xdoc else [1] * 4 + [2] * 4 + [0, 0])
    assert batch["attention_mask"] is selected
    assert batch["packed_token_indices"].tolist() == [list(range(8)) + [-1, -1]] * 2
    assert batch["cu_seqlens"].tolist() == ([[0, 8]] * 2 if xdoc else [[0, 4, 8]] * 2)
    assert batch["max_seqlen"] == (8 if xdoc else 4)
    # Each PP microbatch must retain local offsets; flattening before splitting
    # would incorrectly make the second microbatch index the first one's tokens.
    for indices, boundaries in zip(batch["packed_token_indices"].split(1), batch["cu_seqlens"].split(1)):
        flat_indices, cu = flatten_packed_sequence_metadata(indices, boundaries, batch_size=1, sequence_length=10)
        assert flat_indices.tolist() == list(range(8))
        assert cu.tolist() == ([0, 8] if xdoc else [0, 4, 8])
    assert batch["labels"][:, 7:].eq(-100).all()


@pytest.mark.parametrize("xdoc", [False, True])
def test_kimi_full_attention_honors_selected_ids_with_original_boundaries(config, tokenizer, xdoc):
    # This CPU check covers Kimi's full-attention consumer. Its actual recurrent
    # kernels are validated separately on CUDA, rather than replaced by stubs.
    from nemo_automodel.components.models.kimi_linear.model import KimiLinear48BForCausalLM
    from tests.unit_tests.models.kimi_linear.test_model import _backend_config, _tiny_kimi_config

    torch.manual_seed(5)
    model = KimiLinear48BForCausalLM(_tiny_kimi_config(), backend=_backend_config()).eval()
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    config = replace(
        config,
        batch_size=1,
        zephon=replace(config.zephon, training=replace(config.zephon.training, allow_cross_document_attention=xdoc)),
    )
    contract = configure_packing_for_models([model])
    with closing(config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1, packing_contract=contract)) as loader:
        batch = next(loader)
    perturbed = batch["input_ids"].clone()
    perturbed[:, :4] += 10
    original = model(**batch).logits[:, 4:]
    changed = model(**{**batch, "input_ids": perturbed}).logits[:, 4:]
    if xdoc:
        assert not torch.allclose(original, changed, atol=1e-6)
    else:
        torch.testing.assert_close(original, changed, atol=1e-6, rtol=1e-5)
    original.square().mean().backward()
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()
