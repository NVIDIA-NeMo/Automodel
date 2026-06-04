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

"""Unit tests for P-EAGLE COD sampling, mask, registry, and draft forward."""

import pytest
import torch
from transformers import LlamaConfig

from nemo_automodel.components.speculative.eagle.draft_llama_peagle import LlamaPEagleDraftModel
from nemo_automodel.components.speculative.eagle.peagle_attention import create_peagle_mask_mod
from nemo_automodel.components.speculative.eagle.peagle_core import PEagleTrainerModule
from nemo_automodel.components.speculative.eagle.peagle_data import generate_cod_sample_indices
from nemo_automodel.components.speculative.eagle.registry import resolve_peagle_draft_spec

# ── COD sampling ─────────────────────────────────────────────────────────


def test_cod_depth0_covers_full_sequence():
    seq_len = 16
    loss_mask = torch.ones(1, seq_len)
    anchor_pos, depth = generate_cod_sample_indices(seq_len, loss_mask, num_depths=4)
    assert int((depth == 0).sum()) == seq_len
    # orig position recovery stays in range.
    orig = anchor_pos + depth
    assert int(orig.min()) >= 0 and int(orig.max()) < seq_len


def test_cod_depths_shrink_geometrically():
    seq_len = 64
    loss_mask = torch.ones(1, seq_len)
    _, depth = generate_cod_sample_indices(
        seq_len, loss_mask, num_depths=4, down_sample_ratio=0.5, down_sample_ratio_min=0.0
    )
    counts = [int((depth == d).sum()) for d in range(int(depth.max()) + 1)]
    # Each deeper group retains no more positions than the previous one.
    for prev, cur in zip(counts, counts[1:]):
        assert cur <= prev


def test_cod_respects_num_depths_cap():
    seq_len = 32
    loss_mask = torch.ones(1, seq_len)
    _, depth = generate_cod_sample_indices(seq_len, loss_mask, num_depths=3)
    assert int(depth.max()) <= 2


# ── Mask mod ─────────────────────────────────────────────────────────────


def test_peagle_mask_blocks_cross_document_and_future():
    anchor_pos = torch.tensor([0, 1, 2, 0, 1])
    depth = torch.tensor([0, 0, 0, 1, 1])
    lengths = torch.tensor([3])  # one document of 3 tokens, total_seq_len padded to 5
    mask_mod = create_peagle_mask_mod(anchor_pos, depth, lengths, total_seq_len=5)

    def m(q, kv):
        return bool(mask_mod(0, 0, torch.tensor(q), torch.tensor(kv)))

    # depth-0 causal: q at orig 2 may attend to depth-0 kv at orig 0, not orig 1 future of itself? 1<=2 ok.
    assert m(2, 0) is True
    assert m(0, 2) is False  # future depth-0 key
    # same rollout (anchor 0) earlier depth visible.
    assert m(3, 0) is True
    # different rollout, non depth-0 key is blocked.
    assert m(3, 4) is False


# ── Registry ─────────────────────────────────────────────────────────────


def test_peagle_registry_resolves_supported_archs():
    for arch in ("LlamaForCausalLM", "Qwen3ForCausalLM", "Phi3ForCausalLM"):
        assert resolve_peagle_draft_spec([arch]).draft_cls is LlamaPEagleDraftModel


def test_peagle_registry_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_peagle_draft_spec(["TotallyUnknownForCausalLM"])


# ── Draft forward + trainer (CUDA-only: FlexAttention requires a GPU) ─────


def _tiny_draft_config():
    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
    )
    cfg.target_hidden_size = 64
    cfg.num_aux_hidden_states = 3
    cfg.draft_vocab_size = 256
    cfg.pad_token_id = 0
    return cfg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention requires a CUDA device")
def test_peagle_trainer_forward_backward():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    cfg = _tiny_draft_config()
    draft = LlamaPEagleDraftModel(cfg).to(device=device, dtype=dtype)

    vocab = cfg.vocab_size
    module = PEagleTrainerModule(
        draft,
        selected_token_ids=torch.arange(vocab),
        selected_token_mask=torch.ones(vocab, dtype=torch.bool),
        num_depths=4,
        mask_token_id=0,
    ).to(device)

    seq_len = 32
    input_ids = torch.randint(0, vocab, (1, seq_len), device=device)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
    loss_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
    aux_hidden_states = torch.randn(1, seq_len, cfg.hidden_size * 3, device=device, dtype=dtype)
    target_logits = torch.randn(1, seq_len, vocab, device=device, dtype=dtype)

    metrics = module(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        aux_hidden_states=aux_hidden_states,
        target_logits=target_logits,
    )
    assert torch.isfinite(metrics.loss)
    metrics.loss.backward()
    assert draft.mask_hidden.grad is not None
    assert draft.lm_head.weight.grad is not None
