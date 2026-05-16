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

"""Functional CUDA coverage for EAGLE-3 FlashAttention-2 draft attention."""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig

from nemo_automodel.components.speculative.eagle.draft_llama import _HAS_FA, LlamaEagle3DraftModel

_SKIP_REASON = "EAGLE-3 FlashAttention-2 functional test requires CUDA and flash-attn"


def _build_eagle3_config(attn_implementation: str) -> LlamaConfig:
    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=4,
        vocab_size=1024,
        max_position_embeddings=128,
    )
    config.torch_dtype = torch.bfloat16
    config.draft_vocab_size = 128
    config.target_hidden_size = 256
    config.attn_implementation = attn_implementation
    return config


@pytest.mark.skipif(not torch.cuda.is_available() or not _HAS_FA, reason=_SKIP_REASON)
def test_eagle3_flash_attention_matches_eager_with_right_padding():
    """FA2 and eager should match on valid tokens for right-padded CUDA batches."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    eager_draft = LlamaEagle3DraftModel(_build_eagle3_config("eager")).to(device=device, dtype=dtype)
    fa_draft = LlamaEagle3DraftModel(_build_eagle3_config("flash_attention_2")).to(device=device, dtype=dtype)
    fa_draft.load_state_dict(eager_draft.state_dict())

    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, eager_draft.config.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.tensor(
        [
            [1] * 48 + [0] * 16,
            [1] * 32 + [0] * 32,
        ],
        dtype=torch.long,
        device=device,
    )
    aux_hidden_states = torch.randn(batch_size, seq_len, eager_draft.config.hidden_size * 3, device=device, dtype=dtype)

    projected_eager = eager_draft.project_hidden_states(aux_hidden_states)
    projected_fa = fa_draft.project_hidden_states(aux_hidden_states)
    cache_eager: list[list[torch.Tensor]] = [[], []]
    cache_fa: list[list[torch.Tensor]] = [[], []]

    with torch.no_grad():
        h_eager = projected_eager
        h_fa = projected_fa
        for _ in range(3):
            h_eager = eager_draft(
                input_ids=input_ids,
                projected_hidden_states=h_eager,
                attention_mask=attention_mask,
                cache_hidden=cache_eager,
            )
            h_fa = fa_draft(
                input_ids=input_ids,
                projected_hidden_states=h_fa,
                attention_mask=attention_mask,
                cache_hidden=cache_fa,
            )

    valid = attention_mask.bool().unsqueeze(-1).expand_as(h_eager)
    torch.testing.assert_close(h_eager[valid], h_fa[valid], atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available() or not _HAS_FA, reason=_SKIP_REASON)
def test_eagle3_flash_attention_rejects_left_padding_on_cuda():
    """FA2 path should fail loudly when the batch is not right-padded."""
    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    draft = LlamaEagle3DraftModel(_build_eagle3_config("flash_attention_2")).to(device=device, dtype=dtype)
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, draft.config.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.tensor(
        [
            [0] * 8 + [1] * 24,
            [0] * 4 + [1] * 28,
        ],
        dtype=torch.long,
        device=device,
    )
    aux_hidden_states = torch.randn(batch_size, seq_len, draft.config.hidden_size * 3, device=device, dtype=dtype)

    with pytest.raises(ValueError, match="right-padded attention_mask"):
        draft(
            input_ids=input_ids,
            projected_hidden_states=draft.project_hidden_states(aux_hidden_states),
            attention_mask=attention_mask,
        )
