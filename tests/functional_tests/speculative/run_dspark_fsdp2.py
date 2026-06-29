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

"""2-GPU FSDP2 (DTensor) smoke for the DSpark draft.

Run with torchrun:

    torchrun --nproc_per_node=2 tests/functional_tests/speculative/run_dspark_fsdp2.py

Shards the draft per decoder layer with ``fully_shard`` (DTensor), confirms the
parameters become DTensors, and trains a few steps to confirm the objective runs
and the loss goes down under data-parallel sharding.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from transformers import Qwen3Config

from nemo_automodel.components.speculative.dspark import (
    Qwen3DSparkModel,
    build_draft_config,
    compute_dspark_loss,
)

VOCAB = 1024
HIDDEN = 256
TARGET_LAYER_IDS = [1, 3, 5]


class _Args(dict):
    def __getattr__(self, key):
        return self[key]


def _model_args() -> _Args:
    return _Args(
        num_draft_layers=4,
        target_layer_ids=list(TARGET_LAYER_IDS),
        block_size=7,
        mask_token_id=7,
        num_anchors=64,
        markov_rank=64,
        markov_head_type="vanilla",
        confidence_head_alpha=1.0,
        confidence_head_with_markov=True,
    )


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dtype = torch.bfloat16

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    log(f"world_size={world}  FSDP2 data-parallel")

    target_config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    draft = Qwen3DSparkModel(build_draft_config(target_config, _model_args())).to(
        device=device, dtype=dtype
    )
    draft.initialize_embeddings_and_head(
        embed_tokens=nn.Embedding(VOCAB, HIDDEN),
        lm_head=nn.Linear(HIDDEN, VOCAB, bias=False),
        freeze=True,
    )

    # FSDP2: shard each transformer block, then the root module (DTensor sharding).
    for layer in draft.layers:
        fully_shard(layer)
    fully_shard(draft)

    sharded = [n for n, p in draft.named_parameters() if isinstance(p, DTensor)]
    assert sharded, "no parameters were converted to DTensor by fully_shard"
    log(f"sharded params (DTensor): {len(sharded)} tensors, e.g. {sharded[0]}")

    # Per-rank fixed batch + synthetic target supervision (isolates FSDP2 from the target).
    gen = torch.Generator().manual_seed(100 + rank)
    seq = 48
    input_ids = torch.randint(0, VOCAB, (1, seq), generator=gen).to(device)
    loss_mask = torch.ones(1, seq, dtype=torch.uint8, device=device)
    target_hidden_states = torch.randn(
        1, seq, len(TARGET_LAYER_IDS) * HIDDEN, generator=gen
    ).to(device=device, dtype=dtype)
    target_last_hidden_states = torch.randn(1, seq, HIDDEN, generator=gen).to(
        device=device, dtype=dtype
    )

    optim = torch.optim.AdamW([p for p in draft.parameters() if p.requires_grad], lr=5e-3)
    draft.train()
    losses = []
    for step in range(20):
        optim.zero_grad()
        torch.manual_seed(7)  # fixed anchors -> clean overfit signal
        out = draft(
            input_ids=input_ids,
            target_hidden_states=target_hidden_states,
            loss_mask=loss_mask,
            target_last_hidden_states=target_last_hidden_states,
        )
        loss = compute_dspark_loss(
            outputs=out,
            loss_decay_gamma=4.0,
            ce_loss_alpha=0.1,
            l1_loss_alpha=0.9,
            confidence_head_alpha=1.0,
        )
        loss.backward()
        optim.step()
        losses.append(loss.item())
        log(f"step {step:2d}  loss {loss.item():.4f}")

    assert all(torch.isfinite(torch.tensor(x)) for x in losses), "non-finite loss"
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    log(f"PASS  first={losses[0]:.4f} last={losses[-1]:.4f}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
