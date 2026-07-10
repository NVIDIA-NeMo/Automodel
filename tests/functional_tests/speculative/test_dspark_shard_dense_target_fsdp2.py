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

"""2-GPU functional test for the FSDP2-sharded dense DSpark target path.

Exercises the real distributed contract behind ``recipe_args.shard_dense_target``
on two NCCL ranks, with a tiny random Qwen3 target saved to disk (no download):

1. sharded loading: ``create_distributed_setup_from_config`` +
   ``NeMoAutoModelForCausalLM.from_pretrained(distributed_setup=...)`` must
   return a target whose parameters are FSDP2 DTensors (not replicated);
2. embedding / head copying: ``gather_full_weight_module`` must reassemble the
   sharded ``embed_tokens`` / ``lm_head`` into full plain tensors that are
   bit-identical to the checkpoint on disk, and the draft must seed from them;
3. hidden-state capture: ``HFDSparkTargetModel.generate_batch`` through the
   FSDP2-wrapped target must match the same capture through the default
   replicated target;
4. training: the draft trains against that captured supervision with finite,
   decreasing loss.

Requires 2 GPUs (FlexAttention has no CPU backward); spawns its own NCCL ranks.
"""

import math
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, Qwen3Config

from nemo_automodel._transformers import NeMoAutoModelForCausalLM
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.speculative.dspark import (
    Qwen3DSparkModel,
    build_draft_config,
    compute_dspark_loss,
)
from nemo_automodel.components.speculative.dspark.target import HFDSparkTargetModel
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config
from nemo_automodel.recipes.llm._dspark_target_build import gather_full_weight_module

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs 2 GPUs (FSDP2 sharding + FlexAttention backward)"
)

_VOCAB = 1024
_HIDDEN = 128
_SEQ = 32
_TARGET_LAYER_IDS = [1, 3, 5]
_WORLD_SIZE = 2


class _Args(dict):
    def __getattr__(self, key):
        return self[key]


def _model_args() -> _Args:
    return _Args(
        num_draft_layers=2,
        target_layer_ids=list(_TARGET_LAYER_IDS),
        block_size=4,
        mask_token_id=7,
        num_anchors=16,
        markov_rank=32,
        markov_head_type="vanilla",
        confidence_head_alpha=1.0,
        confidence_head_with_markov=True,
    )


def _tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=2 * _HIDDEN,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )


def _capture(wrapper: HFDSparkTargetModel, input_ids: torch.Tensor):
    """Run one no-grad target capture.

    Args:
        wrapper: Target wrapper to capture from.
        input_ids: ``[B, S]`` int64 token ids (``B`` = batch, ``S`` = sequence).

    Returns:
        ``DSparkTargetBatch`` with ``target_hidden_states`` ``[B, S, F*H]``
        (``F`` = number of captured feature layers, ``H`` = hidden size) and
        ``target_last_hidden_states`` ``[B, S, H]``.
    """
    ones = torch.ones_like(input_ids)
    return wrapper.generate_batch(input_ids=input_ids, attention_mask=ones, loss_mask=ones)


def _rank_worker(rank: int, world_size: int, init_file: str, model_dir: str) -> None:
    """One NCCL rank of the sharded dense-target smoke (spawned; must be top-level)."""
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dtype = torch.bfloat16
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        # 1. Sharded loading through the same path the recipe uses for
        # shard_dense_target (ConfigNode top-level config, default FSDP2 block).
        setup = create_distributed_setup_from_config(
            ConfigNode({"distributed": {"strategy": "fsdp2"}}), world_size=world_size
        )
        target = NeMoAutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, distributed_setup=setup)
        target.requires_grad_(False)
        params = dict(target.named_parameters())
        sharded = {name: p for name, p in params.items() if isinstance(p, DTensor)}
        assert sharded, "distributed_setup load produced no DTensor parameters (target came back replicated)"
        assert any(p.to_local().numel() < p.numel() for p in sharded.values()), (
            "DTensor parameters are not actually sharded across ranks"
        )

        # 2. Embedding / head gathering: full plain tensors, bit-identical to the
        # checkpoint on disk, then the draft seeds from them.
        wrapper = HFDSparkTargetModel(target, target_layer_ids=list(_TARGET_LAYER_IDS))
        embed_src = gather_full_weight_module(wrapper.get_input_embeddings())
        head_src = gather_full_weight_module(wrapper.get_output_embeddings())
        reference = load_file(os.path.join(model_dir, "model.safetensors"))
        for src, key in ((embed_src, "model.embed_tokens.weight"), (head_src, "lm_head.weight")):
            assert not isinstance(src.weight, DTensor)
            torch.testing.assert_close(src.weight.detach().cpu(), reference[key], rtol=0, atol=0)

        draft = Qwen3DSparkModel(build_draft_config(_tiny_config(), _model_args())).to(device=device, dtype=dtype)
        draft.initialize_embeddings_and_head(embed_tokens=embed_src, lm_head=head_src, freeze=True)
        torch.testing.assert_close(
            draft.embed_tokens.weight.detach().cpu(), reference["model.embed_tokens.weight"], rtol=0, atol=0
        )

        # 3. Hidden-state capture through the FSDP2-wrapped target matches the
        # default replicated load (same class, same kernels, same dtype).
        generator = torch.Generator().manual_seed(100 + rank)  # per-rank batch, data-parallel style
        input_ids = torch.randint(0, _VOCAB, (1, _SEQ), generator=generator).to(device)
        batch = _capture(wrapper, input_ids)
        replicated = NeMoAutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype).to(device)
        replicated.requires_grad_(False)
        reference_batch = _capture(HFDSparkTargetModel(replicated, target_layer_ids=list(_TARGET_LAYER_IDS)), input_ids)
        assert batch.target_hidden_states.shape == (1, _SEQ, len(_TARGET_LAYER_IDS) * _HIDDEN)
        assert batch.target_last_hidden_states.shape == (1, _SEQ, _HIDDEN)
        torch.testing.assert_close(batch.target_hidden_states, reference_batch.target_hidden_states)
        torch.testing.assert_close(batch.target_last_hidden_states, reference_batch.target_last_hidden_states)

        # 4. The draft trains against the captured supervision (FSDP2-sharded like
        # the recipe) with finite, decreasing loss.
        for layer in draft.layers:
            fully_shard(layer)
        fully_shard(draft)
        optim = torch.optim.AdamW([p for p in draft.parameters() if p.requires_grad], lr=5e-3)
        draft.train()
        losses = []
        for _ in range(20):
            optim.zero_grad()
            torch.manual_seed(7)  # fixed anchors -> clean overfit signal
            outputs = draft(
                input_ids=batch.input_ids.to(device),
                target_hidden_states=batch.target_hidden_states.to(device=device, dtype=dtype),
                loss_mask=batch.loss_mask.to(device),
                target_last_hidden_states=batch.target_last_hidden_states.to(device=device, dtype=dtype),
            )
            loss = compute_dspark_loss(
                outputs=outputs,
                loss_decay_gamma=4.0,
                ce_loss_alpha=0.1,
                l1_loss_alpha=0.9,
                confidence_head_alpha=1.0,
            )
            loss.backward()
            optim.step()
            losses.append(loss.item())
        assert all(math.isfinite(x) for x in losses), f"non-finite loss: {losses}"
        assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_shard_dense_target_two_rank_contract(tmp_path):
    """Sharded loading, DTensor embed/head gathering, capture parity, one training run."""
    torch.manual_seed(0)
    model_dir = str(tmp_path / "tiny_qwen3_target")
    AutoModelForCausalLM.from_config(_tiny_config()).to(torch.bfloat16).save_pretrained(model_dir)

    init_file = str(tmp_path / "dist_init")
    mp.spawn(_rank_worker, args=(_WORLD_SIZE, init_file, model_dir), nprocs=_WORLD_SIZE, join=True)
