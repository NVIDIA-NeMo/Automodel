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

"""Activation-checkpoint replay of the frozen GLM DSA indexer selection."""

import copy

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.rope_utils import freqs_cis_from_position_ids
from nemo_automodel.components.models.glm_moe_dsa.model import GlmMoeDsaForCausalLM

pytestmark = pytest.mark.cpu_only


def _model() -> GlmMoeDsaForCausalLM:
    config = GlmMoeDsaConfig(
        vocab_size=256,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=4,
        intermediate_size=128,
        moe_intermediate_size=64,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        attention_bias=False,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=8,
        mlp_layer_types=["dense", "dense", "sparse", "sparse"],
        indexer_types=["full", "full", "shared", "shared"],
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )
    torch.manual_seed(1234)
    model = GlmMoeDsaForCausalLM(config, backend=backend)
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    return model.train()


def _count_indexer_calls(block) -> list[int]:
    calls: list[int] = []
    indexer_forward = block.self_attn.indexer.forward

    def counting_forward(*args, **kwargs):
        calls.append(1)
        return indexer_forward(*args, **kwargs)

    block.self_attn.indexer.forward = counting_forward
    return calls


def _run(block, x: torch.Tensor, freqs_cis: torch.Tensor):
    x = x.clone().requires_grad_(True)
    out, topk_indices = block(x, freqs_cis=freqs_cis)
    out.float().square().sum().backward()
    grads = {name: param.grad.clone() for name, param in block.named_parameters() if param.grad is not None}
    return out.detach(), topk_indices, x.grad.clone(), grads


def test_full_layer_replays_selection_on_recompute_and_matches_plain_block():
    model = _model()
    block = model.model.layers["0"]  # dense MLP, full indexer
    torch.manual_seed(7)
    batch, seq_len = 2, 12  # longer than index_topk so the selection is a real top-k
    x = torch.randn(batch, seq_len, model.config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    freqs_cis = freqs_cis_from_position_ids(position_ids, model.model.freqs, qkv_format="bshd", for_fused_rope=False)

    reference = copy.deepcopy(block)
    reference_calls = _count_indexer_calls(reference)
    ref_out, ref_idx, ref_x_grad, ref_grads = _run(reference, x, freqs_cis)
    assert len(reference_calls) == 1

    plain = copy.deepcopy(block)
    plain_calls = _count_indexer_calls(plain)
    plain_wrapped = checkpoint_wrapper(plain, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    _run(plain_wrapped, x, freqs_cis)
    assert len(plain_calls) == 2, "without replay the recompute runs the indexer again"

    replayed = copy.deepcopy(block)
    replayed_calls = _count_indexer_calls(replayed)
    replayed_wrapped = checkpoint_wrapper(
        replayed,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        context_fn=replayed.nemo_checkpoint_context_fn(None),
    )
    out, topk_indices, x_grad, grads = _run(replayed_wrapped, x, freqs_cis)
    assert len(replayed_calls) == 1, "the recompute must take the recorded selection"
    assert torch.equal(topk_indices, ref_idx)
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(x_grad, ref_x_grad)
    assert grads.keys() == ref_grads.keys()
    for name, grad in grads.items():
        torch.testing.assert_close(grad, ref_grads[name], msg=lambda m, n=name: f"{n}: {m}")
    # Indexer weights receive no gradient on either path: the selection is discrete.
    assert not any(name.startswith("self_attn.indexer.") for name in ref_grads)

    # A second checkpointed call gets its own recorder: still one indexer run per call.
    _run(replayed_wrapped, x, freqs_cis)
    assert len(replayed_calls) == 2


def test_shared_layer_hook_leaves_context_unchanged():
    model = _model()
    shared_block = model.model.layers["2"]
    assert shared_block.skip_topk
    sentinel = object()
    assert shared_block.nemo_checkpoint_context_fn(sentinel) is sentinel
    full_block = model.model.layers["1"]
    assert full_block.nemo_checkpoint_context_fn(None) is not None
