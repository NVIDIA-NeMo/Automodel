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


"""Actual CUDA FA4 parity with checkpoint-local replay and the native SAC policy."""

import copy
import json

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.distributed.activation_checkpointing import (
    make_selective_checkpoint_context_fn,
)
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLTextConfig
from nemo_automodel.components.models.minimax_m3_vl.cp_sparse_attn import (
    MiniMaxM3CPSparseAttention,
)
from nemo_automodel.components.models.minimax_m3_vl.layers import Block
from nemo_automodel.components.models.minimax_m3_vl.recompute import sparse_route_replay
from nemo_automodel.components.moe.config import MoEConfig

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FA4"),
    pytest.mark.timeout(300),
]


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Compare matching tensor layouts with a global relative L2 error.

    Args:
        actual: Output or gradient tensor of arbitrary shape.
        expected: Reference tensor with the same shape.

    Returns:
        Relative L2 error against the reference norm.
    """
    return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize("length,selective", [(385, False), (385, True), (4096, True)])
def test_fa4_sparse_replay(monkeypatch: pytest.MonkeyPatch, length: int, selective: bool) -> None:
    import nemo_automodel.components.models.minimax_m3_vl.cp_sparse_attn as cp

    torch.manual_seed(419)
    real_shape = length == 4096
    hidden, heads, kv_heads, dim = (6144, 64, 4, 128) if real_shape else (128, 4, 2, 64)
    config = MiniMaxM3VLTextConfig(
        hidden_size=hidden,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=dim,
        rotary_dim=dim // 2,
        num_hidden_layers=1,
        num_mtp_modules=0,
        dense_intermediate_size=256,
        intermediate_size=256,
        moe_layer_freq=[0],
        sparse_attention_config={
            "sparse_num_index_heads": kv_heads,
            "sparse_index_dim": dim,
            "sparse_block_size": 128,
            "sparse_topk_blocks": 16 if real_shape else 2,
            "sparse_init_block": 0,
            "sparse_local_block": 1,
            "sparse_score_type": "max",
        },
    )
    backend = BackendConfig(
        attn="fa4",
        linear="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
    )
    if real_shape:
        model = MiniMaxM3CPSparseAttention(config, backend)
    else:
        moe = MoEConfig(
            n_routed_experts=2,
            n_shared_experts=0,
            n_activated_experts=1,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=hidden,
            inter_dim=256,
            moe_inter_dim=256,
            norm_topk_prob=True,
        )
        model = Block(0, config, moe, backend)
    model = model.cuda().to(torch.bfloat16)
    model.init_weights(torch.device("cuda"))
    baseline = copy.deepcopy(model)
    positions = torch.arange(length, device="cuda", dtype=torch.float32)
    angles = positions[:, None] / (10000 ** (torch.arange(0, dim // 2, 2, device="cuda") / (dim // 2)))
    freqs = torch.cat((angles.cos(), angles.sin()), -1).unsqueeze(0)
    inputs = [
        torch.randn(1, length, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(1 if real_shape else 2)
    ]
    references = [x.detach().clone().requires_grad_() for x in inputs]
    upstream = [torch.randn_like(x) for x in inputs]
    selected: list[torch.Tensor] = []
    snapshots: list[torch.Tensor] = []
    select = cp.select_sparse_blocks

    def capture(*args: torch.Tensor, **kwargs: int | str | torch.Tensor) -> torch.Tensor:
        """Retain final selections for same-input comparison.

        Args:
            args: Index queries [B,S,H,D] and keys [B,S,1,D].
            kwargs: Selector settings and query positions [S].

        Returns:
            Selected blocks [B,H,S,blocks].
        """
        result = select(*args, **kwargs)
        selected.append(result)
        snapshots.append(result.detach().clone())
        return result

    monkeypatch.setattr(cp, "select_sparse_blocks", capture)
    expected = [baseline(x, freqs_cis=freqs) for x in references]
    expected_selection = selected.copy()
    selected.clear()
    snapshots.clear()
    inner = make_selective_checkpoint_context_fn() if selective else None
    contexts = (
        sparse_route_replay.checkpoint_context_fn(inner) if real_shape else model.nemo_checkpoint_context_fn(inner)
    )
    actual = [checkpoint(model, x, freqs_cis=freqs, use_reentrant=False, context_fn=contexts) for x in inputs]
    assert len(selected) == len(inputs)
    for got, reference in zip(selected, expected_selection):
        torch.testing.assert_close(got, reference, rtol=0, atol=0)
        assert bool((got[..., -1, :-1].sum(-1) < got.shape[-1] - 1).all())
    if len(selected) == 2:
        assert not torch.equal(selected[0], selected[1])
    errors = {}
    for i in reversed(range(len(inputs))):
        torch.testing.assert_close(actual[i], expected[i], rtol=0, atol=0)
        actual[i].backward(upstream[i])
        expected[i].backward(upstream[i])
        errors[f"input_{i}"] = _relative_l2(inputs[i].grad, references[i].grad)
    assert len(selected) == len(inputs), "frozen indexer must not rerun during recompute"
    assert sparse_route_replay.current() is None
    for selection, snapshot in zip(selected, snapshots):
        torch.testing.assert_close(selection, snapshot, rtol=0, atol=0)
    for (name, parameter), (_, reference) in zip(model.named_parameters(), baseline.named_parameters()):
        if "indexer" in name:
            assert parameter.grad is reference.grad is None
        else:
            assert parameter.grad is not None, name
            errors[name] = _relative_l2(parameter.grad, reference.grad)
    # Same FA4 kernel/inputs/selected sets; allow its observed nondeterministic
    # atomic-reduction envelope, much tighter than the independent BF16 math gate.
    assert max(errors.values()) < 0.001, errors
    print(
        "M3_REPLAY_METRICS "
        + json.dumps(
            {
                "length": length,
                "selective": selective,
                "scope": "full_width_attention" if real_shape else "complete_dense_mlp_block",
                "microbatches": len(inputs),
                "selector_calls": len(selected),
                "gradient_relative_l2": errors,
            }
        )
    )
