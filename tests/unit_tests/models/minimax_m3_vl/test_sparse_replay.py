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


"""CPU checkpoint lifecycle tests; CUDA kernel execution is covered separately."""

import copy
import gc
import weakref
from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import (
    CheckpointPolicy,
    SelectiveCheckpointContext,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLTextConfig
from nemo_automodel.components.models.minimax_m3_vl.cp_sparse_attn import (
    MiniMaxM3CPSparseAttention,
)
from nemo_automodel.components.models.minimax_m3_vl.layers import Block
from nemo_automodel.components.models.minimax_m3_vl.recompute import sparse_route_replay
from nemo_automodel.components.moe.config import MoEConfig


def _block() -> Block:
    config = MiniMaxM3VLTextConfig(
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        rotary_dim=8,
        num_hidden_layers=1,
        num_mtp_modules=0,
        dense_intermediate_size=64,
        intermediate_size=64,
        moe_layer_freq=[0],
        sparse_attention_config={
            "sparse_num_index_heads": 1,
            "sparse_index_dim": 16,
            "sparse_block_size": 4,
            "sparse_topk_blocks": 2,
            "sparse_init_block": 0,
            "sparse_local_block": 1,
            "sparse_score_type": "max",
            "sparse_attention_freq": [1],
        },
    )
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
        dim=32,
        inter_dim=64,
        moe_inter_dim=64,
        norm_topk_prob=True,
    )
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
    )
    block = Block(0, config, moe, backend).float()
    block.init_weights(torch.device("cpu"))
    return block


def _cpu_forward(
    self: MiniMaxM3CPSparseAttention,
    x: torch.Tensor,
    *,
    freqs_cis: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Enter the local path on CPU for checkpoint bookkeeping coverage.

    Args:
        x: Hidden states [batch, sequence, hidden].
        freqs_cis: Rotary values [batch, sequence, rotary_dim].
        attention_mask: Optional boolean keep mask [batch, sequence].
        padding_mask: Unused token padding indicator [batch, sequence].

    Returns:
        Output [batch, sequence, hidden].
    """
    return self._local_sparse_forward(x, freqs_cis=freqs_cis, attention_mask=attention_mask)


def _cpu_attention(
    self: MiniMaxM3CPSparseAttention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_sel: torch.Tensor,
    q_positions: torch.Tensor,
    keep_mask: torch.Tensor | None,
    rescue_pad_queries: bool,
) -> torch.Tensor:
    """CPU SDPA substitute for the CUDA kernel, retaining all trainable paths.

    Args:
        q: Queries [batch, sequence, query_heads, head_dim].
        k: Keys [batch, sequence, kv_heads, head_dim].
        v: Values [batch, sequence, kv_heads, head_dim].
        block_sel: Boolean selected blocks [batch, index_heads, sequence, blocks].
        q_positions: Query token slots [sequence].
        keep_mask: Optional caller keep mask [batch, heads, sequence, sequence].
        rescue_pad_queries: Local path must pass False.

    Returns:
        Attention values [batch, sequence, query_heads, head_dim].
    """
    assert not rescue_pad_queries
    length = k.shape[1]
    keys = torch.arange(length, device=q.device)
    mask = block_sel[..., keys // self.indexer.block_size]
    mask = mask.repeat_interleave(self.num_heads // self.indexer.num_index_heads, dim=1)
    mask = mask & (keys[None, :] <= q_positions[:, None])
    if keep_mask is not None:
        mask = mask & keep_mask
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=mask,
        enable_gqa=True,
    ).transpose(1, 2)


@pytest.mark.parametrize("selective", [False, True])
def test_sparse_replay_interleaved_microbatches(monkeypatch: pytest.MonkeyPatch, selective: bool) -> None:
    """Two outstanding calls must keep distinct routes and matching full-block gradients."""
    import nemo_automodel.components.models.minimax_m3_vl.cp_sparse_attn as cp

    torch.manual_seed(72)
    monkeypatch.setattr(MiniMaxM3CPSparseAttention, "forward", _cpu_forward)
    monkeypatch.setattr(MiniMaxM3CPSparseAttention, "_flex_sparse_attention", _cpu_attention)
    block = _block()
    baseline = copy.deepcopy(block)
    positions = torch.arange(17, dtype=torch.float32)
    angles = positions[:, None] / (10000 ** (torch.arange(0, 8, 2) / 8))
    freqs = torch.cat((angles.cos(), angles.sin()), -1).unsqueeze(0)
    inputs = [torch.randn(1, 17, 32, requires_grad=True) for _ in range(2)]
    references = [x.detach().clone().requires_grad_() for x in inputs]
    weights = [torch.randn_like(x) for x in inputs]
    selected: list[torch.Tensor] = []
    original_select = cp.select_sparse_blocks

    def capture(*args: torch.Tensor, **kwargs: int | str | torch.Tensor) -> torch.Tensor:
        """Record selector output without retaining quadratic intermediate tensors.

        Args:
            args: Index queries [B,S,H,D] and keys [B,S,1,D].
            kwargs: Selector scalar settings and query positions [S].

        Returns:
            Selected blocks [B,H,S,blocks].
        """
        result = original_select(*args, **kwargs)
        selected.append(result.detach().clone())
        return result

    monkeypatch.setattr(cp, "select_sparse_blocks", capture)
    expected = [baseline(x, freqs_cis=freqs) for x in references]
    expected_selections = selected.copy()
    selected.clear()
    score_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def policy(
        ctx: SelectiveCheckpointContext, op: torch._ops.OpOverload, *args: torch.Tensor, **kwargs: object
    ) -> CheckpointPolicy:
        """Save trainable matmuls while auditing visible operand layouts.

        Args:
            ctx: PyTorch forward/recompute policy context.
            op: Dispatched operator.
            args: Operator inputs with operator-defined tensor layouts.
            kwargs: Operator keyword arguments with operator-defined layouts.

        Returns:
            MUST_SAVE for mm/bmm and PREFER_RECOMPUTE otherwise.
        """
        if op in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default):
            if not ctx.is_recompute:
                score_shapes.append((tuple(args[0].shape), tuple(args[1].shape)))
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    inner = (lambda: create_selective_checkpoint_contexts(policy)) if selective else None
    contexts = block.nemo_checkpoint_context_fn(inner)
    actual = [checkpoint(block, x, freqs_cis=freqs, use_reentrant=False, context_fn=contexts) for x in inputs]
    assert len(selected) == 2
    assert not torch.equal(selected[0], selected[1])
    for got, ref in zip(selected, expected_selections):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
        # S=17 exceeds the 2-block budget: later queries really discard valid blocks.
        assert bool((got[..., -1, :-1].sum(-1) < 4).all())
    for i in (1, 0):
        torch.testing.assert_close(actual[i], expected[i], rtol=0, atol=0)
        (actual[i] * weights[i]).sum().backward()
        (expected[i] * weights[i]).sum().backward()
        torch.testing.assert_close(inputs[i].grad, references[i].grad, rtol=2e-5, atol=2e-6)
    assert len(selected) == 2, "checkpoint recompute must not rerun the frozen indexer"
    assert sparse_route_replay.current() is None
    for (name, parameter), (_, reference) in zip(block.named_parameters(), baseline.named_parameters()):
        if "indexer" in name:
            assert parameter.grad is reference.grad is None
        else:
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=2e-5, atol=2e-6)
    if selective:
        assert score_shapes, "negative control: trainable matmuls must remain visible to SAC"
        assert ((1, 17, 16), (1, 16, 17)) not in score_shapes, "quadratic indexer scores must not enter SAC"


def test_no_checkpoint_retention() -> None:
    """A normal forward must not retain a selection in the replay channel."""
    attn = _block().self_attn
    x = torch.randn(1, 17, 32)
    freqs = torch.cat((torch.ones(1, 17, 4), torch.zeros(1, 17, 4)), -1)
    selection = attn._select_local_blocks(x, freqs_cis=freqs, q_positions=torch.arange(17))
    reference = weakref.ref(selection)
    del selection
    gc.collect()
    assert reference() is None
    assert sparse_route_replay.current() is None


@pytest.mark.parametrize("exclude_region", [False, True])
def test_quadratic_sac_cache_negative_control(monkeypatch: pytest.MonkeyPatch, exclude_region: bool) -> None:
    """The SAC audit must detect an indexer score if region exclusion is removed."""
    import nemo_automodel.components.models.minimax_m3_vl.cp_sparse_attn as cp

    if not exclude_region:
        monkeypatch.setattr(cp, "_disable_current_modes", nullcontext)
    shapes = []

    def policy(
        ctx: SelectiveCheckpointContext, op: torch._ops.OpOverload, *args: torch.Tensor, **kwargs: object
    ) -> CheckpointPolicy:
        """Audit BMM operands [batch, rows, reduction] and [batch, reduction, columns].

        Args:
            ctx: Checkpoint policy context.
            op: Dispatched operator.
            args: Operator inputs with operator-defined layouts.
            kwargs: Operator keyword arguments with operator-defined layouts.

        Returns:
            MUST_SAVE to make the control expose the quadratic score.
        """
        if op == torch.ops.aten.bmm.default:
            shapes.append((tuple(args[0].shape), tuple(args[1].shape)))
        return CheckpointPolicy.MUST_SAVE

    attn = _block().self_attn
    freqs = torch.cat((torch.ones(1, 17, 4), torch.zeros(1, 17, 4)), -1)
    forward, _ = create_selective_checkpoint_contexts(policy)
    with forward:
        attn._select_local_blocks(torch.randn(1, 17, 32), freqs_cis=freqs, q_positions=torch.arange(17))
    assert (((1, 17, 16), (1, 16, 17)) in shapes) is not exclude_region


def test_checkpoint_selection_released_with_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed checkpoint graph must release its private selection recorder."""
    import nemo_automodel.components.models.minimax_m3_vl.cp_sparse_attn as cp

    monkeypatch.setattr(MiniMaxM3CPSparseAttention, "forward", _cpu_forward)
    monkeypatch.setattr(MiniMaxM3CPSparseAttention, "_flex_sparse_attention", _cpu_attention)
    references: list[weakref.ReferenceType[torch.Tensor]] = []
    original = cp.select_sparse_blocks

    def capture(*args: torch.Tensor, **kwargs: int | str | torch.Tensor) -> torch.Tensor:
        """Track lifetime without retaining the selection.

        Args:
            args: Index queries [B,S,H,D] and keys [B,S,1,D].
            kwargs: Selector settings and query positions [S].

        Returns:
            Selected blocks [B,H,S,blocks].
        """
        result = original(*args, **kwargs)
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(cp, "select_sparse_blocks", capture)
    block = _block()
    x = torch.randn(1, 17, 32, requires_grad=True)
    freqs = torch.cat((torch.ones(1, 17, 4), torch.zeros(1, 17, 4)), -1)
    output = checkpoint(
        block, x, freqs_cis=freqs, use_reentrant=False, context_fn=block.nemo_checkpoint_context_fn(None)
    )
    assert len(references) == 1 and references[0]() is not None
    output.sum().backward()
    assert len(references) == 1
    del output
    gc.collect()
    assert references[0]() is None
    assert sparse_route_replay.current() is None
