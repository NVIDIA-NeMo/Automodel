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

"""CUDA local sparse attention against independently enumerated selected-key masks."""

import copy
import json

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLTextConfig
from nemo_automodel.components.models.minimax_m3_vl.cp_sparse_attn import MiniMaxM3CPSparseAttention
from nemo_automodel.components.models.minimax_m3_vl.layers import MiniMaxM3RMSNorm

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention"),
    pytest.mark.timeout(300),
]


def _project(
    module: torch.nn.Linear,
    x: torch.Tensor,
    heads: int,
    norm: MiniMaxM3RMSNorm | None,
    freqs: torch.Tensor | None,
) -> torch.Tensor:
    """Independently project, normalize and apply partial half-split RoPE.

    Args:
        module: Linear projection with weight [heads * dim, hidden].
        x: Hidden states [batch, sequence, hidden].
        heads: Number of projected heads.
        norm: Per-head Gemma RMSNorm with weight [dim], or no normalization.
        freqs: Rotary cos/sin [batch, sequence, rotary_dim], or no rotation.

    Returns:
        Projected states [batch, sequence, heads, dim].
    """
    dim = module.out_features // heads
    y = F.linear(x, module.weight).reshape(*x.shape[:2], heads, dim)
    if norm is not None:
        yf = y.float()
        y = (yf / torch.sqrt((yf * yf).mean(-1, keepdim=True) + norm.eps) * (1 + norm.weight.float())).to(y.dtype)
    if freqs is not None:
        cos, sin = freqs.chunk(2, -1)
        cos, sin = cos.unsqueeze(2).to(y.dtype), sin.unsqueeze(2).to(y.dtype)
        half = freqs.shape[-1] // 2
        first, second, rest = y[..., :half], y[..., half : 2 * half], y[..., 2 * half :]
        y = torch.cat((first * cos - second * sin, second * cos + first * sin, rest), -1)
    return y


def _reference(
    attn: MiniMaxM3CPSparseAttention,
    x: torch.Tensor,
    freqs: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    dense: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return output [B,S,H], selected blocks [B,index_heads,S,blocks] and dense keep-mask [B,heads,S,S].

    Args:
        attn: Independently copied attention weights/configuration.
        x: Hidden states [batch, sequence, hidden].
        freqs: Partial rotary cos/sin [batch, sequence, rotary_dim].
        mask: Caller padding [B,S] or explicit broadcastable [B,H,S,S] keep-mask.
        dense: Negative control that deliberately discards sparse selection.
    """
    q = _project(attn.q_proj, x, attn.num_heads, attn.q_norm, freqs)
    k = _project(attn.k_proj, x, attn.num_kv_heads, attn.k_norm, freqs)
    v = _project(attn.v_proj, x, attn.num_kv_heads, None, None)
    batch, length = x.shape[:2]
    with torch.no_grad():
        idx = attn.indexer
        iq = _project(idx.index_q_proj, x, idx.num_index_heads, idx.index_q_norm, freqs)
        ik = _project(idx.index_k_proj, x, 1, idx.index_k_norm, freqs)
        # Enumerate key blocks independently. Unlike production, each reduction
        # slices the actual causal key range; no padded score view/helper is used.
        positions = torch.arange(length, device=x.device)
        block_scores = []
        for start in range(0, length, idx.block_size):
            stop = min(start + idx.block_size, length)
            score = torch.einsum("bqhd,bkd->bhqk", iq.float(), ik[:, start:stop, 0].float())
            score = score * idx.index_head_dim**-0.5
            score.masked_fill_(positions[start:stop] > positions[:, None], float("-inf"))
            block_scores.append(score.amax(-1))
        ranking = torch.stack(block_scores, -1)
        own = positions // idx.block_size
        if idx.local_blocks:
            ranking.scatter_(-1, own.view(1, 1, -1, 1).expand(batch, idx.num_index_heads, -1, 1), float("inf"))
        ranking[..., : idx.init_blocks] = float("inf")
        picked = ranking.topk(min(idx.topk_blocks, ranking.shape[-1]), dim=-1).indices
        selected = torch.zeros_like(ranking, dtype=torch.bool).scatter_(-1, picked, True)
        selected &= torch.arange(ranking.shape[-1], device=x.device) <= own[:, None]
        keep = selected[..., own] & (positions <= positions[:, None])
        if dense:
            keep = torch.ones_like(keep).tril()
        keep = keep.repeat_interleave(attn.num_heads // idx.num_index_heads, dim=1)
        if mask is not None:
            keep = keep & (mask[:, None, None, :] if mask.dim() == 2 else mask)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2).repeat_interleave(attn.num_heads // attn.num_kv_heads, dim=1)
    v = v.transpose(1, 2).repeat_interleave(attn.num_heads // attn.num_kv_heads, dim=1)
    logits = (q.float() @ k.float().transpose(-1, -2)) * attn.head_dim**-0.5
    logits = logits.masked_fill(~keep, float("-inf"))
    valid = keep.any(-1, keepdim=True)
    # Fully masked rows have zero attention/output and zero gradient, as SDPA.
    probs = torch.softmax(torch.where(valid, logits, torch.zeros_like(logits)), -1)
    probs = torch.where(valid, probs, torch.zeros_like(probs))
    # Mathematical oracle: preserve FP32 probabilities and PV accumulation.
    # Rounding normalized probabilities before PV adds a second BF16 rounding
    # absent from the mathematical attention definition.
    out = (probs @ v.float()).to(q.dtype).transpose(1, 2).flatten(2)
    return F.linear(out, attn.o_proj.weight), selected, keep


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Report error scale and near-zero counts for identically shaped tensors.

    Args:
        actual: Actual output or gradient tensor.
        expected: Independent mathematical reference with the same shape.

    Returns:
        Absolute/relative errors, norms, and diagnostics of the old elementwise
        BF16 gradient gate. Counts are retained even when the norm gate passes.
    """
    delta = actual.detach().float() - expected.detach().float()
    return {
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / expected.detach().float().norm().clamp_min(1e-12)).item(),
        "actual_norm": actual.detach().float().norm().item(),
        "expected_norm": expected.detach().float().norm().item(),
        "error_norm": delta.norm().item(),
        "near_zero_elements": (expected.detach().abs() < 0.06).sum().item(),
        "original_elementwise_gate_failures": (delta.abs() > 0.06 + 0.06 * expected.detach().abs()).sum().item(),
    }


@pytest.mark.parametrize(
    "dtype,mask_kind,local_blocks,length",
    [
        (torch.float32, "none", 1, 384),
        (torch.bfloat16, "padding", 1, 385),
        (torch.bfloat16, "packed", 1, 384),
        (torch.float32, "none", 0, 384),
        (torch.float32, "key_broadcast_true", 1, 384),
        (torch.float32, "key_broadcast_false", 1, 384),
        (torch.float32, "key_broadcast_queries", 1, 384),
        (torch.bfloat16, "none", 1, 4096),
    ],
)
def test_local_flex_forward_backward(
    dtype: torch.dtype, mask_kind: str, local_blocks: int, length: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(119)
    torch.backends.cuda.matmul.allow_tf32 = False
    real_shape = length == 4096
    cfg = MiniMaxM3VLTextConfig(
        hidden_size=6144 if real_shape else 128,
        num_attention_heads=64 if real_shape else 4,
        num_key_value_heads=4 if real_shape else 2,
        head_dim=128 if real_shape else 64,
        rotary_dim=64 if real_shape else 32,
        num_hidden_layers=1,
        num_mtp_modules=0,
        sparse_attention_config={
            "sparse_num_index_heads": 4 if real_shape else 2,
            "sparse_index_dim": 128 if real_shape else 64,
            "sparse_block_size": 128,
            "sparse_topk_blocks": 16 if real_shape else 2,
            "sparse_init_block": 0,
            "sparse_local_block": local_blocks,
            "sparse_score_type": "max",
        },
    )
    backend = BackendConfig(attn="sdpa", linear="torch", rope_fusion=False, experts="torch", dispatcher="torch")
    attn = MiniMaxM3CPSparseAttention(cfg, backend).cuda().to(dtype)
    with torch.no_grad():
        for p in attn.parameters():
            if real_shape and p.ndim == 2:
                p.normal_(mean=0.0, std=0.02)
            else:
                p.uniform_(-0.1, 0.1)
    reference = copy.deepcopy(attn)
    x = torch.randn(1, length, cfg.hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    positions = torch.arange(length, device="cuda", dtype=torch.float32)
    if mask_kind == "packed":
        positions[128:] -= 128
    angles = positions[:, None] / (10000 ** (torch.arange(0, cfg.rotary_dim, 2, device="cuda") / cfg.rotary_dim))
    freqs = torch.cat((angles.cos(), angles.sin()), -1).unsqueeze(0)
    mask = None
    if mask_kind == "padding":
        mask = torch.ones(1, length, dtype=torch.bool, device="cuda")
        mask[:, 129:133] = False
    elif mask_kind.startswith("key_broadcast"):
        mask = torch.ones(1, 1, 1, 1, dtype=torch.bool, device="cuda")
        if mask_kind == "key_broadcast_false":
            mask = ~mask
        elif mask_kind == "key_broadcast_queries":
            mask = (torch.arange(length, device="cuda") % 3 != 0)[None, None, :, None]
    elif mask_kind == "packed":
        ids = torch.arange(length, device="cuda") >= 128
        mask = (ids[:, None] == ids[None, :])[None, None].expand(1, 1, length, length).clone()
        mask[:, :, -1] = False
        mask[:, :, :, -1] = False

    import nemo_automodel.components.models.minimax_m3_vl.cp_sparse_attn as cp
    import nemo_automodel.components.models.minimax_m3_vl.layers as layers

    selections = []
    mask_metrics = {}
    original_select = cp.select_sparse_blocks
    original_flex_getter = cp._get_compiled_flex_attention

    def record_flex():
        flex = original_flex_getter()

        def run(q, k, v, **kwargs):
            block_mask = kwargs["block_mask"]
            visited = block_mask.kv_num_blocks.sum()
            if block_mask.full_kv_num_blocks is not None:
                visited = visited + block_mask.full_kv_num_blocks.sum()
            mask_batch, mask_heads = block_mask.kv_num_blocks.shape[:2]
            rows = (length + 127) // 128
            causal = mask_batch * mask_heads * rows * (rows + 1) // 2
            mask_metrics.update(
                {
                    "batch": mask_batch,
                    "heads": mask_heads,
                    "visited_tiles": visited.item(),
                    "causal_tiles": causal,
                    "extra_tile_skip_fraction": 1 - visited.item() / causal,
                }
            )
            return flex(q, k, v, **kwargs)

        return run

    def record_selection(*args, **kwargs):
        result = original_select(*args, **kwargs)
        selections.append(result.detach())
        return result

    def reject_dense_mask(*args, **kwargs):
        raise AssertionError("local CUDA attention materialized a quadratic per-head keep-mask")

    monkeypatch.setattr(cp, "_get_compiled_flex_attention", record_flex)
    monkeypatch.setattr(cp, "select_sparse_blocks", record_selection)
    monkeypatch.setattr(layers, "build_block_sparse_attn_mask", reject_dense_mask)
    actual = attn(x, freqs_cis=freqs, attention_mask=mask)
    expected, selected, keep = _reference(reference, xr, freqs, mask)
    assert len(selections) == 1
    torch.testing.assert_close(selections[0], selected, rtol=0, atol=0)
    boundary = attn.indexer.topk_blocks * attn.indexer.block_size
    dropped = keep[0, :, boundary:].sum(-1) < torch.arange(boundary + 1, length + 1, device="cuda")
    assert dropped.any()
    metrics = {
        "length": length,
        "dtype": str(dtype),
        "mask": mask_kind,
        "dropped_query_heads": dropped.sum().item(),
        "selected_set_agreement": 1.0,
        "prefix": _error(actual[:, :boundary], expected[:, :boundary]),
        "sparse_tail": _error(actual[:, boundary:], expected[:, boundary:]),
        "block_mask": dict(mask_metrics),
    }
    if mask_kind == "packed":
        torch.testing.assert_close(actual[:, -1], torch.zeros_like(actual[:, -1]), rtol=0, atol=0)
    tolerance = 3e-5 if dtype == torch.float32 else 0.02
    for start, end in ((0, boundary), (boundary, length)):
        torch.testing.assert_close(actual[:, start:end], expected[:, start:end], rtol=tolerance, atol=tolerance)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    grad_tolerance = 2e-4 if dtype == torch.float32 else 0.06
    metrics["input_gradient"] = _error(x.grad, xr.grad)
    if dtype == torch.float32:
        torch.testing.assert_close(x.grad, xr.grad, rtol=grad_tolerance, atol=grad_tolerance)
    else:
        assert metrics["input_gradient"]["relative_l2"] <= 0.006
    metrics["parameter_gradients"] = {}
    for (name, parameter), (ref_name, ref_parameter) in zip(attn.named_parameters(), reference.named_parameters()):
        assert name == ref_name
        if name.startswith("indexer."):
            assert parameter.grad is None and ref_parameter.grad is None
        else:
            metrics["parameter_gradients"][name] = _error(parameter.grad, ref_parameter.grad)
            if dtype == torch.float32:
                torch.testing.assert_close(parameter.grad, ref_parameter.grad, rtol=grad_tolerance, atol=grad_tolerance)
            else:
                # Real-width baseline SDPA and both Flex layouts show the same
                # 0.3-0.5% norm-error envelope against the FP32 oracle. BF16
                # accumulation near cancellation makes a fixed elementwise
                # absolute threshold unsuitable for large weight gradients.
                assert metrics["parameter_gradients"][name]["relative_l2"] <= 0.006
    if real_shape:
        # With no padding and a forced current block, the CP self-rescue is
        # provably a no-op and selects the original ungrouped Flex layout.
        own = torch.arange(length, device="cuda") // attn.indexer.block_size
        assert selected.gather(-1, own.view(1, 1, -1, 1).expand(1, 4, -1, 1)).all()
        control = copy.deepcopy(attn)
        control.zero_grad(set_to_none=True)
        xc = x.detach().clone().requires_grad_()
        original_attention = control._flex_sparse_attention

        def ungrouped_attention(*args, **kwargs):
            kwargs["rescue_pad_queries"] = True
            return original_attention(*args, **kwargs)

        with monkeypatch.context() as context:
            context.setattr(control, "_flex_sparse_attention", ungrouped_attention)
            control_output = control(xc, freqs_cis=freqs)
        torch.testing.assert_close(actual, control_output, rtol=0, atol=0)
        control_output.backward(gradient)
        torch.testing.assert_close(x.grad, xc.grad, rtol=0, atol=0)
        for (name, parameter), (control_name, control_parameter) in zip(
            attn.named_parameters(), control.named_parameters()
        ):
            assert name == control_name
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, control_parameter.grad, rtol=0, atol=0)
        metrics["grouped_vs_ungrouped_forward_and_gradients_bitwise_equal"] = True
    if mask_kind == "none":
        with torch.no_grad():
            wrong, _, _ = _reference(reference, xr, freqs, mask, dense=True)
        assert (wrong[:, boundary:] - expected[:, boundary:]).abs().max() > 0.01
    print(json.dumps(metrics, sort_keys=True))
