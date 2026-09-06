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

"""Real SM100 packed, sparse, checkpoint and pipeline training parity."""

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m3_vl import _msa as msa
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLTextConfig
from nemo_automodel.components.models.minimax_m3_vl.layers import MiniMaxM3Attention, MiniMaxM3Indexer
from nemo_automodel.components.models.minimax_m3_vl.model import MiniMaxM3SparseForCausalLM
from nemo_automodel.shared.import_utils import UnavailableError

_BLOCK, _HEADS, _KV_HEADS, _DIM, _TOPK = 128, 64, 4, 128, 16
_SCALE = _DIM**-0.5
_PP_WORKER = __name__ == "__main__" and "--pp-worker" in sys.argv


def _unavailable() -> str | None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        return "requires an SM100 GPU"
    try:
        msa._require_msa()
        msa._require_msa_backward()
    except UnavailableError:
        return "requires uv sync --extra msa"
    return None


_SKIP_REASON = None if _PP_WORKER else _unavailable()
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "SM100 available")


def _backend(sparse_attn: str = "msa") -> BackendConfig:
    return BackendConfig(
        attn="sdpa",
        sparse_attn=sparse_attn,
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def _config() -> MiniMaxM3VLTextConfig:
    return MiniMaxM3VLTextConfig(
        hidden_size=32,
        intermediate_size=32,
        dense_intermediate_size=48,
        shared_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        head_dim=_DIM,
        vocab_size=64,
        max_position_embeddings=32,
        rotary_dim=64,
        num_local_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=0,
        moe_layer_freq=[0, 0],
        num_mtp_modules=0,
        attention_dropout=0.0,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_num_index_heads": _KV_HEADS,
            "sparse_index_dim": _DIM,
            "sparse_block_size": _BLOCK,
            "sparse_topk_blocks": _TOPK,
            "sparse_init_block": 0,
            "sparse_local_block": 1,
            "sparse_score_type": "max",
            "sparse_attention_freq": [0, 1],
            "sparse_disable_index_value": [0, 1],
        },
    )


def _assert_error(actual: torch.Tensor, expected: torch.Tensor, absolute: float, relative: float) -> None:
    """Compare same-shaped tensors of arbitrary rank using max-absolute and L2-relative error."""
    difference = (actual.float() - expected.float()).abs()
    max_abs = difference.max().item()
    norm_rel = (difference.norm() / expected.float().norm().clamp_min(1e-12)).item()
    assert max_abs <= absolute and norm_rel <= relative, (max_abs, norm_rel)


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    support: torch.Tensor,
    lengths: tuple[int, ...],
    rows: torch.Tensor,
) -> torch.Tensor:
    """FP32 q[tokens,64,128], k/v[tokens,4,128], support[4,tokens,16], rows[selected] -> O[selected,64,128]."""
    device = q.device
    lengths_tensor = torch.tensor(lengths, device=device)
    documents = torch.repeat_interleave(torch.arange(len(lengths), device=device), lengths_tensor)
    starts = torch.repeat_interleave(lengths_tensor.cumsum(0) - lengths_tensor, lengths_tensor)
    positions = torch.arange(q.shape[0], device=device)
    local_blocks = (positions - starts) // _BLOCK
    selected_keys = (support[:, rows, :, None] == local_blocks[None, None, None, :]).any(dim=2)
    same_document = documents[rows, None] == documents[None, :]
    causal = positions[None, :] <= rows[:, None]
    keep = (selected_keys & same_document & causal).repeat_interleave(_HEADS // _KV_HEADS, dim=0)
    query = q[rows].transpose(0, 1)
    key = k.repeat_interleave(_HEADS // _KV_HEADS, dim=1).transpose(0, 1)
    value = v.repeat_interleave(_HEADS // _KV_HEADS, dim=1).transpose(0, 1)
    scores = (query @ key.transpose(-1, -2) * _SCALE).masked_fill(~keep, float("-inf"))
    return (scores.softmax(-1) @ value).transpose(0, 1)


def _check_flat_attention(
    layout: msa._MSAPackedLayout,
    support: torch.Tensor,
    lengths: tuple[int, ...],
    rows: torch.Tensor,
) -> None:
    """Check compact O/dQ/dK/dV for layout, support[4,tokens,16] and selected query rows[selected]."""
    device = support.device
    generator = torch.Generator(device=device).manual_seed(20260902)
    qkv = [
        torch.randn(sum(lengths), h, _DIM, device=device, dtype=torch.bfloat16, generator=generator, requires_grad=True)
        for h in (_HEADS, _KV_HEADS, _KV_HEADS)
    ]
    out = msa._MSAFlatAttention(_SCALE)(*qkv, support, layout=layout)
    selected_grad = torch.randn(rows.numel(), _HEADS, _DIM, device=device, dtype=torch.bfloat16, generator=generator)
    out.backward(torch.zeros_like(out).index_copy(0, rows, selected_grad))
    reference_qkv = [tensor.detach().float().requires_grad_() for tensor in qkv]
    reference = _reference(*reference_qkv, support, lengths, rows)
    reference.backward(selected_grad.float())
    _assert_error(out[rows], reference, 0.025, 0.006)
    for actual, expected, absolute in zip(qkv, reference_qkv, (0.035, 0.075, 0.1), strict=True):
        _assert_error(actual.grad, expected.grad, absolute, 0.007)


def test_packed_forward_backward_parity() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    lengths = (127, 129, 1, 2, 3, 5, 128)
    documents = torch.zeros(2, 260, dtype=torch.int64, device=device)
    documents[0, :127], documents[0, 127:256] = 1, 2
    position = 0
    for document, length in enumerate(lengths[2:], start=1):
        documents[1, position : position + length] = document
        position += length
    support_rows = []
    for length in lengths:
        current = torch.arange(length, device=device)[:, None] // _BLOCK
        blocks = torch.arange(_TOPK, device=device)[None, :]
        support_rows.append(torch.where(blocks <= current, blocks, -1))
    support = torch.cat(support_rows).expand(_KV_HEADS, -1, -1).to(torch.int32).contiguous()
    _check_flat_attention(
        msa._MSAPackedLayout.build(documents), support, lengths, torch.arange(sum(lengths), device=device)
    )


def test_top16_truncation_large_schedule_parity() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    tokens = 17 * _BLOCK + 1
    layout = msa._MSAPackedLayout.build(torch.ones(1, tokens, dtype=torch.int64, device=device))
    config = _config()
    with device:
        indexer = MiniMaxM3Indexer(config, config.sparse_attention_config, _backend())
    index_q = torch.zeros(tokens, _KV_HEADS, _DIM, dtype=torch.bfloat16, device=device)
    index_q[..., 0] = 1
    scores = torch.tensor(
        (-90, 90, -80, 80, -70, 70, -60, 60, -50, 50, -40, 40, -30, 30, -20, 20, -10, 0),
        dtype=torch.bfloat16,
        device=device,
    )
    index_k = torch.zeros(tokens, 1, _DIM, dtype=torch.bfloat16, device=device)
    index_k[:, 0, 0] = scores.repeat_interleave(_BLOCK)[:tokens]
    support = indexer._select_msa_blocks(index_q, index_k, layout=layout)
    final = support[0, -1]
    assert set(final.tolist()) == set(range(18)) - {0, 2}
    _check_flat_attention(layout, support, (tokens,), torch.arange(tokens - 8, tokens, device=device))


def test_checkpointed_layer_projection_gradient_parity() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    config = _config()
    torch.manual_seed(20260903)
    with device:
        actual_layer = MiniMaxM3Attention(config, _backend(), is_sparse_attention_layer=True)
        reference_layer = MiniMaxM3Attention(config, _backend("generic"), is_sparse_attention_layer=True)
    reference_layer.load_state_dict(actual_layer.state_dict())
    documents = torch.ones(2, 16, dtype=torch.int64, device=device)
    documents[0, -1] = 0
    keep = documents > 0
    layout = msa._MSAPackedLayout.build(documents)
    positions = torch.arange(16, device=device).view(1, -1, 1)
    inv_freq = 10_000 ** (-torch.arange(0, 64, 2, dtype=torch.float32, device=device) / 64)
    angles = positions * inv_freq
    frequencies = torch.cat((angles.cos(), angles.sin()), dim=-1).expand(2, -1, -1)
    actual_x = torch.randn(2, 16, config.hidden_size, dtype=torch.bfloat16, device=device, requires_grad=True)
    reference_x = actual_x.detach().clone().requires_grad_()
    upstream = torch.randn_like(actual_x).masked_fill(~keep.unsqueeze(-1), 0)

    def recompute(hidden: torch.Tensor) -> torch.Tensor:
        """Map BF16 hidden[batch,sequence,hidden] to attention output with the same shape."""
        return actual_layer(hidden, freqs_cis=frequencies, attention_mask=keep, _msa_layout=layout)

    actual = checkpoint(recompute, actual_x, use_reentrant=False)
    expected = reference_layer(reference_x, freqs_cis=frequencies, attention_mask=keep)
    actual.backward(upstream)
    expected.backward(upstream)
    assert torch.count_nonzero(actual[~keep]) == 0
    assert torch.count_nonzero(actual_x.grad[~keep]) == torch.count_nonzero(reference_x.grad[~keep]) == 0
    _assert_error(layout.pack(actual), layout.pack(expected), 0.04, 0.008)
    _assert_error(layout.pack(actual_x.grad), layout.pack(reference_x.grad), 0.04, 0.012)
    actual_parameters = dict(actual_layer.named_parameters())
    expected_parameters = dict(reference_layer.named_parameters())
    for name, tolerance in (
        ("q_proj.weight", 0.125),
        ("k_proj.weight", 0.5),
        ("v_proj.weight", 0.5),
        ("o_proj.weight", 0.125),
    ):
        gradient = actual_parameters[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
        _assert_error(gradient, expected_parameters[name].grad, tolerance, 0.015)


def _run_pp_worker() -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.distributed.pipelining import AutoPipeline

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    assert _unavailable() is None
    dist.init_process_group("nccl")
    try:
        torch.manual_seed(20260906)
        with device:
            model = MiniMaxM3SparseForCausalLM(_config(), backend=_backend())
            model.initialize_weights(buffer_device=device, dtype=torch.bfloat16)
        model.train()
        reference = copy.deepcopy(model)

        def loss_fn(output: torch.Tensor, upstream: torch.Tensor) -> torch.Tensor:
            """Reduce BF16 output/upstream[microbatch,sequence,vocab] to an FP32 scalar inner product."""
            return (output.float() * upstream.float()).sum()

        pipeline = AutoPipeline(
            world_mesh=init_device_mesh("cuda", (2, 1), mesh_dim_names=("pp", "dp")),
            pp_axis_name="pp",
            dp_axis_names=("dp",),
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=2,
            device=device,
            dtype=torch.bfloat16,
            pp_seq_len=8,
        ).build(model, loss_fn=loss_fn)
        generator = torch.Generator().manual_seed(20260907)
        input_ids = torch.randint(1, 64, (2, 8), generator=generator).to(device)
        documents = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2], [7, 7, 7, 9, 9, 9, 9, 9]], device=device)
        same_document = documents.unsqueeze(-1) == documents.unsqueeze(-2)
        mask = (same_document & torch.ones(8, 8, dtype=torch.bool, device=device).tril()).unsqueeze(1)
        upstream = torch.randn(2, 8, 64, generator=generator).to(device=device, dtype=torch.bfloat16)
        losses = [] if pipeline.info.has_last_stage else None
        actual = pipeline.step(input_ids, target=upstream, losses=losses, attention_mask=mask)
        expected = reference(input_ids, attention_mask=mask)
        expected_loss = loss_fn(expected, upstream)
        expected_loss.backward()
        if pipeline.info.has_last_stage:
            assert actual is not None and losses is not None and torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
            torch.testing.assert_close(torch.stack(losses).sum(), expected_loss, rtol=0.01, atol=0.01)
        else:
            assert actual is None
        expected_parameters = dict(reference.named_parameters())
        compared = 0
        for name, parameter in pipeline.parts[0].named_parameters():
            expected_gradient = expected_parameters[name].grad
            if expected_gradient is None:
                assert parameter.grad is None, name
            else:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                torch.testing.assert_close(
                    parameter.grad.float(),
                    expected_gradient.float(),
                    rtol=0.04,
                    atol=0.02,
                    msg=lambda message: f"[rank {dist.get_rank()}] {name}: {message}",
                )
                compared += 1
        assert compared > 0
        if dist.get_rank() == 0:
            print("MINIMAX_M3_MSA_PP_PASS", flush=True)
    finally:
        dist.destroy_process_group()


def test_two_rank_pipeline_forward_backward_parity() -> None:
    if torch.cuda.device_count() < 2 or any(torch.cuda.get_device_capability(i) != (10, 0) for i in range(2)):
        pytest.skip("requires two SM100 GPUs")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(env.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")[:2])
    root = str(Path(__file__).resolve().parents[4])
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", __file__, "--pp-worker"],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "MINIMAX_M3_MSA_PP_PASS" in result.stdout, result.stdout


if _PP_WORKER:
    _run_pp_worker()
