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

import inspect
import logging

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaIndexer as HFIndexer

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm_moe_dsa.layers import GlmMoeDsaIndexer

# HF 5.18 requires this unused compatibility argument; earlier releases omit it.
_HF_INDEXER_KWARGS = {"position_ids": None} if "position_ids" in inspect.signature(HFIndexer.forward).parameters else {}


class _CaptureTopKScores(TorchDispatchMode):
    """Retain the reference's actual pre-top-k scores for boundary diagnostics."""

    def __init__(self) -> None:
        super().__init__()
        self.scores: torch.Tensor | None = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """Preserve the dispatch contract for arbitrary operator tensor layouts."""
        if func is torch.ops.aten.topk.default:
            self.scores = args[0].detach()
        return func(*args, **(kwargs or {}))


@pytest.mark.gpu
@pytest.mark.parametrize(
    "dtype,seq_len,packed,compiled",
    [
        (torch.float32, 1, False, False),
        (torch.float32, 17, True, False),
        (torch.float16, 17, False, False),
        (torch.bfloat16, 17, True, False),
        (torch.bfloat16, 17, False, True),
        (torch.bfloat16, 4096, False, False),
    ],
)
def test_dense_indexer_preserves_hf_selection_on_cuda(dtype, seq_len, packed, compiled):
    torch.manual_seed(19)
    config = GlmMoeDsaConfig(
        hidden_size=6144 if seq_len == 4096 else 128,
        q_lora_rank=2048 if seq_len == 4096 else 64,
        index_n_heads=32,
        index_head_dim=128,
        qk_rope_head_dim=64,
        index_topk=2048 if seq_len == 4096 else 4,
        torch_dtype=dtype,
    )
    backend = BackendConfig(linear="torch", attn="sdpa", rms_norm="torch", experts="torch", dispatcher="torch")
    indexer = GlmMoeDsaIndexer(config, backend).cuda().to(dtype)
    reference = HFIndexer(config, layer_idx=0).cuda().to(dtype)
    reference.load_state_dict(indexer.state_dict())
    batch = 1 if packed or seq_len == 4096 else 2
    x = torch.randn(batch, seq_len, config.hidden_size * 2, device="cuda", dtype=dtype)[..., ::2]
    q_resid = torch.randn(batch, seq_len, config.q_lora_rank * 2, device="cuda", dtype=dtype)[..., ::2]
    if seq_len > 1:
        # Every adjacent pair is separated by at most one representable input
        # step, exercising nearly tied keys rather than only broad random scores.
        x[:, 1::2] = torch.nextafter(x[:, :-1:2], torch.full_like(x[:, :-1:2], float("inf")))
    x.requires_grad_()
    q_resid.requires_grad_()
    freqs = torch.ones(batch, seq_len, config.qk_rope_head_dim // 2, dtype=torch.complex64, device="cuda")
    cos = torch.ones(batch, seq_len, config.qk_rope_head_dim, device="cuda", dtype=dtype)
    sin = torch.zeros_like(cos)
    causal = torch.zeros(seq_len, seq_len, device="cuda").masked_fill(
        torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool).triu(1), float("-inf")
    )
    # HF 5.18 returns int32 indices; AutoModel's sparse-mask contract is int64.
    with _CaptureTopKScores() as capture:
        expected = reference(x, q_resid, (cos, sin), causal, **_HF_INDEXER_KWARGS).to(torch.int64)
    run = torch.compile(indexer, fullgraph=True) if compiled else indexer
    if packed:
        actual = run(x[0], q_resid[0], freqs[0]).unsqueeze(0)
    else:
        actual = run(x, q_resid, freqs)
    # Initial rows can contain equally masked future-key fillers. For later
    # queries, permit only boundary ties within one input-dtype resolution.
    prefix = min(config.index_topk, seq_len) - 1
    actual, expected = actual[:, prefix:], expected[:, prefix:]
    different = (actual.sort(-1).values != expected.sort(-1).values).any(-1)
    if config.index_topk >= seq_len:
        assert not different.any()
        return
    assert capture.scores is not None
    scores = capture.scores[:, prefix:].float()
    boundary = scores.topk(config.index_topk + 1, dim=-1).values[..., -2:]
    margin = boundary[..., 0] - boundary[..., 1]
    resolution = boundary.abs().amax(-1) * torch.finfo(dtype).eps
    selected = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, actual, True)
    reference_selected = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, expected, True)
    changed = selected ^ reference_selected
    changed_distance = torch.where(changed, (scores - boundary[..., :1]).abs(), 0).amax(-1)
    diagnostics = (
        f"different queries={different.sum().item()}/{different.numel()}, "
        f"min k/k+1 margin among differing queries={margin[different].min().item() if different.any() else 'n/a'}, "
        f"max changed-key distance={changed_distance.max().item()}"
    )
    logging.getLogger(__name__).info(diagnostics)
    assert torch.all(
        ~different | (torch.isfinite(margin) & (margin <= resolution) & (changed_distance <= resolution))
    ), diagnostics
