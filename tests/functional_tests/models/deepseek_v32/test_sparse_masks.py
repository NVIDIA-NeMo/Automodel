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

"""Exercise DeepSeek V3.2 sparse masks through actual CUDA attention backends."""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v32.config import DeepseekV32Config
from nemo_automodel.components.models.deepseek_v32.layers import DeepseekV32MLA

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="This functional test requires CUDA attention.")


@pytest.mark.parametrize("backend", ["sdpa", "te"])
@pytest.mark.parametrize("padding", [False, True])
def test_sparse_mla_cuda_mask_invariance(backend: str, padding: bool) -> None:
    torch.manual_seed(73)
    config = DeepseekV32Config(
        hidden_size=64,
        num_attention_heads=2,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_head_dim=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=64,
        index_n_heads=2,
        index_head_dim=64,
        index_topk=4,
        max_position_embeddings=64,
    )
    mla = DeepseekV32MLA(config, BackendConfig(attn=backend, linear="torch", rms_norm="torch"))
    mla = mla.cuda().to(torch.bfloat16)
    mla.init_weights(torch.device("cuda"), init_std=0.1)
    x = torch.randn(2, 32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    freqs = torch.polar(torch.ones(2, 32, 16, device="cuda"), torch.randn(2, 32, 16, device="cuda"))
    mask = None
    if padding:
        mask = torch.ones(2, 32, device="cuda")
        mask[0, -8:] = 0
        mask[1, -4:] = 0

    output = mla(x, freqs, attention_mask=mask)
    changed = x.detach().clone()
    changed[:, 16:] = torch.randn_like(changed[:, 16:]) * 10
    other = mla(changed, freqs, attention_mask=mask)
    torch.testing.assert_close(output[:, :16], other[:, :16], rtol=0, atol=0)
    assert torch.isfinite(output).all()
    (output[:, :16] * torch.randn_like(output[:, :16])).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[:, 16:]) == 0
    assert torch.count_nonzero(x.grad[:, :16]) > 0

    # Each sample must have the same sparse attention result alone and in a batch.
    # Allow ordinary BF16 backend rounding, but not the old union of both key sets.
    for batch in range(2):
        individual = mla(
            x.detach()[batch : batch + 1],
            freqs[batch : batch + 1],
            attention_mask=None if mask is None else mask[batch : batch + 1],
        )
        torch.testing.assert_close(output[batch : batch + 1, :16], individual[:, :16], rtol=0.02, atol=0.002)
