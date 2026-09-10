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

"""CUDA regression tests for packed CSA2 training with the TileLang backend."""

import pytest
import torch

from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.moe.parallelizer import apply_ac
from tests.unit_tests.models.deepseek_v41.conftest import build_tiny_model, tiny_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="TileLang attention requires CUDA")


@pytest.mark.parametrize("fake_quant", [False, True])
def test_tilelang_packed_causality_and_checkpoint_backward(fake_quant):
    torch.manual_seed(0)
    config = tiny_config(engram_enabled=False, kv_cache_fake_quant=fake_quant, index_n_heads=16)
    model = build_tiny_model(config, attn="tilelang").cuda()
    cast_model_to_dtype(model, torch.bfloat16)
    embeddings = torch.randn(1, 8, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    kwargs = dict(
        qkv_format="thd",
        seq_lens=torch.tensor([3, 5], device="cuda"),
        seq_lens_padded=torch.tensor([3, 5], device="cuda"),
    )
    hidden = model.model(inputs_embeds=embeddings, **kwargs)
    (gradient,) = torch.autograd.grad(hidden[:, 4].float().square().sum(), embeddings)
    assert torch.count_nonzero(gradient[:, 5:]) == 0
    assert torch.count_nonzero(gradient[:, :3]) == 0
    assert gradient[:, 3:5].abs().sum() > 0

    apply_ac(model)
    model.train()
    tokens = torch.tensor([[5, 6, 7, 11, 12, 13, 14, 15]], device="cuda")
    model(tokens, **kwargs).logits.square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(grad).all() for grad in gradients)
    assert model.model.layers["2"].self_attn.compressor.wkv.weight.grad.abs().sum() > 0
