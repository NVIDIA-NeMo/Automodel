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

"""Real TileLang attention output and trainable-gradient parity for compact masks."""

import copy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.layers import (
    DeepseekV4Attention,
    DeepseekV4RotaryEmbedding,
    build_causal_padding_mask,
    build_packed_causal_padding_mask,
)
from nemo_automodel.components.models.deepseek_v4.optimized_kernels import is_dsv4_kernel_available
from nemo_automodel.components.models.deepseek_v4.sparse_mask import build_dsv4_sparse_mask

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or not is_dsv4_kernel_available("sparse_attn"),
        reason="CUDA and TileLang required",
    ),
    pytest.mark.timeout(300),
]


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_attention_compact_vs_dense(ratio, packed, use_checkpoint):
    torch.manual_seed(2026)
    cfg = DeepseekV4Config(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=16,
        head_dim=128,
        qk_rope_head_dim=32,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        compress_ratios=[ratio],
        sliding_window=32,
        index_head_dim=128,
        index_n_heads=16,
        index_topk=8,
        max_position_embeddings=512,
        torch_dtype="bfloat16",
    )
    backend = BackendConfig(
        attn="tilelang", linear="torch", rms_norm="torch", rope_fusion=False, experts="torch", dispatcher="torch"
    )
    module = DeepseekV4Attention(cfg, 0, backend).cuda()
    module.init_weights(torch.device("cuda"))
    module = module.bfloat16().train()
    module.sinks_param.float()
    if module.compressor is not None:
        for compressor in (module.compressor, module.compressor.indexer):
            if compressor is not None:
                compressor.wkv.float()
                compressor.wgate.float()
                compressor.ape_param.float()
    # The H100 indexer kernel's shared-memory requirement exceeds this device's
    # limit. Both controls use the unchanged torch indexer, while attention
    # forward/backward execute real TileLang kernels.
    if ratio == 4:
        module.compressor.backend = BackendConfig(attn="sdpa")
        module.compressor.indexer.backend = BackendConfig(attn="sdpa")
    reference = copy.deepcopy(module)
    length = 264
    x = torch.randn(2, length, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    slots = torch.arange(length, device="cuda")[None].expand(2, -1)
    positions = slots.clone()
    docs = None
    if packed:
        positions[:, 128:] -= 128
        docs = torch.ones_like(positions)
        docs[:, 128:] = 2
        docs[:, -4:] = 0
        dense = build_packed_causal_padding_mask(
            torch.tensor([[128, 132], [128, 132]], device="cuda"), length, x.dtype, x.device, 32
        )
        pad = None
    else:
        keep = torch.ones(2, length, device="cuda", dtype=torch.bool)
        keep[:, 5:9] = False
        dense = build_causal_padding_mask(keep, length, x.dtype, x.device, 2, 32)
        pad = ~keep
    sparse = build_dsv4_sparse_mask(
        slots, key_len=length, packed_seq_ids=docs, padding_mask=pad, sliding_window=32, cp_group=None
    )
    rotary = DeepseekV4RotaryEmbedding(rope_theta=10000.0, head_dim=128, partial_rotary_factor=0.25).cuda()
    rotary_c = DeepseekV4RotaryEmbedding(rope_theta=160000.0, head_dim=128, partial_rotary_factor=0.25).cuda()
    kwargs = dict(
        position_embeddings=rotary(x, positions),
        position_embeddings_compress=rotary_c(x, positions),
        rotary_compress=rotary_c,
        position_ids=positions,
        packed_seq_ids=docs,
    )

    def call(inputs):
        return module(inputs, sparse_mask=sparse, **kwargs)[0]

    actual = checkpoint(call, x, use_reentrant=False) if use_checkpoint else call(x)
    expected = reference(xr, attention_mask=dense, **kwargs)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    dy = torch.randn_like(actual)
    actual.backward(dy)
    expected.backward(dy)
    for a, b in [(x.grad, xr.grad)] + [(p.grad, r.grad) for p, r in zip(module.parameters(), reference.parameters())]:
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=0.01, atol=0.002)


@pytest.mark.parametrize("packed,mtp_layers", [(False, 0), (True, 0), (False, 1)])
def test_model_and_mtp_skip_dense_builders(packed, mtp_layers, monkeypatch):
    from nemo_automodel.components.models.deepseek_v4 import model as model_module

    torch.manual_seed(19)
    cfg = DeepseekV4Config(
        vocab_size=64,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=16,
        head_dim=128,
        qk_rope_head_dim=32,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        n_routed_experts=4,
        n_shared_experts=0,
        num_experts_per_tok=2,
        num_hash_layers=0,
        hc_mult=2,
        compress_ratios=[0, 0],
        sliding_window=32,
        max_position_embeddings=256,
        num_nextn_predict_layers=mtp_layers,
        torch_dtype="bfloat16",
    )
    backend = BackendConfig(
        attn="tilelang",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    module = model_module.DeepseekV4ForCausalLM(cfg, backend=backend).cuda().train()
    module.initialize_weights(dtype=torch.bfloat16)
    reference = copy.deepcopy(module)
    # Change only the model-level mask choice. Its attention modules keep
    # TileLang, including the MTP attention, so this is the legacy dense-mask
    # path with exactly the same kernels and parameters.
    reference.backend = BackendConfig(attn="sdpa")
    reference.model.backend = reference.backend
    ids = torch.randint(0, 64, (1, 128), device="cuda")
    positions = torch.arange(128, device="cuda")[None]
    kwargs = dict(position_ids=positions)
    if packed:
        kwargs.update(qkv_format="thd", seq_lens=torch.tensor([[64, 64]], device="cuda"))
        kwargs["position_ids"] = positions % 64
    else:
        keep = torch.ones_like(ids, dtype=torch.bool)
        keep[:, 50:53] = False
        kwargs["attention_mask"] = keep
    expected = reference(ids, **kwargs)

    def reject(*args, **kwargs):
        raise AssertionError("text TileLang materialized a dense mask")

    with monkeypatch.context() as context:
        context.setattr(model_module, "build_causal_padding_mask", reject)
        context.setattr(model_module, "build_packed_causal_padding_mask", reject)
        actual = module(ids, **kwargs)
    a_tensors = [actual.logits] + (actual.mtp_per_depth_h or [])
    b_tensors = [expected.logits] + (expected.mtp_per_depth_h or [])
    assert len(a_tensors) == len(b_tensors)
    for a, b in zip(a_tensors, b_tensors):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    gradients = [torch.randn_like(x) for x in a_tensors]
    torch.autograd.backward(a_tensors, gradients)
    torch.autograd.backward(b_tensors, gradients)
    original_grads = {name: p.grad for name, p in reference.named_parameters()}
    reference.zero_grad(set_to_none=True)
    repeated = reference(ids, **kwargs)
    torch.autograd.backward([repeated.logits] + (repeated.mtp_per_depth_h or []), gradients)
    for (name, a), (_, b) in zip(module.named_parameters(), reference.named_parameters()):
        target = original_grads[name]
        assert (a.grad is None) == (target is None) == (b.grad is None), name
        if target is not None:
            scale = target.float().norm().clamp_min(1e-12)
            relative = ((a.grad.float() - target.float()).norm() / scale).item()
            aa = ((b.grad.float() - target.float()).norm() / scale).item()
            assert torch.isfinite(a.grad).all()
            assert relative < 0.004 and aa < 0.004, (name, relative, aa)
