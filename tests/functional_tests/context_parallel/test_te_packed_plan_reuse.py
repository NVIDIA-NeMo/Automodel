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

"""Actual TE forward/backward and checkpoint parity for virtual documents.

Run in separate processes with NVTE_FLASH_ATTN=0,NVTE_FUSED_ATTN=1 for
cuDNN and NVTE_FLASH_ATTN=1,NVTE_FUSED_ATTN=0 for FlashAttention. The latter
supports vanilla softmax without physical padding; use
``-k "(vanilla and unpadded) or recompute"`` for its parity cases.
"""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.attention.utils import initialize_attn_module_and_func

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="real TE parity requires CUDA")


def _metadata(documents: int, *, physical_padding: bool) -> dict[str, torch.Tensor | int | str | bool]:
    """Make THD offsets [documents + 1]; padded cases also have unequal Q/KV lengths."""
    lengths_q = torch.tensor([16 + 16 * (i % 2) for i in range(documents)], device="cuda", dtype=torch.int32)
    # FlashAttention cannot implement top-left causal attention with unequal
    # Q/KV lengths. Keep that contract in the cuDNN-only physical-padding cases.
    lengths_kv = lengths_q + (16 if physical_padding else 0)
    physical_q = torch.full_like(lengths_q, 32) if physical_padding else lengths_q
    physical_kv = torch.full_like(lengths_kv, 48) if physical_padding else lengths_kv
    metadata = {
        "qkv_format": "thd",
        "attn_mask_type": "padding_causal",
        "cu_seqlens_q": torch.cat((lengths_q.new_zeros(1), lengths_q.cumsum(0, dtype=torch.int32))),
        "cu_seqlens_kv": torch.cat((lengths_kv.new_zeros(1), lengths_kv.cumsum(0, dtype=torch.int32))),
        "cu_seqlens_q_padded": torch.cat((physical_q.new_zeros(1), physical_q.cumsum(0, dtype=torch.int32))),
        "cu_seqlens_kv_padded": torch.cat((physical_kv.new_zeros(1), physical_kv.cumsum(0, dtype=torch.int32))),
        "max_seqlen_q": 32,
        "max_seqlen_kv": 48 if physical_padding else 32,
        "pad_between_seqs": physical_padding,
    }
    if not physical_padding:
        # TE also infers absence of physical padding from these aliases when
        # the explicit flag is omitted. The production adapter must retain it.
        metadata["cu_seqlens_q_padded"] = metadata["cu_seqlens_q"]
        metadata["cu_seqlens_kv_padded"] = metadata["cu_seqlens_kv"]
        del metadata["pad_between_seqs"]
    return metadata


@pytest.mark.parametrize("softmax", ["vanilla", "learnable"])
@pytest.mark.parametrize("window", [(-1, 0), (15, 0)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "documents,physical_padding",
    [(10, False), (129, False), (129, True)],
    ids=["small_unpadded", "large_unpadded", "large_padded"],
)
def test_native_te_parity(
    softmax: str, window: tuple[int, int], dtype: torch.dtype, documents: int, physical_padding: bool
) -> None:
    """Compare Q/K/V and sink gradients with exact-boundary native TE."""
    pytest.importorskip("transformer_engine.pytorch")
    torch.manual_seed(89)
    module, call = initialize_attn_module_and_func(
        "te",
        4,
        64,
        64,
        1 / 8,
        num_gqa_groups=2,
        qkv_format="thd",
        attn_mask_type="padding_causal",
        softmax_type=softmax,
        attention_dropout=0.0,
    )
    module.cuda()
    metadata = _metadata(documents, physical_padding=physical_padding)
    original = {key: value.clone() for key, value in metadata.items() if key.startswith("cu_")}
    qtokens = int(metadata["cu_seqlens_q_padded"][-1])
    ktokens = int(metadata["cu_seqlens_kv_padded"][-1])
    inputs = [
        torch.randn(tokens, heads, 64, device="cuda", dtype=dtype, requires_grad=True)
        for tokens, heads in [(qtokens, 4), (ktokens, 2), (ktokens, 2)]
    ]
    upstream = torch.randn(qtokens, 256, device="cuda", dtype=dtype)
    # Physical padding has no query output contract; zero its upstream gradient.
    valid = torch.zeros(qtokens, device="cuda", dtype=torch.bool)
    for i in range(documents):
        start = int(metadata["cu_seqlens_q_padded"][i])
        length = int(metadata["cu_seqlens_q"][i + 1] - metadata["cu_seqlens_q"][i])
        valid[start : start + length] = True
    upstream[~valid] = 0
    reference = module(*inputs, **metadata, window_size=window)
    reference_grads = torch.autograd.grad(reference, inputs + list(module.parameters()), upstream)
    candidate = call(*inputs, **metadata, window_size=window)
    candidate_grads = torch.autograd.grad(candidate, inputs + list(module.parameters()), upstream)
    torch.testing.assert_close(candidate[valid], reference[valid], atol=2e-2, rtol=2e-2)
    for actual, expected in zip(candidate_grads, reference_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    for key, value in original.items():
        torch.testing.assert_close(metadata[key], value, atol=0, rtol=0)


def test_vanilla_recompute_after_larger_pack() -> None:
    """A later 129-document forward cannot change a pending checkpoint's shape."""
    pytest.importorskip("transformer_engine.pytorch")
    module, call = initialize_attn_module_and_func("te", 2, 64, 64, 1 / 8, qkv_format="thd", attention_dropout=0.0)
    module.cuda()
    torch.manual_seed(49)
    inputs = []
    metadata = []
    for count in (3, 129):
        cu = torch.arange(count + 1, device="cuda", dtype=torch.int32) * 16
        metadata.append(
            dict(
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                max_seqlen_q=16,
                max_seqlen_kv=16,
                attn_mask_type="padding_causal",
                qkv_format="thd",
            )
        )
        inputs.append(
            [torch.randn(count * 16, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        )
    reference = [call(*values, **kwargs) for values, kwargs in zip(inputs, metadata)]
    upstream = [torch.randn_like(value) for value in reference]
    expected = torch.autograd.grad(reference, [value for row in inputs for value in row], upstream)
    actual = [checkpoint(call, *values, **kwargs, use_reentrant=False) for values, kwargs in zip(inputs, metadata)]
    grads = torch.autograd.grad(actual, [value for row in inputs for value in row], upstream)
    for result, baseline in zip(actual, reference, strict=True):
        torch.testing.assert_close(result, baseline, atol=0, rtol=0)
    for result, baseline in zip(grads, expected, strict=True):
        torch.testing.assert_close(result, baseline, atol=0, rtol=0)
