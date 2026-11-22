# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""CUDA memory and numerical regressions for chunked output projection."""

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.loss.chunked_ce import ChunkedCrossEntropy


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA allocator profiling requires a GPU")
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_chunked_projection_peak_memory(compiled, training):
    torch.manual_seed(19)
    device = "cuda"
    vocab, hidden_dim, chunk_len = 8192, 128, 128
    weight = torch.randn(vocab, hidden_dim, device=device, dtype=torch.bfloat16, requires_grad=training)
    chunked = ChunkedCrossEntropy(chunk_len, compile=compiled)

    def measure(tokens, use_chunked):
        hidden = torch.randn(tokens, hidden_dim, device=device, dtype=torch.bfloat16, requires_grad=training)
        labels = torch.randint(vocab, (tokens,), device=device)

        def step():
            with torch.set_grad_enabled(training):
                loss = (
                    chunked(hidden, labels, weight, num_label_tokens=tokens)
                    if use_chunked
                    else F.cross_entropy(F.linear(hidden, weight).float(), labels)
                )
                if training:
                    loss.backward()
            return loss.detach()

        # Warm kernels/compile before measuring live allocator use.
        step()
        hidden.grad = weight.grad = None
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        loss = step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        hidden.grad = weight.grad = None
        assert torch.isfinite(loss)
        return peak

    small = measure(2048, True)
    large = measure(4096, True)
    dense = measure(4096, False)
    full_logits_bytes = 4096 * vocab * torch.bfloat16.itemsize
    print(f"compiled={compiled} training={training}: chunked_2k={small} chunked_4k={large} dense_4k={dense}")
    # A full BF16 [tokens,vocab] allocation alone exceeds this bound, even if
    # an accidental implementation still chunks its fp32 upcast/softmax.
    assert large < full_logits_bytes * 0.75
    assert large < dense * 0.5
    # Doubling tokens can grow [tokens,hidden] gradients, not vocabulary buffers.
    assert large - small < full_logits_bytes * 0.1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA BF16 and autocast")
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_cuda_gradient_parity(compiled, autocast):
    torch.manual_seed(23)
    dtype = torch.float32 if autocast else torch.bfloat16
    hidden = (torch.randn(2, 17, 64, device="cuda", dtype=dtype) * 0.2).requires_grad_()
    weight = (torch.randn(257, 64, device="cuda", dtype=dtype) * 0.2).requires_grad_()
    labels = torch.randint(257, (2, 17), device="cuda")
    labels[0, :7] = -100
    ref_hidden = hidden.detach().clone().requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        loss = ChunkedCrossEntropy(7, compile=compiled)(hidden, labels, weight, num_label_tokens=27)
        reference = F.cross_entropy(F.linear(ref_hidden, ref_weight).float().flatten(0, 1), labels.flatten())
    # Backward deliberately runs outside autocast, as in normal AMP training.
    (loss * 1.7).backward()
    (reference * 1.7).backward()
    torch.testing.assert_close(loss, reference, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(hidden.grad.float(), ref_hidden.grad.float(), rtol=2e-2, atol=2e-4)
    torch.testing.assert_close(weight.grad.float(), ref_weight.grad.float(), rtol=2e-2, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA mixed-dtype projection")
@pytest.mark.parametrize("compiled", [False, True])
def test_fp32_head_preserves_bf16_logits_rounding(compiled):
    torch.manual_seed(67)
    hidden = torch.randn(1, 19, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(67, 32, device="cuda", requires_grad=True)
    labels = torch.randint(67, (1, 19), device="cuda")
    ref_hidden = hidden.detach().clone().requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    loss = ChunkedCrossEntropy(4, compile=compiled)(hidden, labels, weight, logits_dtype=torch.bfloat16)
    reference = F.cross_entropy(
        F.linear(ref_hidden.float(), ref_weight).bfloat16().float().flatten(0, 1), labels.flatten(), reduction="sum"
    )
    loss.backward()
    reference.backward()
    torch.testing.assert_close(loss, reference, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(weight.grad, ref_weight.grad, rtol=1e-5, atol=1e-5)
