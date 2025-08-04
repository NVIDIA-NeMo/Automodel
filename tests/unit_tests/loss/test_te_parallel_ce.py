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
import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.loss.te_parallel_ce import (
    HAVE_TE_PARALLEL_CE,
    TEParallelCrossEntropy,
)

@pytest.mark.skipif(not HAVE_TE_PARALLEL_CE, reason="TE parallel cross entropy is not available")
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_te_parallel_cross_entropy(reduction):
    """Tests te_parallel_cross_entropy against PyTorch's CE.

    * has close output with PyTorch's cross_entropy
    * handles strided label format correctly
    * works with tensor parallel groups
    * works with different reduction methods
    """
    if not torch.cuda.is_available():
        pytest.skip("This test requires a GPU")

    device = torch.device("cuda")
    batch_size = 8
    seq_length = 2048
    vocab_size = 128256
    dtype = torch.bfloat16

    logits = torch.randn(batch_size, seq_length, vocab_size, dtype=dtype, device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)

    # Measure memory for PyTorch implementation
    torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        pytorch_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), reduction=reduction)
        if reduction == "none":
            pytorch_loss = pytorch_loss.view(batch_size, seq_length)

    pytorch_memory = torch.cuda.max_memory_allocated()

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # Measure memory for TE implementation
    torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        te_loss = TEParallelCrossEntropy(tp_group=None, reduction=reduction)(logits, targets)

    te_memory = torch.cuda.max_memory_allocated()

    print("\nTE Parallel CE Memory usage comparison:")
    print(f"PyTorch implementation: {pytorch_memory / 1024**2:.2f} MB")
    print(f"TE parallel implementation: {te_memory / 1024**2:.2f} MB")
    
    if te_memory < pytorch_memory:
        print(f"Memory savings: {(pytorch_memory - te_memory) / 1024**2:.2f} MB")
    else:
        print(f"Memory overhead: {(te_memory - pytorch_memory) / 1024**2:.2f} MB")

    pytorch_loss = pytorch_loss.float()
    te_loss = te_loss.float()

    if reduction == "none":
        assert torch.allclose(te_loss, pytorch_loss, rtol=1e-2, atol=1e-2), (
            f"Loss mismatch: PyTorch shape={pytorch_loss.shape}, TE shape={te_loss.shape}\n"
            f"PyTorch mean={pytorch_loss.mean().item()}, TE mean={te_loss.mean().item()}"
        )
    else:
        assert torch.allclose(te_loss, pytorch_loss, rtol=1e-2, atol=1e-2), (
            f"Loss mismatch with reduction={reduction}: PyTorch={pytorch_loss}, TE={te_loss}"
        )


@pytest.mark.skipif(not HAVE_TE_PARALLEL_CE, reason="TE parallel cross entropy is not available")
def test_te_parallel_cross_entropy_with_masking():
    """Tests te_parallel_cross_entropy with loss masking against masked_cross_entropy."""
    if not torch.cuda.is_available():
        pytest.skip("This test requires a GPU")

    from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

    device = torch.device("cuda")
    batch_size = 8
    seq_length = 2048
    vocab_size = 128256
    dtype = torch.bfloat16

    logits = torch.randn(batch_size, seq_length, vocab_size, dtype=dtype, device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    loss_mask = torch.randint(0, 2, (batch_size, seq_length), device=device)

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)
    
    # Compare against masked_cross_entropy (the actual baseline)
    torch.cuda.synchronize()
    mem_before_masked = torch.cuda.memory_allocated(device)
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        # MaskedCrossEntropy fills in -100 for masked positions in-place, so we need to clone the targets for test correctness
        masked_ce_loss = MaskedCrossEntropy()(logits, targets.clone(), mask=loss_mask)
    torch.cuda.synchronize()
    mem_after_masked = torch.cuda.memory_allocated(device)
    masked_ce_peak = torch.cuda.max_memory_allocated(device)

    # Reset for TE measurement
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    mem_before_te = torch.cuda.memory_allocated(device)
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        te_loss = TEParallelCrossEntropy()(logits, targets.clone(), mask=loss_mask)

    torch.cuda.synchronize()
    mem_after_te = torch.cuda.memory_allocated(device)
    te_peak = torch.cuda.max_memory_allocated(device)

    # Print memory comparison
    print(f"Memory usage comparison:")
    print(f"  Masked CE: (peak: {masked_ce_peak / 1024**2:.2f} MB)")
    print(f"  Masked CE: {mem_after_masked / 1024**2:.2f} MB)")
    print(f"  TE Parallel: (peak: {te_peak / 1024**2:.2f} MB)")
    print(f"  TE Parallel: {mem_after_te / 1024**2:.2f} MB)")

    # Accuracy comparison
    assert torch.allclose(te_loss, masked_ce_loss, rtol=1e-2, atol=1e-2), (
        f"Masked loss mismatch: masked_cross_entropy={masked_ce_loss.item():.4f}, TE={te_loss.item():.4f}"
    )

    # Memory efficiency check - TE should use similar or less memory
    assert te_peak <= masked_ce_peak * 1.5, (
        f"TE parallel CE uses significantly more memory: {te_peak / 1024**2:.2f} MB vs {masked_ce_peak / 1024**2:.2f} MB"
    )
