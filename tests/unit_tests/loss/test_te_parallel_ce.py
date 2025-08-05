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

from nemo_automodel.components.loss.te_parallel_ce import TEParallelCrossEntropy, HAVE_TE_PARALLEL_CE, MISSING_TE_PARALLEL_CE_MSG

@pytest.mark.skipif(not HAVE_TE_PARALLEL_CE, reason=MISSING_TE_PARALLEL_CE_MSG)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("ignore_index", [-100, -199])
def test_te_parallel_cross_entropy(reduction, ignore_index):
    """Tests te_parallel_cross_entropy against PyTorch's CE.

    * has close output with PyTorch's cross_entropy
    * works with different reduction methods
    * works with different ignore_index values
    """

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
        pytorch_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), reduction=reduction, ignore_index=ignore_index)
        if reduction == "none":
            pytorch_loss = pytorch_loss.view(batch_size, seq_length)

    pytorch_memory = torch.cuda.max_memory_allocated()

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # Measure memory for TE implementation
    torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        te_loss = TEParallelCrossEntropy(tp_group=None, reduction=reduction, ignore_index=ignore_index)(logits, targets)

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

@pytest.mark.skipif(not HAVE_TE_PARALLEL_CE, reason=MISSING_TE_PARALLEL_CE_MSG)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_te_parallel_cross_entropy_with_masking(reduction):
    """Tests te_parallel_cross_entropy with loss masking against masked_cross_entropy."""

    from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

    device = torch.device("cuda")
    batch_size = 4
    seq_length = 100
    vocab_size = 128
    dtype = torch.bfloat16

    logits = torch.randn(batch_size, seq_length, vocab_size, dtype=dtype, device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    loss_mask = torch.randint(0, 2, (batch_size, seq_length), device=device)

    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        # MaskedCrossEntropy fills in ignore_index for masked positions in-place, so we need to clone the targets for test correctness
        masked_ce_loss = MaskedCrossEntropy(reduction=reduction)(logits, targets.clone(), mask=loss_mask)
        if reduction == "none":
            masked_ce_loss = masked_ce_loss.view(batch_size, seq_length)

    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        te_loss = TEParallelCrossEntropy(reduction=reduction)(logits, targets.clone(), mask=loss_mask)

    masked_ce_loss = masked_ce_loss.float()
    te_loss = te_loss.float()

    if reduction == "none":
        assert torch.allclose(te_loss, masked_ce_loss, rtol=1e-2, atol=1e-2), (
            f"Loss mismatch: MaskedCrossEntropy shape={masked_ce_loss.shape}, TEParallelCrossEntropy shape={te_loss.shape}\n"
            f"MaskedCrossEntropy mean={masked_ce_loss.mean().item()}, TEParallelCrossEntropy mean={te_loss.mean().item()}"
        )
    else:
        assert torch.allclose(te_loss, masked_ce_loss, rtol=1e-2, atol=1e-2), (
            f"Loss mismatch with reduction={reduction}: MaskedCrossEntropy={masked_ce_loss}, TEParallelCrossEntropy={te_loss}"
        )
