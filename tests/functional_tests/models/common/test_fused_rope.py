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

"""Correctness tests for the standalone fused RoPE CUDA kernel."""

from __future__ import annotations

from typing import Literal

import pytest
import torch

from nemo_automodel.components.models.common.fused_rope import apply_fused_rope

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="standalone fused RoPE correctness requires compiling and executing CUDA kernels",
)

_TensorFormat = Literal["sbhd", "bshd", "thd"]
_TOLERANCES = {
    torch.float32: 2e-6,
    torch.float16: 2e-3,
    torch.bfloat16: 2e-2,
}


def _make_freqs(
    positions: int,
    rotary_dim: int,
    *,
    interleaved: bool,
    device: torch.device,
) -> torch.Tensor:
    """Build raw RoPE angles without using Transformer Engine.

    Args:
        positions: Number of position rows.
        rotary_dim: Even width of the rotary prefix.
        interleaved: Whether equal angles occupy adjacent rotary dimensions.
        device: CUDA device on which to create the table.

    Returns:
        Contiguous float32 tensor of shape [positions, 1, 1, rotary_dim].
    """
    inv_freq = 1.0 / (10_000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
    half_angles = torch.outer(torch.arange(positions, dtype=torch.float32, device=device), inv_freq)
    if interleaved:
        angles = torch.repeat_interleave(half_angles, 2, dim=-1)
    else:
        angles = torch.cat((half_angles, half_angles), dim=-1)
    return angles[:, None, None, :].contiguous()


def _cp_position_indices(
    local_sequence: int,
    cp_size: int,
    cp_rank: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return global positions owned by one mirrored two-chunk CP shard.

    Args:
        local_sequence: Number of sequence elements stored on this rank.
        cp_size: Number of context-parallel shards.
        cp_rank: Rank of the local context-parallel shard.
        device: CUDA device on which to create the indices.

    Returns:
        Int64 tensor of shape [local_sequence] in local storage order.
    """
    if cp_size == 1:
        return torch.arange(local_sequence, device=device)
    if local_sequence % 2:
        raise ValueError("local_sequence must be even when cp_size > 1")
    chunk = local_sequence // 2
    global_sequence = local_sequence * cp_size
    first = torch.arange(cp_rank * chunk, (cp_rank + 1) * chunk, device=device)
    second = torch.arange(
        global_sequence - (cp_rank + 1) * chunk,
        global_sequence - cp_rank * chunk,
        device=device,
    )
    return torch.cat((first, second))


def _rotate_reference(
    input_tensor: torch.Tensor,
    angles: torch.Tensor,
    *,
    interleaved: bool,
) -> torch.Tensor:
    """Apply an explicit eager PyTorch RoPE reference in FP32.

    Args:
        input_tensor: Tensor of shape [..., heads, head_dim], with arbitrary
            leading dimensions.
        angles: Float32 tensor broadcastable to shape [..., heads, rotary_dim].
            Values are raw angles arranged in the kernel's rotary pairing layout.
        interleaved: Whether rotary pairs occupy adjacent head-dimension elements.

    Returns:
        Tensor with the same shape, dtype, and device as `input_tensor`.
    """
    rotary_dim = angles.shape[-1]
    source = input_tensor[..., :rotary_dim].float()
    if interleaved:
        rotated = torch.stack((-source[..., 1::2], source[..., 0::2]), dim=-1).flatten(-2)
    else:
        first, second = source.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
    rotary_output = source * torch.cos(angles) + rotated * torch.sin(angles)
    if rotary_dim == input_tensor.shape[-1]:
        return rotary_output.to(input_tensor.dtype)
    return torch.cat((rotary_output, input_tensor[..., rotary_dim:].float()), dim=-1).to(input_tensor.dtype)


def _reference_rope(
    input_tensor: torch.Tensor,
    freqs: torch.Tensor,
    *,
    tensor_format: _TensorFormat,
    interleaved: bool,
    cu_seqlens: torch.Tensor | None,
    cp_size: int,
    cp_rank: int,
    tokenwise_freqs: bool,
) -> torch.Tensor:
    """Apply RoPE with explicit PyTorch math and independent CP indexing.

    Args:
        input_tensor: Tensor in SBHD [sequence, batch, heads, head_dim], BSHD
            [batch, sequence, heads, head_dim], or THD [tokens, heads, head_dim]
            layout.
        freqs: Float32 tensor of raw angles with shape [positions, 1, 1, rotary_dim].
        tensor_format: Semantic layout of `input_tensor`.
        interleaved: Whether rotary pairs occupy adjacent head-dimension elements.
        cu_seqlens: For THD, int32 offsets of shape [batch + 1] describing global
            padded sequence spans; otherwise `None`.
        cp_size: Number of context-parallel shards.
        cp_rank: Rank of the local context-parallel shard.
        tokenwise_freqs: Whether THD frequencies use global physical-token indexing.

    Returns:
        Reference tensor with the same shape, dtype, and device as `input_tensor`.
    """
    if tensor_format != "thd":
        padded = input_tensor.transpose(0, 1) if tensor_format == "sbhd" else input_tensor
        positions = _cp_position_indices(padded.shape[1], cp_size, cp_rank, device=input_tensor.device)
        angles = freqs[positions, 0, 0, :][None, :, None, :]
        output = _rotate_reference(padded, angles, interleaved=interleaved)
        return output.transpose(0, 1).contiguous() if tensor_format == "sbhd" else output

    if cu_seqlens is None:
        raise ValueError("cu_seqlens is required for THD reference input")
    offsets = cu_seqlens.detach().cpu().tolist()
    pieces = []
    local_start = 0
    for global_start, global_end in zip(offsets, offsets[1:]):
        global_sequence = global_end - global_start
        if global_sequence % cp_size:
            raise ValueError("each THD sequence must be divisible by cp_size")
        local_sequence = global_sequence // cp_size
        positions = _cp_position_indices(local_sequence, cp_size, cp_rank, device=input_tensor.device)
        if tokenwise_freqs:
            positions = positions + global_start
        angles = freqs[positions, 0, 0, :][:, None, :]
        local_input = input_tensor.narrow(0, local_start, local_sequence)
        pieces.append(_rotate_reference(local_input, angles, interleaved=interleaved))
        local_start += local_sequence
    if local_start != input_tensor.shape[0]:
        raise ValueError("cu_seqlens does not cover the local THD input")
    return torch.cat(pieces, dim=0)


def _assert_matches_reference(
    input_tensor: torch.Tensor,
    freqs: torch.Tensor,
    *,
    tensor_format: _TensorFormat,
    interleaved: bool,
    cu_seqlens: torch.Tensor | None = None,
    cp_size: int = 1,
    cp_rank: int = 0,
    tokenwise_freqs: bool = False,
) -> None:
    """Compare fused forward and input gradient against the eager reference.

    Args:
        input_tensor: Contiguous CUDA tensor in SBHD [sequence, batch, heads,
            head_dim], BSHD [batch, sequence, heads, head_dim], or THD
            [tokens, heads, head_dim] layout.
        freqs: Contiguous float32 CUDA tensor of shape [positions, 1, 1, rotary_dim].
        tensor_format: Semantic layout of `input_tensor`.
        interleaved: Whether rotary pairs occupy adjacent head-dimension elements.
        cu_seqlens: For THD, contiguous int32 CUDA offsets of shape [batch + 1];
            otherwise `None`.
        cp_size: Number of context-parallel shards.
        cp_rank: Rank of the local context-parallel shard.
        tokenwise_freqs: Whether THD frequencies use global physical-token indexing.

    Returns:
        None.
    """
    actual_input = input_tensor.detach().clone().requires_grad_(True)
    reference_input = input_tensor.detach().clone().requires_grad_(True)
    actual = apply_fused_rope(
        actual_input,
        freqs,
        tensor_format=tensor_format,
        interleaved=interleaved,
        cu_seqlens=cu_seqlens,
        cp_size=cp_size,
        cp_rank=cp_rank,
        tokenwise_freqs=tokenwise_freqs,
    )
    reference = _reference_rope(
        reference_input,
        freqs,
        tensor_format=tensor_format,
        interleaved=interleaved,
        cu_seqlens=cu_seqlens,
        cp_size=cp_size,
        cp_rank=cp_rank,
        tokenwise_freqs=tokenwise_freqs,
    )

    assert actual.shape == reference.shape == input_tensor.shape
    assert actual.dtype == reference.dtype == input_tensor.dtype
    assert actual.device == reference.device == input_tensor.device
    assert actual.is_contiguous()

    tolerance = _TOLERANCES[input_tensor.dtype]
    torch.testing.assert_close(actual, reference, rtol=0, atol=tolerance)

    grad_storage = torch.randn(
        (*actual.shape[:-1], actual.shape[-1] * 2),
        device=actual.device,
        dtype=actual.dtype,
    )
    grad_output = grad_storage[..., ::2]
    assert not grad_output.is_contiguous()
    actual_grad = torch.autograd.grad(actual, actual_input, grad_output)[0]
    reference_grad = torch.autograd.grad(reference, reference_input, grad_output)[0]
    torch.testing.assert_close(actual_grad, reference_grad, rtol=0, atol=tolerance)

    rotary_dim = freqs.shape[-1]
    if rotary_dim < input_tensor.shape[-1]:
        torch.testing.assert_close(actual[..., rotary_dim:], input_tensor[..., rotary_dim:], rtol=0, atol=0)
        torch.testing.assert_close(actual_grad[..., rotary_dim:], grad_output[..., rotary_dim:], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("tensor_format", ("sbhd", "bshd", "thd"))
@pytest.mark.parametrize("interleaved", (False, True))
@pytest.mark.parametrize("rotary_dim", (32, 64))
def test_forward_and_backward_match_reference(
    dtype: torch.dtype,
    tensor_format: _TensorFormat,
    interleaved: bool,
    rotary_dim: int,
) -> None:
    """Cover every supported layout, dtype, pairing, and partial-rotation path."""
    torch.manual_seed(1439)
    sequence, batch, heads, head_dim = 16, 2, 3, 64
    if tensor_format == "sbhd":
        shape = (sequence, batch, heads, head_dim)
        cu_seqlens = None
    elif tensor_format == "bshd":
        shape = (batch, sequence, heads, head_dim)
        cu_seqlens = None
    else:
        shape = (batch * sequence, heads, head_dim)
        cu_seqlens = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * sequence

    input_tensor = torch.randn(shape, device="cuda", dtype=dtype)
    freqs = _make_freqs(sequence, rotary_dim, interleaved=interleaved, device=input_tensor.device)
    _assert_matches_reference(
        input_tensor,
        freqs,
        tensor_format=tensor_format,
        interleaved=interleaved,
        cu_seqlens=cu_seqlens,
    )


@pytest.mark.parametrize("tensor_format", ("sbhd", "bshd", "thd"))
@pytest.mark.parametrize("cp_size,cp_rank", ((2, 0), (2, 1), (4, 0), (4, 3)))
def test_context_parallel_forward_and_backward_match_reference(
    tensor_format: _TensorFormat,
    cp_size: int,
    cp_rank: int,
) -> None:
    """Cover mirrored dual-chunk indexing for padded and variable-length THD input."""
    torch.manual_seed(811 + cp_rank)
    heads, head_dim, rotary_dim = 2, 80, 64
    interleaved = cp_rank % 2 == 1
    if tensor_format == "thd":
        global_lengths = (8 * cp_size, 12 * cp_size)
        local_tokens = sum(global_lengths) // cp_size
        shape = (local_tokens, heads, head_dim)
        cu_seqlens = torch.tensor(
            (0, global_lengths[0], sum(global_lengths)),
            device="cuda",
            dtype=torch.int32,
        )
        positions = max(global_lengths)
    else:
        global_sequence, batch = 32, 2
        local_sequence = global_sequence // cp_size
        shape = (
            (local_sequence, batch, heads, head_dim)
            if tensor_format == "sbhd"
            else (batch, local_sequence, heads, head_dim)
        )
        cu_seqlens = None
        positions = global_sequence

    input_tensor = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    freqs = _make_freqs(positions, rotary_dim, interleaved=interleaved, device=input_tensor.device)
    _assert_matches_reference(
        input_tensor,
        freqs,
        tensor_format=tensor_format,
        interleaved=interleaved,
        cu_seqlens=cu_seqlens,
        cp_size=cp_size,
        cp_rank=cp_rank,
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("interleaved", (False, True))
@pytest.mark.parametrize("cp_size,cp_rank", ((1, 0), (2, 0), (2, 1)))
def test_tokenwise_packed_thd_forward_and_backward_match_reference(
    dtype: torch.dtype,
    interleaved: bool,
    cp_size: int,
    cp_rank: int,
) -> None:
    """Cover distinct raw-angle rows for every packed physical token."""
    torch.manual_seed(1931 + cp_rank)
    global_lengths = (8 * cp_size, 12 * cp_size, 16 * cp_size)
    total_global_tokens = sum(global_lengths)
    local_tokens = total_global_tokens // cp_size
    heads, head_dim, rotary_dim = 2, 64, 48
    input_tensor = torch.randn((local_tokens, heads, head_dim), device="cuda", dtype=dtype)
    half_angles = torch.randn(
        (total_global_tokens, rotary_dim // 2),
        device="cuda",
        dtype=torch.float32,
    )
    if interleaved:
        angles = torch.repeat_interleave(half_angles, 2, dim=-1)
    else:
        angles = torch.cat((half_angles, half_angles), dim=-1)
    freqs = angles[:, None, None, :].contiguous()
    cu_seqlens = torch.tensor(
        (0, global_lengths[0], sum(global_lengths[:2]), total_global_tokens),
        device="cuda",
        dtype=torch.int32,
    )
    _assert_matches_reference(
        input_tensor,
        freqs,
        tensor_format="thd",
        interleaved=interleaved,
        cu_seqlens=cu_seqlens,
        cp_size=cp_size,
        cp_rank=cp_rank,
        tokenwise_freqs=True,
    )


def test_rejects_unpadded_offsets_for_padded_thd_storage() -> None:
    """Reject offsets that would leave physical THD storage unwritten."""
    input_tensor = torch.randn((32, 2, 64), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    freqs = _make_freqs(32, 64, interleaved=False, device=input_tensor.device)
    wrong_offsets = torch.tensor((0, 15, 29), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match=r"cu_seqlens\[-1\].*input\.size\(0\) \* cp_size"):
        apply_fused_rope(
            input_tensor,
            freqs,
            tensor_format="thd",
            cu_seqlens=wrong_offsets,
        )


def test_rejects_frequency_gradients() -> None:
    """Reject an unsupported gradient instead of silently returning None."""
    input_tensor = torch.randn((8, 1, 2, 64), device="cuda")
    freqs = _make_freqs(8, 64, interleaved=False, device=input_tensor.device).requires_grad_(True)
    with pytest.raises(ValueError, match="does not compute frequency gradients"):
        apply_fused_rope(input_tensor, freqs, tensor_format="sbhd")


def test_rejects_cpu_tensors_before_compilation() -> None:
    """Reject CPU input through the public contract before loading the extension."""
    input_tensor = torch.randn((8, 1, 2, 64))
    freqs = _make_freqs(8, 64, interleaved=False, device=input_tensor.device)
    with pytest.raises(RuntimeError, match="must be CUDA tensors"):
        apply_fused_rope(input_tensor, freqs, tensor_format="sbhd")


def test_rejects_odd_local_sequence_with_context_parallelism() -> None:
    """Reject padded CP metadata that would trigger a device-side assertion."""
    input_tensor = torch.randn((7, 1, 2, 64), device="cuda")
    freqs = _make_freqs(14, 64, interleaved=False, device=input_tensor.device)
    with pytest.raises(RuntimeError, match="local sequence length must be even"):
        apply_fused_rope(
            input_tensor,
            freqs,
            tensor_format="sbhd",
            cp_size=2,
            cp_rank=0,
        )


@pytest.mark.parametrize(
    "tensor_format,with_offsets,tokenwise_freqs,cp_size,cp_rank,message",
    (
        ("thd", False, False, 1, 0, "cu_seqlens is required"),
        ("sbhd", True, False, 1, 0, "must be omitted"),
        ("sbhd", False, True, 1, 0, "only supported with THD"),
        ("sbhd", False, False, 2, 2, "cp_rank"),
    ),
)
def test_rejects_inconsistent_layout_metadata(
    tensor_format: _TensorFormat,
    with_offsets: bool,
    tokenwise_freqs: bool,
    cp_size: int,
    cp_rank: int,
    message: str,
) -> None:
    """Reject invalid public metadata before launching a CUDA kernel."""
    input_tensor = torch.randn((8, 1, 2, 64), device="cuda")
    freqs = _make_freqs(8, 64, interleaved=False, device=input_tensor.device)
    cu_seqlens = torch.empty(0, device="cuda", dtype=torch.int32) if with_offsets else None
    with pytest.raises(ValueError, match=message):
        apply_fused_rope(
            input_tensor,
            freqs,
            tensor_format=tensor_format,
            cu_seqlens=cu_seqlens,
            cp_size=cp_size,
            cp_rank=cp_rank,
            tokenwise_freqs=tokenwise_freqs,
        )
