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

"""Validate fused Transformer Engine RoPE against an FP32 reference in THD format.

Usage:
    torchrun --nproc_per_node=2 tests/functional_tests/context_parallel/run_te_rope_thd.py
"""

import os

import torch
import torch.distributed as dist

# Fused BF16 RoPE should match correctly rounded FP32 except for sparse one-ULP differences.
RELATIVE_L2_TOLERANCE = 5e-5


def _relative_l2_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Compute relative L2 error in FP32.

    Args:
        actual: Tensor of arbitrary shape produced by fused RoPE.
        expected: Tensor of the same shape produced by the reference implementation.

    Returns:
        Scalar relative L2 error.
    """
    error = actual.float() - expected.float()
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(torch.finfo(torch.float32).tiny)
    return float((torch.linalg.vector_norm(error) / denominator).item())


def _build_freqs(
    global_tokens: int,
    head_dim: int,
    base: float,
    interleaved: bool,
    device: torch.device,
) -> torch.Tensor:
    """Build the FP32 angle table expected by Transformer Engine.

    Args:
        global_tokens: Number of tokens in the unsharded packed THD tensor.
        head_dim: Per-head rotary dimension.
        base: RoPE frequency base.
        interleaved: Whether adjacent dimension pairs form rotary pairs.
        device: CUDA device on which to create the table.

    Returns:
        FP32 tensor of shape [global_tokens, 1, 1, head_dim].
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    positions = torch.arange(global_tokens, dtype=torch.float32, device=device)
    angles = torch.outer(positions, inv_freq)
    if interleaved:
        duplicated_angles = torch.stack((angles, angles), dim=-1).flatten(-2)
    else:
        duplicated_angles = torch.cat((angles, angles), dim=-1)
    return duplicated_angles.view(global_tokens, 1, 1, head_dim).contiguous()


def _apply_fp32_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    freqs: torch.Tensor,
    interleaved: bool,
) -> torch.Tensor:
    """Apply RoPE in FP32 using explicit trigonometric operations.

    Args:
        x: Tensor of shape [local_tokens, heads, head_dim].
        positions: Integer tensor of shape [local_tokens] containing document-local positions.
        freqs: FP32 tensor of shape [global_tokens, 1, 1, head_dim].
        interleaved: Whether adjacent dimension pairs form rotary pairs.

    Returns:
        FP32 tensor of shape [local_tokens, heads, head_dim].
    """
    if interleaved:
        angles = freqs[positions, 0, 0, 0::2]
        first = x.float()[..., 0::2]
        second = x.float()[..., 1::2]
    else:
        angles = freqs[positions, 0, 0, : x.shape[-1] // 2]
        first, second = x.float().chunk(2, dim=-1)

    cos = angles.cos().unsqueeze(1)
    sin = angles.sin().unsqueeze(1)
    rotated_first = first * cos - second * sin
    rotated_second = second * cos + first * sin
    if not interleaved:
        return torch.cat((rotated_first, rotated_second), dim=-1)

    output = torch.empty_like(x, dtype=torch.float32)
    output[..., 0::2] = rotated_first
    output[..., 1::2] = rotated_second
    return output


def _run_case(
    case_name: str,
    layout_name: str,
    heads: int,
    head_dim: int,
    base: float,
    interleaved: bool,
    cu_seqlens: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    global_rank: int,
) -> None:
    """Validate one fused THD RoPE tensor layout through forward and backward.

    Args:
        case_name: THD or context-parallel THD case name.
        layout_name: Query or key tensor layout name.
        heads: Number of local attention heads.
        head_dim: Per-head rotary dimension.
        base: RoPE frequency base.
        interleaved: Whether adjacent dimension pairs form rotary pairs.
        cu_seqlens: Int32 tensor of shape [documents + 1] containing padded global token offsets.
        cp_size: Context-parallel world size.
        cp_rank: Context-parallel rank used by Transformer Engine.
        global_rank: Distributed rank used for diagnostic output.
    """
    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb

    global_tokens = int(cu_seqlens[-1].item())
    indices = tex.thd_get_partitioned_indices(cu_seqlens, global_tokens, cp_size, cp_rank)
    full_positions = torch.cat(
        [
            torch.arange(int(end - start), dtype=torch.long, device=cu_seqlens.device)
            for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])
        ]
    )
    local_positions = full_positions.index_select(0, indices.long())
    freqs = _build_freqs(global_tokens, head_dim, base, interleaved, cu_seqlens.device)

    torch.manual_seed(2468 + heads + head_dim)
    x = torch.randn(indices.numel(), heads, head_dim, dtype=torch.bfloat16, device=cu_seqlens.device)
    grad_output = torch.randn_like(x)

    x_fused = x.detach().clone().requires_grad_(True)
    fused_output = apply_rotary_pos_emb(
        x_fused,
        freqs,
        tensor_format="thd",
        interleaved=interleaved,
        fused=True,
        cu_seqlens=cu_seqlens,
        cp_size=cp_size,
        cp_rank=cp_rank,
    )
    fused_grad = torch.autograd.grad(fused_output, x_fused, grad_output)[0]

    x_reference = x.detach().float().requires_grad_(True)
    reference_output = _apply_fp32_rope(x_reference, local_positions, freqs, interleaved)
    reference_grad = torch.autograd.grad(reference_output, x_reference, grad_output.float())[0]
    rounded_output = reference_output.to(torch.bfloat16)
    rounded_grad = reference_grad.to(torch.bfloat16)

    if not torch.isfinite(fused_output).all() or not torch.isfinite(fused_grad).all():
        raise AssertionError(f"{case_name}.{layout_name}: fused RoPE produced non-finite values")

    output_error = _relative_l2_error(fused_output, rounded_output)
    grad_error = _relative_l2_error(fused_grad, rounded_grad)
    if global_rank == 0:
        print(
            f"{case_name}.{layout_name}: output_rel_l2={output_error:.9g}, "
            f"grad_rel_l2={grad_error:.9g}, "
            f"output_max_abs={(fused_output.float() - rounded_output.float()).abs().max().item():.9g}, "
            f"grad_max_abs={(fused_grad.float() - rounded_grad.float()).abs().max().item():.9g}"
        )
    if output_error > RELATIVE_L2_TOLERANCE or grad_error > RELATIVE_L2_TOLERANCE:
        raise AssertionError(
            f"{case_name}.{layout_name}: relative L2 error exceeds {RELATIVE_L2_TOLERANCE}: "
            f"output={output_error}, grad={grad_error}"
        )


def main() -> None:
    """Run fused RoPE reference checks for plain THD and two-rank CP."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"This test requires exactly two GPUs, got {world_size}")

    cases = (
        ("thd", torch.tensor([0, 257, 1024, 2048, 4096], dtype=torch.int32, device=device), 1, 0),
        ("thd_cp2", torch.tensor([0, 256, 1280, 2048, 4096], dtype=torch.int32, device=device), 2, rank),
    )
    layouts = (
        ("qwen_q", 32, 128, 1_000_000.0, False),
        ("qwen_k", 4, 128, 1_000_000.0, False),
        ("mla_kpe", 1, 64, 10_000.0, True),
    )
    for case_name, cu_seqlens, cp_size, cp_rank in cases:
        for layout_name, heads, head_dim, base, interleaved in layouts:
            _run_case(
                case_name,
                layout_name,
                heads,
                head_dim,
                base,
                interleaved,
                cu_seqlens,
                cp_size,
                cp_rank,
                rank,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
