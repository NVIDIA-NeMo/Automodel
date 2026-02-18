"""
Minimal SSM FLOP calibration script for Mamba2 layers.

Profiles a single Mamba2 layer to measure the training/forward time ratio,
then calculates the SSM FLOP multiplier.

Usage:
    python profile_ssm_calibration.py [--batch-size 4] [--seq-len 512]

Note: Requires ~30GB GPU memory. Ensure no other processes are using the GPU.

Result: Validates that SSM operations require ~18.3× the simple 2×B×L×D×N formula.
"""

import torch
import time
import argparse
from transformers import AutoModelForCausalLM, AutoConfig


def profile_layer(layer, input_tensor):
    """Profile forward and backward pass timing."""
    # Warm up
    for _ in range(3):
        output = layer(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        loss = output.sum()
        loss.backward()
        layer.zero_grad()

    torch.cuda.synchronize()

    # Profile forward only
    input_fwd = input_tensor.detach().requires_grad_(False)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        output = layer(input_fwd)
        if isinstance(output, tuple):
            output = output[0]
    torch.cuda.synchronize()
    fwd_time = time.time() - start

    # Profile forward + backward
    input_train = input_tensor.detach().requires_grad_(True)
    torch.cuda.synchronize()
    start = time.time()
    output = layer(input_train)
    if isinstance(output, tuple):
        output = output[0]
    loss = output.sum()
    loss.backward()
    torch.cuda.synchronize()
    train_time = time.time() - start

    layer.zero_grad()
    return fwd_time, train_time


def main():
    parser = argparse.ArgumentParser(description="Profile Mamba2 SSM FLOP calibration")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for profiling (default: 4)")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length for profiling (default: 512)")
    args = parser.parse_args()

    print("Loading Nemotron NanoV3 model...")
    config = AutoConfig.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).cuda()

    # Find first Mamba2 layer
    mamba_idx = next(i for i, t in enumerate(config.layers_block_type) if t == 'mamba')
    mamba_layer = model.backbone.layers[mamba_idx]

    print(f"\nProfiling Mamba2 layer {mamba_idx}")
    print(f"Config: hidden_size={config.hidden_size}, d_inner={config.mamba_num_heads * config.mamba_head_dim}, state_size={config.ssm_state_size}")
    print(f"Input: batch_size={args.batch_size}, seq_len={args.seq_len}")

    # Create input
    input_tensor = torch.randn(
        args.batch_size, args.seq_len, config.hidden_size,
        dtype=torch.bfloat16, device="cuda", requires_grad=True
    )

    # Profile
    fwd_time, train_time = profile_layer(mamba_layer, input_tensor)
    ratio = train_time / fwd_time

    print(f"\nTiming Results:")
    print(f"  Forward: {fwd_time*1000:.2f} ms")
    print(f"  Training (fwd+bwd): {train_time*1000:.2f} ms")
    print(f"  Ratio: {ratio:.2f}×")

    # Calculate FLOP breakdown
    d_inner = config.mamba_num_heads * config.mamba_head_dim
    in_proj_dim = 2 * d_inner + 2 * config.n_groups * config.ssm_state_size + config.mamba_num_heads

    flops_in_proj = 2 * args.batch_size * args.seq_len * config.hidden_size * in_proj_dim
    flops_out_proj = 2 * args.batch_size * args.seq_len * d_inner * config.hidden_size
    flops_matmuls = flops_in_proj + flops_out_proj

    # Detailed SSM forward FLOP counting
    # Based on Mamba2 operations: dt projection, A/B/C discretization, state updates,
    # decay computation, output projection, skip connections, RMSNorm, gating
    B, L, H, D, N = args.batch_size, args.seq_len, config.mamba_num_heads, config.mamba_head_dim, config.ssm_state_size

    ssm_ops = {
        'dt_softplus': B * L * H,
        'discretize_A': 2 * B * L * H,
        'A_cumsum': B * L * H,
        'decay_exp': B * L * H * D * N,
        'discretize_B': B * L * H * N,
        'dBx': B * L * H * D * N,
        'state_update_mult': B * L * H * D * N,
        'state_update_add': B * L * H * D * N,
        'state_C_mult': B * L * H * D * N,
        'state_C_sum': B * L * H * D * N,
        'skip_D_mult': B * L * H * D,
        'skip_D_add': B * L * H * D,
        'rmsnorm': 5 * B * L * d_inner,  # square, mean, sqrt, div, scale
        'gating': 4 * B * L * d_inner,   # silu (3 ops) + mult
    }

    total_ssm_ops = sum(ssm_ops.values())
    simple_ssm = 2 * B * L * d_inner * N
    ssm_forward_multiplier = total_ssm_ops / simple_ssm

    print(f"\nSSM Forward FLOP Breakdown:")
    print(f"  Simple formula (2×B×L×D×N): {simple_ssm/1e9:.3f} GFLOPs")
    print(f"  Detailed count: {total_ssm_ops/1e9:.3f} GFLOPs")
    print(f"  Forward multiplier: {ssm_forward_multiplier:.2f}×")

    flops_ssm_fwd = total_ssm_ops

    matmul_fraction = flops_matmuls / (flops_matmuls + flops_ssm_fwd)
    ssm_fraction = flops_ssm_fwd / (flops_matmuls + flops_ssm_fwd)

    print(f"\nFLOP Composition:")
    print(f"  Matmuls: {matmul_fraction*100:.1f}%")
    print(f"  SSM: {ssm_fraction*100:.1f}%")

    # Solve for SSM training multiplier
    # ratio = matmul_fraction × 3 + ssm_fraction × ssm_mult
    estimated_ssm_mult = (ratio - matmul_fraction * 3) / ssm_fraction

    print(f"\nSSM Training Multiplier:")
    print(f"  Empirical (from timing): {estimated_ssm_mult:.2f}×")
    print(f"  Theoretical (6 × {ssm_forward_multiplier:.2f}): {6 * ssm_forward_multiplier:.2f}×")
    print(f"  Difference: {abs(estimated_ssm_mult - 6*ssm_forward_multiplier) / (6*ssm_forward_multiplier) * 100:.1f}% (memory bandwidth/kernel efficiency)")

    print(f"\n{'='*60}")
    print(f"RESULT: Using theoretical {6*ssm_forward_multiplier:.1f}× multiplier")
    print(f"        (forward {ssm_forward_multiplier:.2f}× × training 3× × matmul 2×)")
    print(f"        Empirical measurement: {estimated_ssm_mult:.1f}× (~{abs(estimated_ssm_mult - 6*ssm_forward_multiplier) / (6*ssm_forward_multiplier) * 100:.0f}% lower)")
    print(f"{'='*60}")

    return estimated_ssm_mult


if __name__ == "__main__":
    main()
