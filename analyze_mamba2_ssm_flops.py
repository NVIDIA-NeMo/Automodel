"""
Detailed FLOP analysis for Mamba2 SSM operations based on HuggingFace implementation.

Based on analysis of:
- transformers/models/mamba2/modeling_mamba2.py (Mamba2Mixer class)
- mamba_ssm/ops/triton/ssd_combined.py (mamba_chunk_scan_combined kernel)

Architecture parameters for Nemotron-3 Nano:
- hidden_size (hs) = 2688
- d_inner = num_heads × head_dim = 64 × 64 = 4096
- state_size (N) = 128
- n_groups (G) = 8
- num_heads (H) = 64
- head_dim (D) = 64
"""

def count_mamba2_ssm_flops(batch_size=4, seq_len=512, num_heads=64, head_dim=64,
                           state_size=128, n_groups=8):
    """
    Count FLOPs for Mamba2 SSM scan operations (forward pass only).

    Based on mamba_chunk_scan_combined kernel and torch_forward reference implementation.
    """

    B = batch_size
    L = seq_len
    H = num_heads
    D = head_dim
    N = state_size
    G = n_groups
    d_inner = H * D

    print(f"Configuration:")
    print(f"  Batch size (B): {B}")
    print(f"  Sequence length (L): {L}")
    print(f"  Num heads (H): {H}")
    print(f"  Head dim (D): {D}")
    print(f"  State size (N): {N}")
    print(f"  Groups (G): {G}")
    print(f"  d_inner = H × D: {d_inner}")
    print()

    flops = {}

    # 1. dt projection and softplus
    # dt comes from in_proj, shape (B, L, H)
    # softplus(dt + dt_bias) - elementwise ops, count as 1 FLOP per element
    flops['dt_softplus'] = B * L * H

    # 2. Discretize A: dA = exp(A * dt)
    # A has shape (H,), dt has shape (B, L, H)
    # Result: (B, L, H) - broadcast multiply + exp
    flops['discretize_A'] = 2 * B * L * H  # multiply + exp

    # 3. Reshape operations (no FLOPs, just memory)
    # x: (B, L, d_inner) → (B, L, H, D)
    # B: (B, L, G*N) → (B, L, H, N) (with broadcasting)
    # C: (B, L, G*N) → (B, L, H, N) (with broadcasting)

    # 4. SSM scan - the core operation
    # This is the chunked scan implementation

    # 4a. Compute cumulative A (for segment_sum)
    # A_cumsum = cumsum(A * dt) along L dimension
    # Shape: (B, L, H)
    flops['A_cumsum'] = B * L * H  # cumsum is L adds per sequence

    # 4b. Compute decay factors: exp(A_cumsum)
    # Shape: (B, L, H, D, N) in expanded form
    flops['decay_exp'] = B * L * H * D * N

    # 4c. Discretize B: dB = dt[:, :, :, None] * B
    # dt: (B, L, H, 1), B: (B, L, H, N)
    # Result: (B, L, H, N)
    flops['discretize_B'] = B * L * H * N

    # 4d. Compute dBx = dB[:, :, :, None, :] * x[:, :, :, :, None]
    # dB: (B, L, H, 1, N), x: (B, L, H, D, 1)
    # Result: (B, L, H, D, N)
    flops['dBx'] = B * L * H * D * N

    # 4e. State update: state = state * decay + dBx
    # This is done per-chunk with recurrence, but total operations:
    # state: (B, H, D, N), operations per token
    # Effectively O(L) sequential updates of (H, D, N) state
    flops['state_update_mult'] = B * L * H * D * N  # multiply by decay
    flops['state_update_add'] = B * L * H * D * N   # add dBx

    # 4f. Output projection: y = sum(state * C, axis=-1)
    # state: (B, L, H, D, N), C: (B, L, H, N)
    # Result: (B, L, H, D)
    # This is: state[:, :, :, :, :] * C[:, :, :, None, :] then sum over N
    flops['state_C_mult'] = B * L * H * D * N
    flops['state_C_sum'] = B * L * H * D * N

    # 4g. Skip connection: y += x * D
    # x: (B, L, H, D), D: (H, D) or (H,)
    # Assuming D is (H, D) for full skip connection
    flops['skip_D_mult'] = B * L * H * D
    flops['skip_D_add'] = B * L * H * D

    # 5. RMSNorm with gating (if used)
    # norm_weight: (d_inner,)
    # RMS computation: sqrt(mean(x^2))
    # Normalization: x / RMS * norm_weight
    flops['rmsnorm_square'] = B * L * d_inner
    flops['rmsnorm_mean'] = B * L * d_inner  # sum + divide
    flops['rmsnorm_sqrt'] = B * L  # one sqrt per (B, L) position
    flops['rmsnorm_div'] = B * L * d_inner
    flops['rmsnorm_scale'] = B * L * d_inner

    # If gating is used: x * silu(gate)
    # silu(x) = x * sigmoid(x) - 3 ops per element
    flops['gate_silu'] = 3 * B * L * d_inner
    flops['gate_mult'] = B * L * d_inner

    # Print breakdown
    print("Forward FLOP breakdown:")
    print("-" * 60)
    total_flops = 0
    for name, count in flops.items():
        print(f"  {name:30s}: {count/1e9:8.4f} GFLOPs")
        total_flops += count

    print("-" * 60)
    print(f"  {'TOTAL SSM':30s}: {total_flops/1e9:8.4f} GFLOPs")
    print()

    # Compare to simple approximation
    simple_approx = 2 * B * L * d_inner * N
    print(f"Simple approximation (2×B×L×d_inner×N):")
    print(f"  {simple_approx/1e9:.4f} GFLOPs")
    print()
    print(f"Detailed count is {total_flops/simple_approx:.2f}× the simple approximation")
    print()

    # Now compute what we SHOULD be using for the formula
    # The simple approximation is missing factor
    multiplier = total_flops / simple_approx
    print(f"Recommended: Use {multiplier:.2f}× for SSM forward FLOPs")
    print(f"  i.e., SSM forward FLOPs = {multiplier:.2f} × 2 × B × L × d_inner × state_size")

    return total_flops, simple_approx, multiplier


if __name__ == "__main__":
    print("="*80)
    print("Mamba2 SSM FLOP Analysis for Nemotron-3 Nano")
    print("="*80)
    print()

    # Use the exact config from profiling script
    total, simple, mult = count_mamba2_ssm_flops(
        batch_size=4,
        seq_len=512,
        num_heads=64,
        head_dim=64,
        state_size=128,
        n_groups=8
    )

    print()
    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()
    print(f"The current formula uses: 2 × B × L × d_inner × state_size")
    print(f"But should use: {mult:.2f} × 2 × B × L × d_inner × state_size")
    print()
    print(f"This means SSM operations account for ~{mult:.1f}× more FLOPs than")
    print(f"the simple matmul approximation.")
    print()
    print("After updating the formula with this multiplier, we should re-run")
    print("profiling to verify the training/forward ratio gives a reasonable")
    print("ssm_scan_mult in the [1, 5] range (accounting for backward pass overhead).")
