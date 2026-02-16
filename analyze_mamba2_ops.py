"""
Analyze what operations happen in a Mamba2 layer to get better FLOP estimates.
"""

from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    trust_remote_code=True
)

print("=" * 80)
print("Mamba2 Operations Analysis")
print("=" * 80)

# Config
hs = config.hidden_size  # 2688
d_inner = config.mamba_num_heads * config.mamba_head_dim  # 4096
d_state = config.ssm_state_size  # 128
ngroups = config.n_groups  # 8
nheads = config.mamba_num_heads  # 64
conv_kernel = config.conv_kernel  # 4

print(f"\nConfig:")
print(f"  hidden_size (hs): {hs}")
print(f"  d_inner: {d_inner}")
print(f"  d_state: {d_state}")
print(f"  ngroups: {ngroups}")
print(f"  nheads: {nheads}")
print(f"  conv_kernel: {conv_kernel}")

# in_proj outputs split into: [z, x, B, C, dt]
z_dim = d_inner  # 4096
x_dim = d_inner  # 4096
B_dim = ngroups * d_state  # 8 * 128 = 1024
C_dim = ngroups * d_state  # 8 * 128 = 1024  
dt_dim = nheads  # 64

in_proj_dim = z_dim + x_dim + B_dim + C_dim + dt_dim
print(f"\nin_proj output dimensions:")
print(f"  z: {z_dim}")
print(f"  x: {x_dim}")
print(f"  B: {B_dim}")
print(f"  C: {C_dim}")
print(f"  dt: {dt_dim}")
print(f"  Total: {in_proj_dim}")

# Test dimensions
batch_size = 4
seq_len = 512

print(f"\n{'='*80}")
print("Forward FLOPs Breakdown (per token operations)")
print(f"{'='*80}")

# 1. in_proj: Linear(hs, in_proj_dim)
flops_in_proj = 2 * batch_size * seq_len * hs * in_proj_dim
print(f"\n1. in_proj Linear({hs}, {in_proj_dim}):")
print(f"   FLOPs: {flops_in_proj/1e12:.4f} TFLOPs")

# 2. Conv1d operations
# Conv input is x + B + C = d_inner + 2*ngroups*d_state = 4096 + 2048 = 6144
conv_input_dim = d_inner + 2 * ngroups * d_state
flops_conv = 2 * batch_size * seq_len * conv_input_dim * conv_kernel  # Depthwise conv
print(f"\n2. Conv1d (depthwise, kernel={conv_kernel}):")
print(f"   Input dim: {conv_input_dim}")
print(f"   FLOPs: {flops_conv/1e12:.4f} TFLOPs")

# 3. SSM operations (this is where it gets complex!)
print(f"\n3. SSM Scan Operations:")
print(f"   This involves:")
print(f"   - Discretization of continuous params (A, B, C, dt)")
print(f"   - Selective scan (recurrent state updates)")
print(f"   - State space computation")

# The SSM scan involves multiple steps per token:
# a) dt projection and softplus
flops_dt_proj = 2 * batch_size * seq_len * nheads * d_inner  # Broadcast/project dt
print(f"   a) dt projection: ~{flops_dt_proj/1e12:.4f} TFLOPs")

# b) A, B, C transformations (discretization)
# A: (nheads, d_state) → (batch, seq, nheads, d_state)
# B: (batch, seq, ngroups, d_state) → (batch, seq, nheads, d_state) via repeat
# C: similar to B
flops_abc = 2 * batch_size * seq_len * nheads * d_state * 3  # Rough estimate for transforms
print(f"   b) A,B,C discretization: ~{flops_abc/1e12:.4f} TFLOPs")

# c) Selective scan (the core SSM operation)
# This is a sequential scan over sequence dimension with state updates
# Per time step: state_{t} = A * state_{t-1} + B * x_{t}
#                 y_{t} = C * state_{t}
# FLOPs per step: d_state operations × nheads
# Total: seq_len steps × batch × nheads × d_state operations
flops_scan = 2 * batch_size * seq_len * nheads * d_state * 3  # State update + output
print(f"   c) Selective scan (core SSM): ~{flops_scan/1e12:.4f} TFLOPs")

# d) Normalization (RMSNorm or similar)
flops_norm = batch_size * seq_len * d_inner  # Elementwise ops, not matmul
print(f"   d) Normalization: ~{flops_norm/1e12:.4f} TFLOPs (elementwise)")

# Total SSM
total_ssm = flops_dt_proj + flops_abc + flops_scan + flops_norm
print(f"   TOTAL SSM: ~{total_ssm/1e12:.4f} TFLOPs")

# 4. SiLU activation (elementwise)
flops_silu = batch_size * seq_len * d_inner
print(f"\n4. SiLU activation:")
print(f"   FLOPs: ~{flops_silu/1e12:.4f} TFLOPs (elementwise)")

# 5. Elementwise multiply (z * output)
flops_mult = batch_size * seq_len * d_inner
print(f"\n5. Elementwise multiply (z * output):")
print(f"   FLOPs: ~{flops_mult/1e12:.4f} TFLOPs")

# 6. out_proj: Linear(d_inner, hs)
flops_out_proj = 2 * batch_size * seq_len * d_inner * hs
print(f"\n6. out_proj Linear({d_inner}, {hs}):")
print(f"   FLOPs: {flops_out_proj/1e12:.4f} TFLOPs")

# Total
total_flops = (flops_in_proj + flops_conv + total_ssm + 
               flops_silu + flops_mult + flops_out_proj)

print(f"\n{'='*80}")
print("TOTAL FORWARD FLOPs")
print(f"{'='*80}")
print(f"Matmuls (in_proj + out_proj): {(flops_in_proj + flops_out_proj)/1e12:.4f} TFLOPs ({(flops_in_proj + flops_out_proj)/total_flops*100:.1f}%)")
print(f"Conv1d: {flops_conv/1e12:.4f} TFLOPs ({flops_conv/total_flops*100:.1f}%)")
print(f"SSM operations: {total_ssm/1e12:.4f} TFLOPs ({total_ssm/total_flops*100:.1f}%)")
print(f"Other (activations, etc): {(flops_silu + flops_mult)/1e12:.4f} TFLOPs ({(flops_silu + flops_mult)/total_flops*100:.1f}%)")
print(f"TOTAL: {total_flops/1e12:.4f} TFLOPs")

print(f"\n{'='*80}")
print("Comparison with Our Simple Formula")
print(f"{'='*80}")
simple_ssm = 2 * batch_size * seq_len * d_inner * d_state
print(f"Our simple SSM estimate: {simple_ssm/1e12:.4f} TFLOPs")
print(f"Better SSM estimate: {total_ssm/1e12:.4f} TFLOPs")
print(f"Ratio: {total_ssm/simple_ssm:.2f}x")

print(f"\nThis explains why the estimated multiplier was {total_ssm/simple_ssm:.2f}x!")
print(f"We were missing {(total_ssm/simple_ssm - 1)*100:.0f}% of the SSM operations.")

