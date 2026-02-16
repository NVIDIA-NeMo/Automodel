"""
Profile Mamba2 SSM operations to calibrate the ssm_scan_mult constant.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
import time

def profile_layer(layer, input_tensor, name="Layer"):
    """Profile a layer's forward and backward pass."""
    print(f"\n{'='*80}")
    print(f"Profiling: {name}")
    print(f"{'='*80}")
    
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
    input_tensor_fwd = input_tensor.detach().requires_grad_(False)
    
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    
    with torch.no_grad():
        output = layer(input_tensor_fwd)
        if isinstance(output, tuple):
            output = output[0]
    
    torch.cuda.synchronize()
    fwd_time = time.time() - start
    fwd_mem = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"Forward only:")
    print(f"  Time: {fwd_time*1000:.2f} ms")
    print(f"  Memory: {fwd_mem:.2f} GB")
    
    # Profile forward + backward
    input_tensor_train = input_tensor.detach().requires_grad_(True)
    
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    
    output = layer(input_tensor_train)
    if isinstance(output, tuple):
        output = output[0]
    loss = output.sum()
    loss.backward()
    
    torch.cuda.synchronize()
    train_time = time.time() - start
    train_mem = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"\nForward + Backward:")
    print(f"  Time: {train_time*1000:.2f} ms")
    print(f"  Memory: {train_mem:.2f} GB")
    print(f"  Training/Forward time ratio: {train_time/fwd_time:.2f}x")
    
    layer.zero_grad()
    
    return fwd_time, train_time, train_time/fwd_time

def main():
    print("Loading Nemotron NanoV3 model...")
    
    config = AutoConfig.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        trust_remote_code=True
    )
    
    # Load just the model on GPU
    model = AutoModelForCausalLM.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).cuda()
    
    print(f"\nModel config:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  ssm_state_size: {config.ssm_state_size}")
    print(f"  mamba_num_heads: {config.mamba_num_heads}")
    print(f"  mamba_head_dim: {config.mamba_head_dim}")
    print(f"  n_groups: {config.n_groups}")
    
    # Find a Mamba2 layer
    mamba_layer_idx = None
    for idx, layer_type in enumerate(config.layers_block_type):
        if layer_type == 'mamba':
            mamba_layer_idx = idx
            break
    
    if mamba_layer_idx is None:
        print("ERROR: No Mamba layer found!")
        return
    
    print(f"\nUsing layer {mamba_layer_idx} (type: mamba)")
    
    # Extract the Mamba2 layer
    mamba_layer = model.backbone.layers[mamba_layer_idx]
    print(f"Layer type: {type(mamba_layer)}")
    
    # Create input tensor
    batch_size = 4
    seq_len = 512
    hidden_size = config.hidden_size
    
    print(f"\nInput shape: ({batch_size}, {seq_len}, {hidden_size})")
    
    input_tensor = torch.randn(
        batch_size, seq_len, hidden_size,
        dtype=torch.bfloat16,
        device="cuda:0",
        requires_grad=True
    )
    
    # Profile the full Mamba2 block
    fwd_time, train_time, ratio = profile_layer(
        mamba_layer,
        input_tensor,
        name=f"Full Mamba2 Block (layer {mamba_layer_idx})"
    )
    
    # Calculate theoretical FLOPs
    d_inner = config.mamba_num_heads * config.mamba_head_dim  # 4096
    in_proj_dim = 2 * d_inner + 2 * config.n_groups * config.ssm_state_size + config.mamba_num_heads  # 10304
    
    print(f"\n{'='*80}")
    print("FLOP Analysis")
    print(f"{'='*80}")
    
    # Forward FLOPs (factor of 2 for matmul)
    flops_in_proj = 2 * batch_size * seq_len * hidden_size * in_proj_dim
    flops_out_proj = 2 * batch_size * seq_len * d_inner * hidden_size
    flops_matmuls = flops_in_proj + flops_out_proj
    
    # SSM scan (forward only, with 3.05× multiplier from detailed analysis)
    # See analyze_mamba2_ssm_flops.py for breakdown
    ssm_forward_multiplier = 3.05
    flops_ssm_scan_fwd = ssm_forward_multiplier * 2 * batch_size * seq_len * d_inner * config.ssm_state_size
    
    print(f"\nForward FLOPs breakdown:")
    print(f"  in_proj:  {flops_in_proj/1e12:.3f} TFLOPs")
    print(f"  out_proj: {flops_out_proj/1e12:.3f} TFLOPs")
    print(f"  Matmuls total: {flops_matmuls/1e12:.3f} TFLOPs")
    print(f"  SSM scan (approx): {flops_ssm_scan_fwd/1e12:.3f} TFLOPs")
    print(f"  TOTAL: {(flops_matmuls + flops_ssm_scan_fwd)/1e12:.3f} TFLOPs")
    
    # Training FLOPs with 3x multiplier for matmuls
    flops_matmuls_train = flops_matmuls * 3
    
    print(f"\nIf we assume:")
    print(f"  - Matmuls use 3× training multiplier")
    print(f"  - SSM scan uses X× training multiplier (unknown)")
    
    print(f"\nThen training FLOPs = {flops_matmuls_train/1e12:.3f} (matmuls) + X × {flops_ssm_scan_fwd/1e12:.3f} (SSM)")
    
    # Estimate based on timing ratio
    print(f"\n{'='*80}")
    print("SSM SCAN MULTIPLIER ESTIMATION")
    print(f"{'='*80}")
    
    # Observed training/forward ratio
    print(f"\nObserved training/forward time ratio: {ratio:.2f}x")
    
    # More refined estimate accounting for matmuls
    matmul_fraction = flops_matmuls / (flops_matmuls + flops_ssm_scan_fwd)
    ssm_fraction = flops_ssm_scan_fwd / (flops_matmuls + flops_ssm_scan_fwd)
    
    print(f"\nFLOP composition:")
    print(f"  Matmuls: {matmul_fraction*100:.1f}%")
    print(f"  SSM:     {ssm_fraction*100:.1f}%")
    
    # Solve for ssm_mult:
    # ratio = (matmul_frac * 3 + ssm_frac * ssm_mult)
    # ssm_mult = (ratio - matmul_frac * 3) / ssm_frac
    
    if ssm_fraction > 0:
        estimated_ssm_mult = (ratio - matmul_fraction * 3) / ssm_fraction
        print(f"\nEstimated ssm_scan_mult: {estimated_ssm_mult:.2f}")
        print(f"  (Assuming matmuls use 3× and solving for SSM multiplier)")
        
        # Sanity check
        if estimated_ssm_mult < 1 or estimated_ssm_mult > 5:
            print(f"\n  ⚠️  WARNING: Estimated multiplier {estimated_ssm_mult:.2f} is outside expected range [1, 5]")
            print(f"      This might indicate measurement noise or incorrect assumptions.")
        else:
            print(f"\n  ✓ Estimated multiplier {estimated_ssm_mult:.2f} is in reasonable range")
            
        # Calculate what our current formula predicts
        current_formula_flops = (
            6 * batch_size * seq_len * hidden_size * in_proj_dim +
            6 * batch_size * seq_len * d_inner * config.ssm_state_size +
            6 * batch_size * seq_len * d_inner * hidden_size
        )
        
        # Better formula
        better_formula_flops = (
            6 * batch_size * seq_len * hidden_size * in_proj_dim +
            2 * estimated_ssm_mult * batch_size * seq_len * d_inner * config.ssm_state_size +
            6 * batch_size * seq_len * d_inner * hidden_size
        )
        
        print(f"\n{'='*80}")
        print("FORMULA COMPARISON")
        print(f"{'='*80}")
        print(f"Current formula (using 6× for SSM): {current_formula_flops/1e12:.3f} TFLOPs")
        print(f"Better formula (using {estimated_ssm_mult:.2f}× for SSM): {better_formula_flops/1e12:.3f} TFLOPs")
        print(f"Difference: {(current_formula_flops - better_formula_flops)/current_formula_flops*100:.1f}%")
        
        return estimated_ssm_mult

if __name__ == "__main__":
    ssm_mult = main()
    if ssm_mult:
        print(f"\n{'='*80}")
        print(f"✓ RECOMMENDED: Use ssm_scan_mult = {ssm_mult:.2f} in the FLOP formula")
        print(f"{'='*80}")
