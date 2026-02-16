"""
Verify the updated FLOP formula gives reasonable MFU.
"""
from transformers import AutoConfig
from nemo_automodel.components.utils.flops_utils import nemotronh_flops

# Load config
config = AutoConfig.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    trust_remote_code=True
)

# Benchmark parameters (from profiling)
gbs = 4
seq_len = 512

# Calculate FLOPs
total_flops = nemotronh_flops(config, gbs=gbs, seq_len=seq_len)

print(f"Configuration:")
print(f"  Global batch size: {gbs}")
print(f"  Sequence length: {seq_len}")
print()

# Count layers
layer_types = config.layers_block_type
num_mamba = sum(1 for t in layer_types if t == 'mamba')
num_moe = sum(1 for t in layer_types if t == 'moe')
num_attn = sum(1 for t in layer_types if t == 'attention')

print(f"Model architecture:")
print(f"  Mamba layers: {num_mamba}")
print(f"  MOE layers: {num_moe}")
print(f"  Attention layers: {num_attn}")
print(f"  Total layers: {len(layer_types)}")
print()

print(f"Total FLOPs: {total_flops/1e12:.3f} TFLOPs")
print()

# Simulated timing from profiling
time_per_iter = 0.076  # seconds (from benchmark)
h100_peak_tflops = 989  # BF16 peak

achieved_tflops = (total_flops / 1e12) / time_per_iter
mfu = achieved_tflops / h100_peak_tflops * 100

print(f"Performance analysis:")
print(f"  Time per iteration: {time_per_iter:.3f} seconds")
print(f"  Achieved: {achieved_tflops:.1f} TFLOPs/s")
print(f"  H100 peak: {h100_peak_tflops} TFLOPs/s")
print(f"  MFU: {mfu:.1f}%")
print()

if mfu > 100:
    print(f"⚠️  MFU is still >100%, formula needs more tuning")
elif mfu > 50:
    print(f"✓ MFU is in reasonable range (<100%)")
else:
    print(f"⚠️  MFU is quite low, formula may be overestimating FLOPs")
