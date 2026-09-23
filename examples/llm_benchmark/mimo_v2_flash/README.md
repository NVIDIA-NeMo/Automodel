# MiMo V2.6 text throughput benchmark

This example runs the dedicated LLM benchmark recipe with the complete 48-layer
text backbone of [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL).
It requires the MiMo V2.6 implementation in [PR #3992](https://github.com/NVIDIA-NeMo/Automodel/pull/3992).
No checkpoint weights, images, audio or MTP heads are loaded.

| Configuration | Sequence length | GPUs | EP / CP / PP | Local / global batch | Accumulation |
| --- | ---: | ---: | --- | --- | ---: |
| [4K mock](mimo_v26_4k_mock.yaml) | 4096 | 64 H100 SXM | 64 / 1 / 1 | 2 / 128 | 1 |

The benchmark runs 30 iterations, including 10 warmup iterations. Mock data is
non-packed; each row is one full-length sequence. Fake balanced routing and static routing
metadata are enabled for throughput measurement, not convergence measurement.

## Run

Launch inside a scheduler allocation of eight nodes with eight GPUs per node:

```bash
automodel examples/llm_benchmark/mimo_v2_flash/mimo_v26_4k_mock.yaml --nnodes 8 --nproc-per-node 8
```

The recipe reports per-iteration MFU and writes a JSON summary under
`training_logs/`. W&B is commented out; enable the block with your own project/entity to log a run.

The config retains the measured kernel choices: TE attention, linear and RMSNorm;
`torch_mm` experts; HybridEP with 20 communication SMs; fused linear cross entropy;
TE FusedAdam; FSDP2 prefetch and optimizations; activation checkpointing; and
`reshard_after_forward: true` for both dense and expert parameters. It uses BF16,
not FP8. HybridEP needs a working DeepEP/DOCA multi-node installation.

The recorded run used cuDNN 9.26 with Transformer Engine. This is a non-packed
BSHD/CP1 example. The separate packed THD sink-attention path requires cuDNN
9.26 or newer; see the runtime notes in PR #3992.

## FLOPs accounting

The model-owned [formula](../../../nemo_automodel/components/models/mimo_v2_flash/flops.py)
is selected through the MiMo config's `flops_formula` hook. The inherited hook
also covers `MiMoV2Config`; the common utility contains no MiMo architecture logic.
The benchmark's `flops_scope: text` makes the scope explicit.

Dimensions below come from the checkpoint's
[config.json at revision 5711b268](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/blob/5711b268169967567844e1e560e8a3966da959b1/config.json).
Projection shapes and layer selection follow
[model.py](../../../nemo_automodel/components/models/mimo_v2_flash/model.py).

| Quantity | Checkpoint value / source field |
| --- | --- |
| Hidden / vocabulary | 4096 / 152576: `hidden_size`, `vocab_size` |
| Attention layers | 9 full + 39 SWA: first 48 entries of `hybrid_layer_pattern` |
| Full Q / KV heads; QK / V dimensions | 64 / 4; 192 / 128 |
| SWA Q / KV heads; QK / V dimensions | 64 / 8; 192 / 128 |
| Causal SWA window | 128 including the current token: `sliding_window` |
| Dense / MoE layers | 1 / 47: `moe_layer_freq` |
| Dense / expert intermediate size | 16384 / 2048 |
| Routed / activated / shared experts | 256 / 8 / 0 |

Let `B` be global batch size, `S` sequence length, `H` hidden size and `V`
vocabulary size. Every trainable matrix MAC costs six FLOPs for forward plus
backward. For each layer, with query heads `Nq`, KV heads `Nkv`, QK dimension
`Dqk`, value dimension `Dv`, and visible causal query/key pairs `P`:

| Operation | Forward + backward FLOPs |
| --- | --- |
| Q | `6 B S H Nq Dqk` |
| K | `6 B S H Nkv Dqk` |
| V | `6 B S H Nkv Dv` |
| O | `6 B S H Nq Dv` |
| QK / PV | `6 B P Nq Dqk` / `6 B P Nq Dv` |
| Dense SwiGLU, three projections | `18 B S H intermediate_size` |
| Routed + shared SwiGLU | `18 B S H moe_intermediate_size (top_k + shared)` |
| Learned router | `6 B S H n_routed_experts` |
| Vocabulary projection, once after all layers | `6 B S H V` |

Full attention has `P = S(S+1)/2`. SWA has
`P = w(w+1)/2 + (S-w)w`, where `w = min(S, sliding_window)`.
This includes the causal ramp at the start of each sequence. QK and V have
different dimensions; TE's padded V width is not model work.

At GBS 128 this gives **48.125875807322112 PFLOPs/step at 4K**. MFU is:

```text
100 * global_step_FLOPs / (GPU_count * step_seconds * peak_FLOPs_per_GPU)
```

The YAML uses the H100 SXM BF16 dense peak of 989 TFLOPs/s, rather than the
1979 TFLOPs/s FP8 peak or a sparse peak. See the
[NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/).

This is nominal **model** FLOPs accounting. It includes the learned router even
when synthetic routing bypasses it, and counts the vocabulary projection at all
input positions even when fused CE skips ignored labels. For this mock dataset,
removing the router and the one ignored label per sequence gives
47.970364705210368 PFLOPs/step at 4K, about 0.32% less. That runtime-work variant
is not the number emitted by the registered model formula.

Norms, RoPE, softmax/sinks, SwiGLU activations, residuals, routing weights and
embedding lookups are elementwise or lookup work and are outside this
matrix-FLOPs convention. Communication, optimizer updates, activation
recomputation and backend padding are also excluded. Packed batches with
multiple constituent sequences need their actual lengths for attention FLOPs;
this fixed-length example does not claim packed or multimodal FLOPs coverage.

## Recorded measurements

The original 4K run completed all 30 iterations:
[W&B jo1ktifp](https://wandb.ai/Nemo-automodel/automodel-mimo-v26/runs/jo1ktifp).
Over measured steps 10–29, averaging each step's slowest-rank duration gives
3.3088171 seconds, **2475.8 tokens/s/GPU**, and **22.98% MFU** with this formula.
Rank 0 reported 61.15 GiB maximum allocated memory. The W&B run predates this
formula; its generic-formula MFU is incorrect and has not been rewritten.
