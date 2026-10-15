# Step 5 Preview full-depth benchmark on 128 H100 GPUs

This recipe benchmarks the full 92-layer
[Step 5 Preview](https://huggingface.co/stepfun-ai/Step-5-Preview-BF16) language
backbone with the native `nemo_automodel.recipes.llm.benchmark` entry point. It
constructs the model with `NeMoAutoModelForCausalLM.from_config` from the
checked-in `model-config/config.json`; it does not load a checkpoint.

## Configuration

| Setting | Value |
| --- | --- |
| Hardware | 16 nodes × 8 H100 80GB |
| Parallelism | EP32 / PP4 / CP1 / TP1, DP32 |
| Pipeline | `1f1b`, 24 microbatches, layer partitions [0,24), [24,46), [46,70), [70,92) |
| Sequence / global batch | 8,192 tokens / 768 sequences |
| Data | Mock single documents of 8,192 tokens |
| MoE routing | Fake balanced, static routing over HybridEP |
| Attention | FA4; learned CSA selection; SWA varlen and CSA forward/backward full blocks |
| Precision | BF16 compute; FP32 norms and residuals |
| Steps | 17 total; the first seven are warmup |

The model has hidden size 4096, 88 MoE layers with 352 experts and top-8
routing, four dense layers, 64 query heads, 4 key/value heads, and head
dimension 192. The multi-token prediction layers are disabled.

## Result

| Metric | Value |
| --- | ---: |
| Mean step time | 41.86 s |
| Model FLOPs utilization | 20.5% |

MFU uses 989 TFLOP/s per H100 and excludes activation recomputation and
physical attention tile padding. This is a mock-data throughput measurement;
it does not measure real-data training or checkpoint loading.

## Running the recipe

The FA4 backend requires a FlashAttention build with SM90 head-dimension-192
GQA backward support (see `FLASH_ATTN_REF` in `docker/Dockerfile`). Within a
16-node allocation, launch one task per node from the repository root, with
`MASTER_ADDR` set to the first node:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
torchrun --nnodes=16 --nproc_per_node=8 --node_rank="$SLURM_NODEID" \
  --master_addr="$MASTER_ADDR" --master_port=29630 \
  -m nemo_automodel.recipes.llm.benchmark \
  --config examples/llm_benchmark/step/step5_full92_mock_h100/benchmark.yaml
```

Only the tokenizer is downloaded from the Hugging Face checkpoint. Outputs go
to `outputs/step5-full92-mock-h100`; override `checkpoint.checkpoint_dir` and
`benchmark.json_output_path` to change them. See the
[Slurm launcher guide](../../../../docs/launcher/slurm.mdx) for multi-node
allocation.
