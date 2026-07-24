# Gemma 4 31B — Tulu-3 Convergence

Dense 31B **multimodal** base (`google/gemma-4-31B`) — vision + audio + text towers; we do
**text-only SFT** with the vision/audio towers frozen. ~60 transformer layers with
Gemma's alternating **local sliding-window / global attention** (heterogeneous head dims: 256 local,
512 global), bf16. `recipe: FinetuneRecipeForVLM`, entry `examples/vlm_finetune/finetune.py`. Base,
not `-it`.

**Model:** [google/gemma-4-31B](https://huggingface.co/google/gemma-4-31B)

## Config

The single validated convergence recipe:

| Config | batch | optimizer | packing | loss |
|--------|-------|-----------|---------|------|
| [`gemma4_31b_tulu3_packed2k_cp1_gbs32_4node.yaml`](gemma4_31b_tulu3_packed2k_cp1_gbs32_4node.yaml) | cp1·dp32·**GBS 32** (4×8 GPUs) | TE FusedAdam fp32-master, **lr 5e-6** cosine, warmup 100, wd 0 | 2k, `gemma4_inject_thinking_prefix` | 0.79 → 0.50 |

Multi-node (4×8 GPUs) — launch with `sbatch` using a script adapted from the repo's reference
[`slurm.sub`](../../../../../slurm.sub).

## Important notes

The HP choices that give a clean, monotonic descent (0.79 → 0.50):
- **GBS = 32** — a large effective batch avoids the per-step loss swing of a small batch (`gbs=1`
  swings 0.19–1.60 and buries the descent). GBS=32 = cp1·dp32 across 4 nodes (grad_acc 1; ~26 GiB/GPU). 
  `gbs>1` OOMs on single node.
- **2k packing** — multiple conversations per sample (~2× more supervised tokens/step than unpacked) → cleaner gradient, cleaner descent.
- **lr 5e-6 + warmup 100 + wd 0** — monotonic and reaches the lowest floor (0.50).

Why the alternatives don't work:
- **Higher lr (1e-5)** perturbs the base upward before the cosine recovers (a hump) and ends higher.
- **Unpacked** at the same GBS gives a shallower descent (~2× fewer supervised tokens/step).
- **Single-node `gbs>1`** OOMs (the dense-CP grad-accum path isn't memory-neutral) → needs multi-node
  data parallelism.

## Data

Tulu-3 is pulled straight from the Hub at load time via `make_tulu3_dataset`
(`allenai/tulu-3-sft-mixture`) — **no pre-filter step**. Unlike the moonlight/qwen LLM paths (which
pre-filter to `seq_length=2048` to avoid OOM from variable-length batching with a large vocabulary),
the VLM packed path handles over-length samples at pack time: `packed_sequence.max_length: 2048` +
`drop_long_samples: true` pack multiple conversations into fixed 2k blocks and drop the ~4% of samples
that exceed 2k. So there is no meta-JSON / prefilter / cache step to run.

## Training

4 nodes × 8 H100. Launch with `sbatch` using a SLURM script adapted from the repo's reference
[`slurm.sub`](../../../../../slurm.sub) — point it at `examples/vlm_finetune/finetune.py -c <this
config>` with a 4-node `torchrun` rendezvous.

The base ships no chat template — the recipe supplies the `-it` template via
`dataset.chat_template` (the VLM loader applies it to the processor), so no manual staging onto the
model dir is needed. Runs 1000 steps, `save_consolidated: final`.

## Evaluate

```bash
CKPT="$(readlink -f checkpoints_convergence/gemma4_31b_tulu3_packed2k_cp1_gbs32_4node_lr5e6/LATEST)/model/consolidated"

bash examples/convergence/tulu3/eval/run_eval.sh \
    --model-path "$CKPT" \
    --tokenizer "$CKPT" \
    --tasks ifeval \
    --tp-size 2 --dp-size 1 \
    --extra-model-args "max_model_len=8192"
```

**Note:** `--tp-size 2` is required (the 31B VLM won't leave KV-cache room on 1 GPU).
`--tokenizer "$CKPT"` uses the trained checkpoint's tokenizer, which carries the chat template (the
base ships none). The eval env pins `nvidia-cutlass-dsl==4.5.0` (see Gotchas).

## Results

### Training Loss

| Config | Step 0 | Step 999 |
|--------|-------:|---------:|
| packed2k gbs32 lr5e-6 | 0.79 | 0.50 |

### Training Curve

<p align="center">
  <img src="https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/examples/convergence/tulu3/models/gemma4-31b/gemma4_31b_tulu3_sft.png" alt="Gemma4 31B SFT loss curve on Tulu3" width="700">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/examples/convergence/tulu3/models/gemma4-31b/gemma4_31b_tulu3_sft_grad_norm.png" alt="Gemma4 31B Tulu3 grad nrom" width="700">
</p>

[W&B run](https://wandb.ai/nvidia/convergence/runs/p0h8ls77)

### IFEval Results (lm-eval + vLLM, tp=2, max_model_len=8192)

| Model | prompt_strict | prompt_loose | inst_strict | inst_loose |
|-------|-------------:|-------------:|------------:|-----------:|
| gemma-4-31B base | 0.2477 ±0.0186 | 0.2976 ±0.0197 | 0.3921 ±N/A | 0.4365 ±N/A |
| **This run — packed2k gbs32 lr5e-6** | **0.5287 ±0.0215** | 0.6044 ±0.0210 | 0.6595 ±N/A | 0.7134 ±N/A |
| Δ (SFT − base) | +0.281 | +0.307 | +0.267 | +0.277 |

`±` is lm-eval's bootstrap **Stderr**; `inst_level_*` are `±N/A` (no bootstrap stderr for the
instruction-level aggregation).

## Weekly convergence CI

This recipe's `ci.downstream_eval` block registers it in the weekly convergence flow
(`tests/ci_tests/configs/convergence/convergence_recipes.yml`): train 1000 steps → IFEval → **pass iff
`|prompt_level_strict_acc − 0.5287| < 2·0.0215`** (band `[0.486, 0.572]`). The base ships no chat
template; the recipe supplies the `-it` template via `dataset.chat_template`. Kept `allow_failure`
until a green run verifies train + eval (`tokenizer: checkpoint` carries the template) + gate, then
flip to blocking like moonlight/qwen.

## Gotchas

- **Base ships no chat template** → the recipe supplies the `-it` template via
  `dataset.chat_template` (the VLM loader applies it to the processor); no manual staging needed.
- **Eval flashinfer/cutlass drift** → `setup_lm_eval.sh` pins `nvidia-cutlass-dsl==4.5.0` (flashinfer
  0.6.13's `>=4.5.0` is too loose; 4.5.2 breaks the CuTeDSL MLIR API).
