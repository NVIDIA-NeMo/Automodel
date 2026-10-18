# Decision-model post-training examples

| File | Recipe | Description |
|---|---|---|
| `clm_soc_triage.yaml` | `TrainCLMHeadsRecipe` | CLM-v0.1-8B: frozen Qwen3-8B encoder (embedded once, cached), projection-head eval / SFT / RL |
| `clef_soc_triage_heads.yaml` | `TrainClefDecisionRecipe` | CLEF frozen backbone + joint schema head, using the same eval / SFT / reward loop |
| `clef_soc_triage_lora.yaml` | `TrainClefDecisionRecipe` | Clef 27B: LoRA + joint schema head with FSDP2, eval / SFT / RL, stage chaining with `init_from` |
| `d1_posttrain.py` | `D1PostTraining` | Experimental frozen-backbone D1-3B answer-token rows or Omni marker-head adaptation, with a publisher-baseline gate |
| `export_clef_release.py` | — | Merge a Clef stage (LoRA + head) into a directory with the `Cloudflare/clef` release layout |

From the repository root, prepare fresh fictional data first (128 train, 32 validation and 32 test records):

```bash
PYTHONPATH=tutorials/decision_models uv run --no-project python -m decision_tutorial.data \
    --output-dir data/soc_triage --seed 7
```

The generator rejects an existing directory. It needs no model, API key or GPU. These tutorial labels and utilities
come from a fictional policy and do not measure operational security performance. Use the framework's supported
Linux GPU environment for the training commands, and choose the process count for your available hardware.

Every run takes `--mode eval|train` and `--loss.mode sft|rl`, and writes `metrics_{validation,test}.json` to
`output_dir`. Running `eval` on the released checkpoint gives the *before* numbers to compare stages against:

```bash
automodel examples/decision/clm_soc_triage.yaml --nproc-per-node 8 --mode eval --output_dir runs/clm/base
automodel examples/decision/clm_soc_triage.yaml --nproc-per-node 8 --mode train --loss.mode sft --output_dir runs/clm/sft
automodel examples/decision/clm_soc_triage.yaml --nproc-per-node 8 --mode train --loss.mode rl \
    --heads.init_checkpoint runs/clm/sft/heads.pt --train.lr 1.0e-4 --train.epochs 10 \
    --train.eval_metric reward --output_dir runs/clm/rl
```

Data records use the SystemOne request format plus supervision:
`{"id", "state", "questions", "answers": {qid: option}, "rewards": {qid: {option: reward}}, "meta": {...}}`.

The CLM feature cache requires immutable encoder and tokenizer commit IDs. Its identity includes truncation,
pooling, normalization, precision and library settings, so changing those settings generates a new cache.
CLM/CLEF example configurations pin the public model versions whose prompt and checkpoint contracts they use.
The CLEF frozen-backbone configuration installs no adapters and optimizes only the external joint head.
Stages save weights for staged initialization; they are not full optimizer/scheduler resume checkpoints.

D1 examples are experimental. Tiny configuration tests exercise causal/readout/gradient contracts; released-size
GPU acceptance remains required for D1-3B. A prior released-size D1-3B baseline comparison exceeded the declared probability
tolerance, and no successful fresh full-size rerun has been collected for this contribution. The contributed NeMo
implementation follows this repository's Apache 2.0 license. Downloaded publisher code and model artifacts retain
their own license terms:
[model](https://huggingface.co/LiquidAI/d1-3B) and [Omni](https://huggingface.co/LiquidAI/d1-omni-600M).
No publisher weights are included in these examples.

CLEF uses complete optimizer windows and drops an incomplete final window. Its global batch size must be positive,
divisible by `local_batch_size * DP world size`, and small enough to fit at least one training window. For the
16-record checked-in fixture, reduce the global batch size (for example, local batch 1, global batch 4, one or two
processes). An undersized split fails rather than producing a successful zero-step run. Losses normalize over
eligible question weights across the complete accumulation window and DP group. A rank with no local supervision
participates with zero local loss; a globally feedback-free window fails on every rank before backward.
