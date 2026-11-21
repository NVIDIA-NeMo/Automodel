# Decision-Model Post-Training Examples

| File | Recipe | Description |
|---|---|---|
| `clm_soc_triage.yaml` | `TrainCLMHeadsRecipe` | CLM-v0.1-8B: frozen Qwen3-8B encoder (embedded once, cached), projection-head eval / SFT / RL |
| `clef_soc_triage_heads.yaml` | `TrainClefDecisionRecipe` | CLEF frozen backbone + joint schema head, using the same eval / SFT / reward loop |
| `clef_soc_triage_lora.yaml` | `TrainClefDecisionRecipe` | Clef 27B: LoRA + joint schema head with FSDP2, eval / SFT / RL, stage chaining with `init_from` |
| `d1_posttrain.py` | `D1PostTraining` | Experimental frozen-backbone D1-3B answer-token rows or Omni marker-head adaptation, with a publisher-baseline gate |
| `gs1_distribution_sft.yaml` | `TrainGS1DecisionRecipe` | Native GS1 compiler, hybrid backbone, and decision head with distribution NLL + expected Brier SFT |
| `export_clef_release.py` | — | Merge a Clef stage (LoRA + head) into a directory with the `Cloudflare/clef` release layout |

From the repository root, prepare fresh fictional data first (128 train, 32 validation, and 32 test records):

```bash
PYTHONPATH=tutorials/decision_models uv run --no-project python -m decision_tutorial.data \
    --output-dir data/soc_triage --seed 7
```

The generator rejects an existing directory. It needs no model, API key, or GPU. These tutorial labels and utilities
come from a fictional policy and do not measure operational security performance. Use the framework's supported
Linux GPU environment for the training commands, and choose the process count for your available hardware.

CLM and CLEF runs take `--mode eval|train` and `--loss.mode sft|rl`, and write `metrics_{validation,test}.json` to
`output_dir`. Running `eval` on the released checkpoint gives the *before* numbers to compare stages against:

```bash
automodel examples/decision/clm_soc_triage.yaml --nproc-per-node 8 --mode eval --output_dir runs/clm/base
automodel examples/decision/clm_soc_triage.yaml --nproc-per-node 8 --mode train --loss.mode sft --output_dir runs/clm/sft
automodel examples/decision/clm_soc_triage.yaml --nproc-per-node 8 --mode train --loss.mode rl \
    --heads.init_checkpoint runs/clm/sft/heads.pt --optimizer.lr 1.0e-4 --train.epochs 10 \
    --train.eval_metric reward --output_dir runs/clm/rl
```

Data records use the SystemOne request format plus supervision:
`{"id", "state", "questions", "answers": {qid: option}, "rewards": {qid: {option: reward}}, "meta": {...}}`.

The CLM feature cache requires immutable encoder and tokenizer commit IDs. Its identity includes truncation,
pooling, normalization, precision, and library settings, so changing those settings generates a new cache.
CLM embeds across data-parallel ranks, then trains its projection heads on rank zero. Its `encode.cache_dir`
must be shared by all ranks, including across nodes, for shard assembly and cache reuse. Inconsistent cache
visibility fails on every rank. The encoding process group is released before rank-zero head training so
idle ranks do not wait in a collective for the duration of training.
CLM and CLEF example configurations pin the public model versions whose prompt and checkpoint contracts they use.
Set `train.eval_metric` to `accuracy`, `exact_match`, `reward`, or `expected_reward`; native stage selection
maximizes that metric.
CLEF training requires `checkpoint.enabled: true` so final reports use the validation-selected saved stage.
The CLEF frozen-backbone configuration installs no adapters and optimizes only the external joint head.
CLEF stages use native AutoModel PEFT checkpoints under `model/` and DCP joint-head state under `head/`.
Immutable directories such as `step_00000000` and `step_00000025` are published through the existing checkpoint lifecycle;
`LATEST` selects the last saved stage and `LOWEST_VAL` selects the highest validation score (stored as its negative).
Set `init_from: runs/clef/sft/LOWEST_VAL` to initialize the next stage with fresh optimizer and scheduler state.
This is weights-only stage continuation; full optimizer and scheduler resume is not provided. Keep the base revision,
PEFT configuration, and joint-head architecture consistent. Use `head_optimizer.lr` for the external head,
`optimizer.lr` for the backbone adapters, and `lr_scheduler.lr_warmup_steps` for integer optimizer-step warmup.
The decision recipe does not prune these `step_*` stage directories automatically; manage their retention explicitly.
Publisher-layout merged export remains the explicit `examples/decision/export_clef_release.py` operation.

D1 examples are experimental. Tiny configuration tests exercise causal, readout, and gradient contracts. The
[D1 acceptance notes](../../tests/functional_tests/decision_models/d1/README.md) record released-size D1-3B BF16
and Omni FP32 checks on a small fictional fixture, with their numerical tolerances and remaining limitations.
These checks do not establish adaptation quality or Omni FP16 acceptance. The contributed NeMo
implementation follows this repository's Apache 2.0 license. Downloaded publisher code and model artifacts retain
their own license terms:
[D1-3B](https://huggingface.co/LiquidAI/d1-3B) and [D1 Omni](https://huggingface.co/LiquidAI/d1-omni-600M).
No publisher weights are included in these examples.

GS1 checkpoints expose their native `AutoModel` implementation, prompt compiler, hybrid backbone, and cross-attention
decision head through their Hub code. The recipe loads that checkpoint through NeMo AutoModel's DDP path and samples up
to 3,000 text-only questions from a native `request`/`targets` JSONL file. Its single-stage objective is
`distribution_NLL + brier_weight * expected_Brier`; `brier_weight` defaults to 1.0. Image/video states and tensor,
context, pipeline, and expert parallelism are not supported by this initial recipe.

The GS1 family provides [0.8B](https://huggingface.co/bigai/GS1-0.8B),
[2B](https://huggingface.co/bigai/GS1-2B), [4B](https://huggingface.co/bigai/GS1-4B), and
[9B](https://huggingface.co/bigai/GS1-9B) checkpoints; the example YAML pins GS1-0.8B.

The checkpoint code requires Transformers 5.3 or newer and `typesafe-sdk==0.7.0`. Install the optional GS1 runtime
dependency with `uv pip install "typesafe-sdk==0.7.0"`, then set `data.train` in
`examples/decision/gs1_distribution_sft.yaml` to your `distribution-v4.5/train.jsonl` path and run:

```bash
automodel examples/decision/gs1_distribution_sft.yaml --nproc-per-node 8
```

CLEF uses complete optimizer windows and drops an incomplete final window. Its global batch size must be positive,
divisible by `local_batch_size * DP world size`, and small enough to fit at least one training window. For the
16-record checked-in fixture, reduce the global batch size (for example, local batch 1, global batch 4, one or two
processes). An undersized split fails rather than producing a successful zero-step run. Losses normalize over
eligible question weights across the complete accumulation window and DP group. A rank with no local supervision
participates with zero local loss; a globally feedback-free window fails on every rank before backward.
