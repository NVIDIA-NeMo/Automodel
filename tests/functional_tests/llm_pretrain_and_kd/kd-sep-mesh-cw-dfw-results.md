# Separate-Mesh KD BF16 Numerics on cw-dfw

Validated on 2026-07-10 for PR 2954 (`akoumpa/separate_mesh_kd`). The runs used
`nemo_auto_nightly_15_may_2026.sqsh` with the current Automodel worktree mounted
at `/workspace`. A Transformers 5.12.1 overlay was placed before `/workspace` on
`PYTHONPATH` because that is the version pinned by the checked-out code.

## Numeric fixes

The original KD values near 10 were not zero-based KL divergence. Two independent
issues inflated them:

1. `KDLoss` computed teacher-to-student soft cross-entropy, which includes teacher
   entropy, despite documenting forward KL. It now computes
   `KL(P_teacher || P_student) = sum(P_teacher * (log P_teacher - log P_student))`
   in the normal, chunked, and tensor-parallel paths. This changes the reported
   scalar but leaves student gradients unchanged.
2. The student fixture rendered the smaller model's chat template and allowed
   sliding windows to start after the teacher's generation prompt. The 31B/26B
   teacher then assigned near-unit probability to a missing channel token. The
   fixture now uses the teacher tokenizer and generation prompt, masks prompt
   labels, and repeats a complete deterministic transcript.

With the corrected prompt, the MoE custom teacher and stock Hugging Face
teacher produce true KL values of `1.83363` and `1.82491`, respectively (0.48%
difference), confirming that the custom teacher backend is not the source of the
previous high value.

## Method

- Student: `google/gemma-4-E2B-it` in BF16.
- Dense teacher: `google/gemma-4-31B-it` in BF16.
- MoE teacher: `google/gemma-4-26B-A4B-it` in BF16.
- All layouts use the same teacher-tokenized, assistant-supervised transcript.
- Optimizer learning rate is zero, so every step intentionally repeats and only
  topology changes between paired runs.
- `kd_ratio=0.5`, so `Total = 0.5 * CE + 0.5 * KL`.
- Pass criterion: `math.isclose` with `rtol=1%` and `atol=0.05` for total, CE,
  zero-based KL, and gradient norm at every step.

## Topologies

| Layout | Allocation | Student mesh | Teacher mesh |
|---|---:|---|---|
| Shared dense | 1 node, 8 GPUs | Shared `DP8` | Shared `DP8` |
| Separate TP2 | 4 nodes, 32 GPUs | Ranks 0-7, `DP8`, node 0 only | Ranks 8-31, `DP12 x TP2`, nodes 1-3 |
| Separate PP3 | 4 nodes, 32 GPUs | Ranks 0-7, `DP8`, node 0 only | Ranks 8-31, `DP8 x PP3`, nodes 1-3 |
| Shared MoE | 1 node, 8 GPUs | Shared `DP8` | Shared Hugging Face `DP8` |
| Separate EP8 | 5 nodes, 40 GPUs | Ranks 0-7, `DP8`, node 0 only | Ranks 8-39, `DP32 x EP8 x expert-shard4`, nodes 1-4 |

The student always occupies exactly one node. Every separate teacher occupies
at least three nodes, and the matrix exercises teacher TP, PP, and EP.

## Losses

| Layout | Step | Total | CE | Zero-based KL | Grad norm |
|---|---:|---:|---:|---:|---:|
| Shared dense | 0 | 6.4569998 | 10.4323130 | 2.4816835 | 824.0 |
| Shared dense | 1 | 6.4569998 | 10.4323130 | 2.4816835 | 824.0 |
| Shared dense | 2 | 6.4569998 | 10.4323130 | 2.4816835 | 824.0 |
| Separate TP2 | 0 | 6.4361105 | 10.4323130 | 2.4399059 | 824.0 |
| Separate TP2 | 1 | 6.4361105 | 10.4323130 | 2.4399059 | 824.0 |
| Separate TP2 | 2 | 6.4361105 | 10.4323130 | 2.4399059 | 824.0 |
| Separate PP3 | 0 | 6.4485483 | 10.4323130 | 2.4647844 | 820.0 |
| Separate PP3 | 1 | 6.4485483 | 10.4323130 | 2.4647844 | 820.0 |
| Separate PP3 | 2 | 6.4485483 | 10.4323130 | 2.4647844 | 820.0 |
| Shared MoE | 0 | 6.1531248 | 10.4813356 | 1.8249131 | 812.0 |
| Shared MoE | 1 | 6.1531248 | 10.4813356 | 1.8249131 | 812.0 |
| Shared MoE | 2 | 6.1531248 | 10.4813356 | 1.8249131 | 812.0 |
| Separate EP8 | 0 | 6.1574821 | 10.4813356 | 1.8336279 | 820.0 |
| Separate EP8 | 1 | 6.1574821 | 10.4813356 | 1.8336279 | 820.0 |
| Separate EP8 | 2 | 6.1574821 | 10.4813356 | 1.8336279 | 820.0 |

CE matches exactly in every pair. The maximum total-loss difference is 0.3235%,
and the maximum gradient-norm difference is 0.9852%. TP2 has the largest KL
difference: `0.04178` absolute and 1.6834% relative. A clean TP2 restart
reproduced `2.4399059` exactly, showing that this is deterministic BF16 reduction
order rather than instability. The absolute tolerance is important for the
smaller zero-based KL scalar because it is obtained by subtracting two close
log-probability terms.

## Evidence

| Layout | Slurm job | State | W&B run |
|---|---:|---|---|
| Shared dense | 13689977 | Completed | [eekmgoh9](https://wandb.ai/Nemo-automodel/kd_sep_mesh/runs/eekmgoh9) |
| Separate TP2 | 13689979 | Completed | [q09df5b1](https://wandb.ai/Nemo-automodel/kd_sep_mesh/runs/q09df5b1) |
| Separate PP3 | 13689980 | Completed | [djpndi6m](https://wandb.ai/Nemo-automodel/kd_sep_mesh/runs/djpndi6m) |
| Shared MoE | 13689813 | Completed | [4lr58vi9](https://wandb.ai/Nemo-automodel/kd_sep_mesh/runs/4lr58vi9) |
| Separate EP8 | 13689817 | Completed | [xu5gzzf3](https://wandb.ai/Nemo-automodel/kd_sep_mesh/runs/xu5gzzf3) |
| Parity summary | 13690273 | Completed | [rcaa6vbb](https://wandb.ai/Nemo-automodel/kd_sep_mesh/runs/rcaa6vbb) |

Project: [Nemo-automodel/kd_sep_mesh](https://wandb.ai/Nemo-automodel/kd_sep_mesh)

Remote artifacts are under
`/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/akoumparouli/what/codex_remote_work/kd_sep_mesh`:

- `results/kd_sep_mesh/ship_parity_summary.json`
- `results/kd_sep_mesh/ship_*/training.jsonl`
- `logs/kd_sep_mesh_*`

The final comparison is reproducible with:

```bash
python tests/functional_tests/llm_pretrain_and_kd/compare_kd_sep_mesh_losses.py \
  --shared-dense results/kd_sep_mesh/ship_shared_dense/training.jsonl \
  --separate-tp results/kd_sep_mesh/ship_separate_tp/training.jsonl \
  --separate-pp results/kd_sep_mesh/ship_separate_pp/training.jsonl \
  --shared-moe results/kd_sep_mesh/ship_shared_moe/training.jsonl \
  --separate-ep results/kd_sep_mesh/ship_separate_ep/training.jsonl \
  --relative-tolerance 0.01 \
  --absolute-tolerance 0.05 \
  --output results/kd_sep_mesh/ship_parity_summary.json
```
