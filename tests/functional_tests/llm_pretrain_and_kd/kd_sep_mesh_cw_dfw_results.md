# Separate-Mesh KD bf16 Results on cw-dfw

Validated on 2026-07-09 PDT (2026-07-10 UTC) for PR 2954
(`akoumpa/separate_mesh_kd`). The runs used
`nemo_auto_nightly_15_may_2026.sqsh` with the current Automodel worktree mounted
at `/workspace` and selected through `PYTHONPATH=/workspace`.

## Method

- Student: `google/gemma-3-270m` in bf16.
- Dense teacher: `google/gemma-4-31B-it` in bf16.
- MoE teacher: `google/gemma-4-26B-A4B-it` in bf16.
- Loss reductions use the recipe's configured fp32 upcast.
- Optimizer learning rate is zero so topology is the only changing variable.
- Dense runs use three complete global batches of 24 samples (`local_batch_size=3`).
- MoE runs repeat one deterministic tokenizer-encoded natural-text sample so the
  one-GPU shared reference and eight-way student DP run consume identical data.
- Pass criterion: `math.isclose` with `rtol=1%` and `atol=0.01` for total, CE,
  and KD loss at every step.

## Topologies

| Layout | Allocation | Student mesh | Teacher mesh |
|---|---:|---|---|
| Shared dense | 1 node, 8 GPUs | Shared `DP8` | Shared `DP8` |
| Separate TP2 | 4 nodes, 32 GPUs | Ranks 0-7, `DP8`, node 0 only | Ranks 8-31, `DP12 x TP2`, nodes 1-3 |
| Separate PP3 | 4 nodes, 32 GPUs | Ranks 0-7, `DP8`, node 0 only | Ranks 8-31, `DP8 x PP3`, nodes 1-3 |
| Shared MoE | 1 node, 1 GPU | Shared `DP1` | Shared custom backend, no EP |
| Separate EP8 | 5 nodes, 40 GPUs | Ranks 0-7, `DP8`, node 0 only | Ranks 8-39, `EP-shard4 x EP8`, nodes 1-4 |

The three-node teacher layout was invalid for Gemma4 MoE because expert feature
dimension 2816 cannot be evenly FSDP-sharded three ways. The final EP run uses
four teacher nodes, satisfying the requested greater-than-two-node teacher shape
and evenly sharding the feature dimension.

## Losses

| Layout | Step | Total | CE | KD |
|---|---:|---:|---:|---:|
| Shared dense | 0 | 10.3772888 | 10.1918383 | 10.5627394 |
| Shared dense | 1 | 9.7734632 | 9.8107901 | 9.7361355 |
| Shared dense | 2 | 10.3017607 | 10.4417124 | 10.1618090 |
| Separate TP2 | 0 | 10.3645096 | 10.1918383 | 10.5371799 |
| Separate TP2 | 1 | 9.7611465 | 9.8107901 | 9.7115040 |
| Separate TP2 | 2 | 10.2936783 | 10.4417124 | 10.1456451 |
| Separate PP3 | 0 | 10.3875456 | 10.1918383 | 10.5832520 |
| Separate PP3 | 1 | 9.7826176 | 9.8107901 | 9.7544470 |
| Separate PP3 | 2 | 10.2908812 | 10.4417124 | 10.1400490 |
| Shared MoE | 0-2 | 7.9366655 | 5.7125373 | 10.1607933 |
| Separate EP8 | 0-2 | 7.9366651 | 5.7125368 | 10.1607933 |

The maximum relative difference across the full matrix is 0.1261% for total
loss and 0.2530% for KD loss. EP8 KD loss is identical at float precision; its
maximum total-loss relative difference is `6.01e-8`.

The configured `KDLoss` reports teacher-to-student cross-entropy, which includes
the teacher distribution's entropy. Its absolute value is therefore expected to
be materially larger than a zero-based KL divergence; the parity comparison
uses the same objective and pretrained checkpoints in every layout.

## Evidence

| Layout | Slurm job | State | W&B run |
|---|---:|---|---|
| Shared dense | 13682961 | Completed | [rs4m2jin](https://wandb.ai/Nemo-automodel/kd_sep_mesh/runs/rs4m2jin) |
| Separate TP2 | 13683877 | Completed | [rar0y57q](https://wandb.ai/Nemo-automodel/kd_sep_mesh/runs/rar0y57q) |
| Separate PP3 | 13683632 | Completed | [b0ty4ngv](https://wandb.ai/Nemo-automodel/kd_sep_mesh/runs/b0ty4ngv) |
| Shared MoE | 13685292 | Completed | [dulb2mcv](https://wandb.ai/Nemo-automodel/kd_sep_mesh/runs/dulb2mcv) |
| Separate EP8 | 13685397 | Completed | [0ipqqi24](https://wandb.ai/Nemo-automodel/kd_sep_mesh/runs/0ipqqi24) |
| Parity summary | 13686234 | Completed | [xgjsx2y0](https://wandb.ai/Nemo-automodel/kd_sep_mesh/runs/xgjsx2y0) |

Project: [Nemo-automodel/kd_sep_mesh](https://wandb.ai/Nemo-automodel/kd_sep_mesh)

Remote artifacts are under
`/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/akoumparouli/what/codex_remote_work/kd_sep_mesh`:

- `results/kd_sep_mesh/parity_summary.json`
- `results/kd_sep_mesh/*/training.jsonl`
- `logs/kd_sep_mesh_*`

The final comparison is reproducible with:

```bash
python tests/functional_tests/llm_pretrain_and_kd/compare_kd_sep_mesh_losses.py \
  --shared-dense results/kd_sep_mesh/shared_dense/training.jsonl \
  --separate-tp results/kd_sep_mesh/separate_tp/training.jsonl \
  --separate-pp results/kd_sep_mesh/separate_pp/training.jsonl \
  --shared-moe results/kd_sep_mesh/shared_moe/training.jsonl \
  --separate-ep results/kd_sep_mesh/separate_ep/training.jsonl \
  --output results/kd_sep_mesh/parity_summary.json
```
