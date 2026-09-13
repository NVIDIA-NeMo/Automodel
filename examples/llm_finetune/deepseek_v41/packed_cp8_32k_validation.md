# Packed CP8 32K validation attempt

The full pretrained DeepSeek V4.1 Flash text model completed one optimizer
update, then failed with CUDA out of memory during the second update. This
attempt does not establish stable 32K training or 32K CP parity.

The recipe is deepseek_v41_flash_tulu3_packed_cp8_32k.yaml. The implementation
and recipe used for this run were pinned to commit
2ab674a988310bab347c41a2af4ee3f38d1b98d3. Subsequent documentation commits do
not change the model implementation.

| Setting or result | Value |
| --- | --- |
| Slurm job | 7112881 |
| Account / partition | coreai_dlalgo_llm / batch |
| GPUs / nodes | 64 GB200 / 16 |
| Model | All 40 text layers, hidden 5120, vocabulary 129280, Engram 1/14 |
| CP / EP / Engram owners | 8 / 64 / 64 |
| Global / local sequence length | 32768 / 4096 |
| GBS / LBS / gradient accumulation | 64 / 1 / 8 |
| Activation checkpointing | Per block |
| defer_fsdp_grad_sync | false |
| Requested limit / target | One hour / 100 optimizer updates |
| Actual start / end (PDT, 2026-09-12) | 22:56:09 / 23:02:56 |
| Actual elapsed / Slurm result | 6m 47s / FAILED, exit 143:0 |
| Completed updates | 1 (logged step 0) |
| Training loss / gradient norm | 0.825686216 / 2.703560350 |
| Supervised tokens in completed update | 1,210,247 |
| Validation updates / saved checkpoints | 0 / 0 |

All 16 nodes completed prewarm. All 64 initial parameter audit files exist.
Input audits cover steps 0 and 1, each with 64 unique packed samples and
gradient accumulation 8. Only step 0 completed. Its audited supervised-token
denominator matches the native loss logger exactly.

## Failure and memory evidence

The earliest recorded failure is rank 41 at attention.py line 329:

    scores = (scores.relu() * weights.unsqueeze(-1)).sum(dim=2)

CUDA could not allocate an additional 8.00 GiB for the ReLU result. That device
had 184.31 GiB capacity and 5.25 GiB free. The error reported 179.04 GiB process
memory including non-PyTorch allocations, with 130.33 GiB allocated by PyTorch
and 17.42 GiB reserved by PyTorch but unallocated.

The indexer first materializes scores with dimensions
[batch, local_query, index_head, global_compressed_key], before reducing the
head dimension. A local query length of 4096 therefore does not bound this
allocation independently of global sequence length. Global 32K increases the
key dimension relative to the completed 4K training runs.

The completed step's logged allocated-memory value was 111.010609 GiB on the
reference GPU. It is not the whole-job peak or a maximum across ranks and
must not replace the OOM memory evidence above. The allocation was terminated
after rank failures, well before the one-hour limit or minute-50 checkpoint
signal. No memory optimization was implemented or validated in this attempt.

## Native metrics and disposition

[W&B run](https://wandb.ai/Nemo-automodel/huiyingl_workspace/runs/ds41packed32kcp820260912a)
has state failed and exactly one native history row. Every original payload
field matches training.jsonl exactly; no validation row exists.

Cluster artifacts are under
/lustre/fsw/portfolios/coreai/users/huiyingl/ds41/logs/packed32k_20260912a/cp8/:
final_report.json, wandb_native_verification.json, oom_rank41_traceback.txt,
training.jsonl, input_audit.jsonl and run_scope.json.
The full Slurm log is
/lustre/fsw/portfolios/coreai/users/huiyingl/ds41/logs/ds41-pack32k-cp8-7112881.log.

The completed [packed CP1/CP2 4K validation](packed_cp_validation.md) remains
separate: its primary training-parity gates passed, while the predeclared
5% gradient-norm diagnostic failed and remains recorded.

Per the user's single-allocation limit, this run was not extended, requeued
or continued. Both monitoring controls were disabled after terminal-result
review. No new GPU job was submitted, and no commits were pushed.
