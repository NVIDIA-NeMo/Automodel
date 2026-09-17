# Titans paper reproduction target

This directory targets the NeurIPS 2025 proceedings version of
**Titans: Learning to Memorize at Test Time** (Behrouz, Zhong, and Mirrokni),
while retaining the original arXiv v1 experiments. The proceedings paper is
the authoritative target for the added 1.3B/100B-token and RULER results.

## Pinned language-model setup

- Data: FineWeb-Edu.
- Tokenizer: Llama 2, vocabulary size 32K.
- Training sequence length: 4096 tokens (2048 for sliding-window attention).
- Neural-memory chunk size: 16.
- Deep-memory update: public `titans-pytorch` chunk-aggregated semantics. The
  canonical LMM uses one 4224-token gradient-anchor batch (128 persistent
  vectors followed by the 4096 data tokens); the `no_persistent` ablation uses
  one 4096-token anchor batch.
- Persistent memory: 128 learned vectors prepended once to the model input,
  following Equation 6. Their hidden states are discarded before the LM head.
  This is the recorded operational choice for LMM; hybrids will use persistent
  attention K/V slots and receive separate parity tests.
- Long-term-memory output: 256 memory tokens for hybrid architectures.
- Memory MLP: two layers by default, expansion factor 4, GELU, residual
  connection, and layer normalization.
- Token mixer: Llama macro architecture with SwiGLU, RoPE, RMSNorm, causal
  depthwise convolution of width 4 after Q/K/V projections, L2-normalized
  queries and keys, and gated normalization before the output projection.
- Optimizer: AdamW, weight decay 0.1, cosine learning-rate schedule.
- Effective global batch: approximately 0.5M tokens.

The proceedings architecture table gives these reported scale points:

- 170M: 12 blocks, hidden size 768, 16 heads, peak LR 3e-3, 15B tokens.
- 340M: 24 blocks, hidden size 1024, 16 heads, peak LR 1.5e-3, 15B tokens.
- 760M: 24 blocks, hidden size 1536, 16 heads, peak LR 1.25e-3, 30B tokens.
- 1.3B: 18 blocks, hidden size 2048, 8 heads, peak LR 7e-4, 100B tokens.

There is an internal inconsistency in those reported dimensions. With the
LMM block implemented here, the 24-block 1024- and 1536-dimensional shapes
contain 507,620,608 and 1,091,731,968 parameters. Sixteen blocks contain
360,303,104 and 760,655,360 parameters respectively, matching Figure 7's
"360M" label and the reported 760M parameter class. The larger LMM recipes
therefore use 16 blocks and record this as an operational choice; runs using
them must not claim fidelity to the appendix's 24-block row.

## Reproduction order

1. Validate the neural-memory module and its Gated DeltaNet reduction.
2. Train the 170M memory-only LMM as the scale and convergence gate.
3. Reproduce the memory-depth and component ablations.
4. Train the 340M and 760M LMM and hybrid variants (MAC, MAG, MAL).
5. Reproduce language-model perplexity and common-sense evaluations.
6. Reproduce S-NIAH and the proceedings RULER evaluation.
7. Attempt the 1.3B/100B-token proceedings extension.

## Unresolved details

No official implementation was released. The paper does not fully specify:

- the 400M model dimensions reported in the language-model table;
- whether the architecture-table peak learning rates supersede the prose
  statement that training uses a learning rate of 4e-4;
- whether the 24-block entries for the 340M/760M rows are erroneous, since
  parameter-count-matched LMMs use 16 blocks in this implementation;
- all tensor shapes and parameterizations of the channel-wise forget gate;
- exact chunk-boundary retrieval alignment and streaming-state semantics;
- all implementation details needed to reconstruct MAC, MAG, and MAL.

Do not silently guess these details. Record each operational choice in the
resolved run configuration and validate it against an independent reference
or ablation before labeling a run as a paper reproduction.

## Acceptance gates

A full-scale run may start only after:

- tiny-model forward, backward, and checkpoint round-trip tests pass;
- deep-memory chunk size 1 matches an independent per-token autograd reference;
- momentum disabled matches the Gated DeltaNet recurrence;
- state carried across two calls matches one concatenated causal call;
- the configured parameter count matches the intended scale;
- a one-GPU smoke and one-node distributed smoke produce finite loss.

## Preflight status

Validated on 2026-09-08 with RTX 6000 Ada GPUs:

- The deterministic 760,784-parameter smoke model completed two training and
  validation steps on one GPU, wrote DCP checkpoints, and resumed from
  `LATEST` for a third step.
- The same smoke completed one FSDP2 step on two GPUs, including validation
  and checkpointing. Titans uses a dtype-aware FSDP strategy so its fp32 decay
  parameters are isolated from bf16 parameters.
- The full 173,597,376-parameter model completed one forward/backward optimizer
  step at sequence length 128 on one GPU. It allocated 6.68 GiB and processed
  180.7 tokens/s. This is a short-sequence viability probe, not an estimate
  that establishes 4096-token fit.
- The NanoGPT writer/reader round trip now records BOS locations consistently
  as token offsets. Validation datasets use `repeat: false`; otherwise the
  iterable validation loop never terminates.
- The first cw-dfw 4K attempt exposed an invalid one-step warmup and the second
  exposed uncheckpointed neural-memory activations at 79 GiB/H100. Both are
  fixed. Checkpointing reduced the measured allocation to about 22.4 GiB/H100,
  but the per-chunk Python reference backend then exceeded the 30-minute NCCL
  watchdog. The production recipe now selects the vectorized public backend
  and its Triton associative scan.
- cw-dfw job `18245782` completed the ten-step production-shaped pilot on
  2026-09-09: 5,242,880 FineWeb-Edu tokens, loss 16.2104 → 8.5571,
  approximately 89.5K aggregate tokens/s, 4.51 GiB/GPU peak allocation,
  finite distributed validation, and a resumable checkpoint.
- The first 15B chain (`titans170m15bv1`) was launched before persistent
  prefixing landed. Preserve it as the paper's `no_persistent` component
  ablation; do not report it as the canonical LMM baseline.

The next execution gate is a full-shape, 4096-token, one-node FSDP2 smoke on
the target cluster. Do not start the 15B-token run until that job establishes
memory headroom and step throughput.

### cw-dfw H100 gate

The preferred workflow uses the configured `cw-dfw` profile from `slurm-cli`.
From a clean, pushed AutoModel branch:

```bash
tools/submit_titans_cwdfw.sh
# Or submit, monitor to a terminal state, and print the final log:
tools/submit_titans_cwdfw.sh --wait
```

The wrapper verifies that local `HEAD` equals the pushed branch, clones or
fast-forwards the Lustre checkout through `slurm-cli shell`, submits through
`slurm-cli job submit`, and prints reproducible status/log commands. It refuses
dirty or unpushed source trees. The initial submission uses `--auto-account`:
Slurm account associations are discovered with `sacctmgr`, the exact allocation
is projected with `sbatch --test-only`, and `sshare`/`sprio` fair-share priority
breaks near ties. Continuation segments retain the selected account.

After the 4K gate, launch the production-shaped pilot with:

```bash
tools/submit_titans_cwdfw.sh --pilot
```

This submits an eight-H100 pilot that first streams
`HuggingFaceFW/fineweb-edu` (`sample-10BT`) through AutoModel's NanoGPT
processor with the public `NousResearch/Llama-2-7b-hf` tokenizer when the
cached 64M-token shard is absent. It then trains on 5,242,880 tokens, validates
distributed control flow, and writes a job-specific checkpoint. Dataset
preparation uses the GPU allocation because `cw-dfw` does not currently grant
this account CPU-partition capacity. The pilot is not a paper result.

The full run uses the same wrapper after the pilot gate:

```bash
tools/submit_titans_cwdfw.sh --full
```

The first job creates sixteen document-aligned ~1B-token training shards from
FineWeb-Edu `sample-100BT` and one exact 4,194,304-token held-out shard. It then
starts a chain of 1,800-step jobs against one fixed checkpoint directory.
AutoModel restores model, optimizer, LR scheduler, dataloader, RNG, and global
step from `LATEST`; the global cosine horizon remains 28,610 steps across every
segment. W&B uses the stable run ID `titans170m15bv1`:

<https://wandb.ai/nvidia/titans-paper-reproduction/runs/titans170m15bv1>

### Multi-node component ablations

The Table 4 component runners use immutable, commit-addressed cluster checkouts
so preparing the next experiments cannot alter an active chain:

```bash
tools/submit_titans_ablation_cwdfw.sh baseline --pilot --nodes 2
tools/submit_titans_ablation_cwdfw.sh no_convolution --pilot --nodes 2
tools/submit_titans_ablation_cwdfw.sh no_momentum --pilot --nodes 2
tools/submit_titans_ablation_cwdfw.sh no_weight_decay --pilot --nodes 2
tools/submit_titans_ablation_cwdfw.sh depth3 --pilot --nodes 2
tools/submit_titans_ablation_cwdfw.sh depth4 --pilot --nodes 2
```

Replace `--pilot` with `--full` only after the two-node checkpoint-resume gate
passes. Each node runs one `torchrun` launcher with eight workers and c10d
rendezvous. Global batch remains 128, so gradient accumulation changes from 16
on one node to 8 on two nodes without changing tokens per optimizer step.

The `linear_memory` entry is intentionally launch-blocked: with momentum
enabled, that path still uses a 4096-iteration Python recurrence. It needs a
vectorized kernel before a paper-scale allocation is responsible.

Defaults:

- account: automatically selected from the user's authorized `cw-dfw`
  associations;
- partition: `batch`;
- current LMM allocation: one node with eight GPUs for four hours per segment;
- ablation allocation: configurable one or two nodes, eight GPUs per node;
- image: `nvcr.io#nvidia/nemo-automodel:26.04`;
- work root:
  `/lustre/fsw/portfolios/coreai/users/ffrujeri/titans-automodel`.

Override `AUTOMODEL_CHECKOUT`, `TITANS_WORK_ROOT`, or `TITANS_IMAGE` with
exported environment variables when the checkout, storage root, or approved
container differs. Success requires a finite train and validation loss plus
the logged peak memory and tokens/s from all eight workers.

### Blackwell scale and ablation pilots

Blackwell pilots use one portable total-GPU request instead of assuming eight
GPUs per node:

```bash
tools/submit_titans_blackwell.sh 170m baseline --pilot --total-gpus 8
tools/submit_titans_blackwell.sh 170m no_convolution --pilot --total-gpus 8
tools/submit_titans_blackwell.sh 340m baseline --pilot --total-gpus 8
tools/submit_titans_blackwell.sh 760m baseline --pilot --total-gpus 16
```

For a throughput comparison rather than a ten-step gate, set
`TITANS_PILOT_STEPS=100`; the effective local batch and checkpointing mode are
logged at startup.

The 170M model can also benchmark replicated data parallelism:

```bash
TITANS_DISTRIBUTED_STRATEGY=ddp TITANS_PILOT_STEPS=100 \
  tools/submit_titans_blackwell.sh 170m baseline --pilot --total-gpus 8
```

DDP uses BF16 autocast, gradient bucket views, and no activation checkpointing.
Static-graph mode remains disabled because Titans' custom neural-memory
backward is incompatible with the PyTorch DDP reducer's static-graph hooks.
FSDP2 remains the full-run default until this pilot demonstrates that DDP fits
and improves steady-state throughput.

The submitter asks `slurm-cli` to rank configured Blackwell clusters and their
authorized accounts. Eight total GPUs resolve to one node on
`nsc-svg-slurm-1` (B200) or `aws-pdx-slurm-1` (B300), and two nodes on the
four-GPU NVL72 partitions at `aws-cmh-slurm-1` and `oci-hsg-cs-001`. The
resolved topology is then fixed for the submitted job. Account prefixes are
mapped to verified compute-visible FSW roots such as
`/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_eval/users/ffrujeri`;
targets without a configured writable shared root are rejected. `/lustre` may
be a login-node alias for this physical path and is therefore not used by the
portable workflow. Data, immutable checkouts, checkpoints, caches, and job
logs remain under the shared FSW hierarchy, never `$HOME`.
Each pilot uses an
immutable AutoModel commit, prepares a reusable 64M-token shard if needed,
runs ten optimizer steps, writes a checkpoint, and logs to
`titans-paper-reproduction`.

Full runs use the same router with `--full`. The selected cluster first
prepares a document-aligned 15B-token dataset for the 170M/340M scales or a
30B-token dataset for 760M, including a real 4M-token FineWeb-Edu validation
split. It then starts a resumable training chain on the resolved topology:

```bash
tools/submit_titans_blackwell.sh 170m baseline --full --total-gpus 8
```

All new pilots and full runs log to the consolidated
`nvidia/titans-paper-reproduction` W&B project. Groups separate canonical
LMM scales (`lmm-170m`, `lmm-340m`, `lmm-760m`) from `ablations-170m`; cluster
and GPU topology remain properties of each run rather than separate projects.

Blackwell runs trade their large memory capacity for fewer gradient
accumulation rounds. Activation checkpointing is disabled, and the default
local batch is 8 for 170M, 4 for 340M, and 2 for 760M, capped by
`global_batch_size / world_size`. The global batch remains 128, so token
budgets and LR schedules are unchanged. Set `TITANS_LOCAL_BATCH_SIZE` or
`TITANS_ACTIVATION_CHECKPOINTING` to override these hardware defaults.
Set `TITANS_RUN_SUFFIX` when restarting with a new configuration so its
checkpoints and W&B run remain separate from an earlier canonical attempt.

Each completed full-run segment measures its optimizer-step throughput and
adjusts the next segment by at most 25%, targeting 13,200 seconds (3h40m) of
the four-hour allocation. This preserves a 20-minute checkpoint/runtime
margin while adapting `max_steps_per_run` to B200, B300, or GB200 throughput.
The global `max_steps` and cosine schedule never change.
