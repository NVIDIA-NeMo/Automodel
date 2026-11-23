# Multi-Token Prediction (MTP) Fine-Tuning

Several model families (Qwen3.5 dense and MoE, nemotron_v3, step3p7, minimax_m3_vl,
deepseek_v4, GLM-5.2, and nemotron_omni) ship a native MTP head: extra decoder
depths that each predict one additional future token, trained alongside the
main next-token loss. MTP fine-tuning improves that head's self-speculative
accept length for a target domain without retraining the base model.

This is **not** a separate speculative-decoding recipe or draft model (see
`examples/speculative/` for those). MTP already trains through the standard
`TrainFinetuneRecipeForNextTokenPrediction` recipe whenever the loaded model
supports it. The only two pieces this directory adds are:

1. **`freeze_config`** to freeze everything except the MTP block, so training
   only touches the MTP depths:

   ```yaml
   freeze_config:
     freeze_modules:
     - glob: "*"
     unfreeze_modules:
     - glob: "*mtp*"
   ```

2. **`val_mtp_tau_sim`**, a validation-only metric (see
   `nemo_automodel/components/speculative/mtp/core.py`) reporting the expected
   self-speculative accept length under teacher-forced decoding, the same
   `tau_sim` convention the EAGLE-3 recipe already logs. Acceptance is judged
   against the model's **own main-head greedy prediction** (the actual
   verifier in self-speculative decoding), not the dataset label, since those
   two can disagree whenever the main head is imperfect.

Only models that opt back into computing MTP outputs during `model.eval()`
report this metric. Today, that is `nemotron_v3` (and `nemotron_omni`, which
delegates to it) through `model.compute_mtp_in_eval` (the recipe toggles it on for
validation and restores it afterward). Other MTP-capable families currently
gate MTP on `self.training` unconditionally, so `val_mtp_tau_sim` stays absent
for them until they add the same hook.

`nemotron_nano_v3_5_lightning_mtp_finetune.yaml` runs this on a single GPU
against `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, which ships a
real pretrained MTP layer (`mtp.*` weights in its safetensors index):

```bash
automodel examples/llm_finetune/mtp_finetune/nemotron_nano_v3_5_lightning_mtp_finetune.yaml --nproc-per-node 1
```

Watch `val_mtp_tau_sim` across validation runs: it should trend upward as the
MTP head adapts to the fine-tuning domain. A value near `1.0` means the MTP
head is barely contributing beyond the main head's own next token; a value
approaching `1 + num_mtp_layers` means every MTP depth is being accepted.

## Gated-LoRA MTP for Models without an MTP Head

`qwen3_0p6b_gated_lora_mtp.yaml` adds MTP to a model that has no MTP layers,
following [Your LLM Knows the Future](https://arxiv.org/abs/2507.11851). A
`gated_mtp:` block lists one mask token id per MTP depth:

```yaml
gated_mtp:
  mask_token_ids: [151700, 151701, 151702]
```

The dense layout computes one block of mask tokens per clean position. A LoRA
adapter is active only on those masks, and clean queries never attend mask keys.
The clean decoder path stays frozen during training. A shared sampler implements
Section 2.3's two `Linear -> SiLU -> LayerNorm` blocks and projects their output
through the frozen LM head. Depth one starts from the base NTP prediction;
later depths use the preceding sampler prediction. There is no residual
connection or zero-initialized projection in this sampler.

Select the paper's Section 2.5 objective explicitly:

```yaml
gated_mtp:
  mask_token_ids: [151700, 151701, 151702]
  objective: paper
```

It trains base-head CE and sampler-head CE over supervised clean and mask
positions, plus latent consistency MSE toward detached clean hidden states.
Masks after unsupervised anchors do not contribute loss. CE is averaged over
both kinds of positions; consistency first averages matching masks for each
clean target, then averages over supervised clean positions. Samples are
averaged across the full accumulation window, including a partial final window.
`gated_mtp.scaling_factor` scales this complete objective (default `1.0`).
Do not combine it with `mtp.distillation_mode: soft`.

```bash
automodel examples/llm_finetune/mtp_finetune/qwen3_0p6b_gated_lora_mtp.yaml --nproc-per-node 1
```

The 0.6B example is a small validation setup. The
`qwen3_4b_gated_lora_mtp_paper.yaml` example uses eight depths, LoRA rank 128,
AdamW at `2e-4`, 5,000 warmup steps, and 50,000 updates. A single GPU accumulates
an effective batch of eight. Qwen3-4B, Dolly data, sequence length 512, and one
GPU are adaptations; this does not reproduce the paper's speedup experiments.

For compatibility, omitting `objective` retains `objective: mtp`: the existing
hard-MTP CE baseline (without a sampler), or experimental soft distillation
when `mtp.distillation_mode: soft` is supplied. The hard baseline supports
FSDP2/HSDP and DDP. The paper objective and soft sampler currently support only
one GPU because the sampler is attached after the base model is sharded.
All variants require `flex_attention`, `sdpa`, or `eager`, and exclude tensor,
context, and pipeline parallelism, packed sequences, DoRA, sliding-window
attention, and a LoRA-adapted LM head. The paper objective additionally requires
`MaskedCrossEntropy` and excludes domain-mixture loss weights.

Validation reports frozen-base CE and the sampler's teacher-forced prefix-hit
metric `val_mtp_tau_sim`; validation CE does not measure sampler training loss.
The metric is a proxy for acceptance, not an end-to-end generation benchmark.

Mask ids reuse reserved vocabulary rows. Setup deterministically randomizes
those input rows and freezes them. With tied embeddings this also rewrites the
corresponding output rows, changing mask-token logits and the softmax
normalizer: exact distribution parity with the untouched checkpoint is not
guaranteed. The gated path still preserves the clean distribution of this
initialized verifier, provided clean inputs do not contain the reserved ids.

PEFT checkpoints save LoRA and sampler tensors, but not the frozen randomized
mask rows. An external loader must recreate those rows with the same seed and
load the sampler explicitly. Start a new training run for this sampler:
checkpoints from the former zero-initialized residual sampler have different
semantics and must not be resumed as paper-sampler checkpoints.
