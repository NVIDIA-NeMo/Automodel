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

Every position in the batch is an anchor: one mask token per depth is appended
after it, and a LoRA adapter that is active only on the mask tokens learns to
draft the next tokens from them. All other tokens use the frozen base weights,
so the model's own next-token distribution is unchanged and it verifies the
drafts losslessly. The mask tokens' hidden states are fed to the same MTP loss
and `val_mtp_tau_sim` metric as a native MTP head. `mtp.distillation_mode: soft`
distills every depth toward the frozen base model's distribution, the
verifier at decode time:

```bash
automodel examples/llm_finetune/mtp_finetune/qwen3_0p6b_gated_lora_mtp.yaml --nproc-per-node 8
```

The validation loss stays constant during training, since it is the base
model's loss. The dense layout multiplies the sequence length by
`1 + len(mask_token_ids)` and needs an attention backend that accepts a custom
mask (`flex_attention`, preferred at scale, `sdpa`, or `eager`). It supports
FSDP2/HSDP and DDP, but not yet tensor, context, or pipeline parallelism,
packed sequences, DoRA, or LoRA on MoE experts.
