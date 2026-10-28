# MTP (Multi-Token Prediction) Finetuning

Several model families (Qwen3.5 dense/MoE, nemotron_v3, step3p7, minimax_m3_vl,
deepseek_v4, GLM-5.2, nemotron_omni) ship a native MTP head: extra decoder
depths that each predict one additional future token, trained alongside the
main next-token loss. MTP finetuning improves that head's self-speculative
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
   self-speculative accept length under teacher-forced decoding — the same
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
