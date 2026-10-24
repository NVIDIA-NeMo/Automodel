# MTP (Multi-Token Prediction) Finetuning

Several model families (Qwen3.5 dense/MoE, nemotron_v3, step3p7, minimax_m3_vl,
deepseek_v4, GLM-5.2, nemotron_omni) ship a native MTP head: extra decoder
depths that each predict one additional future token, trained alongside the
main next-token loss. MTP finetuning improves that head's self-speculative
accept length for a target domain without retraining the base model.

This is **not** a separate speculative-decoding recipe or draft model (see
`examples/speculative/` for those). MTP already trains through the standard
`TrainFinetuneRecipeForNextTokenPrediction` recipe whenever the loaded model
supports it — the only two pieces this directory adds are:

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
   `tau_sim` convention the EAGLE-3 recipe already logs, computed here from the
   MTP depths' own argmax against ground truth. It appears automatically in
   `_run_validation_epoch`'s metrics whenever the model's MTP head is enabled;
   no config is needed to turn it on.

`nemotron_nano_v3_mtp_finetune.yaml` runs this on a single GPU against
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`:

```bash
automodel examples/llm_finetune/mtp_finetune/nemotron_nano_v3_mtp_finetune.yaml --nproc-per-node 1
```

Watch `val_mtp_tau_sim` across validation runs: it should trend upward as the
MTP head adapts to the finetuning domain. A value near `1.0` means the MTP
head is barely contributing beyond the main head's own next token; a value
approaching `1 + num_mtp_layers` means every MTP depth is being accepted.
