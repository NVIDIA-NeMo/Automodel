# Mixed-Precision Training

NeMo AutoModel uses FSDP2's [`MixedPrecisionPolicy`](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) to control compute precision during forward and backward, and the model's storage dtype (`model.torch_dtype`) to control the precision of the *resident* sharded parameter. Together, these decide what numeric precision the optimizer state ends up in, which is the part that determines whether long full-parameter training runs converge cleanly.

This page describes the precision patterns we recommend, and the trap that sits between them. For any long full-parameter training run (pre-training or extended fine-tuning), the key rule is: do not combine `torch.optim.AdamW` with bf16 resident parameters unless you have explicitly accepted bf16 Adam state.

## Storage Dtype vs. Compute Dtype

Precision settings have distinct effects:

| Setting | Controls | Effect on optimizer state |
| --- | --- | --- |
| `model.torch_dtype` | Storage dtype of the sharded parameter that PyTorch holds. | For `torch.optim`, the optimizer reads `param.data`, so the EMA buffers (`exp_avg`, `exp_avg_sq` for AdamW) end up in this dtype. (TE FusedAdam keeps fp32 state regardless.) |
| `mp_policy.param_dtype` | Compute dtype FSDP2 casts to during forward/backward. | None directly; this only affects matmul / activation precision. |
| `mp_policy.reduce_dtype` | Dtype used for gradient reduce-scatter / all-reduce across DP ranks. | None directly; only affects how gradients are summed. |
| `mp_policy.output_dtype` | Dtype FSDP2 casts module outputs to. | None directly; this affects activation tensors, including tensors that cross pipeline-parallel boundaries. |

When `model.torch_dtype: bfloat16` is used with `torch.optim.AdamW`, the AdamW EMA buffers (`exp_avg` and `exp_avg_sq`) are also stored in bf16. This is fragile for long full-parameter training runs: bf16 has only a 7-bit mantissa, so small EMA updates can be rounded away even though the values themselves are still in range. Symptoms range from silent degradation (slower convergence, i.e., higher final loss at the same step count, with no visible instability) to overt failure: unstable `grad_norm`, loss bumps, loss spikes, or divergence.

## Recommended fp32 Master Weight and Optimizer-State Patterns

Use one of these patterns for long full-parameter training (pre-training or extended fine-tuning):

| Pattern | Model storage dtype | Optimizer state | When to use |
| --- | --- | --- | --- |
| TE FusedAdam with bf16 model storage | bf16 | fp32 master weights and fp32 Adam EMA buffers | When TE has a validated memory/runtime path for the model and you want training checkpoints to stay bf16. |
| torch AdamW with `model.torch_dtype: float32` | fp32 | fp32 master weight (the resident param) and fp32 Adam EMA buffers | Robust starting point for new or precision-sensitive models — params, gradients, and optimizer state are all fp32. Trade-off: writes fp32 training checkpoints. |

Both patterns keep forward/backward compute in bf16 through FSDP2 mixed precision. They differ in where the fp32 master weight lives, how much peak memory they use for a specific model, and what dtype is written to the training checkpoint.

### Pattern A: TE FusedAdam and bf16 Model Storage

Use Transformer Engine FusedAdam when it has been validated for the model. The resident model parameters remain bf16, so model checkpoints can stay bf16, while TE keeps the optimizer's master weights and Adam EMA buffers in fp32.

```yaml
model:
  torch_dtype: bfloat16             # resident sharded parameter + model checkpoint in bf16

optimizer:
  _target_: transformer_engine.pytorch.optimizers.fused_adam.FusedAdam
  lr: 3.0e-4
  adam_w_mode: true
  bias_correction: true
  master_weights: true
  store_param_remainders: true
  exp_avg_dtype: torch.float32
  exp_avg_sq_dtype: torch.float32

distributed:
  strategy: fsdp2
  # Defaults already provide bf16 forward/backward + fp32 gradient reduction; this block is shown explicitly for clarity.
  mp_policy:
    _target_: torch.distributed.fsdp.MixedPrecisionPolicy
    param_dtype: bfloat16
    reduce_dtype: float32
    output_dtype: bfloat16
    cast_forward_inputs: true
```

TE FusedAdam is the cleanest way to request fp32 optimizer state without making the resident model parameter fp32.

It is a common misconception that TE costs extra memory for a second fp32 master. With `store_param_remainders: true` (as above), TE does **not** keep a full extra fp32 master: it stores the bf16 parameter plus a 16-bit remainder that together reconstruct the fp32 master, costing the same ~4 bytes/param as torch AdamW's fp32 resident parameter. The fp32 Adam EMA buffers (`exp_avg`, `exp_avg_sq`) are the same in both patterns, so the two optimizers' steady-state footprints are essentially equal. The practical difference is then simplicity (torch AdamW: no TE dependency, fewer moving parts) vs. keeping model storage and checkpoints in bf16 (TE).

Where the two *can* differ is the gradient buffer: TE keeps the resident parameter in bf16, so its gradients are bf16, whereas torch AdamW with fp32 storage keeps parameters and gradients in fp32 (~2 bytes/param more on the gradient buffer). In practice, what we measure is *peak* memory, which is usually dominated by activations rather than the optimizer step, so depending on the model (fraction of intrinsically-fp32 params, fragmentation, where the peak falls), TE can come out lower, equal, or higher. Validate memory per model before making it the default.

### Pattern B: torch AdamW and fp32 Model Storage

```yaml
model:
  torch_dtype: float32              # sharded parameter + AdamW state in fp32

optimizer:
  _target_: torch.optim.AdamW
  lr: 3.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.1

distributed:
  strategy: fsdp2
  # Defaults already provide bf16 forward/backward + fp32 gradient reduction; this block is shown explicitly for clarity.
  mp_policy:
    _target_: torch.distributed.fsdp.MixedPrecisionPolicy
    param_dtype: bfloat16           # forward/backward compute in bf16 (fast)
    reduce_dtype: float32           # safe gradient reduction
    output_dtype: bfloat16
    cast_forward_inputs: true
```

This is the PyTorch AdamW version of the master-weights pattern. Forward/backward run in bf16, the all-reduce / reduce-scatter runs in fp32, and the optimizer applies updates against fp32 resident parameters. With `torch.optim.AdamW`, the resident fp32 sharded parameter is the master weight, so there is no separate fp32 master-weight copy.

The trade-off is the dtype of the *training* checkpoint: AutoModel's training (DCP) checkpoint stores the resident model parameters, so this pattern writes them in fp32, which you keep for exact resume. This concerns only the training checkpoint; a consolidated HF checkpoint for inference or release is exported separately and follows the model's intended dtype (for fine-tuning, this matches the original HF checkpoint, typically bf16). See [checkpointing](checkpointing.md) for details.

### Precision and Robustness

Both patterns keep the master weights and Adam EMA in fp32, so for most models they converge equivalently. The remaining difference is the gradient: under TE (Pattern A) the resident parameter is bf16, so the gradient feeding the Adam update carries bf16 rounding, while torch AdamW with fp32 storage (Pattern B) keeps the parameter, gradient, and optimizer state all in fp32.

This is usually negligible. When bringing up a new model, fp32 storage (Pattern B) is a robust starting point because every part of the update path is fp32; move to TE (Pattern A) after it is validated for that model and you want bf16 checkpoints.

## Risky Pattern: torch AdamW with bf16 Model Storage

This pattern is easy to enter accidentally. In several AutoModel paths, leaving `model.torch_dtype` unset, or setting it to `auto`, resolves the resident model parameter dtype to bf16. If that is paired with `torch.optim.AdamW`, the AdamW EMA buffers are also bf16 because the optimizer initializes state from the parameter dtype.

```yaml
model:
  # torch_dtype omitted, or torch_dtype: auto / bfloat16

optimizer:
  _target_: torch.optim.AdamW

distributed:
  strategy: fsdp2
```

This keeps the resident model parameters in bf16 instead of fp32, so it can reduce memory usage compared with the torch AdamW and `model.torch_dtype: float32` pattern. It is common in existing fine-tuning example configs and is probably acceptable for short fine-tuning runs or LoRA/PEFT. It is **not** recommended for long full-parameter training (pre-training or extended fine-tuning): bf16 EMA quantization can quietly slow convergence (higher final loss at the same step count) and, in worse cases, produce unstable `grad_norm`, loss bumps, loss spikes, or divergence.

## Example Configs

- [`examples/llm_pretrain/llama3_70b_pretrain.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_pretrain/llama3_70b_pretrain.yaml): TE FusedAdam example. This keeps model storage and training checkpoints in bf16 while using fp32 master weights and optimizer state.
- [`examples/llm_pretrain/megatron_pretrain_moonlight_16b_te_slurm.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_pretrain/megatron_pretrain_moonlight_16b_te_slurm.yaml): torch AdamW example. This uses `model.torch_dtype: float32` so AdamW state stays fp32 while compute remains bf16 through FSDP2 mixed precision.
