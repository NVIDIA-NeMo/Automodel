# Mixed-Precision Training

NeMo AutoModel uses FSDP2's
[`MixedPrecisionPolicy`](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.MixedPrecisionPolicy)
to control compute precision during forward and backward, and the model's
storage dtype (`model.torch_dtype`) to control the precision of the *resident*
sharded parameter. Together these decide what numeric precision the optimizer
state ends up in -- which is the part that determines whether long pre-training
runs converge cleanly.

This page describes the two precision patterns we recommend, and the trap that
sits between them.

## Storage dtype vs compute dtype

Two settings, two distinct effects:

| Setting | Controls | Effect on optimizer state |
| --- | --- | --- |
| `model.torch_dtype` | Storage dtype of the sharded parameter that PyTorch holds. | The optimizer reads `param.data`, so the EMA buffers (`exp_avg`, `exp_avg_sq` for AdamW) end up in this dtype. |
| `mp_policy.param_dtype` | Compute dtype FSDP2 casts to during forward/backward. | None directly; this only affects matmul / activation precision. |
| `mp_policy.reduce_dtype` | Dtype used for gradient reduce-scatter / all-reduce across DP ranks. | None directly; only affects how gradients are summed. |

When `model.torch_dtype: bfloat16` is used with `torch.optim.AdamW`, the
AdamW EMA buffers (`exp_avg` and `exp_avg_sq`) are also stored in bf16. This is
fragile for long pre-training runs: bf16 has only a 7-bit mantissa, so small
EMA updates can be rounded away even though the values themselves are still in
range. As training progresses and per-parameter updates become more specialized
and smaller, the AdamW state can become quantized/stale, producing periodic
`grad_norm` oscillations and matching loss bumps. See
[issue #1679](https://github.com/NVIDIA-NeMo/Automodel/issues/1679) for the
full reproduction.

## Pattern A (recommended for pre-training): fp32 master weights + bf16 compute

```yaml
model:
  torch_dtype: float32              # sharded parameter + Adam state in fp32

distributed:
  _target_: nemo_automodel.components.distributed.config.FSDP2Config
  # Defaults already provide bf16 forward/backward + fp32 gradient reduction;
  # this block is shown explicitly for clarity.
  mp_policy:
    _target_: torch.distributed.fsdp.MixedPrecisionPolicy
    param_dtype: bfloat16           # forward/backward compute in bf16 (fast)
    reduce_dtype: float32           # safe gradient reduction
    output_dtype: bfloat16
    cast_forward_inputs: true
```

This is the same arrangement Megatron-LM and Lingua use. Forward/backward run
in bf16 on the SM, the all-reduce / reduce-scatter runs in fp32, and the
optimizer applies updates against an fp32 weight. The defaults `FSDP2Config`
applies when no `mp_policy:` is set are exactly the values above -- you only
need to set `model.torch_dtype: float32` to opt into the master-weights side
of the pattern.

If you set the model dtype to bf16 storage, but need the precision of fp32
optimizer state, you can also use the Transformer Engine fused optimizer instead -- it
keeps `master_weights`, `exp_avg`, and `exp_avg_sq` in fp32 explicitly:

```yaml
optimizer:
  _target_: transformer_engine.pytorch.optimizers.fused_adam.FusedAdam
  lr: 3.0e-4
  master_weights: true
  exp_avg_dtype: torch.float32
  exp_avg_sq_dtype: torch.float32
```

## Pattern B (memory-constrained, accepts the precision risk): all-bf16

```yaml
model:
  torch_dtype: bfloat16             # sharded parameter + Adam state in bf16

distributed:
  _target_: nemo_automodel.components.distributed.config.FSDP2Config
  mp_policy:
    _target_: torch.distributed.fsdp.MixedPrecisionPolicy
    param_dtype: bfloat16
    reduce_dtype: bfloat16
    output_dtype: bfloat16
    cast_forward_inputs: true
```

This saves the fp32 master-weight slice. It is probably acceptable for short
fine-tuning runs or LoRA / PEFT. It is **not** recommended for pre-training
Llama-style architectures: the bf16 EMA quantization compounds and produces
the periodic `grad_norm` / loss artefacts described above.

## See also

- [`examples/llm_pretrain/llama3_70b_pretrain.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_pretrain/llama3_70b_pretrain.yaml)
  -- canonical pre-training recipe using Pattern A.
- [Issue #1679](https://github.com/NVIDIA-NeMo/Automodel/issues/1679) -- the
  Llama-3.2-1B pre-training instability that motivated this guide.
