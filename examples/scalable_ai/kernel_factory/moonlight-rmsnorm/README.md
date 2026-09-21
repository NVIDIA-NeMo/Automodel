# moonlight_v4_rmsnorm_h2048 — a Kernel Factory campaign against the Automodel RMSNorm

This is a **backend migration**, not a DeepSeek-V4 model rewrite. Kernel Factory
searches for the operator implementation on a remote GPU; Automodel keeps the
model contract and decides whether the result actually improves training.

## Correction to the earlier slide

The campaign is **`moonlight_v4_rmsnorm_h2048`**, not `dsv4_rmsnorm_h4096`. The
recipe under `examples/scalable_ai/` runs
[`akoumpa/Moonlight-V4-16B-A3B`](https://huggingface.co/akoumpa/Moonlight-V4-16B-A3B),
whose resolved checkpoint config is:

| field | value |
|---|---|
| `hidden_size` | 2048 |
| `rms_norm_eps` | 1e-6 |
| `num_hidden_layers` | 27 |
| `num_nextn_predict_layers` | 0 |

H=4096 and eps=1e-5 in the public walkthrough come from Llama-3.1, not from this
checkpoint. Searching at the wrong hidden size tunes tile shapes and vector widths
for an operator this recipe never runs.

## Migration boundary

```
current                                POC
-------                                ---
DeepSeek-V4 block / final norm         same model call sites
  initialize_rms_norm_module(             initialize_rms_norm_module(
      "torch_fp32")                           "kf_triton_h2048")
    Float32RMSNorm                          KFTritonRMSNorm  (adapter, hand-written)
                                              kf_rms_norm_kernel.run  (generated)
                                              reference fp32 backward
```

No DeepSeek-V4 model wiring changes. The block and final norms already go through
the common backend factory — `deepseek_v4/model.py:242`, `:245` and `:650` — and the
attention-specific q/kv, compressor and indexer norms are pinned to `torch_fp32`
in `deepseek_v4/layers.py:732`, `:877` and `:1226`, so they are untouched and stay
on the baseline.

## Contract

| | NVIDIA walkthrough | this campaign |
|---|---|---|
| Hidden dimension | 4096 | **2048** |
| Epsilon | 1e-5 | **1e-6** |
| Input | `[rows, 4096]` | `[tokens, 2048]` |
| Arithmetic | fp32 accumulation | fp32 accumulation |
| Output | bf16 | bf16 |
| Scope | forward only | forward now; backward is a separate campaign |

Token counts come from the actual Automodel runs, not from the walkthrough's Llama
workload:

| tokens | source |
|---|---|
| 1024 | the small nsys stage configuration |
| 4096 | `examples/scalable_ai/profile_layer.py` default |
| 8192 | local batch 4 x sequence 2048, `configs/moonlight_v4_16b_torch.yaml` |

## Files here

| file | what it is |
|---|---|
| `definition.json` | the operator: bf16, `hidden` const 2048, `tokens` var, eps scalar; reference is exactly `nemo_automodel::float32_rms_norm` |
| `workload.jsonl` | the three token counts above, eps pinned to 1e-6, tolerance at one bf16 ulp; 8192 carries double weight because it decides the e2e claim |
| `prompt.md` | numerical and integration constraints handed to the agents |
| `baseline.solution.json` | the shipped Automodel path (`torch.compile`d fp32 RMSNorm), so the scoreboard is against what we actually run rather than against eager PyTorch |

## Remote GPU handoff

```bash
kf gpu register \
  --server-url http://remote-gpu.example.com:8001 \
  --label moonlight-h100 \
  --test

kf campaign init moonlight-v4-rmsnorm \
  --definition definition.json \
  --workloads workload.jsonl \
  --gpu-endpoint <endpoint-id> \
  --prompt-file prompt.md \
  --language triton \
  --baseline-solution baseline.solution.json

kf campaign prepare --from moonlight-v4-rmsnorm/campaign.yaml
kf campaign start   --from moonlight-v4-rmsnorm/campaign.yaml --watch
kf campaign results moonlight-v4-rmsnorm --output-dir ./winner
```

Search on the **same GPU architecture** used for the final benchmark. A B200 winner
is not automatically an H100 winner. `baseline.solution.json` names
`"target_hardware": ["H100"]` and `"definition": "rmsnorm_bf16_h2048_eps1e6"`; set
both to whatever `kf campaign prepare` reports for your registered definition.

## Installing the winner

One file changes:

```bash
cp ./winner/<solution>/kernel.py \
   nemo_automodel/components/models/common/kf_rms_norm_kernel.py   # keep the header + run() contract
```

`kf_rms_norm_kernel.py` currently holds a hand-written placeholder Triton kernel so
the backend is runnable and testable before the campaign returns. It is untuned, and
it is nonetheless the fastest of the four backends on GPU kernel time (0.0238 ms,
84% of HBM peak) — which is the strongest argument for questioning whether this
campaign is the right use of the GPU budget.

Everything else is already in place:

1. `kf_triton_rms_norm.py` — the adapter. Owns the `weight` parameter (checkpoint
   keys unchanged), flattens `[B, S, 2048]` to `[B*S, 2048]`, calls the generated
   kernel, restores the shape, registers the DTensor sharding rule, and rejects
   unsupported dims, dtypes, epsilon and layouts explicitly.
2. `BackendConfig.rms_norm` and `initialize_rms_norm_module()` accept
   `kf_triton_h2048` (`components/models/common/utils.py`).
3. `profile_layer.py --backend-rms-norm` accepts it.
4. `configs/moonlight_v4_16b_kf_rmsnorm.yaml` is the e2e A/B twin of
   `configs/moonlight_v4_16b_torch.yaml`; the only difference is the backend string.

## Verification

Isolated layer first:

```bash
nsys profile -c cudaProfilerApi -t cuda,nvtx -o rmsnorm_torch \
  python examples/scalable_ai/profile_layer.py \
  --layer rmsnorm --batch-size 4 --seq-len 2048 --backend-rms-norm torch_fp32

nsys profile -c cudaProfilerApi -t cuda,nvtx -o rmsnorm_candidate \
  python examples/scalable_ai/profile_layer.py \
  --layer rmsnorm --batch-size 4 --seq-len 2048 --backend-rms-norm kf_triton_h2048
```

Then end to end, `moonlight_v4_16b_torch.yaml` against `moonlight_v4_16b_kf_rmsnorm.yaml`.

Promotion gates:

- forward output parity within one bf16 ulp
- `grad_x` and `grad_weight` parity
- bf16 in and out, fp32 internal arithmetic
- `torch.compile`, FSDP2, DTensor and checkpoint compatibility
- repeatable layer-level improvement
- end-to-end step-time improvement outside normal run variation

`tests/unit_tests/models/common/test_kf_triton_rms_norm.py` covers the first four:
18 tests pass on 2x H100 80 GB (torch 2.11.0a0, Triton 3.6.0), including the
DTensor sharding test and `torch.library.opcheck`.

### Measured numbers

The four-way backend comparison (`torch` / `te` / `kf_triton_h2048` / `torch_fp32`),
the memory-bandwidth roofline, and what the measurements revealed about the baseline
live in the POC record: [`../README.md`](../README.md).

Three results matter before launching this campaign:

- **The forward kernel is already at 84% of the memory roofline.** The placeholder
  Triton kernel runs in 0.0238 ms against a 0.0200 ms roofline, faster than both TE
  (0.0256) and `nn.RMSNorm` (0.0271). The entire remaining search space is 19% of
  24 microseconds.
- **The dispatch stack costs 60 microseconds, 2.5x the kernel.** That is what a
  campaign cannot touch and `BackendConfig.cuda_graph` can.
- **The bar is not `torch_fp32`.** It runs four kernels per call because its opaque
  custom op materializes a full fp32 copy of `x`. Re-point `baseline.solution.json`
  at `te` or `torch` before scoring anything.

## The caveat

The walkthrough generates a **forward-only** kernel. That is enough to demonstrate
the integration; it is not a training-speed claim. The adapter keeps the fp32
reference gradient, so today's backward cost is unchanged by construction. A proper
training campaign needs a second definition taking `x`, `weight`, `grad_output` and
returning `grad_x`, `grad_weight`.
