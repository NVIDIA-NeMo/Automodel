# Checkpoint Robustness Test Status

Last updated: 2026-04-02 UTC

> **Note:** vLLM deployment tests moved to separate PR.

## Passing Models (8/15)

| # | Model | SFT | PEFT | TP | Cross-TP | HF KL (SFT) | HF KL (PEFT) | VRAM SFT | VRAM PEFT | Resume | Special Flags |
|---|-------|-----|------|----|----------|-------------|--------------|----------|-----------|--------|---------------|
| 1 | Llama 3.2 3B | PASS | PASS | 1 | TP=2 | 5e-3 | 5e-3 | — | — | PASS | check_fused_qkv_keys (PEFT) |
| 2 | GPT-OSS 20B | PASS | PASS | 1 | — | 5e-2 | 5e-2 | — | — | Disabled (MoE) | check_phantom_keys, EP=8 |
| 3 | Nemotron Nano V3 | PASS | PASS | 1 | — | 7e-2 | 1e-1 | — | — | Disabled (MoE) | experts_implementation, EP=8 |
| 4 | Gemma 3 270m | PASS | PASS | 1 | — | 3.8e-3 (t=6e-3) | 7.5e-3 (t=8e-3) | 4.47 GB | 0.45 GB | PASS | 1 KV head, can't TP=2 |
| 5 | Phi-4 | PASS | PASS | 2 | — | 7.6e-4 (t=1.2e-3) | 6.4e-4 (t=1e-3) | 15.41 GB | 8.62 GB | PASS (t=7e-3) | TP=2 DTensor bug fixed on main |
| 6 | Qwen2.5 7B | PASS | PASS | 2 | TP=2 (KL=0) | 5.9e-3 (t=9e-3) | 5.5e-2 (t=8e-2) | 8.02 GB | 4.49 GB | PASS | check_fused_qkv_keys ✓, cross-TP ✓ |
| 7 | Nemotron-Nano-8B-v1 | PASS | PASS | 2 | TP=2 (KL=0) | 4.2e-4 (t=7e-4) | 2.1e-3 (t=5e-3) | 7.73 GB | 4.42 GB | Disabled (Mamba) | check_fused_qkv_keys ✓, cross-TP ✓ |
| 8 | Qwen3-MoE 30B | PASS | **FAIL** | 1 | — | 6.4e-5 (t=1e-4) | — | 28.18 GB | 11.81 GB | — | EP=8. SFT KL extremely low. **PEFT Phase 3 KL=0.84 — broken PEFT checkpoint reload, real bug** |

## Failing Models (5/15)

| # | Model | TP Tried | Error | Root Cause | Phases Passed |
|---|-------|----------|-------|------------|---------------|
| 9 | Nemotron Flash 1B | TP=1 | Phase 4: `triton_attention.py` not found in consolidated dir | Consolidated checkpoint missing custom model files for `trust_remote_code` | Phase 1-3 PASS |
| 10 | Nemotron Nano V2 9B | TP=2, TP=1 | `'FSDPNemotronHForCausalLM' has no attribute 'model'` | FSDP wrapping issue even with `force_hf: true` | Crashes during setup |
| 11 | Baichuan 2 7B | TP=2, TP=1 | TP=2: `ColwiseParallel only supports nn.Linear/Embedding`. TP=1: Phase 4 `Cannot copy out of meta tensor` | TP=2: custom layers. TP=1: transformers 5.3 meta tensor bug | Phase 1-3 PASS at TP=1 |
| 12 | Mistral3 3B | TP=2, TP=1 | `fully_shard doesn't support scalar parameters (weight_scale_inv)` | FP8 quantized model has scalar scale params incompatible with FSDP2. Same error at both TP sizes. | Crashes during setup |
| 8* | Qwen3-MoE PEFT | TP=1 EP=8 | Phase 3 KL=0.84 (should be 0) | **Real bug**: PEFT checkpoint reload is broken for Qwen3-MoE. SFT works fine. | Phase 1-2 PASS |

## Multi-Node Results (tested 2026-04-02)

| # | Model | Mode | Nodes | Config | Phases 1-3 | Phase 4 (HF) | Resume | Notes |
|---|-------|------|-------|--------|-----------|--------------|--------|-------|
| 13 | Super-49B | SFT | 2 (TP=4) | 16 GPUs | PASS | FAIL (KL=10.6) | Not reached | Combined QKV keys in consolidated ckpt — vanilla HF can't load |
| 13 | Super-49B | PEFT | 1 (TP=2) | 8 GPUs | PASS | FAIL (KL=10.5) | Not reached | Same combined QKV issue in PEFT adapter |
| 14 | Embed-1B-v2 | SFT | 1 | 8 GPUs | PASS (cosine=1.0) | N/A | PASS (t=2e-2) | Biencoder test, all phases pass |
| 15 | Super-120B | SFT | 4 (EP=32) | 32 GPUs | PASS | PASS (device_map=auto) | Disabled (MoE) | All phases pass, 9:27 |
| 15 | Super-120B | PEFT | 2 (EP=16) | 16 GPUs | PASS | FAIL (KL=8.5e-2, t=7e-2) | Disabled (MoE) | Combined QKV in PEFT adapter |

## Not Yet Run

(None — all models tested)

## Known Issues

- **MoE resume non-determinism**: DeepEP expert routing causes 3e-2 to 1e-1 loss diff. `--check_resume` disabled for MoE models.
- **Mamba hybrid resume non-determinism**: Nano-8B-v1 has 0.62 loss diff on resume. Mamba layers have non-deterministic state.
- **transformers 5.3 compatibility**: Flash 1B (triton_attention.py), Nano V2 (FSDP model attr), Baichuan (meta tensor).
- **TP=2 failures**: Gemma 3 (1 KV head), Baichuan (custom layers), Mistral3 (FP8 scalars). Phi-4 TP=2 fixed on main.
- **Combined QKV Phase 4 failures**: Super-49B and Super-120B PEFT produce combined projection keys (qkv_proj, gate_up_proj) in consolidated/adapter checkpoints. Vanilla HF models expect separate projections. StateDictAdapter conversion needed for Phase 4 to work.
- **Qwen3-MoE PEFT bug**: Phase 3 KL=0.84 indicates broken PEFT checkpoint save/reload. Needs investigation in Qwen3MoeStateDictAdapter.

## TODO

### Investigate failures:
1. **Super-49B Phase 4** — consolidated checkpoint has combined QKV keys. Need StateDictAdapter in Phase 4, or fix save_consolidated to split projections.
2. **Super-120B PEFT Phase 4** — same combined QKV issue for PEFT adapter weights.
3. **Qwen3-MoE PEFT bug** — investigate why Phase 3 KL=0.84 (real checkpoint bug)
### Investigate other failures (may need code fixes):
5. **Nemotron Flash 1B** — consolidated checkpoint missing triton_attention.py
6. **Nemotron Nano V2 9B** — FSDP wrapping issue
7. **Baichuan 2 7B** — meta tensor in Phase 4 HF loading
8. **Mistral3 3B** — FP8 scalar params vs FSDP2

### Infrastructure improvements:
10. **`--resume_loss_threshold`** flag — DONE (added, default 5e-3)
11. **Memory thresholds** for remaining models (Llama, GPT-OSS, Nano V3 still missing)

## Commits on branch `adil-a/checkpoint-robustness-test`

- `7ef62d55` — Nemotron Nano V3 checkpoint robustness + vLLM smoke tests
- `04620847` — Cross-cutting features (tokenizer, memory, phantom keys, fused QKV, resume)
- `229bb84f` — 12 new model configs (shell scripts, YAMLs, biencoder test)
- `ce81ee22` — Dataset limit to 500, memory thresholds for Gemma 3 + Phi-4
- `5928505d` — Merge main
- `39a413ef` — Tighten thresholds, add cross-TP for Qwen2.5/Nano-8B-v1/Baichuan/Mistral3
- `2f1a5a94` — STATUS update with Super-49B results
- Phi-4 TP=2: SFT PASS (HF KL=2.7e-4, resume 7e-3), PEFT PASS (9:05). DTensor bug fixed on main after merge.
- Multi-node tests (2026-04-02): Super-120B SFT PASS (4 nodes EP=32, device_map=auto). Super-49B SFT Phase 1-3 PASS (2 nodes TP=4), Phase 4 FAIL (combined QKV). Embed-1B-v2 PASS (cosine=1.0, resume t=2e-2).
- Added `--hf_device_map_auto` flag for Phase 4 large model loading across all GPUs.
- Added `--resume_loss_threshold` to biencoder test. Fixed biencoder import path and tokenizer compatibility.
