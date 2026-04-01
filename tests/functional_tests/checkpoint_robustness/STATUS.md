# Checkpoint Robustness Test Status

Last updated: 2026-04-01 09:00 UTC

## Passing Models (8/15)

| # | Model | SFT | PEFT | TP | Cross-TP | HF KL (SFT) | HF KL (PEFT) | VRAM SFT | VRAM PEFT | vLLM SFT | vLLM PEFT | Resume | Special Flags |
|---|-------|-----|------|----|----------|-------------|--------------|----------|-----------|----------|-----------|--------|---------------|
| 1 | Llama 3.2 3B | PASS | PASS | 1 | TP=2 | 5e-3 | 5e-3 | — | — | PASS | PASS | PASS | check_fused_qkv_keys (PEFT) |
| 2 | GPT-OSS 20B | PASS | PASS | 1 | — | 5e-2 | 5e-2 | — | — | PASS (smoke) | PASS (smoke) | Disabled (MoE) | check_phantom_keys, EP=8 |
| 3 | Nemotron Nano V3 | PASS | PASS | 1 | — | 7e-2 | 1e-1 | — | — | PASS (smoke) | PASS (smoke) | Disabled (MoE) | experts_implementation, EP=8 |
| 4 | Gemma 3 270m | PASS | PASS | 1 | — | 3.8e-3 (t=6e-3) | 7.5e-3 (t=8e-3) | 4.47 GB | 0.45 GB | FAIL (vLLM) | — | PASS | 1 KV head, can't TP=2 |
| 5 | Phi-4 | PASS | PASS | 1 | — | 7.6e-4 (t=1.2e-3) | 6.4e-4 (t=1e-3) | 15.41 GB | 8.62 GB | PASS | PASS | PASS | DTensor bug at TP=2 |
| 6 | Qwen2.5 7B | PASS | PASS | 2 | TP=2 (KL=0) | 5.9e-3 (t=9e-3) | 5.5e-2 (t=8e-2) | 8.02 GB | 4.49 GB | PASS | FAIL (lm_head LoRA) | PASS | check_fused_qkv_keys ✓, cross-TP ✓ |
| 7 | Nemotron-Nano-8B-v1 | PASS | PASS | 2 | TP=2 (KL=0) | 4.2e-4 (t=7e-4) | 2.1e-3 (t=5e-3) | 7.73 GB | 4.42 GB | FAIL (token mismatch) | — | Disabled (Mamba) | check_fused_qkv_keys ✓, cross-TP ✓ |
| 8 | Qwen3-MoE 30B | PASS | **FAIL** | 1 | — | 6.4e-5 (t=1e-4) | — | 28.18 GB | 11.81 GB | NOT RUN | NOT RUN | — | EP=8. SFT KL extremely low. **PEFT Phase 3 KL=0.84 — broken PEFT checkpoint reload, real bug** |

## Failing Models (5/15)

| # | Model | TP Tried | Error | Root Cause | Phases Passed |
|---|-------|----------|-------|------------|---------------|
| 9 | Nemotron Flash 1B | TP=1 | Phase 4: `triton_attention.py` not found in consolidated dir | Consolidated checkpoint missing custom model files for `trust_remote_code` | Phase 1-3 PASS |
| 10 | Nemotron Nano V2 9B | TP=2, TP=1 | `'FSDPNemotronHForCausalLM' has no attribute 'model'` | FSDP wrapping issue even with `force_hf: true` | Crashes during setup |
| 11 | Baichuan 2 7B | TP=2, TP=1 | TP=2: `ColwiseParallel only supports nn.Linear/Embedding`. TP=1: Phase 4 `Cannot copy out of meta tensor` | TP=2: custom layers. TP=1: transformers 5.3 meta tensor bug | Phase 1-3 PASS at TP=1 |
| 12 | Mistral3 3B | TP=2, TP=1 | `fully_shard doesn't support scalar parameters (weight_scale_inv)` | FP8 quantized model has scalar scale params incompatible with FSDP2. Same error at both TP sizes. | Crashes during setup |
| 8* | Qwen3-MoE PEFT | TP=1 EP=8 | Phase 3 KL=0.84 (should be 0) | **Real bug**: PEFT checkpoint reload is broken for Qwen3-MoE. SFT works fine. | Phase 1-2 PASS |

## Deferred — Multi-Node (2/15)

| # | Model | Config | Why |
|---|-------|--------|-----|
| 13 | Llama-3.3-Super-49B | TP=4 PP=2 (2 nodes, 16 GPUs) | OOM on 8 GPUs in all configs. PEFT Phase 1-3 passed at TP=1 but Phase 4 OOMs (49B HF on 1 GPU). |
| 15 | Nemotron Super 120B | EP=32 (4 nodes, 32 GPUs) | SFT needs 4 nodes. PEFT (EP=8) could fit 1 node but not tested yet. |

## vLLM Failures

| # | Model | Mode | Error | Root Cause |
|---|-------|------|-------|------------|
| 4 | Gemma 3 270m | SFT full | `rope_parameters should have a 'rope_type' key` | vLLM version incompatible with Gemma 3's RoPE config |
| 6 | Qwen2.5 7B | PEFT full | `expected target modules in {'k_proj',...} but received ['lm_head']` | `match_all_linear: true` includes lm_head which vLLM LoRA doesn't support |
| 7 | Nemotron-Nano-8B-v1 | SFT full | Token mismatch on prompt 1 | Mamba hybrid layers produce different outputs between HF and vLLM. Should use smoke test mode. |

## Not Yet Run (1/15)

| # | Model | Notes |
|---|-------|-------|
| 14 | Embed-1B-v2 | Biencoder, separate test file (test_checkpoint_robustness_biencoder.py), no vLLM |

## Known Issues

- **MoE resume non-determinism**: DeepEP expert routing causes 3e-2 to 1e-1 loss diff. `--check_resume` disabled for MoE models.
- **Mamba hybrid resume non-determinism**: Nano-8B-v1 has 0.62 loss diff on resume. Mamba layers have non-deterministic state.
- **Mamba hybrid vLLM mismatch**: Nano-8B-v1 greedy tokens differ between HF and vLLM. Should use `--vllm_smoke_test`.
- **transformers 5.3 compatibility**: Flash 1B (triton_attention.py), Nano V2 (FSDP model attr), Baichuan (meta tensor).
- **TP=2 failures**: Gemma 3 (1 KV head), Phi-4 (DTensor assertion), Baichuan (custom layers), Mistral3 (FP8 scalars).
- **vLLM Gemma 3**: vLLM in `/adasif/vllm-venv/` doesn't support Gemma 3's RoPE config.
- **vLLM Qwen2.5 PEFT**: `match_all_linear: true` includes `lm_head` as LoRA target which vLLM rejects.
- **Qwen3-MoE PEFT bug**: Phase 3 KL=0.84 indicates broken PEFT checkpoint save/reload. Needs investigation in Qwen3MoeStateDictAdapter.

## TODO

### Runnable on this machine:
1. **Embed-1B-v2** — biencoder test, last remaining model
2. **Qwen3-MoE PEFT bug** — investigate why Phase 3 KL=0.84 (real checkpoint bug)
3. **vLLM fixes** — Nano-8B-v1 switch to smoke test; Qwen2.5 PEFT exclude lm_head

### Investigate failures (may need code fixes):
4. **Nemotron Flash 1B** — consolidated checkpoint missing triton_attention.py
5. **Nemotron Nano V2 9B** — FSDP wrapping issue
6. **Baichuan 2 7B** — meta tensor in Phase 4 HF loading
7. **Mistral3 3B** — FP8 scalar params vs FSDP2

### Multi-node (need 2+ nodes):
8. **Super-49B SFT** — TP=4 PP=2 on 16 GPUs
9. **Super 120B** — EP=32 on 32 GPUs (PEFT EP=8 could run on 1 node)

### Infrastructure improvements:
10. **`--resume_loss_threshold`** flag for MoE/Mamba models
11. **Memory thresholds** for remaining models (Llama, GPT-OSS, Nano V3 still missing)

## Commits on branch `adil-a/checkpoint-robustness-test`

- `7ef62d55` — Nemotron Nano V3 checkpoint robustness + vLLM smoke tests
- `04620847` — Cross-cutting features (tokenizer, memory, phantom keys, fused QKV, resume)
- `229bb84f` — 12 new model configs (shell scripts, YAMLs, biencoder test)
- `ce81ee22` — Dataset limit to 500, memory thresholds for Gemma 3 + Phi-4
- `5928505d` — Merge main
- `39a413ef` — Tighten thresholds, add cross-TP for Qwen2.5/Nano-8B-v1/Baichuan/Mistral3
- `2f1a5a94` — STATUS update with Super-49B results
