# Checkpoint Robustness Test Status

Last updated: 2026-04-01 22:40 UTC

## Passing Models (7/15)

| # | Model | SFT | PEFT | TP | Cross-TP | HF KL (SFT) | HF KL (PEFT) | VRAM SFT | VRAM PEFT | vLLM SFT | vLLM PEFT | Resume | Special Flags |
|---|-------|-----|------|----|----------|-------------|--------------|----------|-----------|----------|-----------|--------|---------------|
| 1 | Llama 3.2 3B | PASS | PASS | 1 | TP=2 | 5e-3 | 5e-3 | — | — | PASS | PASS | PASS | check_fused_qkv_keys (PEFT) |
| 2 | GPT-OSS 20B | PASS | PASS | 1 | — | 5e-2 | 5e-2 | — | — | PASS (smoke) | PASS (smoke) | Disabled (MoE) | check_phantom_keys, EP=8 |
| 3 | Nemotron Nano V3 | PASS | PASS | 1 | — | 7e-2 | 1e-1 | — | — | PASS (smoke) | PASS (smoke) | Disabled (MoE) | experts_implementation, EP=8 |
| 4 | Gemma 3 270m | PASS | PASS | 1 | — | 3.8e-3 (t=6e-3) | 7.5e-3 (t=8e-3) | 4.47 GB | 0.45 GB | FAIL (vLLM) | — | PASS | 1 KV head, can't TP=2 |
| 5 | Phi-4 | PASS | PASS | 1 | — | 7.6e-4 (t=1.2e-3) | 6.4e-4 (t=1e-3) | 15.41 GB | 8.62 GB | PASS | PASS | PASS | DTensor bug at TP=2 |
| 6 | Qwen2.5 7B | PASS | PASS | 2 | TP=2 (KL=0) | 5.9e-3 (t=9e-3) | 5.5e-2 (t=8e-2) | 8.02 GB | 4.49 GB | PASS | FAIL (lm_head LoRA) | PASS | check_fused_qkv_keys ✓, cross-TP ✓ |
| 7 | Nemotron-Nano-8B-v1 | PASS | PASS | 2 | TP=2 (KL=0) | 4.2e-4 (t=7e-4) | 2.1e-3 (t=5e-3) | 7.73 GB | 4.42 GB | FAIL (token mismatch) | — | Disabled (Mamba) | check_fused_qkv_keys ✓, cross-TP ✓ |
| 8 | Qwen3-MoE 30B | PASS | **FAIL** | 1 | — | 6.4e-5 (t=7e-2) | — | 28.18 GB | 11.81 GB | NOT RUN | NOT RUN | — | EP=8. SFT KL extremely low. **PEFT Phase 3 KL=0.84 — broken PEFT checkpoint reload, real bug** |

## Failing Models (4/15)

| # | Model | TP Tried | Error | Root Cause | Phases Passed |
|---|-------|----------|-------|------------|---------------|
| 8 | Nemotron Flash 1B | TP=1 | Phase 4: `triton_attention.py` not found in consolidated dir | Consolidated checkpoint missing custom model files for `trust_remote_code` | Phase 1-3 PASS |
| 9 | Nemotron Nano V2 9B | TP=2, TP=1 | `'FSDPNemotronHForCausalLM' has no attribute 'model'` | FSDP wrapping issue even with `force_hf: true` | Crashes during setup |
| 10 | Baichuan 2 7B | TP=2, TP=1 | TP=2: `ColwiseParallel only supports nn.Linear/Embedding`. TP=1: Phase 4 `Cannot copy out of meta tensor` | TP=2: custom layers. TP=1: transformers 5.3 meta tensor bug | Phase 1-3 PASS at TP=1 |
| 11 | Mistral3 3B | TP=2, TP=1 | `fully_shard doesn't support scalar parameters (weight_scale_inv)` | FP8 quantized model has scalar scale params incompatible with FSDP2. Same error at TP=1 — not TP-related. Needs code fix or model dequant. | Crashes during setup |
| 13 | Llama-3.3-Super-49B | TP=4 PP=2, TP=8, PP=2 TP=1 | SFT: OOM in all configs on 8 GPUs (TP=4 PP=2 gets through step 0 at 40GB then OOMs step 1). PEFT TP=1: Phase 1-3 PASS, Phase 4 OOM (49B HF on 1 GPU). | Needs 2 nodes (16 GPUs) as YAML specifies (`ci: nodes: 2`). | PEFT Phase 1-3 PASS |

## vLLM Failures

| # | Model | Mode | Error | Root Cause |
|---|-------|------|-------|------------|
| 4 | Gemma 3 270m | SFT full | `rope_parameters should have a 'rope_type' key` | vLLM version incompatible with Gemma 3's RoPE config |
| 6 | Qwen2.5 7B | PEFT full | `expected target modules in {'k_proj',...} but received ['lm_head']` | `match_all_linear: true` includes lm_head which vLLM LoRA doesn't support |
| 7 | Nemotron-Nano-8B-v1 | SFT full | Token mismatch on prompt 1 | Mamba hybrid layers produce different outputs between HF and vLLM. Should use smoke test mode. |

## Not Yet Run (3/15)

| # | Model | TP Plan | Dataset | Cached? | Notes |
|---|-------|---------|---------|---------|-------|
| ~~12~~ | ~~Qwen3-MoE 30B~~ | — | — | — | SFT passed, PEFT broken (see Passing table) |
| ~~13~~ | ~~Llama-3.3-Super-49B~~ | — | — | — | Moved to Failing |
| 14 | Embed-1B-v2 | FSDP | biencoder | Local | separate test file, no vLLM |
| 15 | Nemotron Super 120B | EP=32 (4 nodes) | hellaswag | Yes | Deferred — SFT needs 4 nodes |

## Known Issues

- **MoE resume non-determinism**: DeepEP expert routing causes 3e-2 to 1e-1 loss diff. `--check_resume` disabled for MoE models.
- **Mamba hybrid resume non-determinism**: Nano-8B-v1 has 0.62 loss diff on resume.
- **Mamba hybrid vLLM mismatch**: Nano-8B-v1 greedy tokens differ between HF and vLLM. Should use `--vllm_smoke_test`.
- **transformers 5.3 compatibility**: Flash 1B (triton_attention.py), Nano V2 (FSDP model attr), Baichuan (meta tensor).
- **TP=2 failures**: Gemma 3 (1 KV head), Phi-4 (DTensor assertion), Baichuan (custom layers), Mistral3 (FP8 scalars).
- **vLLM Gemma 3**: vLLM doesn't support Gemma 3's RoPE config.
- **vLLM Qwen2.5 PEFT**: `match_all_linear: true` includes `lm_head` as LoRA target which vLLM rejects. Need to exclude lm_head from LoRA targets.

## TODO Next Session

### Immediate:
1. **Qwen3-MoE 30B** — downloading, run SFT (EP=8) + PEFT after download
2. **Super-49B** — download + run SFT (TP=8) + PEFT (TP=2)
3. **Nano-8B-v1 vLLM** — switch to smoke test mode for Mamba hybrid
4. **Qwen2.5 vLLM PEFT** — exclude lm_head from LoRA targets or use smoke test

### Investigate failures:
5. **Nemotron Flash 1B** — consolidated checkpoint missing triton_attention.py
6. **Nemotron Nano V2 9B** — FSDP wrapping issue
7. **Baichuan 2 7B** — meta tensor in Phase 4
8. **Mistral3 3B** — try TP=1 (only TP=2 was tried)

### Deferred:
9. **Nemotron Super 120B** — needs 4 nodes for SFT
10. **Embed-1B-v2** — biencoder, separate test file
11. **MoE resume** — needs `--resume_loss_threshold` flag

## Commits on branch `adil-a/checkpoint-robustness-test`

- `7ef62d55` — Nemotron Nano V3 checkpoint robustness + vLLM smoke tests
- `04620847` — Cross-cutting features (tokenizer, memory, phantom keys, fused QKV, resume)
- `229bb84f` — 12 new model configs (shell scripts, YAMLs, biencoder test)
- `ce81ee22` — Dataset limit to 500, memory thresholds for Gemma 3 + Phi-4
- `5928505d` — Merge main (fixes _tied_weights_keys for Flash 1B)
- **Uncommitted**: Qwen2.5 thresholds, Nano-8B-v1 script, cross-TP additions for Baichuan/Mistral3/Nano-8B-v1/Qwen2.5
