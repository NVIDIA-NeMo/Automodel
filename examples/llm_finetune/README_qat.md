# QAT Examples

The [QAT guide](../../docs/guides/quantization-aware-training.mdx) is the canonical
reference for configuration, run commands, local checkpoint paths, distributed
requirements, reference export, and evaluation limitations.

## Choose a Mode and Model

QAT supports the following modes:

- **Legacy TorchAO INT4:** Full-parameter BF16 SFT without PEFT.
- **Targeted LoRA FP8/MXFP4:** Trainable adapters on frozen floating-point bases;
  PEFT and QAT selectors are independent. Do not mix the two modes.

| Example | Scope | Launch |
| --- | --- | --- |
| [Llama 3.2 INT4](llama3_2/llama3_2_1b_squad_qat.yaml) | Legacy full-parameter SQuAD, no PEFT | 8 GPUs; use the guide's delay override |
| [Qwen3-0.6B](qwen/qwen3_0p6b_lora_qat.yaml) | Two-step pretrained SQuAD, q/v LoRA FP8 | Exactly one GPU |
| [DeepSeek-V4-Flash](deepseek_v4/deepseek_v4_flash_hellaswag_experts_qat.yaml) | Experimental full pretrained backbone, expert-only LoRA MXFP4 | 4 nodes × 8 GPUs, EP32 |
| [GLM-5.3-Flash](glm/glm_5.3_flash_hellaswag_experts_qat.yaml) | Experimental full pretrained backbone, expert-only LoRA FP8/FP32 scales | 4 nodes × 8 GPUs, EP32 |

V4 and GLM are **experimental reference configurations**. Earlier full-model smoke
tests used synthetic overrides; they do not establish that these HellaSwag YAMLs
ran unchanged, convergence, task-quality gains, or native serving parity.

## Reference Train-Inference Consistency Evidence

For targeted LoRA QAT, the frozen base $W$ and trainable adapters $A,B$ form

$$
W_{\mathrm{eff}} = W + \frac{\alpha}{r}BA, \qquad
W_{\mathrm{qdq}} = D(Q(W_{\mathrm{eff}})).
$$

The targeted training forward uses $W_{\mathrm{qdq}}$ with straight-through
gradients to the adapters. Reference inference decodes the saved payload and
scales into the same floating-point weight. This is **weight-only** FP8/MXFP4:
there is no activation quantization or native low-bit GEMM in this comparison.
Matching dtype, quantization settings, and operation order are required. The
following checks provide bounded evidence, not universal serving equivalence.

### Reproducible Repository Checks (CPU)

The repository includes the following CPU checks:

- [Independent scalar oracle](../../tests/unit_tests/quantization/test_weight_qat.py):
  checks FP8/MXFP4 codes, scales, rounding boundaries, packing, and decoded values
  independently of the vectorized implementation.
- [Trained-adapter local checkpoint test](../../tests/unit_tests/_transformers/test_lora_qat.py):
  `test_optimizer_export_reload_exact_eval_and_byte_roundtrip` performs an
  optimizer step, verifies adapter updates and an unchanged base, exports to
  disk, and reloads into a **fresh non-LoRA model**. It asserts exact eval outputs
  (`rtol=atol=0`) and byte-stable re-export across FP32/BF16/FP16, FP8 with
  E8M0/FP32 scales, and MXFP4 with E8M0 scales.

From the AutoModel repository root in an installed test environment (including
PyTorch, safetensors, pytest, and pytest-timeout), without model or data downloads:

```bash
CUDA_VISIBLE_DEVICES="" uv run --no-sync pytest tests/unit_tests/quantization/test_weight_qat.py -q --timeout=60
CUDA_VISIBLE_DEVICES="" uv run --no-sync pytest tests/unit_tests/_transformers/test_lora_qat.py::test_optimizer_export_reload_exact_eval_and_byte_roundtrip -q --timeout=60
```

### Recorded Full-Model Experimental Measurements

These are **recorded experimental measurements, not a CI guarantee**. Both arms
used the same pretrained base, seed 42, and initial adapters (rank 4, alpha 8),
matched source code, data, optimizer, and topology: AdamW, learning rate 1e-5,
two steps, global/local batch 32/1, sequence length 64, BF16, and
4 nodes × 8 H100 GPUs with FSDP2 and EP32. LoRA and QAT targeted routed experts
only; V4 used MXFP4 group32/E8M0, GLM FP8 block128/FP32 scales.
Inputs were **synthetic arithmetic text**, not unchanged HellaSwag recipes.
The original backbone depth, width, vocabulary, and expert counts were retained;
MTP was off, and GLM's vision branch stayed frozen for text-only evaluation.

Each arm's post-training eval logits were compared with **its own** saved and
decoded reference, over 32 ranks × 8 tokens = 256 token positions. RMSE is over
logits; KL is mean per-token divergence from eval to decoded-reference
distributions; top-1 is token agreement, not task accuracy.

| Original Full Model | Training | Logit RMSE | KL/Token | Top-1 Agreement |
| --- | --- | ---: | ---: | ---: |
| DeepSeek-V4-Flash, 43 layers | Ordinary LoRA | 0.2418006364 | 0.0211047611 | 233/256 (91.015625%) |
| DeepSeek-V4-Flash, 43 layers | QAT-LoRA | 0 | 0 | 256/256 (100%) |
| GLM-5.3-Flash, 45 text layers | Ordinary LoRA | 0.8229582466 | 0.4842822026 | 173/256 (67.578125%) |
| GLM-5.3-Flash, 45 text layers | QAT-LoRA | 0 | 0 | 256/256 (100%) |

Unlike the local CPU test, full-model reference inference reused the **same
distributed model and topology**, not a fresh non-LoRA model. Each rank saved
and decoded its local expert payload and scales, copied decoded weights into its
base parameters, zeroed LoRA, and used identity quantization to preserve the
projection and routing path. Merged-QDQ and disk-decoded outputs were exact in
both arms. This does not extend the local-only `QAT.export` API to FSDP/EP shards.

The matched GLM baseline included the corrected routing-weight placement.
Ordinary-LoRA discrepancies include **merge rounding and operation-order changes
as well as quantization**; an independent merge-only control is required to
isolate quantization error. QAT's zero discrepancy is expected by construction
under the matching QDQ contract, not evidence of better learned quality.
Native serving, convergence, and task-quality gains have **not** been validated.

Full-model measurements require accompanying run artifacts for independent
reproduction; commands above validate the repository regression contract only,
not this full-model experiment. The
[historical evidence bundle](https://github.com/hasuoshenyun/Automodel/releases/tag/qat-pr3904-evidence-20261110)
provides source/config fingerprints, initial-adapter checks, per-rank metrics,
and hash-matched drivers as accompanying artifacts. It supports offline
recomputation of reported metrics, but does not include the exact historical
AutoModel source snapshot or large checkpoint files.

## Next Steps

Continue with the following topics:

- [Run Qwen and configure distributed expert training](../../docs/guides/quantization-aware-training.mdx#examples-and-supported-training)
- [Slurm allocation and multi-node launch](../../docs/launcher/slurm.mdx)
- [Local-only Qwen reference export](../../docs/guides/quantization-aware-training.mdx#reference-export-is-local-only)
- [Comparison methodology](../../docs/guides/quantization-aware-training.mdx#what-to-compare)
