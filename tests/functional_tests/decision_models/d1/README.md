# D1 publisher-comparison checks

These scripts compare the native AutoModel implementation with pinned publisher code. The contributed NeMo code
follows this repository's Apache 2.0 license. Downloaded publisher code and model artifacts retain their own
[license terms](https://www.liquid.ai/lfm-license). No publisher model weights, historical logs or training datasets
are bundled here. Running a check fetches code at the immutable publisher revisions named in that script.

From the AutoModel repository root, in the supported dependency environment:

```bash
uv run python tests/functional_tests/decision_models/d1/verify_d1_3b_model.py --device cpu
uv run python tests/functional_tests/decision_models/d1/verify_d1_omni_model.py --device cpu
# Repeat in the supported GPU environment:
uv run python tests/functional_tests/decision_models/d1/verify_d1_3b_model.py --device cuda
uv run python tests/functional_tests/decision_models/d1/verify_d1_omni_model.py --device cuda
```

The checks construct tiny models with publisher code and non-unit RMSNorm weights. D1-3B checks exact state-dict
coverage, embedding ties, FP32/BF16 text/image logits and parameter gradients. Omni checks complete tiny weight
coverage, text/image/audio probability readouts, perception prefixes and differentiable text decisions. CPU checks
and full released-size checkpoint acceptance are distinct. Tiny D1-3B publisher comparisons passed CPU FP32/BF16
checks, and both strict publisher scripts passed CUDA checks. The local unit suite uses no publisher downloads.

The released-size baseline comparison also remains in `examples/decision/d1_posttrain.py`. Run `--mode base` first
and require it to pass before starting training. Direct SFT/RL invocations do not rerun that baseline automatically.
Its declared probability tolerance and sampled validation scope are not exhaustive model parity. The separate
released-size checks below passed for D1-3B BF16 and Omni FP32.

## Released-size acceptance

The standalone verifier downloads the same immutable release through the actual native `NeMoAuto` loader and the
unmodified publisher class. It checks every loaded key, tensor shape, dtype and value against the publisher and
serialized checkpoint, including parameter-identity ties. The only normalized serialization alias is D1-3B's tied
`lm_head.weight` / `model.language_model.embed_tokens.weight` pair.

The raw checkpoint check also recognizes the verified Transformers Siglip2 wrapper migration. Its Siglip2VisionModel
conversion uses the CLIPVisionModel rule that removes a `vision_model` wrapper. This verifier permits that removal only
under D1-3B's `model.vision_tower.` and Omni's `vision.tower.` prefixes, and only when the resulting key exists in both
loaded models. Every renamed raw/loaded key pair is retained in `serialized_key_renames`. Unknown keys and collisions
fail, and all tensor shapes, dtypes after the declared load cast, values and final state coverage remain strict.

Both independent loaders use BF16 for D1-3B and FP32 for Omni. Omni's
[pinned model card](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/README.md)
warns against BF16 and documents CPU FP32 / GPU FP16. This correctness verifier uses FP32 on CUDA to avoid that
unsupported BF16 path; it does not claim FP16 acceptance. Raw checkpoint dtypes and requested load dtype are recorded
separately in the result.

Released-size D1-3B BF16 acceptance passed: all 708 state entries matched exactly, with 707 serialized weights and the
input/output parameter tie preserved. Fifteen text decisions met the unchanged `0.02` probability bound (maximum
error `5.96e-8`); all five image questions had zero full-vocabulary logit and answer-alias probability error. A head-only
update changed the additive answer-row adapter while preserving all 708 backbone state entries, and fresh native
model/readout reload met the declared `1e-6` logit tolerance. The native model now recomputes rotary frequencies in FP32
on the execution device after device/dtype moves, matching the pinned publisher. Separate D1-3B base/SFT/RL recipe
stages completed five SFT and five RL optimizer steps with exact readout checkpoint reloads.

Released-size Omni FP32 acceptance passed: all 1,084 state entries matched exactly; 15 text decisions met the `0.02`
probability bound (maximum error `1.49e-7`); five image and five audio decisions had exact perception prefixes and zero
probability error. Its head update produced finite gradients for 31 head parameters, changed the head and preserved
all 1,053 backbone tensors. Fresh native model/readout reload met the declared `1e-6` logit tolerance. Separate base,
SFT and RL recipe stages completed five SFT and five RL optimizer steps with exact readout checkpoint reloads.

Both released-size checks used a newly generated fictional fixture with the tutorial schema: seed 73, four training records, three
validation records and three test records. They cover that fixture and declared precision.

```bash
uv run python tutorials/decision_models/decision_tutorial/data.py \
  --output-dir runs/d1-acceptance-data --seed 73 \
  --train-records 4 --validation-records 3 --test-records 3
uv run python tests/functional_tests/decision_models/d1/verify_released_checkpoint.py \
  --model d1-3b --device cuda --data-dir runs/d1-acceptance-data --output-dir runs/d1-3b-acceptance
uv run python tests/functional_tests/decision_models/d1/verify_released_checkpoint.py \
  --model d1-omni --device cuda --data-dir runs/d1-acceptance-data --output-dir runs/d1-omni-acceptance
# Add media only after establishing the text baseline:
uv run python tests/functional_tests/decision_models/d1/verify_released_checkpoint.py \
  --model d1-omni --device cuda --data-dir runs/d1-acceptance-data \
  --include-media --output-dir runs/d1-omni-media-acceptance
```

Each run uses all five questions in the first three selected validation records (15 independent decisions).
Omitting `--data-dir` selects the distinct bundled smoke fixture; the reported acceptance runs did not use that default.
It compares model-owned readout probabilities with the publisher's single-question API at the unchanged absolute
tolerance **0.02**, with zero relative tolerance. Failed comparisons write direct-chain diagnostics and exit with
failure. A successful baseline is followed by one supervised head update, a check that every backbone tensor stayed
unchanged, and a readout checkpoint reload into a freshly loaded native model/readout. Head state must match exactly;
reloaded decision logits use `atol=rtol=1e-6`.

Optional `--include-media` uses a separately loaded native model at the requested checkpoint dtype, after text parity
passes. For D1-3B, both models receive the same synthetic image and pinned-processor output: full vocabulary logits
use the existing `atol=rtol=0.01` and cosine `>0.9999` criteria, and answer-alias probabilities retain absolute tolerance
`0.02`. Omni requires exact image/audio perception prefixes and compares all five questions with uncalibrated publisher
media probabilities at the existing `atol=rtol=2e-6` criteria. Media results have a separate scope in the result file.
The verifier does not widen tolerances when a check fails.

`acceptance-result.json` records generic correctness evidence; the output directory also holds run-only diagnostics
and readout weights. Keep those artifacts outside source changes. A text-only invocation does not establish media
parity; neither mode establishes multi-device training or full optimizer resume. Adding the verifier does not itself
establish acceptance beyond the recorded checks.
