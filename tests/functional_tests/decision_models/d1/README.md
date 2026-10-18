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
and full released-size GPU checkpoint acceptance are distinct. The strict scripts have not been rerun as part of
this offline source preparation; the local unit suite uses no publisher downloads.

The released-size baseline comparison also remains in `examples/decision/d1_posttrain.py`; it must pass before that
example starts training. Its declared probability tolerance and sampled validation scope are not exhaustive model
parity. Released-size D1-3B probability parity remains unresolved; no successful full-size acceptance is claimed
for this contribution. Tiny-model checks and exact checkpoint reloads do not establish that acceptance.
