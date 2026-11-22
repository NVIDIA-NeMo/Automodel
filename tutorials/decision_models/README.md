# Typed decision-model post-training

These notebooks use framework-owned CLM/CLEF/D1 APIs with a newly authored fictional fixture. They cover released baselines, head adaptation, supervised training, optional policy continuation, validation selection, checkpoint persistence and heldout reporting. No training results, model weights or executed outputs are bundled. The small fixture checks mechanics; it is not a security benchmark.

| Notebook | Workflow |
| --- | --- |
| [00 Fictional data](00_fictional_data.ipynb) | Deterministic typed questions, exact policy labels, per-option utilities and split hashes |
| [01 CLM supervised and utility](01_clm_supervised_and_utility.ipynb) | Frozen encoder, released head baseline, supervised heads, optional exact expected utility or sampled GRPO |
| [02 CLEF heads or LoRA](02_clef_heads_or_lora.ipynb) | Frozen schema head or backbone LoRA plus schema head; optional utility continuation and local merged export |
| [03 CLM contrastive heads](03_clm_contrastive_heads.ipynb) | Bidirectional InfoNCE on cached frozen features, with repeated-positive false-negative masking |
| [04 D1 readouts](04_d1_readouts.ipynb) | Frozen D1-3B answer-token rows or D1 Omni marker head; baseline, SFT and optional utility continuation |
| [05 Optional Data Designer](05_optional_data_designer.ipynb) | Public-provider writer/judge paraphrases, source-case split assignment, identifier filtering and frozen new manifests |
| [06 Experimental calibration continuation](06_experimental_calibration_continuation.ipynb) | Gaussian logit proposals, correctness/confidence reward, group baseline and CE anchor; validation Brier selection |

Paired `.py` files contain the same workflow in percent-cell format. Default `RUN_TRAINING=False` and `RUN_GENERATION=False` create fresh local fixture files and display planned commands. They do not launch subprocesses, load models or call services. Run notebooks from this repository or their tutorial directory so repository discovery succeeds. Generated runs use new timestamped directories under ignored `runs/`; existing datasets are never replaced.

## CPU preview and fixture checks

From the repository root, use an isolated CPU interpreter; these helpers require only the standard library:

```bash
uv run --no-project python -m unittest discover -s tests/unit_tests/tutorials -p 'test_decision_tutorials.py' -v
PYTHONPATH=tutorials/decision_models uv run --no-project python -m decision_tutorial.data   --output-dir tutorials/decision_models/runs/my-fixture --seed 7
uv run --no-project --with jupyterlab --with matplotlib jupyter lab tutorials/decision_models
```

The checked-in `fixtures/smoke/` has 16 train, 8 validation and 8 test records, each with five questions. Training notebooks generate 128/32/32 records. Identifiers use `.example`, `example.com` and RFC 5737 IP addresses. Exact state content and IDs are disjoint across splits; scenario templates intentionally repeat. Gold answers stay outside the model-visible `state`. This policy and all fixture records were authored for this tutorial without copying an operational corpus.

## Enable a model experiment

Prepare the framework's supported Linux GPU environment using its [installation instructions](../../README.md), then use that environment as the notebook kernel. CLM loads a public Qwen3-8B encoder; CLEF loads a 27B backbone even for head-only training. Choose `NPROC` and, where exposed, `TP` for your allocated hardware. Head-only training freezes backbone gradients but still needs memory for forward passes. Download access and model license requirements remain those of the selected publisher. Optional learning curves need `matplotlib` in the kernel environment.

Set `RUN_TRAINING=True` only when ready to run the displayed native commands. CLM caches the baseline's frozen embeddings, then reuses them across stages. CLEF `HEAD_ONLY=True` is the default; `False` enables LoRA plus the joint head. Keep this choice consistent across checkpoint reloads. Saved heads/adapters remain in the run directory. CLEF merged export is an explicit optional action for LoRA stages and may require substantial disk and memory.

`ENABLE_UTILITY=True` continues the selected SFT stage. Exact expected utility is the default because every candidate utility is known. Native sampled GRPO is also available: select `UTILITY_OBJECTIVE="grpo"` in notebooks 01/02 and inspect the recipe's group size, temperature, KL and supervised-anchor settings. This is an experimental configuration choice rather than a claim that sampling is superior. Notebook 06 demonstrates a separate experimental calibration-aware objective inspired by the [JevAny recipe](https://github.com/SimpleJev/JevAny/blob/main/recipes/rlcr.toml); it does not reproduce a proprietary training method or guarantee gains.

## Interpret a run

Use training data for fitting, validation for checkpoint selection and test only for final reporting. Baseline/SFT/continuation stages use the same generated split paths. Retain `data/manifest.json` and verify its hashes when comparing runs. The CSV helper checks common record/decision counts; equal counts do not prove identical heldout records. Preserve model revisions, trainable scope, topology, seed and recipe values with your results. Repeat training seeds before making effectiveness claims.

Native reports retain decision accuracy, exact record match, NLL, Brier, calibration summaries, utility and metadata slices. A utility gain can accompany worse accuracy or probability quality. The notebooks plot actual local objective and validation observations; they make no fixed improvement claim. Objective logs may be per-update/main-rank observations and do not equal a full training-set loss. Checkpoint self-reload, publisher agreement and cross-framework numerical equivalence require separate tests.

## Optional generated prose

Notebook 05 needs `uv pip install data-designer==0.9.4 pandas pyarrow` in the chosen environment and your own `NVIDIA_API_KEY` only when generation is enabled. Inspect its two-record preview before `REVIEWED_PREVIEW=True`. Labels and utility tables remain policy-derived; writer/judge output changes only observation prose. Source cases keep their original split and appear at most once. Automated checks do not prove factual accuracy or absence of personal data; inspect generated text before sharing it. Generated data defines a new experiment and must not replace an existing split silently.

## Experimental D1 readouts

[04 D1 readouts](04_d1_readouts.ipynb) covers D1-3B additive answer-token rows and the D1 Omni marker head through native framework models. It keeps the backbone frozen, selects readouts on validation, and reports baseline/SFT/optional utility stages on the same heldout split. Use a prepared single-GPU environment; notebook source/default-path checks are CPU-only, and the notebook has not been executed end-to-end. Separate released-checkpoint GPU, publisher and media acceptance is documented in the [D1 acceptance notes](../../tests/functional_tests/decision_models/d1/README.md), with its precision and fixture limitations. The fictional observations are text-only and establish no image/audio adaptation result or numerical-parity claim. The implementation follows the repository's Apache 2.0 license; downloaded publisher weights and remote components retain their [publisher license](https://www.liquid.ai/lfm-license).

The tutorial code and newly authored fixture follow this repository's [Apache 2.0 license](../../LICENSE). Downloaded models, publisher runtime components and third-party packages retain their own licenses. Model weights are not bundled in these tutorials. Notebook checks are CPU-only; the notebooks make no API-generation or model-export validation claim.

The native commands use `automodel examples/decision/clm_soc_triage.yaml`, `clef_soc_triage_heads.yaml` and `clef_soc_triage_lora.yaml`. The D1 command is `uv run python examples/decision/d1_posttrain.py`. Recipe definitions remain in `nemo_automodel/recipes/decision/`; model-owned heads remain in their CLM/CLEF packages. See the [example flags and configuration notes](../../examples/decision/README.md).
