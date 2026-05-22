---
name: retrieval-models
description: Work on NeMo AutoModel retrieval encoder support, including bi-encoder embedding and cross-encoder scoring backbones, wrapper and registry wiring, retrieval recipes, and focused validation.
when_to_use: Adding, modifying, or debugging retrieval model support; working with NeMoAutoModelBiEncoder, NeMoAutoModelCrossEncoder, bidirectional causal-decoder backbones, retrieval recipe configs, retrieval dataset/collator shape issues, or encoder save/reload metadata.
---

# Retrieval Models

Use this skill when a task touches retrieval model behavior, not ordinary LLM
generation. Retrieval support has three layers that are easy to mix up:

1. Public entry points: `nemo_automodel.NeMoAutoModelBiEncoder.from_pretrained`
   and `nemo_automodel.NeMoAutoModelCrossEncoder.from_pretrained`.
2. Retrieval wrappers in `nemo_automodel/_transformers/retrieval.py`:
   `BiEncoderModel`, `CrossEncoderModel`, `build_encoder_backbone`, and
   `SUPPORTED_BACKBONES`.
3. Concrete backbone classes under `nemo_automodel/components/models/`, such as
   `llama_bidirectional` and `ministral_bidirectional`.

## References

- `PITFALLS.md`: read when tests fail, save/reload metadata looks wrong, or a
  recipe shape error appears.
- `skills/model-onboarding/SKILL.md`: read before creating a new architecture
  directory or registry entry.
- `skills/recipe-development/SKILL.md`: read before changing retrieval recipe
  flow or YAML config shape.
- `skills/testing/SKILL.md`: read before adding or moving tests.

## First Files

Start with the narrowest surface that matches the task:

- Model construction: `nemo_automodel/_transformers/retrieval.py`
- Public AutoModel wrapper: `nemo_automodel/_transformers/auto_model.py`
- Registry: `nemo_automodel/_transformers/registry.py`
- Existing bidirectional examples:
  `nemo_automodel/components/models/llama_bidirectional/model.py` and
  `nemo_automodel/components/models/ministral_bidirectional/model.py`
- Recipes: `nemo_automodel/recipes/retrieval/train_bi_encoder.py` and
  `nemo_automodel/recipes/retrieval/train_cross_encoder.py`
- Dataset/collator: `nemo_automodel/components/datasets/llm/retrieval_dataset.py`,
  `retrieval_dataset_inline.py`, and `retrieval_collator.py`
- Example YAMLs: `examples/retrieval/bi_encoder/` and
  `examples/retrieval/cross_encoder/`

## Work Checklist

1. Classify the change as backbone, wrapper, recipe/config, or dataset/collator.
2. Read the matching files from the first-files list before planning edits.
3. Preserve the bi-encoder or cross-encoder shape contract while making the
   smallest code change.
4. Add or update the focused unit test that proves the contract changed or still
   holds.
5. Run the smallest validation command from this skill, then broaden only if the
   change touches distributed training, checkpointing, or full recipe execution.

## Choose The Implementation Path

Before editing, decide which path applies:

- Generic encoder or scorer already supported by HuggingFace Auto classes:
  leave `SUPPORTED_BACKBONES` alone unless a custom non-causal backbone is
  required. `build_encoder_backbone` falls back to `AutoModel` for embedding and
  `AutoModelForSequenceClassification` for scoring.
- Causal decoder used for embeddings or reranking:
  add a bidirectional backbone class that disables causal attention and uses a
  bidirectional attention mask.
- Nested model such as a VLM with a text tower:
  use the `extract_submodel` config knob and verify the extracted object has a
  `.config`; the loader preserves the extracted dtype when rebuilding the
  retrieval target class.
- Cross-encoder with custom non-causal behavior:
  provide a sequence-classification retrieval class for the `"score"` task.
  Otherwise the HF sequence-classification fallback may be enough.

## Registration Handshake

Custom retrieval backbones need all of these pieces:

1. Export the model class from the model module with `ModelClass = [...]`.
2. Register every custom retrieval architecture in
   `MODEL_ARCH_MAPPING` in `nemo_automodel/_transformers/registry.py`.
3. Add the optional `{"retrieval"}` tag in `MODEL_ARCH_MAPPING`. This is what
   lets `configure_encoder_metadata` write retrieval `auto_map` metadata for
   saved checkpoints.
4. Add `model_type -> task -> architecture name` entries to
   `SUPPORTED_BACKBONES` in `nemo_automodel/_transformers/retrieval.py`.
   Use `"embedding"` for `BiEncoderModel`; use `"score"` for
   `CrossEncoderModel`.
5. If the config has a new `model_type`, make sure HuggingFace Auto config/model
   reload works. Existing retrieval examples register their bidirectional config
   with `AutoConfig` and `AutoModel`.

## Backbone Rules

For bidirectional causal-decoder backbones, do not stop at setting a config
field. The forward path must actually be non-causal:

- Set each attention layer's `is_causal` flag to `False`.
- Replace the causal mask with `transformers.masking_utils.create_bidirectional_mask`.
- Keep pooling and temperature fields on the retrieval config when the backbone
  needs them.
- Preserve HuggingFace return types such as `BaseModelOutputWithPast` or
  `SequenceClassifierOutputWithPast`.

Use the existing Llama and Ministral bidirectional models as patterns, but copy
only the behavior the target architecture needs.

## Bi-Encoder Contract

Bi-encoder training keeps query and passage encoding separate.

- YAML model target:
  `nemo_automodel.NeMoAutoModelBiEncoder.from_pretrained`
- Dataset: `make_retrieval_dataset(model_type="bi_encoder")`
- Collator: `BiEncoderCollator`
- Dataset example shape: one `question` and `doc_text` as
  `[positive, negative_1, ...]`
- Collated batch:
  - `q_input_ids`, `q_attention_mask`: `[B, Lq]`
  - `d_input_ids`, `d_attention_mask`: `[B * P, Ld]`
  - `labels`: `[B]` zeros for compatibility
- The recipe computes scores `[B, P]` and real CE labels internally. The
  positive passage must be at column 0.

When `do_distributed_inbatch_negative` is enabled, keep `passage_doc_ids` from
the collator so duplicate positives can be masked across gathered passages.
ColBERT pooling does not support distributed in-batch negatives.

## Cross-Encoder Contract

Cross-encoder training jointly encodes a query-passage pair and reshapes scores
back to query groups.

- YAML model target:
  `nemo_automodel.NeMoAutoModelCrossEncoder.from_pretrained`
- Dataset: `make_retrieval_dataset(model_type="cross_encoder")`
- Collator: `CrossEncoderCollator`
- Dataset transform flattens grouped passages into one row per query-passage
  pair and carries `num_labels`.
- Collated batch:
  - `input_ids`, `attention_mask`: `[B * P, L]`
  - `labels`: `[B]` zeros, created from `num_labels`
- The recipe runs the scorer, reshapes `outputs.logits.view(-1, n_passages)`,
  and applies CE with the positive at column 0.

Any change to `n_passages`, `eval_negative_size`, or flattening must preserve
the invariant that flattened rows are divisible by the recipe's
`train_n_passages` or `val_n_passages`.

## Validation

Prefer focused CPU tests first. Use functional or GPU tests only when changing
distributed training, checkpointing, or real recipe execution.

For model/backbone changes, run the relevant subset:

```bash
uv run pytest tests/unit_tests/_transformers/test_retrieval.py -q
uv run pytest tests/unit_tests/models/bi_encoder/test_bi_encoder_model.py -q
uv run pytest tests/unit_tests/models/bi_encoder/test_llama_bidirectional_model.py -q
uv run pytest tests/unit_tests/models/bi_encoder/test_ministral_bidirectional_model.py -q
```

For dataset, recipe, or shape changes:

```bash
uv run pytest tests/unit_tests/datasets/llm/test_bi_encoder_collator.py tests/unit_tests/datasets/llm/test_cross_encoder_collator.py -q
uv run pytest tests/unit_tests/datasets/llm/test_retrieval_dataset.py -q
uv run pytest tests/unit_tests/recipes/test_train_cross_encoder.py -q
```

For a new custom retrieval backbone, add tiny tests that cover:

- config fields and model type,
- all attention layers are non-causal,
- changing a later token affects an earlier token,
- `BiEncoderModel.build` resolves through `SUPPORTED_BACKBONES`,
- `CrossEncoderModel.build` resolves the custom scorer or intentionally falls
  back to HF sequence classification,
- `extract_submodel` rebuilds the retrieval target and preserves dtype,
- saved metadata contains `architectures` and retrieval `auto_map` when the
  architecture has the `{"retrieval"}` tag.

## Trigger Checks

Use this skill for prompts about retrieval encoders, rerankers, bi-encoder
training, cross-encoder scoring, bidirectional retrieval backbones, retrieval
recipe shape errors, and retrieval checkpoint save/reload metadata.

Do not use this skill for unrelated RAG application code, generic causal LM
generation, VLM chunk retrieval, or hard-negative mining unless the task also
touches the model wrapper, dataset/collator contract, or retrieval recipe.

## Evaluation Scenarios

| Prompt | Expected behavior |
| --- | --- |
| "Add retrieval encoder support for a causal decoder backbone." | Uses this skill plus `model-onboarding`, implements a bidirectional backbone, updates registry and `SUPPORTED_BACKBONES`, and adds tiny non-causal and build-resolution tests. |
| "My cross-encoder recipe fails when reshaping logits." | Uses this skill plus `recipe-development`, checks `model_type`, `n_passages`, `val_n_passages`, flattening, and `CrossEncoderCollator` labels before changing model code. |
| "Change the bi-encoder dataset to use inline JSONL." | Uses this skill to choose `retrieval_dataset_inline.make_retrieval_dataset`, verify grouped `[positive, negatives...]` output, and run collator/dataset unit tests. |
| "Build a RAG demo app that calls an existing embedding endpoint." | Does not use this skill unless the task also changes NeMo AutoModel retrieval model, dataset, collator, or recipe code. |
