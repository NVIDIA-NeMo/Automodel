# Retrieval Model Pitfalls

## Incomplete Registration

Adding a class to `MODEL_ARCH_MAPPING` is not enough for retrieval. Custom
retrieval classes also need the `{"retrieval"}` tag and a matching
`SUPPORTED_BACKBONES` entry for each supported task. Without the tag, saved
checkpoints may miss the retrieval `auto_map` metadata. Without
`SUPPORTED_BACKBONES`, `build_encoder_backbone` may silently fall back to HF
Auto classes or reject the task.

## Causal Mask Still Active

For a bidirectional causal-decoder backbone, setting `config.is_causal = False`
is not a sufficient proof. Verify every attention layer has `is_causal = False`
and the forward path uses `create_bidirectional_mask`. Add a tiny test where
changing a later token changes an earlier hidden state.

## Cross-Encoder Labels Look Wrong

Cross-encoder labels are one label per query group, not one label per flattened
query-passage row. The collator emits `[B]` zero labels from `num_labels`, and
the recipe reshapes logits from `[B * P, 1]` to `[B, P]`.

## Positive Passage Order Changed

Both bi-encoder and cross-encoder losses assume the positive passage is first.
If preprocessing changes document order, labels of all zeros become wrong even
though shapes still pass.

## `n_passages` Mismatch

The dataset controls how many positive-plus-negative passages are produced. The
recipes reshape or score using `train_n_passages` and `val_n_passages`. If train
or validation config changes `n_passages` without preserving grouped or
flattened shape, losses and metrics can be incorrect or fail at `view`.

## Wrong Dataset Or Collator Pair

Use `model_type: bi_encoder` with `BiEncoderCollator`; use
`model_type: cross_encoder` with `CrossEncoderCollator`. Mixing them usually
shows up as missing `q_`/`d_` keys, missing labels, or invalid logits reshape.

## Missing Inline Dataset Path

There are two dataset loaders. `retrieval_dataset.py` handles corpus-id JSON and
`hf://` sources. `retrieval_dataset_inline.py` handles inline JSON/JSONL text and
rejects corpus-id format. Functional tests often use the inline loader.

## Pooling Passed To Generic HF Models

Pooling is a retrieval-wrapper or custom-backbone concept. Generic HF
`AutoModel` fallback paths should not receive unsupported pooling kwargs. Let
`build_encoder_backbone` decide which kwargs are safe for supported custom
backbones versus HF fallback classes.

## Nested Model Extraction

When using `extract_submodel`, the dotted path must resolve to an object with a
`.config`. For supported text backbones, the loader rebuilds the registered
retrieval class from the extracted state dict and moves it to the extracted
dtype. Test extraction with a tiny fake or local checkpoint before relying on a
large VLM.

## Save And Reload Metadata

Retrieval wrappers save the inner backbone. `configure_encoder_metadata` sets
`config.architectures` for all backbones and `config.auto_map` only for classes
registered as retrieval architectures. If a saved custom retrieval checkpoint
cannot reload through Auto classes, inspect the registry tag and config
registration first.

## Distributed In-Batch Negatives

Distributed in-batch negatives gather passages across ranks. Keep
`passage_doc_ids` from `BiEncoderCollator` so positives with the same corpus
document id can be masked. This path is not implemented for ColBERT pooling.
