# Retrieval Dataset for Bi-Encoders and Cross-Encoders

NeMo AutoModel supports **retrieval model fine-tuning** using a retrieval-style dataset: each training example is a
**query** paired with **one positive** document and **one or more negative** documents.

This dataset is used by the retrieval recipes (see `examples/retrieval/bi_encoder/` and
`examples/retrieval/cross_encoder/`) together with the retrieval collators. For an end-to-end training workflow, see
[Retrieval Fine-Tuning](retrieval-finetuning.md).

## Raw Records and Runtime Schemas

The dataset factory `nemo_automodel.components.datasets.llm.make_retrieval_dataset` returns a Hugging Face
`datasets.Dataset`. At runtime it transforms each raw record into the training-time schema:

- `question`: query string
- `doc_text`: list of document texts in the order `[positive, negative_1, negative_2, ...]`
- `doc_image`: list of images (or empty strings), aligned with `doc_text`
- `doc_id`: list of document IDs aligned with `doc_text` for corpus-backed and `hf://` sources. Pure inline JSONL
  does not provide real document IDs for duplicate-document masking unless you add them in a custom preprocessing path.
- `query_instruction` / `passage_instruction`: optional, used when `use_dataset_instruction: true` and the corpus
  provides instructions via metadata

The cross-encoder recipe consumes the same raw retrieval records, but sets `model_type: cross_encoder`. Its dataset
transform flattens each query with its positive and negative passages, and `CrossEncoderCollator` serializes each
query-passage pair for reranking.

Training uses exactly one positive passage per example: the first item in `pos_doc`. For datasets with multiple
relevant passages, either choose a canonical positive, expand the record into one example per positive, or add a
multi-positive loss/masking strategy before training. If expanded rows for the same query can share a batch, keep
distributed in-batch negatives disabled unless you also prevent sibling positives from becoming negatives through
qrels-aware sampling or masking. Keep the full set of known positives in your qrels or corpus metadata for evaluation
and false-negative filtering, even when each training row uses one positive.

## Supported Input Formats

NeMo AutoModel supports **two** input schemas across three source types. They use different dataset factories:

- Use `nemo_automodel.components.datasets.llm.make_retrieval_dataset` for corpus ID-based JSON and `hf://` sources.
- Use `nemo_automodel.components.datasets.llm.retrieval_dataset_inline.make_retrieval_dataset` for inline JSONL where
  document text is stored directly in each record.

| Source | Query field | Required document fields | Best for |
|--------|-------------|--------------------------|----------|
| Corpus ID JSON | `question` | `question_id`, `corpus_id`, `pos_doc`, and `neg_doc` IDs resolved through a local corpus | Production data, hard-negative mining, same-document masking |
| `hf://` AutoModel schema | `question` | `pos_doc`, a companion HF corpus split, and `neg_doc` before training with `n_passages > 1` | Tutorial runs and shared AutoModel retrieval datasets |
| Inline JSONL | `query` or `question` | Inline text in `pos_doc` and `neg_doc` | Small custom runs when you do not need mining or document-ID masking |

Separate qrels files are not consumed directly by the training dataset factory. Convert qrels-style data into retrieval
records before training:

1. Put every passage in a corpus split with stable `id` and `text` values.
2. For each query, write one or more training records with `question_id`, `question`, `corpus_id`, `pos_doc`, and
   `neg_doc`. Use unique `question_id` values within each mining file; hard-negative mining writes results back by ID.
3. For training, use the first relevant document in each record as `pos_doc[0]`; expand multi-positive queries into
   multiple records if you want every positive to become a supervised positive.
4. For hard-negative mining, include all known positive document IDs for that query in the row's `pos_doc`. The miner
   excludes only IDs present in the input row, not an external qrels file.
5. If you expand one query into multiple positive rows, keep those sibling-positive rows out of the same in-batch-negative
   training batch or use qrels-aware masking.
6. Preserve the complete qrels separately for full-corpus evaluation and audit mined negatives against them before
   reusing the output for training.

### Corpus ID-Based JSON (Merlin/NeMo-Retriever Style)

This is the format used by NeMo retriever pipelines where documents live in a separate **corpus** and training examples reference documents by **ID**.

**Training file example (single JSON):**

```json
{
  "corpus": [
    { "path": "/abs/path/to/wiki_corpus" }
  ],
  "data": [
    {
      "question_id": "q_001",
      "question": "Explain transformers",
      "corpus_id": "wiki_corpus",
      "pos_doc": [{ "id": "d_123" }],
      "neg_doc": [{ "id": "d_456" }, "d_789"]
    }
  ]
}
```

**Corpus requirements**

Each corpus directory must contain a `merlin_metadata.json` file and a Hugging Face-loadable `train` split with at least
`id` and `text` columns. For `class: TextQADataset`, AutoModel calls `datasets.load_dataset(<corpus path>)["train"]`,
then resolves `pos_doc` and `neg_doc` IDs against that split.

Minimal example:

```json
{ "class": "TextQADataset", "corpus_id": "wiki_corpus" }
```

Minimal local layout:

```text
retrieval-data/
  train.json
  wiki_corpus/
    merlin_metadata.json
    train.parquet   # or another load_dataset-compatible train split with id,text columns
```

The `corpus_id` in `merlin_metadata.json` must match the `corpus_id` in each training record. Relative corpus paths in
`train.json` are resolved relative to the JSON file.

:::{note}
- `pos_doc` and `neg_doc` can be lists of `{"id": ...}` dicts or raw IDs (they are normalized internally).
- Training uses `pos_doc[0]` as the positive. Additional positives are ignored unless you expand the data before
  training.
- To train with corpus instructions, set `use_dataset_instruction: true` on both the dataset and the bi-encoder
  collator. The dataset surfaces `query_instruction` and `passage_instruction` from `merlin_metadata.json`; the collator
  prepends them before tokenization.
:::

### Hugging Face `hf://` Sources

Direct `hf://` loading expects the AutoModel retrieval schema, not arbitrary Hugging Face retrieval datasets. The URI
format is:

```text
hf://<org>/<repo>/<subset>
```

Each subset must provide:

- `<subset>/dataset_metadata.json` with `corpus_id` metadata and `ids_only: false`
- a `<subset>_corpus` train split with `id` and `text` columns
- a `<subset>` train split with `question` and `pos_doc`; `neg_doc` may be absent but must be available before training
  with `n_passages > 1`

Datasets with BEIR, DPR, MS MARCO, MIRACL, or other layouts need a preprocessing step before direct `hf://` loading.

### Inline-Text JSONL (No Corpus Required)

This is convenient for custom fine-tuning pipelines where the documents are included **inline**.

**JSONL example (one example per line):**

```json
{"query":"Explain transformers","pos_doc":"Transformers are a type of neural network...","neg_doc":["RNNs are...","CNNs are..."]}
{"query":"What is Python?","pos_doc":["A programming language."],"neg_doc":"A snake."}
```

:::{note}
- `query` is accepted (`question` is also accepted as an alias).
- `pos_doc` and `neg_doc` can be either:
  - strings (interpreted as document text), or
  - lists of strings, or
  - dicts with at least `text`.
- The current LLM retrieval collators tokenize text only. Do not rely on inline `image` or OCR fields unless you add a
  custom preprocessing and collator path.
- If `corpus_id` is not provided, it defaults to `__inline__`.
- `use_dataset_instruction: true` has no effect for pure inline records (instructions come from corpus metadata).
:::

## YAML Usage (Dataset + Collator)

Use the corpus/HF dataset factory plus the bi-encoder collator for corpus ID-based JSON or `hf://` sources:

```yaml
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  dataset:
    _target_: nemo_automodel.components.datasets.llm.make_retrieval_dataset
    model_type: bi_encoder
    data_dir_list:
      - /abs/path/to/train.json    # or hf://nvidia/embed-nemotron-dataset-v1/FEVER
    data_type: train
    n_passages: 5                 # 1 positive + 4 negatives
    do_shuffle: true
    use_dataset_instruction: false
  collate_fn:
    _target_: nemo_automodel.components.datasets.llm.BiEncoderCollator
    q_max_len: 512
    p_max_len: 512
    query_prefix: "query:"
    passage_prefix: "passage:"
    use_dataset_instruction: false
    pad_to_multiple_of: 8
```

For corpus ID JSON and `hf://` sources, `do_shuffle: true` shuffles rows only when `max_train_samples` is set before
subsampling. Training order is controlled by the dataloader or distributed sampler. For inline JSONL, `do_shuffle: true`
shuffles the loaded rows directly.

Use the inline dataset factory for inline JSONL:

```yaml
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  dataset:
    _target_: nemo_automodel.components.datasets.llm.retrieval_dataset_inline.make_retrieval_dataset
    model_type: bi_encoder
    data_dir_list:
      - /abs/path/to/train.jsonl
    data_type: train
    n_passages: 5                 # 1 positive + 4 negatives
    do_shuffle: true
  collate_fn:
    _target_: nemo_automodel.components.datasets.llm.BiEncoderCollator
    q_max_len: 512
    p_max_len: 512
    query_prefix: "query:"
    passage_prefix: "passage:"
    pad_to_multiple_of: 8
```

For cross-encoder training, keep the same dataset factory, set `model_type: cross_encoder`, and use
`CrossEncoderCollator` arguments:

```yaml
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  dataset:
    _target_: nemo_automodel.components.datasets.llm.retrieval_dataset_inline.make_retrieval_dataset
    model_type: cross_encoder
    data_dir_list:
      - /abs/path/to/train.jsonl
    data_type: train
    n_passages: 5
    do_shuffle: true
  collate_fn:
    _target_: nemo_automodel.components.datasets.llm.CrossEncoderCollator
    rerank_max_length: 512
    prompt_template: "question:{query} \n \n passage:{passage}"
    pad_to_multiple_of: 8
```

## Requirements

- `pos_doc` must be **non-empty**.
- `neg_doc` must be present in local JSON and JSONL training records. It may be empty only when `n_passages: 1`.
- `hf://` sources may omit `neg_doc` in the source dataset, but add negatives before training with `n_passages > 1`.
- If training requests negatives (e.g., `n_passages > 1`), `neg_doc` must contain **at least one** document.

:::{warning}
`n_passages: 1` is a schema escape hatch, not a good default training setup. The standard bi-encoder and cross-encoder
recipes need at least one negative candidate for meaningful contrastive or reranking supervision, unless you add a
custom negative strategy such as qrels-aware in-batch negatives.
:::
