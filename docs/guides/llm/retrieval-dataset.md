# Biencoder Retrieval Dataset (Embedding Fine-tuning)

NeMo Automodel supports **biencoder/embedding model fine-tuning** using a retrieval-style dataset: each training example is a **query** paired with **one positive** document and **one or more negative** documents.

This dataset is used by the biencoder recipes (see `examples/biencoder/`) together with the `RetrievalBiencoderCollator`.

## What the biencoder consumes

The dataset factory `nemo_automodel.components.datasets.llm.make_retrieval_dataset` returns a Hugging Face `datasets.Dataset`. At runtime it transforms each raw record into the training-time schema:

- `question`: query string
- `doc_text`: list of document texts in the order `[positive, negative_1, negative_2, ...]`
- `doc_image`: list of images (or empty strings), aligned with `doc_text`
- `query_instruction` / `passage_instruction`: optional, used when `use_dataset_instruction: true` and the corpus provides instructions via metadata

## Supported input formats

NeMo Automodel supports **two** input schemas:

### 1) Corpus-ID-based JSON (Merlin/NeMo-retriever style)

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

Each corpus directory must contain a `merlin_metadata.json` file.

Minimal example:

```json
{ "class": "TextQADataset", "corpus_id": "wiki_corpus" }
```

Notes:
- `pos_doc` and `neg_doc` can be lists of `{"id": ...}` dicts or raw IDs (they are normalized internally).
- If you set `use_dataset_instruction: true`, optional fields like `query_instruction` and `passage_instruction` in `merlin_metadata.json` will be surfaced to the collator.

### 2) Inline-text JSONL (no corpus required)

This is convenient for custom fine-tuning pipelines where the documents are included **inline**.

**JSONL example (one example per line):**

```json
{"query":"Explain transformers","pos_doc":"Transformers are a type of neural network...","neg_doc":["RNNs are...","CNNs are..."]}
{"query":"What is Python?","pos_doc":["A programming language."],"neg_doc":"A snake."}
```

Notes:
- `query` is accepted (and `question` is also accepted as an alias).
- `pos_doc` and `neg_doc` can be either:
  - strings (interpreted as document text), or
  - lists of strings, or
  - dicts with at least `text` (optionally `image`, `nr_ocr`) for multimodal use cases.
- If `corpus_id` is not provided, it defaults to `__inline__`.
- `use_dataset_instruction: true` has no effect for pure inline records (instructions come from corpus metadata).

## YAML usage (dataset + collator)

Use the dataset factory plus the biencoder collator:

```yaml
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  dataset:
    _target_: nemo_automodel.components.datasets.llm.make_retrieval_dataset
    data_dir_list:
      - /abs/path/to/train.jsonl   # or train.json (corpus-id format)
    data_type: train
    train_n_passages: 5           # 1 positive + 4 negatives
    do_shuffle: true
    use_dataset_instruction: false
  collate_fn:
    _target_: nemo_automodel.components.datasets.llm.RetrievalBiencoderCollator
    q_max_len: 512
    p_max_len: 512
    query_prefix: "query:"
    passage_prefix: "passage:"
    pad_to_multiple_of: 8
```

## Gotchas

- `pos_doc` must be **non-empty**.
- If training requests negatives (e.g., `train_n_passages > 1`), `neg_doc` must contain **at least one** document (the loader will cycle negatives if you provide fewer than needed).
