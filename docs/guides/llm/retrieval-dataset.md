# Retrieval Dataset (Embedding Fine-Tuning)

NeMo AutoModel supports **retrieval model fine-tuning** using a retrieval-style dataset: each training example is a **query** paired with **one positive** document and **one or more negative** documents.

The retrieval recipes use this dataset with the `BiEncoderCollator`. Example implementations are in `examples/retrieval/bi_encoder/` and `examples/retrieval/cross_encoder/`.

## What the Bi-Encoder Consumes

The dataset factory `nemo_automodel.components.datasets.llm.make_retrieval_dataset` returns a Hugging Face `datasets.Dataset`. At runtime, it transforms each raw record into the training-time schema:

- `question`: query string
- `doc_text`: list of document texts in the order `[positive, negative_1, negative_2, ...]`
- `doc_image`: list of images (or empty strings), aligned with `doc_text`
- `query_instruction` / `passage_instruction`: optional, used when `use_dataset_instruction: true` and the corpus provides instructions through metadata

## Supported Input Formats

NeMo AutoModel supports **two** input schemas:

### Corpus ID-Based JSON (Merlin/NeMo-Retriever Style)

This is the format used by NeMo retriever pipelines where documents live in a separate **corpus** and training examples reference documents by **ID**.

#### Training File Example (Single JSON)

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

#### Corpus Requirements

Each corpus directory must contain a `merlin_metadata.json` file.

Minimal example:

```json
{ "class": "TextQADataset", "corpus_id": "wiki_corpus" }
```

:::{note}
- `pos_doc` and `neg_doc` can be lists of `{"id": ...}` dicts or raw IDs (they are normalized internally).
- `cycle_positive_docs` defaults to `false`, which always uses the first positive document. When a training record has multiple positive documents, set `cycle_positive_docs: true` to rotate through them deterministically by epoch order (`epoch % len(pos_doc)`).
- If you set `use_dataset_instruction: true`, optional fields like `query_instruction` and `passage_instruction` in `merlin_metadata.json` are surfaced to the collator.
:::

### Inline-Text JSONL (No Corpus Required)

This is convenient for custom fine-tuning pipelines where the documents are included **inline**.

The format uses one JSON object per line:

```json
{"query":"Explain transformers","pos_doc":"Transformers are a type of neural network...","neg_doc":["RNNs are...","CNNs are..."]}
{"query":"What is Python?","pos_doc":["A programming language."],"neg_doc":"A snake."}
```

:::{note}
- `query` is accepted (`question` is also accepted as an alias).
- `pos_doc` and `neg_doc` can be either:
  - strings (interpreted as document text), or
  - lists of strings, or
  - dicts with at least `text` (optionally `image`, `nr_ocr`) for multimodal use cases.
- If `corpus_id` is not provided, it defaults to `__inline__`.
- `use_dataset_instruction: true` has no effect for pure inline records (instructions come from corpus metadata).
:::

## Configure the Dataset and Collator in YAML

Use the dataset factory with the bi-encoder collator:

```yaml
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  dataset:
    _target_: nemo_automodel.components.datasets.llm.make_retrieval_dataset
    data_dir_list:
      - /abs/path/to/train.jsonl   # or train.json (corpus-id format)
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
    pad_to_multiple_of: 8
```

## Requirements

- `pos_doc` must be **non-empty**.
- If training requests negatives (e.g., `n_passages > 1`), `neg_doc` must contain **at least one** document.

## Normalized VL Retrieval Data

For large vision-language retrieval corpora, the default corpus-id path can spend a long time building Hugging Face
Arrow caches before training starts. To make this preprocessing explicit and portable, create a normalized Arrow bundle
in a CPU job:

```bash
python tools/retrieval/prepare_normalized_vl_retrieval_data.py \
  --config /path/to/retrieval_config.yaml \
  --output-dir /path/to/normalized_vl_retrieval \
  --resume
```

The normalized bundle keeps the same corpus-id data model as the original dataset:

- `train/*.arrow` stores query rows and positive/negative document IDs.
- `corpus/*/*.arrow` stores each referenced document/image once.
- Training still resolves `doc_id -> document` through the same retrieval transform.

This avoids rebuilding Hugging Face caches in every GPU job and avoids duplicating document/image payload in every
training row.

Train from the normalized output by replacing the dataset factory:

```yaml
dataloader:
  dataset:
    _target_: nemo_automodel.components.datasets.llm.make_normalized_retrieval_dataset
    data_dir_list: /path/to/normalized_vl_retrieval
    n_passages: 5
```
