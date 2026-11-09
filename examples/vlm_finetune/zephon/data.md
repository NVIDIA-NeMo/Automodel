# Prepare image documents for Zephon

A Zephon document must contain the conversation and its image payloads. This is
about storage, independently of sequence packing: the processor expands image
tokens and Zephon packs complete documents online.

Native AutoModel datasets can read JSON/JSONL conversations that reference image
files or URLs. That workflow remains available with the native loader. With
Zephon, `fetch()` owns source downloads, caching and eviction; opening those
references later would bypass its cache. Export path-only data before training.
If your Parquet/HF or LitData records already contain image bytes, no storage
conversion is necessary. A native dataset class's arbitrary Python transforms
are not applied by selecting its path in a Zephon configuration.

## Recommended layout

Use self-contained Parquet shards with bounded row groups. Each record can use:

- `conversation` or `messages`: role/content messages with text and image blocks.
  An image block is `{type: "image", image: <JPEG/PNG bytes>}`.
- `conversations`: ShareGPT `from`/`value` messages, plus `image` or `images`.
  Put `<image>` markers at the intended positions in the text.
- `question`, `answer`, and `image` or `images`. Images precede the question.

Separate image columns also work with structured `{type: "image"}` placeholders.
Images bind to placeholders in encounter order. Missing or unused images are
errors; images are never silently appended or discarded. JSON-encoded message
arrays are accepted. Text-only documents are accepted as well.

Image values may be encoded JPEG/PNG bytes, HF `{bytes, path}` structs, or
materialized PIL images in a fetched Python record. If both HF fields exist,
bytes win and path is ignored. Paths/URLs alone are rejected even with
`on_media_error: drop`; that option handles corrupt embedded media, not schema
errors. Video and audio are unsupported.

For nonstandard column names, configure each source independently:

```yaml
sources:
- name: captions
  path: /data/captions
  input:
    column_mapping:
      messages: dialog
      images: photos
```

In Python, import `ZephonVlmSource` and `VlmInputOptions` from
`nemo_automodel.components.datasets.zephon`.
`ZephonVlmSource(name="images", path="…", input=VlmInputOptions(normalizer=my_normalizer))`
accepts a deterministic module-level callable for specialized encodings. It
receives only the fetched payload and returns a materialized `ConversationSample`;
it must not open files or URLs. This keeps filesystem ownership with Zephon.
Container blobs (for example a length-prefixed array of encoded images) need
such a normalizer to unwrap them; a single image field is interpreted as one
encoded image, not an arbitrary binary container.

## Export existing JSONL and local image files

The small converter below supports the schemas above and resolves local image
paths relative to `--media-root` (the JSONL directory by default):

```bash
uv run python examples/vlm_finetune/zephon/prepare_data.py conversations.jsonl /data/embedded --media-root /data/images
```

It streams the JSONL, validates image placement, materializes RGB images and
writes PNG bytes into bounded Parquet shards. It does not download URLs. For
large corpora, export similarly from your existing dataset builder; avoid one
image per tiny shard and tune row-group sizes to your image sizes. HF datasets
can store `datasets.Image` columns with embedded bytes when exporting to Parquet;
check that exported `bytes` are populated rather than retaining only paths.

Source formats are normally auto-detected; use `fmt` only when discovery needs
an override. Zephon cache options are described in the
[integration guide](../../llm_pretrain/zephon/README.md). `cache.cache_limit_bytes`
bounds downloaded source shards; `cache.parquet_rg_cache.limit_bytes` bounds
decoded row groups separately. Neither bounds processor tensors in memory.
See the [Zephon User Guide](https://docs.zephon.io/) for storage, cache and API
details.
