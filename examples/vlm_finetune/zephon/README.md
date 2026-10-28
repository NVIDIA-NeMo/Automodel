# Zephon image VLM training

The existing `FinetuneRecipeForVLM` accepts `ZephonVlmSourceConfig` and
`ZephonVlmDataloaderConfig`. Image models include Qwen2.5-VL, native dense
Qwen3-VL, and Mistral3 with a Mistral/Ministral3 text backbone without sliding
attention, using SDPA/eager or
HF Flash attention. The ordinary `processor` section and optional
`dataset.chat_template` configure both the parent processor and each Zephon
worker. A worker receives configuration-only model position state, never an
`nn.Module` or a bound model method.

From the repository root, with AutoModel installed:

```bash
uv run python examples/vlm_finetune/zephon/prepare_images.py
uv run automodel examples/vlm_finetune/zephon/qwen2_5_vl.yaml --nproc-per-node 1
```

The example needs CUDA and access to the named pretrained model. Its source is
finite; `stop_after_passes: 1` does not repeat the corpus. `max_steps` caps an
unsized stream and does not require it to contain that many optimizer steps.
Incomplete final microbatches are dropped by the shared Zephon training loader.
First-fit/best-fit packing drops oversized documents by default;
`drop_oversized: false` raises instead. The limit is `seq_len + 1`, less any
reserved fake-image segment described below. The processor never truncates an
image into a partial set of image tokens.

For a local scratch Mistral3 model and processor with no model downloads:

```bash
uv run python examples/vlm_finetune/zephon/prepare_mistral3.py
uv run automodel examples/vlm_finetune/zephon/mistral3_tiny.yaml --nproc-per-node 4
```

This GPU recipe uses TP2 × PP2, a genuinely untied tiny Ministral3/Pixtral
configuration, and a finite corpus containing text-only, single-image, and
multiple-image documents. It trains from random weights. The loader never changes
weight tying: AutoModel's native PP validator rejects tied input/output weights,
while ordinary data preparation also supports tied variants. Model and processor
configuration must match when using a pretrained checkpoint.

For dense Qwen3-VL, reuse the Qwen example's Zephon dataset and dataloader
sections, and set the model to `Qwen/Qwen3-VL-4B-Instruct` (or another dense
Qwen3-VL checkpoint). The ordinary processor section loads the matching
processor from that model. AutoModel selects its registered native
`Qwen3VLForConditionalGeneration`, retaining the vision tower's DeepStack
injections; workers reuse the same model-owned mRoPE calculation with only its
configuration. No additional Zephon fields or packing options are needed.
Native dense Qwen3-VL currently declares TP and PP unsupported. Those model
restrictions are separate from loader support; CP remains outside this
integration. Qwen3.5 and Qwen3-VL MoE are not selected by this family entry.

## Gemma4 dense images

The same `ZephonVlmDataloaderConfig` supports Gemma4's dense image variant
(the 31B architecture), using its native model implementation and SDPA/eager.
Select the Gemma4 checkpoint in the ordinary `model` and `processor` sections;
keep the same `dataset`, `dataloader.zephon` source and bin-packing settings as
`qwen2_5_vl.yaml`. The processor expands image tokens and supplies
`image_position_ids`; workers compute scalar decoder positions from configuration.
No additional Gemma-specific data options are required.

The model combines the adapter's document mask with Gemma4's native policy:
full-attention layers stay causal; sliding layers retain their window and
same-image bidirectional attention. Xdoc changes document separation without
merging image groups or changing labels or media. Model position ranges are
joined with the same geometry-preserving translation used for text. Audio, MoE,
per-layer-input embeddings, shared-KV variants, and fully bidirectional text
objectives are not covered by this image integration. HF Flash compact input
support is not implemented: the standard vision variant requires preserving
its native image/sliding policy. Native FA/TE layouts are not certified here.

Gemma4's tied embedding/head weights remain tied. AutoModel's current native
pipeline validator rejects those ties, and its native constructor also rejects
scratch untied configurations. This family therefore has no supported PP claim;
local wrapper parity tests do not override those model restrictions. The
Gemma4 coverage tests processor expansion, dense attention parity and worker
checkpoint replay independently of TP/PP.

## Batch output format

`dataloader.output_format: batched` is the default. It supports the existing
model-specific mask/Flash routes; the label does not name a packing algorithm.
`output_format: thd` adds native `seq_lens`/`seq_lens_padded` metadata in the same
final batch map used for text, preserving expanded image tokens, mRoPE and pixels.
The model/backend must accept native THD input; Qwen3-VL's current implementation,
for example, declares THD unsupported. THD is not required to use Flash Attention.

THD requires `dataset.inject_fake_images: false` and contiguous eligible content
with suffix padding. Processor-excluded real-token slots are rejected because
lengths alone cannot encode their attention policy. VLM pipeline-parallel media
and multi-axis position chunking are not supported by this THD adapter; use
`batched` for VLM PP. Existing batched TP/PP support is unchanged. CP remains
unsupported in both formats. Image-loader THD schema/position parity is tested on
CPU; CUDA TE execution still requires GPU validation.

## Input and packing

Each JSONL document uses AutoModel's native conversation format:

```json
{"conversation": [{"role": "user", "content": [{"type": "image", "image": "/data/photo.png"}, {"type": "text", "text": "Describe the image."}]}, {"role": "assistant", "content": [{"type": "text", "text": "A blue square."}]}]}
```

Image paths are resolved by the existing processor-side media loading logic;
absolute paths avoid worker working-directory dependence. Multiple images and
variable image resolution are supported. By default, each all-text pack gets
one leading fake-image segment so the vision tower participates in sharded
training. A pack containing any real image gets no fake image, even if it also
contains text-only documents. `dataset.inject_fake_images: false` omits this
segment and its reservation, and is appropriate only when the model/parallel
setup permits skipping the vision tower.

The effective processor encodes AutoModel's existing 56×56 white fake image and
its template once during loader construction; its own resize and minimum-grid rules determine
`fake_len`. Every bin reserves that static capacity by packing corpus documents
to `seq_len + 1 - fake_len`. All-text packs prepend the complete synthetic
segment; image-containing packs leave the reservation unused. This costs at most
`fake_len / (seq_len + 1)` of token capacity and can lower the oversized-document
threshold. All synthetic labels are ignored, and the entire synthetic segment
is excluded from real-token attention. Original corpus token/sample counts and
the configured `max_num_documents` cap count corpus documents only. The extra
synthetic segment retains its structural document ID but has zero attention
IDs. It never contributes to source-mixture counts.

The stages are raw conversations → native unshifted processor encoding →
`pack_sequences(first_fit|best_fit, pack_payloads=merge)` → `batch`.
The loader then calls `to_training` and applies attention/output adaptation. Packing measures
expanded language image-token slots, not placeholder count or text-token count.
There is no user-facing flat-packing switch and no wrap packing for images.
`to_training` owns token/label shifting and padding. Field callbacks assemble
pixels/grids and pad/slice already computed scalar or mRoPE coordinates.
Mistral3 retains each image's original `image_sizes` while zero-padding pixels to
the largest height/width in the batch; its native vision model uses those sizes.
Callbacks neither pick bins nor recreate the token collator. Shared media
assembly preserves processor precision; the model owns its input dtype conversion.

Spatial padding can inflate Pixtral pixel storage: the tiny Mistral test's 8×8
and 8×12 images require 1.2 times their original elements, while an 8×2 and 2×8
pair requires 4 times. Prepared packs remain tensors and may be padded again at
batch assembly. Grouping similar image shapes in sources reduces this cost;
there is no claim of an optimized ragged-image layout here.

`tokenize_parallelism` controls the complete VLM processor encoding stage,
including image decoding, expansion and position preparation. It defaults to
three workers per training rank; `fetch_parallelism` defaults to two. Tune both
according to per-rank CPU and memory capacity. Each independent process worker
caps PyTorch intra-op threads and sets `OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1`.
Inline and thread runners leave the trainer's threading environment alone.
`pack_parallelism` controls the packer. Its merge callback concatenates the
selected subdocuments and preserves scalar boundary positions. Zephon recognizes
the merged envelope in `to_training`, so no payload-extraction map is needed.
`map_parallelism` is unused by this image path. Collation and attention adaptation
run in the loader consumer; there is no collation-worker setting. Bin-packed output is shuffled by default;
`shuffle_after_pack: false` disables that final shuffle. No second encoder-worker
setting is needed.

`dataset.on_media_error` defaults to `"raise"`. Set it to `"drop"` to skip
unreadable media and explicit expanded-media-token mismatches deterministically.
The decision depends only on the record and processor settings; no replacement,
drop-fraction threshold, or worker-local counter changes the delivered stream.
The operator emits Zephon tombstones for closing source contributions and logs
one bounded warning per input group, including sample IDs and decode reasons.
Invalid conversations and general processor/template failures still raise,
including unrelated `OSError`s outside media loading. The separate
`drop_oversized` policy concerns successfully encoded document length.

`dataloader.zephon` uses the same source, execution, checkpoint, and optional
`data_config` TOML settings as the text integration. Inline values override TOML
values; relative source/cache/checkpoint paths in a TOML file are relative to
that file. Source objective selection belongs to the dataset target. VLM does
not expose a second tokenizer or special-token policy: its processor/template
owns those choices. PAD resolution reuses the text integration's vocabulary-ID
rules, and padding eligibility is derived from actual lengths even if PAD equals
EOS.

## Attention and checkpointing

`allow_cross_document_attention: false` preserves document separation. Setting
it to `true` joins the adapter's compact model-facing attention IDs after
`to_training` and places each subdocument's model coordinates after the preceding
subdocument's maximum coordinate. The same scalar offset is added to every token
and every RoPE axis of that subdocument, preserving local image/text geometry.
This is the same policy as continuous positions for text, without replacing
mRoPE with a token-index `arange`. Zephon remains model-independent; the existing
model-owned position factory supplies local coordinates. Supervision, labels,
padding eligibility and retained subdocument boundaries stay unchanged. Processor exclusions, including fake
images, remain excluded in either mode. Host attention inputs stay linear in
[batch, sequence]. Separated documents and processor exclusions use compact
attention IDs; the model entry hook materializes a boolean SDPA mask or additive
eager mask on the model device. Ordinary dense xdoc with only trailing padding
uses an all-true 2D mask, selecting HF's normal causal path without inferring
document breaks from rotary positions. Flash retains joined compact IDs for
explicit cumulative metadata, and Gemma keeps them for its native image/window
policy. HF Flash uses the
same document boundaries, materialized as cumulative lengths at decoder entry
after any PP microbatch split. Both backends receive the same joined scalar/mRoPE
coordinates and unchanged media inputs. Its contiguous
segments cannot preserve attention across processor-excluded tokens inside a
subdocument, so those inputs fail explicitly rather than splitting that document.
Excluded prefixes and trailing padding are supported. Xdoc additionally requires
one contiguous eligible span per packed document. Default fake-image preparation
uses its own leading, excluded segment and therefore supports mixed text/image
corpora with Flash in either xdoc mode. Arbitrary processor-excluded holes inside
a real document still require SDPA/eager. No fake slots are unmasked. The separate
`mask_segment_starts` option retains
Zephon's existing boundary-label semantics.

The same `ZephonDataLoader` as the LLM route owns iteration, closing, opaque state
and restore. Every rank participates in checkpointing; distributed runs require
a unique `run_id`. With an initialized training process group, `torch://` is the
default aggregate store; explicit `aggregate_store`/`aggregate_dir` remains
available. A fresh loader can
restore a checkpoint before iteration. Reaching a cap, source exhaustion or a
training exception closes owned workers.

## Current scope and validation

This route rejects unsupported model position factories, video processors,
wrap packing, native AutoModel packing/collator configuration, CP,
and Zephon validation loaders.
Native validation loaders remain available. TP/PP use native model capabilities;
Qwen2.5-VL is still rejected by AutoModel's native PP model validator. Models
with a supported vision-aware pipeline forward use the shared native media
handoff after `to_training`: image counts belong to each packed document, and
pixels/image metadata follow the same pipeline microbatch. Token positions,
modality IDs and masks remain model inputs. Every TP/PP peer with the same DP
coordinate reads identical data and every global rank checkpoints.

Pretokenized/prepacked VLM input
needs a separately validated media/position format before it can be accepted.
VLM token-aware source calibration is not certified; text routes provide token
estimation. Multiple VLM sources require explicit `token_estimation: false`.

Run CPU checks without model/tokenizer downloads:

```bash
uv run pytest tests/unit_tests/datasets/zephon/test_vlm.py tests/unit_tests/datasets/zephon/test_vlm_config.py -q
uv run pytest tests/integration/zephon -q
```

The tests use a real local Qwen processor and tiny real Qwen model. They exercise
expanded image counts, multiple images, variable resolution, text-only input,
assistant supervision, first/best-fit packing, process-worker replay, shared
loader checkpoint/restore, finite loss and nonzero vision gradients. Actual
model perturbation tests verify attention isolation and xdoc influence for
SDPA and eager while checking translated coordinate geometry, unchanged
supervision and opt-in original membership for Gemma image groups.
A CPU SDPA oracle also consumes the public Flash cumulative-length kwargs that
reach real Qwen2.5/Qwen3/Mistral3 decoder attention, comparing valid-token logits,
loss and gradients with dense attention. This verifies metadata handoff, not GPU Flash
kernels.
Native Qwen3-VL tests preserve two real DeepStack injections and compare packed
logits, summed loss, and every parameter gradient with separate-document
training; real process-worker checkpoint replay checks its model-owned state.
Mistral3 tests additionally compare actual two-rank native TP and PP outputs,
loss, and all parameter gradients against an unsplit CPU model. PP runs two
microbatches with empty-first media, mixed image counts/resolutions, and both
xdoc settings. They use the public AutoPipeline validator and schedule.
CUDA/FSDP execution, the four-GPU combined TP2 × PP2 recipe, throughput, and flash
kernels still require hardware validation; CPU tests do not certify those paths.
The loader consumer runs collation after Zephon batching. Upstream processing
still uses the configured runner and operator workers. Throughput measurements should separate batch spatial padding
and pixel concatenation from decode/encoding; no end-to-end speedup is claimed.

Zephon's static validator emits `OP_INSTANCE_RUNTIME_CHECKS_SKIPPED` because it
does not call `BaseOp.setup`, where the worker processor is initialized. The
warning is retained. Separate initialized-operator tests verify singleton vs.
group execution, placeholder-metadata invariance, replay, sentinel preservation
and lack of input mutation; actual process-worker replay covers serialization.


The image loader receives the same model-derived `return_*` contract as text.
Batched output interprets its mask interface and `return_packed_sequence_metadata`
request. THD skips both and derives native lengths; `return_document_ids` can
retain original membership in either format. Contract queries do not install
model hooks or validate packed backends.

The loader converter requests `document_ids` from Zephon, using scalar boundary
positions separately from model-owned multi-axis RoPE. It preserves original
membership and applies processor exclusions only to `attention_document_ids`.
The loader yields selected membership as AutoModel's `_packed_seq_ids`. Original
`document_ids` are retained only for models that request them (Gemma image grouping); it does not rebuild IDs, recollate tensors, or copy the dictionary.
The original SampleBatch and all contributor references/counts stay unchanged;
the loader creates no replacement record or batch and does not merge replay metadata. Dense masks and Flash metadata remain model-entry work
after PP splitting. See the text recipe's [input-interface explanation](../../llm_pretrain/zephon/README.md#document-membership-and-automodel-input-interfaces)
for the gathered-token metadata handoff. Qwen3.5/Kimi text support does not register
new image processors; these families still need model-owned image preparation
before they can be selected by this VLM adapter.
