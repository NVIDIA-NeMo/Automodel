# Step-3.7 text SFT with THD packed sequences

AutoModel's VLM packing pipeline can now consume tokenizer-aware datasets such
as `ChatDataset` without re-tokenizing them through the multimodal processor.

## Data flow

1. `ChatDataset` renders the Step-3.7 chat template and creates unshifted
   `input_ids`, `attention_mask`, and assistant-only `labels`.
2. `PreTokenizedDatasetWrapper` detects those fields and passes them through as
   one-dimensional tensors. It only applies configured truncation.
3. `neat_pack_dataset_vlm` uses exact tokenized lengths for knapsack planning.
4. `PackedDatasetWrapper` applies the causal-LM shift to each document before
   concatenation and gives each document its own attention-mask ID and
   zero-based position IDs.
5. `packed_sequence_thd_vlm_collater` emits `qkv_format: thd`, `seq_lens`, and
   `seq_lens_padded`; Transformer Engine derives `cu_seqlens` from these values.

Documents in the same pack therefore do not attend to each other.

## Config

Smoke config:

```text
examples/vlm_finetune/stepfun/step3p7_single_node_8xh200_text_sft_offload_8k_thd_packed_bs32_smoke.yaml
```

Full-data config:

```text
examples/vlm_finetune/stepfun/step3p7_single_node_8xh200_text_sft_offload_8k_thd_packed_bs32_train.yaml
```

Key settings:

```yaml
dataset:
  seq_length: 8192
  padding: do_not_pad
  truncation: true
  inject_fake_images: false

packed_sequence:
  pretokenize: true
  max_length: 8192
  pack_size: 8192
  packing_ratio: 1.0
  packing_format: thd
  collate_max_length: 8192
```

`local_batch_size` counts packs, not original conversations.

## Validation evidence

On 128 real Step-3.5-Flash-SFT samples, the CPU pipeline produced 54 packs with
99.18% mean utilization and 2.37 documents per pack. A config-built THD batch
had shape `(4, 8192)`, and a multi-document pack emitted sequence lengths
`[7377, 312, 191, 136]`.

GPU forward/backward/optimizer validation must be performed only after the
currently running 8-H200 job releases the GPUs.
