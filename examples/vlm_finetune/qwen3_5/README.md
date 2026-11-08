# Qwen3.5 packed full finetuning

These recipes train the language model and vision tower on MedPix-VQA with
FSDP2 and NEAT sequence packing. Choose the attention kernels for your GPU:

| Recipe | GPUs | Native text full attention | HF vision attention | Pack size |
| --- | --- | --- | --- | --- |
| `qwen3_5_4b_neat_packing.yaml` | Hopper (H100/H200) | Transformer Engine (TE) | FlashAttention 2 | 4096 |
| `qwen3_5_4b_neat_packing_b300.yaml` | One Blackwell B300 | FlashAttention 4 | FlashAttention 4 | 16384 |

Use an AutoModel environment that includes the selected kernels. For the B300
recipe, FlashAttention 4 and its matching CUTLASS dependencies must be installed.
The B300 recipe is a starting configuration; its throughput and peak memory have
not been measured on B300.

```bash
# Eight Hopper GPUs.
automodel examples/vlm_finetune/qwen3_5/qwen3_5_4b_neat_packing.yaml --nproc-per-node 8

# One B300, with activation checkpointing enabled to reduce activation memory.
automodel examples/vlm_finetune/qwen3_5/qwen3_5_4b_neat_packing_b300.yaml --nproc-per-node 1
```

`model.backend.attn` selects the native full-attention backend in the text
backbone. `model.attn_implementation` selects HF attention, including the retained
vision tower. These recipes use the native model; `force_hf` is unnecessary.
Qwen3.5 also contains gated linear-attention layers, which use their own kernels.
MTP auxiliary attention currently uses SDPA even when the backbone uses TE or FA4.

TE and FA4 both preserve document boundaries with `packing_format: neat` for
this model. Keep this setting when switching the native backend; Qwen3.5 does
not accept an externally flattened `packing_format: thd` batch. To try native TE
with the B300 recipe while retaining FA4 in the vision tower:

```bash
automodel examples/vlm_finetune/qwen3_5/qwen3_5_4b_neat_packing_b300.yaml \
  --nproc-per-node 1 --model.backend.attn te
```

`packed_sequence.max_length: 4096` limits each pretokenized sample;
`packed_sequence.pack_size: 16384` in the B300 recipe combines samples into 16K
packs. Adjust the sample limit separately if longer examples should be retained.
Four persistent training workers prefetch two batches each. Tune these values for
the host's CPU and memory capacity. If setting `dataloader.num_workers: 0`, also
set `dataloader.persistent_workers: false` and `dataloader.prefetch_factor: null`.

Both recipes use TE FusedAdam with decoupled weight decay and FP32 Adam moments.
The resident parameters remain FP32 and serve as master weights, so no separate
master-weight copy is enabled. FSDP2 uses its default mixed-precision policy. See the
[mixed-precision guide](../../../docs/guides/mixed-precision-training.mdx) before
changing storage or optimizer precision for a throughput comparison.

MTP follows the pretrained checkpoint configuration in both recipes. AutoModel
trains its auxiliary prediction layers when present. When comparing against a
baseline that omits MTP training, explicitly disable it in the comparison run:

```bash
automodel examples/vlm_finetune/qwen3_5/qwen3_5_4b_neat_packing_b300.yaml \
  --nproc-per-node 1 --model.num_nextn_predict_layers 0
```

Match MTP, precision, batch size, image resolution, packing, and checkpointing
between implementations before comparing tokens per second. Disabling MTP
changes the training objective; it is not the default for these recipes.
