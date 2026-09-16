# Inkling fine-tuning recipes

NeMo AutoModel supports full-parameter fine-tuning for the Inkling multimodal
Mixture-of-Experts model family.

| Recipe | Model | Dataset | Parallelism |
|---|---|---|---|
| [`inkling_medpix.yaml`](./inkling_medpix.yaml) | Inkling | MedPix-VQA | FSDP2, PP8, EP32 |
| [`Inkling_small_medpix_ep64.yaml`](./Inkling_small_medpix_ep64.yaml) | Inkling-Small | MedPix-VQA | FSDP2, EP64 |

## Inkling-Small

[Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small) is a
276B-parameter, 12B-active model. The default topology uses 8 nodes with 8 H100
GPUs per node:

```bash
uv run automodel --nproc-per-node=8 examples/vlm_finetune/inkling/Inkling_small_medpix_ep64.yaml
```

Submit the command through the cluster launcher so all eight nodes receive the
same repository, checkpoint, and dataset paths.

The recipe uses:

- FSDP2 with activation checkpointing
- EP64 with HybridEP dispatch and grouped `torch_mm` experts
- no context or pipeline parallelism
- microbatch size 1 and global batch size 64
- MedPix-VQA padded or truncated to 512 tokens
- frozen vision and audio towers

This topology completed 100 training steps on 64 H100 GPUs. The run used about
50.6 GiB of GPU memory after warm-up and finished with validation loss 2.0430.

See the [Inkling model coverage page](https://docs.nvidia.com/nemo/automodel/nightly/model-coverage/vision-language-models/inkling)
for architecture details.
