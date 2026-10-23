# Retrieval Distillation (Automodel)

`nemotron3_embed_1b_distill.yaml` distills the pretrained
[`nvidia/Nemotron-3-Embed-1B-BF16`](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16)
student from the frozen
[`nvidia/Nemotron-3-Embed-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16)
teacher. The student tokenizer is shared by both encoders; the projection maps
the student embeddings to the teacher's 4,096-dimensional space.

The recipe uses the public `nvidia/embed-nemotron-dataset-v1` dataset and saves
checkpoints under `checkpoints/nemotron3_embed_1b_distill`. Adjust the dataset,
checkpoint directory, and hyperparameters as needed before running.

You can run the distillation training via the command below:

```bash
automodel --nproc-per-node 8 examples/retrieval/distillation/nemotron3_embed_1b_distill.yaml  
```

Training writes checkpoints under `<checkpoint_dir>` in two formats:

- `epoch_<E>_step_<S>/...` standard Automodel checkpoints
- Legacy-compatible HF export under `<checkpoint_dir>/step_<S>/`:
  - `<checkpoint_dir>/step_<S>/student/` — student encoder checkpoint
  - `<checkpoint_dir>/step_<S>/projection.pt` — distillation projection head (sidecar)
