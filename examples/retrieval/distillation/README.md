# Retrieval Distillation (Automodel)

Edit `nemotron3_embed_1b_distill.yaml` to set your dataset paths, checkpoint directory, model settings, and hyperparameters before running.

You can run the distillation training via the command below:

```bash
automodel --nproc-per-node 8 examples/retrieval/distillation/nemotron3_embed_1b_distill.yaml  
```

Training writes checkpoints under `<checkpoint_dir>` in two formats:

- `epoch_<E>_step_<S>/...` standard Automodel checkpoints
- Legacy-compatible HF export under `<checkpoint_dir>/step_<S>/`:
  - `<checkpoint_dir>/step_<S>/student/` — student encoder checkpoint
  - `<checkpoint_dir>/step_<S>/projection.pt` — distillation projection head (sidecar)
