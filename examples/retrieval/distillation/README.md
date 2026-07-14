# Retrieval Distillation (Automodel)

Run the baseline Stage-1 style recipe:

```bash
automodel --nproc-per-node 8 examples/retrieval/distillation/nemotron3_embed_1b_distill.yaml  
```

The recipe writes both:

- `epoch_<E>_step_<S>/...` standard Automodel checkpoints
- `step_<S>/student` + `step_<S>/projection.pt` legacy-compatible HF checkpoint/sidecar


