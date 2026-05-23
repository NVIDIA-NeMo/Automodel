# Serving an Automodel-trained EAGLE drafter with SGLang

The EAGLE / EAGLE-3 drafters trained by the recipes under
`nemo_automodel/recipes/llm/train_eagle{1,2,3}.py` are saved in a recipe-native
layout (`draft_model.pt` plus metadata). The helper script below converts that
checkpoint into an HF/SGLang-readable `model/` directory on first launch, so
you can point it directly at `epoch_<E>_step_<S>`.

## Install SGLang

SGLang is **not** bundled with the NeMo-AutoModel container image and is not
declared as a dependency in `pyproject.toml`. Install it yourself in the same
Python environment where you ran training:

```bash
uv pip install "sglang>=0.5.9"
```

See https://github.com/sgl-project/sglang for the version matching your CUDA
and PyTorch stack.

## Locate the checkpoint

A successful training run leaves the recipe checkpoint at:

```
<output_dir>/checkpoints/epoch_<E>_step_<S>/
  config.json
  draft_model.pt
  eagle3_meta.pt   # EAGLE-3 only, contains the small-vocab token map
  eagle1_meta.pt   # EAGLE-1 / EAGLE-2 only, holds bookkeeping (no token map)
```

On first launch the helper exports:

```
<output_dir>/checkpoints/epoch_<E>_step_<S>/model/
  config.json                # architectures rewritten to LlamaForCausalLMEagle3 for EAGLE3
  model.safetensors
  speculative_token_map.pt   # EAGLE-3 only, materialized from selected_token_ids
```

The helper accepts either the outer `epoch_<E>_step_<S>` directory or the
inner `model/` directory. If you point at the inner `model/` and the
`speculative_token_map.pt` is missing, the helper will look one directory up
for `eagle3_meta.pt` and regenerate the token map from it.

## Launch the server

Use the helper module to start an SGLang HTTP server with the trained drafter:

```bash
python -m nemo_automodel.components.speculative.serve_sglang \
    --target meta-llama/Llama-3.1-8B-Instruct \
    --draft ./checkpoints/epoch_0_step_1000 \
    --algorithm EAGLE3 \
    --num-steps 3 --topk 1 --num-draft-tokens 4 \
    --port 30000
```

Add `--print-only` to inspect the resolved `sglang.launch_server` command
without executing it. In that mode the helper **skips the checkpoint export
step entirely** — the printed `--speculative-draft-model-path` reflects the
path that *would* be produced on a real launch. Pass any extra SGLang flags
after a `--` separator, e.g.:

```bash
python -m nemo_automodel.components.speculative.serve_sglang \
    --target meta-llama/Llama-3.1-8B-Instruct \
    --draft ./checkpoints/epoch_0_step_1000 \
    -- --enable-torch-compile --schedule-conservativeness 1.2
```

## Smoke-test the server

```bash
curl http://localhost:30000/generate \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello, my name is", "sampling_params": {"max_new_tokens": 64}}'
```

The acceptance length (and the resulting tokens/sec speedup) depends on
`--num-steps`, `--topk`, and `--num-draft-tokens`. Sweep them via
`benchmarks/` in SpecForge or the SGLang bench suite to find the best
configuration for your workload.

## Notes & caveats

- **dtype must match training.** Pass `--dtype bfloat16` if you trained the
  drafter in bf16. Mixed precision between target and drafter degrades
  acceptance.
- **EAGLE vs EAGLE3.** Use `--algorithm EAGLE3` for drafters trained with
  `train_eagle3.py`; use `EAGLE` for drafters trained with `train_eagle1.py`
  or `train_eagle2.py`.
- **Tensor parallelism.** `--tp-size` controls SGLang's TP. The drafter
  itself is a single-layer transformer so TP only meaningfully shards the
  target model.
- **Custom target architectures.** Pass `--trust-remote-code` if the target
  model relies on `auto_map` entries in its config.
