# Qwen-Image DMD2

This example runs DMD2 through AutoModel's standard diffusion fine-tuning
trainer. AutoModel continues to own data loading, FSDP2, optimizers, gradient
accumulation, logging, and checkpoint cadence. The top-level `dmd2` block in
the YAML selects Model Optimizer's DMD2 config, Qwen-Image pipeline,
discriminator, and feature hook.

Install the DMD2 dependency:

```bash
uv sync --extra dmd2
```

The negative-prompt embedding is required by CFG, not by DMD2 itself. Because
this example uses `guidance_scale: 4.0`, generate it once with the same
Qwen-Image text encoder used for the data cache. Save a single canonical
floating-point tensor with shape `[sequence, hidden]`; do not save a mapping or
mask. From the repository root:

```bash
uv run --extra dmd2 python - <<'PY'
import torch
from tools.diffusion.processors.qwen_image import QwenImageProcessor

model = "Qwen/Qwen-Image"
output = "/path/to/negative_prompt_embedding.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = QwenImageProcessor()
pipeline = processor.load_models(model, device)["pipeline"]
with torch.no_grad():
    embed, _ = pipeline.encode_prompt(prompt="", device=device)
embed = embed.detach().cpu().to(torch.bfloat16)
if embed.ndim != 3 or embed.shape[0] != 1:
    raise ValueError(f"Expected [1, sequence, hidden], got {tuple(embed.shape)}")
torch.save(embed.squeeze(0).contiguous(), output)
PY
```

Set the three `PATH_TO_*` values in
`examples/diffusion/dmd2/qwen_image_dmd2.yaml`, adjust `dp_size` and
global batch size for the launch, then use the existing fine-tuning entry
point:

```bash
uv run --extra dmd2 torchrun --nproc-per-node=8 \
  examples/diffusion/finetune/finetune.py \
  --config examples/diffusion/dmd2/qwen_image_dmd2.yaml
```

The example mirrors the validated Qwen-Image recipe: CFG 4.0, four student
steps, a 1:4 student-to-fake update pattern, and GAN weight 0.03. It uses the
2e-6 student/fake-score/discriminator learning rate from that run. R1 and EMA
are fully supported but disabled in this YAML because the validated production
run disabled both. EMA requires additional student shadow storage, so enable it
only when that memory cost fits the launch.
