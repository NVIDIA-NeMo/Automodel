# Qwen-Image DMD2

This example keeps data loading, FSDP2, optimizers, gradient accumulation, and
checkpointing in NeMo AutoModel. A thin Qwen-Image adapter connects those
framework pieces to NVIDIA Model Optimizer's DMD2 losses, pipeline,
discriminator, R1, backward simulation, and EMA implementation.

Install the DMD2 dependency. Add AutoModel's media extra when you also need to
build the image cache:

```bash
uv sync --extra dmd2 --extra diffusion-media
```

Build the image/text cache with AutoModel's native Qwen-Image processor. See:

```bash
uv run --extra dmd2 --extra diffusion-media \
  python -m tools.diffusion.preprocessing_multiprocess image --help
```

When `guidance_scale` is set, generate one static negative-prompt embedding
with the same model/text encoder used for the cache. From the repository root:

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
    embed, mask = pipeline.encode_prompt(prompt="", device=device)
embed = embed.detach().cpu().to(torch.bfloat16).squeeze(0)
if mask is None:
    mask = torch.ones(embed.shape[0], dtype=torch.long)
else:
    mask = mask.detach().cpu().to(torch.long).squeeze(0)
torch.save({"embed": embed, "mask": mask}, output)
PY
```

Set the three `PATH_TO_*` values in
`examples/diffusion/dmd2/qwen_image_dmd2.yaml`, adjust `dp_size` and global
batch size for the launch, then run:

```bash
uv run --extra dmd2 torchrun --nproc-per-node=8 \
  examples/diffusion/dmd2/train.py \
  --config examples/diffusion/dmd2/qwen_image_dmd2.yaml
```

The example mirrors the validated Qwen-Image recipe: CFG 4.0, four student
steps, a 1:4 student-to-fake update pattern, and GAN weight 0.03. It uses the
2e-6 student/fake-score/discriminator learning rate from that run. R1 and EMA
are fully supported but disabled in this YAML because the validated production
run disabled both. Exact-resume EMA currently requires `fsdp2: true` and
`mode: full_tensor`; it is expensive for Qwen-Image, so enable it only when the
extra full-model shadow fits on every rank.

The checkpoint includes the student, fake-score model, both model optimizers,
the discriminator and its optimizer, optional EMA, the student-update counter,
RNG, dataloader/sampler, LR scheduler, and outer step scheduler. The frozen
teacher and static negative embedding are reloaded from their configured
sources.
