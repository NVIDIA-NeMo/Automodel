# Train a Qwen3.5-4B MMFlash draft from a single-image cache

`TrainDFlashRecipe` reads a cache when `recipe_args.train_cache_path` is set.
Set `val_cache_path` to a separate cache for validation. AutoModel loads only
the target's frozen input embedding and LM head for training. Each cache
records the expanded token sequence, assistant loss mask, target-layer
activations, and visual scores indexed by **label token**. The same draft,
optimizer, and checkpoint implementation serves ordinary DFlash and MMFlash.

The [example YAML](qwen3_5_4b_mmflash_cached.yaml) uses block size 16, target layers
`[1, 8, 15, 22, 29]`, mask token 248070, and the released draft's five layers,
32 attention heads, eight KV heads, head dimension 128, and FFN width 9728. The cached
feature width is `5 × 2560 = 12800`. The example's 32 anchors and single epoch
keep the smoke run small. [Author draft config](https://github.com/fscdc/MMFlash/blob/5b3fa3fe28f5bc6d7dfe75934a156c52b865dc8c/configs/qwen3.5-4b-mmflash.json)

## Produce a small cache

Use one CUDA device and a work directory with space for the Qwen checkpoint.
The following commands regenerate a few single-turn ChartQA answers with the
pinned target, then run two teacher-forced forwards per image. The scorer
comes from the pinned SpecForge branch in a separate environment. [Cache producer](https://github.com/sgl-project/SpecForge/pull/1225)

```bash
export WORK="$PWD/output/mmflash-smoke"
export TARGET="$WORK/target"
mkdir -p "$WORK"
git clone https://github.com/LUOseu/SpecForge.git "$WORK/SpecForge"
git -C "$WORK/SpecForge" checkout --detach b78a85c8035a91dea74ae97706ee119b74592034
uv venv --python 3.12 "$WORK/scorer-env"
SCORER="$WORK/scorer-env/bin/python"
(
  set -euo pipefail
  cd "$WORK/SpecForge"
  uv pip install --python "$SCORER" -e . "torchvision==0.29.2"
  "$WORK/scorer-env/bin/hf" download Qwen/Qwen3.5-4B \
    --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir "$TARGET"
  "$SCORER" examples/mmflash/prepare_chartqa.py --model "$TARGET" --out "$WORK/chartqa" --train 6 --heldout 1
  "$SCORER" - <<'PY'
import os
from pathlib import Path
import yaml

work = Path(os.environ["WORK"])
cfg = yaml.safe_load(Path("examples/configs/offline/colocated/qwen3.5-4b-mmflash-a100.yaml").read_text())
cfg["model"]["target_model_path"] = os.environ["TARGET"]
cfg["data"]["hidden_states_path"] = str(work / "scored")
(work / "scorer.yaml").write_text(yaml.safe_dump(cfg))
PY
  "$SCORER" scripts/score_visual_kl.py --config "$WORK/scorer.yaml" \
    --input "$WORK/chartqa/train.jsonl" --scores "$WORK/visual_scores.jsonl"
)
```

For this connectivity check, split the scored rows into separate train and
validation directories. The ChartQA `heldout.jsonl` image remains outside
both caches. The validation and training rows share the scorer's calibration
for this smoke test.

```bash
"$SCORER" - <<'PY'
import json
import os
import shutil
from pathlib import Path

work = Path(os.environ["WORK"])
rows = sorted((work / "scored").glob("*.ckpt"))
if len(rows) < 2:
    raise ValueError("Generate more examples so at least two pass the score filter.")
for split, files in (("train", rows[:-1]), ("val", rows[-1:])):
    dest = work / f"{split}-cache"
    dest.mkdir()
    for src in files:
        shutil.copyfile(src, dest / src.name)
    manifest = {
        "format_version": 1,
        "target_model_name_or_path": os.environ["TARGET"],
        "target_layer_ids": [1, 8, 15, 22, 29],
        "target_hidden_size": 2560,
        "target_vocab_size": 248320,
        "block_size": 16,
        "mask_token_id": 248070,
        "hidden_states_dtype": "bfloat16",
        "visual_score_scale": 10000,
        "num_samples": len(files),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
```

Each unpadded `.ckpt` is a weights-only PyTorch record with `input_ids [S]`,
`loss_mask [S]`, `hidden_states [S, 12800]`, and `visual_score [S]`. The latter
is int64 in units of `1 / 10000`. Image records have zeros off the assistant
mask. The `-1` sentinel represents an unscored text row or right padding.
Features concatenate the zero-based decoder block outputs in manifest order,
before final normalization, along the hidden dimension.
The loader checks the ordered layer IDs, width, target metadata, and shape
alignment. It adds per-row document bounds when padding a batch. The scorer
and recipe must agree on sequence length. Prepare new caches when changing
assistant supervision or the target.

With MMFlash, a supervised prediction with draft probability `p` contributes
`s + (1 - s) * p` to its block's prefix product. The objective multiplies this
product by `1 + alpha * g * (1 - p)` and detaches the resulting weight.
The anchor and masked labels make no contribution to the product or loss.
The final reduction divides the weighted CE sum by the number of sequences.
`visual_alpha: 0` keeps the prefix confidence weights.

## Train both objectives

From the AutoModel checkout, use the same train and validation caches for
both arms. The target path must match the value recorded in each manifest.
Keep this directory pinned to the same immutable snapshot used for capture.
The cached arms use `shuffle_seed` for the same initial draft weights.
The `target_embedding_key` in the example locates the frozen token table in
the target checkpoint. When the target ties the LM head and the checkpoint
omits it, the recipe reuses the embedding for the output projection.

```bash
cfg=examples/speculative/dflash/qwen3_5_4b_mmflash_cached.yaml
for arm in dflash mmflash; do
  uv run python -m nemo_automodel.recipes.speculative_decoding.train_dflash -c "$cfg" \
    --recipe_args.target_model_name_or_path "$TARGET" \
    --recipe_args.train_cache_path "$WORK/train-cache" \
    --recipe_args.val_cache_path "$WORK/val-cache" \
    --recipe_args.loss_type "$arm" \
    --recipe_args.output_dir "$WORK/automodel-$arm" \
    --checkpoint.checkpoint_dir "$WORK/automodel-$arm/checkpoints"
done
```

The recipe writes draft, optimizer, scheduler and RNG state. Use the
corresponding checkpoint directory for recipe resume and the
`model/consolidated` directory for the Transformers draft loader.

## Decode a held-out image

Use AutoModel's native `Qwen3DFlashDraftModel.spec_generate` with a stock
Hugging Face Qwen3.5 verifier. The greedy image mode replays committed text
from the image prefill cache for each verification. This preserves recurrent
state when the verifier rejects tokens. The command checks token agreement
and prints the generated IDs.

```bash
uv run --with torchvision python - <<'PY'
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel

work = Path(os.environ["WORK"])
target_path = os.environ["TARGET"]
row = json.loads((work / "chartqa/heldout.jsonl").read_text().splitlines()[0])
question = row["conversations"][0]["content"].removeprefix("<image>").strip()
processor = AutoProcessor.from_pretrained(target_path)
with Image.open(row["image"]) as image:
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": image.convert("RGB")},
                                     {"type": "text", "text": question}]}],
        tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True, enable_thinking=False,
    ).to("cuda")
target = AutoModelForImageTextToText.from_pretrained(
    target_path, dtype=torch.bfloat16, attn_implementation="sdpa",
).to("cuda").eval()
stop = target.generation_config.eos_token_id
stop = [stop] if isinstance(stop, int) else stop
with torch.inference_mode():
    expected = target.generate(**inputs, max_new_tokens=16, do_sample=False)
    for arm in ("dflash", "mmflash"):
        checkpoints = (work / f"automodel-{arm}" / "checkpoints").glob("epoch_*_step_*")
        latest = max(checkpoints, key=lambda p: int(p.name.rsplit("_", 1)[-1]))
        draft = Qwen3DFlashDraftModel.from_pretrained(
            latest / "model/consolidated", dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to("cuda").eval()
        ids = draft.spec_generate(target, **inputs, max_new_tokens=16, stop_token_ids=stop, temperature=0.0)
        generated = ids[:, inputs.input_ids.size(1):]
        assert torch.equal(ids, expected), f"{arm} disagrees with greedy target decoding"
        print(arm, generated.tolist(), "exact greedy sequence match")
PY
```

Run the fixed-anchor author-parity test without downloading any weights.

```bash
uv run --group test pytest tests/unit_tests/speculative/test_mmflash_objective.py
```
