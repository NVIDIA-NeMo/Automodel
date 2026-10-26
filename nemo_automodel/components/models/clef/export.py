# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Merge CLEF adapters and typed heads into the publisher's release layout."""

import json
import logging
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file

from nemo_automodel.components.models.clef.joint_schema_head import save_joint_head
from nemo_automodel.components.models.clef.stage import ClefStageManifest

logger = logging.getLogger(__name__)


def merge_lora_(state_dict: dict[str, torch.Tensor], adapter: dict[str, torch.Tensor], scale: float) -> int:
    """Merge LoRA pairs into ``state_dict`` in place.

    Args:
        state_dict: Base weights by HF name; ``<module>.weight`` tensors have shape [out_features, in_features].
        adapter: ``<module>.lora_A.weight`` of shape [rank, in_features] and ``<module>.lora_B.weight`` of shape
            [out_features, rank].
        scale: LoRA scaling ``alpha / dim``.

    Returns:
        Number of merged modules.

    Raises:
        KeyError: If an adapter module has no matching base weight.
    """
    merged = 0
    for key in adapter:
        if not key.endswith(".lora_A.weight"):
            continue
        module = key[: -len(".lora_A.weight")]
        base_key = f"{module}.weight"
        if base_key not in state_dict:
            raise KeyError(f"no base weight {base_key} for adapter {key}")
        a = adapter[key].float()
        b = adapter[f"{module}.lora_B.weight"].float()
        w = state_dict[base_key]
        state_dict[base_key] = (w.float() + scale * (b @ a)).to(w.dtype)
        merged += 1
    return merged


def export_release(base: str | Path, *, revision: str, stage: Path, out: Path) -> None:
    """Write the trained CLEF release, preserving the publisher's loader and license.

    Args:
        base: Hub repository ID or existing local base release directory.
        revision: Immutable Hub commit ID for remote model files.
        stage: Published native CLEF stage containing DCP head state and optional native PEFT model weights.
        out: Destination for the merged release.
    """
    from huggingface_hub import snapshot_download
    from transformers import Qwen3_5ForConditionalGeneration

    base = Path(base) if Path(base).is_dir() else Path(snapshot_download(base, revision=revision))
    stage, out = Path(stage), Path(out)
    manifest = ClefStageManifest.load(stage)
    if (stage / ".incomplete").exists():
        raise ValueError("cannot export an incomplete stage")
    adapter, scale = {}, 0.0
    if manifest.backbone_adapter:
        meta = json.loads((stage / "model" / "adapter_config.json").read_text())
        if meta.get("use_dora"):
            raise ValueError("CLEF merged export currently supports ordinary LoRA, not DoRA")
        adapter = {
            name.removeprefix("base_model.model."): value
            for name, value in load_file(str(stage / "model" / "adapter_model.safetensors")).items()
        }
        scale = float(meta["lora_alpha"]) / float(meta["r"])

    model = Qwen3_5ForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16)
    state = model.state_dict()
    n = merge_lora_(state, adapter, scale)
    model.load_state_dict(state)
    logger.info("merged %d LoRA modules (scale %.3f)", n, scale)
    model.save_pretrained(out)
    head = manifest.head_config.build()
    head_state = head.state_dict()
    dcp.load(head_state, checkpoint_id=str(stage / ClefStageManifest.HEAD_STATE_NAME))
    head.load_state_dict(head_state, strict=True)
    save_joint_head(head, out, dtype=torch.float32)
    # Tokenizer / processor files and Cloudflare's loader are copied verbatim from the base release.
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "chat_template.jinja",
        "joint_schema_model.py",
        "LICENSE",
    ):
        if (base / name).exists():
            shutil.copy(base / name, out / name)
    logger.info("wrote release directory %s", out)
