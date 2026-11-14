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

"""The public weights by their Hugging Face repos: the checkpoint and the frozen encoders of the reference package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors.torch import load_file
from torch import Tensor

if TYPE_CHECKING:
    from praxis.text_encoder import TextEncoder
    from praxis.vae import Autoencoder

WAN_VAE_FILENAME = "Wan2.2_VAE.pth"


def resolve_repo(path: str | Path, filename: str | None = None) -> Path:
    """Local path of ``path``: itself when it exists, else the Hugging Face repo ``path`` in the Hub cache.

    Args:
        path: Local file or directory, or a Hugging Face repo id such as ``runwayml/praxis-1-droid``.
        filename: With a repo id, the single file to download instead of the whole repo.

    Returns:
        The local file or directory.
    """
    local = Path(path)
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download, snapshot_download

    if filename is not None:
        return Path(hf_hub_download(repo_id=str(path), filename=filename))
    return Path(snapshot_download(repo_id=str(path)))


def load_vae(path: str | Path, device: str | torch.device = "cuda", filename: str = WAN_VAE_FILENAME) -> Autoencoder:
    """The Wan2.2 VAE of the reference package.

    Args:
        path: The ``.pth`` weights file, or the Hugging Face repo that holds it (``Wan-AI/Wan2.2-TI2V-5B``).
        device: Device of the loaded model.
        filename: Name of the weights file inside the repo.
    """
    from praxis.vae import Autoencoder

    return Autoencoder.from_pretrained(resolve_repo(path, filename=filename), device=device)


def load_text_encoder(path: str | Path, device: str | torch.device = "cuda") -> TextEncoder:
    """The Qwen3.5 text encoder of the reference package.

    Args:
        path: Directory in the Hugging Face layout, or the repo id (``Qwen/Qwen3.5-2B``).
        device: Device of the loaded model.
    """
    from praxis.text_encoder import TextEncoder

    return TextEncoder(resolve_repo(path), device=device)


def load_checkpoint_tensors(root: str | Path, device: str | torch.device = "cpu") -> dict[str, Tensor]:
    """Read every shard that ``model.safetensors.index.json`` under ``root`` names.

    Args:
        root: Checkpoint directory.
        device: Device of the loaded tensors.

    Returns:
        The checkpoint tensors under their checkpoint keys.
    """
    root = Path(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json has no weight_map")
    state: dict[str, Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        state.update(load_file(str(root / shard), device=str(device)))
    return state
