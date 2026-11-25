# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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


"""Load only the frozen token tables used by cached DFlash training."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers import PreTrainedConfig
from transformers.utils.hub import cached_file


@dataclass(frozen=True)
class DFlashTargetTables:
    """Target token embedding and output projection, both frozen."""

    embed_tokens: nn.Embedding
    lm_head: nn.Linear


@dataclass(frozen=True, kw_only=True)
class DFlashTargetTablesConfig:
    """Checkpoint keys for the two target tables read by DFlash."""

    embedding_key: str = "model.embed_tokens.weight"
    lm_head_key: str = "lm_head.weight"

    def build(
        self,
        target_model_name_or_path: str,
        *,
        config: PreTrainedConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> DFlashTargetTables:
        """Read the target's input and output tables without building the target.

        Args:
            target_model_name_or_path: Hugging Face safetensors checkpoint.
            config: Target config, with vocabulary, hidden size and tying metadata.
            device: Device for the frozen tables.
            dtype: Compute dtype used by the draft.

        Returns:
            Frozen tables, sharing the same weight when an omitted output head
            is tied to the input embedding.
        """
        keys = (self.embedding_key, self.lm_head_key)
        revision = getattr(config, "_commit_hash", None)
        state = _read_weights(target_model_name_or_path, keys, revision=revision)
        text = config.get_text_config()
        embedding = state.get(self.embedding_key)
        if embedding is None:
            raise ValueError(f"Target checkpoint does not contain {self.embedding_key!r}")
        head = state.get(self.lm_head_key)
        if head is None and not config.tie_word_embeddings:
            raise ValueError(f"Untied target checkpoint does not contain {self.lm_head_key!r}")
        expected = (text.vocab_size, text.hidden_size)
        for key, weight in state.items():
            if weight.shape != expected or not weight.is_floating_point():
                raise ValueError(f"Target table {key!r} must have floating-point shape {expected}")

        embed_tokens = nn.Embedding.from_pretrained(embedding.to(device=device, dtype=dtype), freeze=True)
        lm_head = nn.Linear(expected[1], expected[0], bias=False, device="meta", dtype=dtype)
        lm_head.weight = (
            embed_tokens.weight
            if head is None or self.embedding_key == self.lm_head_key
            else nn.Parameter(head.to(device=device, dtype=dtype), requires_grad=False)
        )
        return DFlashTargetTables(embed_tokens, lm_head)


def _read_weights(source: str, keys: tuple[str, str], *, revision: str | None) -> dict[str, torch.Tensor]:
    index = cached_file(
        source, "model.safetensors.index.json", revision=revision, _raise_exceptions_for_missing_entries=False
    )
    files: dict[str, list[str]] = {}
    if index is None:
        files["model.safetensors"] = list(dict.fromkeys(keys))
    else:
        weight_map = json.loads(Path(index).read_text())["weight_map"]
        for key in dict.fromkeys(keys):
            name = weight_map.get(key)
            if name is not None:
                if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
                    raise ValueError(f"Invalid safetensors shard path for target table {key!r}")
                files.setdefault(name, []).append(key)
    state = {}
    for name, names in files.items():
        path = cached_file(source, name, revision=revision)
        if path is None:
            raise FileNotFoundError(f"Missing {name!r} in target checkpoint {source!r}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for key in names:
                if key in available:
                    state[key] = handle.get_tensor(key)
                elif index is not None:
                    raise ValueError(f"Target table {key!r} is indexed in {name!r} but missing")
    return state
