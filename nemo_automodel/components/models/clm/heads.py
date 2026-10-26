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

"""Contrastive Language Model (CLM) projection heads.

CLM (https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) is a frozen Qwen3-8B encoder (last-token pooled, L2-normalised
hidden state) followed by two small MLP projection heads.  A (state, candidate) pair is scored by
``exp(logit_scale) * cos(state_head(e_state), action_head(e_candidate))`` and a question's answer distribution is the
softmax of those scores over its options.

Source architecture and checkpoint contract: https://github.com/Contrastive-LM/CLM (Apache-2.0;
https://github.com/Contrastive-LM/CLM/blob/main/LICENSE). The NeMo implementation adds typed configuration, validation,
and local checkpoint I/O. The modules reproduce the released architecture parameter-for-parameter so that checkpoints round-trip with the
``contrastive-lm`` package (``clm-serve`` / ``clm.Engine``): the checkpoint is a ``torch.save`` dict with
``state_head``, ``action_head``, ``logit_scale`` and ``cfg``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prompting import clm_pairs

CLM_MAX_LOGIT_SCALE = math.log(100.0)
CLM_HF_REPO = "Contrastive-LM/CLM-v0.1-8B"
CLM_HF_FILE = "CLM_v0.1-8B.pt"
CLM_HF_REVISION = "e939398d4556fcd9400c76fa8c5a513202f42b0a"
CLM_MAX_TOKENS = 2048


@dataclass(frozen=True)
class CLMEmbeddingCacheConfig:
    """Declare the complete frozen-encoder contract used to identify cached features.

    Hub revisions must be immutable commit IDs. Different tokenizer, truncation,
    pooling, normalization, precision, attention, or Transformers settings produce
    different cache names even when the input texts are unchanged.
    """

    encoder_name: str
    encoder_revision: str
    tokenizer_name: str
    tokenizer_revision: str
    max_length: int = 2048
    pooling: str = "last"
    l2_normalize: bool = True
    is_causal: bool = True
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"
    transformers_version: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("encoder_revision", self.encoder_revision),
            ("tokenizer_revision", self.tokenizer_revision),
        ):
            if not re.fullmatch(r"[0-9a-f]{40}", value):
                raise ValueError(f"{name} must be an immutable 40-character Hub commit ID for feature caching")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

    def build(self, *, cache_dir: Path, texts: Sequence[str]) -> Path:
        """Resolve the cache path without creating directories or fetching model files.

        Args:
            cache_dir: Directory that will own the feature cache.
            texts: Ordered input strings, aligned with the embedding table rows.

        Returns:
            Cache path identified by all embedding-producing settings and exact texts.
        """
        identity = (
            2,  # Invalidate features cached before encoder defaults matched this contract.
            self.encoder_name,
            self.encoder_revision,
            self.tokenizer_name,
            self.tokenizer_revision,
            self.max_length,
            self.pooling,
            self.l2_normalize,
            self.is_causal,
            self.dtype,
            self.attention_implementation,
            self.transformers_version,
            tuple(texts),
        )
        digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode("utf-8")).hexdigest()
        return cache_dir / f"embeddings_{digest}.pt"


@dataclass(frozen=True)
class CLMHeadConfig:
    """Declarative configuration for a CLM state/action head pair.

    Args:
        hidden_size: Width of each frozen backbone embedding.
        width: Hidden width of each projection MLP.
        depth: Number of linear layers in each projection MLP; at least two.
        projection_dim: Width of the normalized decision embedding.
        activation: One of ``"gelu"``, ``"relu"``, or ``"silu"``.
        layer_norm: Apply LayerNorm before intermediate hidden layers.
        residual: Add residual connections around intermediate hidden layers.
    """

    hidden_size: int
    width: int
    depth: int
    projection_dim: int
    activation: str
    layer_norm: bool
    residual: bool

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.width <= 0 or self.projection_dim <= 0:
            raise ValueError("CLM head dimensions must be positive")
        if self.depth < 2:
            raise ValueError("CLM head depth must be at least two")
        if self.activation not in {"gelu", "relu", "silu"}:
            raise ValueError(f"Unsupported CLM activation: {self.activation!r}")

    def build(self, *, initial_temperature: float = 0.07) -> CLMHeadPair:
        """Build the configured head pair.

        Args:
            initial_temperature: Positive InfoNCE softmax temperature.

        Returns:
            Trainable CLM state/action projection heads.
        """
        return CLMHeadPair(self, initial_temperature=initial_temperature)


class _ProjectionHead(nn.Module):
    def __init__(self, config: CLMHeadConfig) -> None:
        super().__init__()
        activations: dict[str, type[nn.Module]] = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }
        # Attribute names intentionally match Contrastive-LM's published checkpoint.
        self.inp = nn.Linear(config.hidden_size, config.width)
        self.hidden = nn.ModuleList(nn.Linear(config.width, config.width) for _ in range(config.depth - 2))
        self.norms = nn.ModuleList(
            nn.LayerNorm(config.width) if config.layer_norm else nn.Identity() for _ in range(config.depth - 2)
        )
        self.out = nn.Linear(config.width, config.projection_dim)
        self.act = activations[config.activation]()
        self.residual = config.residual

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project frozen encoder embeddings.

        Args:
            embeddings: Tensor of shape [..., hidden].

        Returns:
            Tensor of shape [..., projection].
        """
        projected = self.act(self.inp(embeddings))
        for linear, norm in zip(self.hidden, self.norms):
            update = self.act(norm(linear(projected)))
            projected = projected + update if self.residual else update
        return self.out(projected)


class CLMHeadPair(nn.Module):
    """Published CLM state and action projections plus learned temperature."""

    def __init__(self, config: CLMHeadConfig, *, initial_temperature: float) -> None:
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        self.config = config
        self.state_head = _ProjectionHead(config)
        self.action_head = _ProjectionHead(config)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / initial_temperature), dtype=torch.float32))

    def set_logit_scale_trainable(self, *, enabled: bool) -> None:
        """Apply the decision-head temperature training policy before optimization.

        Args:
            enabled: Whether the scalar log inverse temperature receives gradients.
        """
        self.logit_scale.requires_grad_(enabled)

    def project_states(self, state_embeddings: torch.Tensor) -> torch.Tensor:
        """Project and normalize state embeddings.

        Args:
            state_embeddings: Tensor of shape [batch, hidden].

        Returns:
            Tensor of shape [batch, projection].
        """
        self._validate_embeddings(state_embeddings, name="state_embeddings")
        return F.normalize(self.state_head(state_embeddings.to(self.logit_scale.dtype)), dim=-1)

    def project_actions(self, action_embeddings: torch.Tensor) -> torch.Tensor:
        """Project and normalize action embeddings.

        Args:
            action_embeddings: Tensor of shape [candidates, hidden].

        Returns:
            Tensor of shape [candidates, projection].
        """
        self._validate_embeddings(action_embeddings, name="action_embeddings")
        return F.normalize(self.action_head(action_embeddings.to(self.logit_scale.dtype)), dim=-1)

    def forward(self, state_embeddings: torch.Tensor, action_embeddings: torch.Tensor) -> torch.Tensor:
        """Score every state against every candidate action.

        Args:
            state_embeddings: Tensor of shape [batch, hidden].
            action_embeddings: Tensor of shape [candidates, hidden].

        Returns:
            Tensor of shape [batch, candidates] containing scaled cosine logits.
        """
        states = self.project_states(state_embeddings)
        actions = self.project_actions(action_embeddings)
        scale = self.logit_scale.clamp(max=CLM_MAX_LOGIT_SCALE).exp()
        return scale * states @ actions.transpose(0, 1)

    @property
    def scale(self) -> torch.Tensor:
        """Return the scalar inverse temperature, capped at the released value 100."""
        return self.logit_scale.clamp(max=CLM_MAX_LOGIT_SCALE).exp()

    def question_logits(
        self, state_emb: torch.Tensor, option_emb: torch.Tensor, option_counts: Sequence[int]
    ) -> list[torch.Tensor]:
        """Score ragged question options in question-major order.

        Args:
            state_emb: Tensor of shape [questions, hidden].
            option_emb: Tensor of shape [sum_options, hidden].
            option_counts: Positive option count for each question, summing to sum_options.

        Returns:
            Question-major tensors, each of shape [options].
        """
        if len(option_counts) != state_emb.shape[0] or any(n <= 0 for n in option_counts):
            raise ValueError("one positive option count is required per question")
        if sum(option_counts) != option_emb.shape[0]:
            raise ValueError("option_counts must sum to the option embedding count")
        states = self.project_states(state_emb)
        actions = self.project_actions(option_emb)
        return [self.scale * (rows @ states[i]) for i, rows in enumerate(actions.split(tuple(option_counts)))]

    @classmethod
    def from_upstream_checkpoint(cls, path: str | Path) -> CLMHeadPair:
        """Load the official Contrastive-LM ``.pt`` head checkpoint.

        Args:
            path: Local ``CLM_v0.1-8B.pt``-compatible checkpoint path.

        Returns:
            Head pair with the published state, action, and temperature tensors.
        """
        raw = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(raw, Mapping):
            raise ValueError("CLM checkpoint must contain a mapping")
        raw_cfg = raw.get("cfg")
        if not isinstance(raw_cfg, Mapping):
            raise ValueError("CLM checkpoint is missing its cfg mapping")
        cfg = cast(Mapping[str, object], raw_cfg)
        config = CLMHeadConfig(
            hidden_size=int(cfg.get("hidden_size", 4096)),
            width=int(cfg["width"]),
            depth=int(cfg["depth"]),
            projection_dim=int(cfg.get("projection_dim", raw.get("projection_dim", 512))),
            activation=str(cfg.get("activation", "gelu")),
            layer_norm=bool(cfg.get("layernorm", False)),
            residual=bool(cfg.get("residual", False)),
        )
        model = cls(config, initial_temperature=0.07)
        model.state_head.load_state_dict(cast(Mapping[str, torch.Tensor], raw["state_head"]))
        model.action_head.load_state_dict(cast(Mapping[str, torch.Tensor], raw["action_head"]))
        with torch.no_grad():
            model.logit_scale.copy_(torch.as_tensor(raw["logit_scale"], dtype=torch.float32))
        return model

    def _validate_embeddings(self, embeddings: torch.Tensor, *, name: str) -> None:
        """Validate the encoder width without modifying its embeddings.

        Args:
            embeddings: Tensor of shape [rows, hidden], with hidden equal to config.hidden_size.
            name: Input name used in validation errors.
        """
        if embeddings.ndim != 2 or embeddings.shape[-1] != self.config.hidden_size:
            raise ValueError(f"{name} must have shape [rows, {self.config.hidden_size}], got {tuple(embeddings.shape)}")


def load_clm_heads(
    path: str | os.PathLike | None = None, map_location: str = "cpu", *, revision: str | None = None
) -> CLMHeadPair:
    """Load a CLM head checkpoint (defaults to the released CLM-v0.1-8B heads from the Hugging Face Hub)."""
    if path is None or str(path).startswith("hf://"):
        from huggingface_hub import hf_hub_download

        repo, filename = CLM_HF_REPO, CLM_HF_FILE
        if path is not None:
            repo, _, filename = str(path)[len("hf://") :].rpartition("/")
        path = hf_hub_download(repo, filename, revision=revision or (CLM_HF_REVISION if repo == CLM_HF_REPO else None))
    return CLMHeadPair.from_upstream_checkpoint(path).to(map_location)


def save_clm_heads(heads: CLMHeadPair, path: str | os.PathLike, extra: dict[str, Any] | None = None) -> None:
    """Save heads in the ``contrastive-lm`` checkpoint format (loadable by ``clm-serve`` and :func:`load_clm_heads`)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ck = {
        "state_head": {k: v.detach().float().cpu() for k, v in heads.state_head.state_dict().items()},
        "action_head": {k: v.detach().float().cpu() for k, v in heads.action_head.state_dict().items()},
        "logit_scale": heads.logit_scale.detach().float().cpu(),
        "cfg": {
            "hidden_size": heads.config.hidden_size,
            "width": heads.config.width,
            "depth": heads.config.depth,
            "projection_dim": heads.config.projection_dim,
            "activation": heads.config.activation,
            "layernorm": heads.config.layer_norm,
            "residual": heads.config.residual,
        },
        "projection_dim": heads.config.projection_dim,
    }
    ck.update(extra or {})
    torch.save(ck, path)


def collect_clm_texts(records: Sequence[dict[str, Any]]) -> list[str]:
    """Every unique text the encoder must embed for these records (state texts first, then candidates)."""
    states, candidates = {}, {}
    for record in records:
        for state, _, texts in clm_pairs(record).values():
            states.setdefault(state, None)
            for text in texts:
                candidates.setdefault(text, None)
    return list(states) + [t for t in candidates if t not in states]


class EmbeddingTable:
    """Text -> row lookup into a table of L2-normalised encoder embeddings."""

    def __init__(self, texts: Sequence[str], embeddings: torch.Tensor) -> None:
        """Create the table.

        Args:
            texts: Unique texts, one per row of ``embeddings``.
            embeddings: Tensor of shape [texts, hidden].
        """
        if len(texts) != embeddings.shape[0]:
            raise ValueError(f"{len(texts)} texts but {embeddings.shape[0]} embeddings")
        self.index = {t: i for i, t in enumerate(texts)}
        self.embeddings = embeddings

    def lookup(self, texts: Sequence[str], device: torch.device | str | None = None) -> torch.Tensor:
        """Return a tensor of shape [len(texts), hidden] (a copy, optionally moved to ``device``)."""
        rows = torch.tensor([self.index[t] for t in texts], dtype=torch.long)
        out = self.embeddings[rows]
        return out.to(device) if device is not None else out

    def save(self, path: str | os.PathLike) -> None:
        """Publish the cache atomically, preserving any prior cache on failure."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            torch.save({"texts": list(self.index), "embeddings": self.embeddings.cpu()}, temporary_path)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "EmbeddingTable":
        """Load a previously published cache using restricted tensor deserialization."""
        ck = torch.load(path, map_location="cpu", weights_only=True)
        return cls(ck["texts"], ck["embeddings"])


def clm_batch(
    records: Sequence[dict[str, Any]], table: EmbeddingTable, device: torch.device | str | None = None
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[str, list[str]]]]:
    """Gather encoder embeddings for a batch of records.

    Returns:
        ``(state_emb, option_emb, option_counts, option_orders)``: ``state_emb`` has shape [questions, hidden] (one row
        per question of every record), ``option_emb`` has shape [candidates, hidden] (question-major), and
        ``option_orders[i]`` is the ``{question_id: option ids}`` mapping of record ``i`` in CLM order.
    """
    states, options, counts, orders = [], [], [], []
    for record in records:
        order = {}
        for qid, (state, keys, texts) in clm_pairs(record).items():
            states.append(state)
            options.extend(texts)
            counts.append(len(texts))
            order[qid] = keys
        orders.append(order)
    return table.lookup(states, device), table.lookup(options, device), counts, orders
