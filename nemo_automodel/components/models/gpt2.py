# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""GPT-2 model utility wrappers for NeMo Automodel.

The canonical way to instantiate a GPT-2 with custom sizes is to pass a
`transformers.GPT2Config` into `NeMoAutoModelForCausalLM.from_config`.  For
YAML-driven workflows, however, specifying the entire nested config can be
verbose.  This module provides a *single-level* builder function that exposes
the most common GPT-2 hyper-parameters directly.

Example (YAML):

```yaml
model:
  _target_: nemo_automodel.components.models.gpt2.build_gpt2_model
  n_layer: 24           # GPT-2 Medium
  n_embd: 1024
  n_head: 16
  vocab_size: 50257
  n_positions: 2048
```
"""

from __future__ import annotations

"""Self-contained GPT-2 (Causal LM) implementation.

This module no longer relies on the HuggingFace *transformers* library or
NeMo’s *NeMoAutoModelForCausalLM*.  Instead, it defines the necessary
building blocks (attention, MLP, transformer block, and language-model head)
using vanilla PyTorch.  The public *build_gpt2_model* helper returns an
``nn.Module`` whose *forward* method mirrors the logits output of
``transformers.GPT2LMHeadModel``.

The implementation purposefully keeps the code concise – it is **not** a 100 %
feature-complete reproduction of GPT-2 (e.g., it omits parallel/sequence
parallelism tricks, rotary embeddings, kv-cache, generation helpers, etc.).
It is, however, perfectly suitable for forward/backward passes, fine-tuning,
and next-token prediction.
"""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["build_gpt2_model"]


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask."""

    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.0):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = attn_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        bsz, seq_len, _ = x.shape

        # Project to QKV and reshape: (B, T, 3*C) → (B, n_head, T, head_dim)
        qkv = self.qkv_proj(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, n_head, T, head_dim)

        # Use torch's optimized SDPA when available (PyTorch ≥2.0)
        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_dropout, is_causal=True
            )  # (B, n_head, T, head_dim)
        else:
            # Fallback implementation with an explicit causal mask
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask, float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)
            attn_output = attn_weights @ v  # (B, n_head, T, head_dim)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class MLP(nn.Module):
    """GPT-2 feed-forward network (GEGLU → Linear)."""

    def __init__(self, embed_dim: int, expansion_factor: int = 4):
        super().__init__()
        hidden_dim = expansion_factor * embed_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """A single transformer block (LN → Attn → Add → LN → MLP → Add)."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2LMHeadModel(nn.Module):
    """Minimal GPT-2 Causal-LM with tied input/output embeddings."""

    def __init__(
        self,
        *,
        vocab_size: int,
        n_positions: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_positions, n_embd)
        self.drop = nn.Dropout(dropout)

        self.h = nn.ModuleList([TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

        # Language model head (weights tied to token embedding matrix)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying

        # Initialize parameters following GPT-2 scheme
        self._init_weights()

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:  # (B, T) → (B, T, V)
        batch_size, seq_len = input_ids.shape

        if seq_len > self.wpe.num_embeddings:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum context size {self.wpe.num_embeddings}.")

        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.wte(input_ids) + self.wpe(pos_ids)
        x = self.drop(x)

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def _init_weights(self):
        """Parameter initialization following GPT-2 conventions."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                # GPT-2 uses normal(0, 0.02)
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)


def build_gpt2_model(
    *,
    vocab_size: int = 50257,
    n_positions: int = 2048,
    n_ctx: int | None = None,
    n_embd: int = 768,
    n_layer: int = 12,
    n_head: int = 12,
    bos_token_id: int = 50256,  # kept for API backward-compat (unused)
    eos_token_id: int = 50256,  # kept for API backward-compat (unused)
    attn_implementation: str = "flash_attention_2",  # retained but ignored
    **extra_cfg: Any,  # ignored to preserve call-sites that used to pass config tweaks
) -> nn.Module:
    """Instantiate and return a *pure-PyTorch* GPT-2 language model.

    The function intentionally keeps the same signature as the original
    wrapper so existing YAML/CLI configurations continue to work.
    Extra keyword arguments are quietly ignored.
    """

    # Map legacy *n_ctx* to *n_positions* if provided.
    if n_ctx is not None and n_ctx != n_positions:
        n_positions = n_ctx

    # Issue a gentle warning if the user passes unused extra kwargs.
    if extra_cfg:
        invalid = ", ".join(extra_cfg.keys())
        print(
            f"[build_gpt2_model] Warning: Ignoring unsupported keyword arguments: {invalid}.",
            flush=True,
        )

    return GPT2LMHeadModel(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
    )
