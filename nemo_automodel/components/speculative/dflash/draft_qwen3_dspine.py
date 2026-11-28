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
#
# Adapted from SpecForge (MIT License):
#   https://github.com/sgl-project/SpecForge
#   revision 2dadea10e317842af21c79e2097dc05abe1e2b5a (head of PR #1038)
#   specforge/modeling/draft/dspine.py
#   (AdjacentInjection, DSpineDraftModel.initialize_transfer_space, refine)
# SpecForge upstream copyright notice: Copyright (c) 2025 sgl-project.
# Licensed under the MIT License - https://github.com/sgl-project/SpecForge/blob/2dadea10e317842af21c79e2097dc05abe1e2b5a/LICENSE
#
# Parameter names, shapes, and config keys follow the released checkpoint layout
# read by Tencent/DSpine (Apache-2.0, Copyright (C) 2026 Tencent):
#   https://github.com/Tencent/DSpine
#   revision 0cc92d857a8f7bf252de67995ad4c19251620143
#   sglang_dspine/models/dspine.py

"""DSpine draft model (Qwen3-style).

DSpine ("Draft in Parallel, Condition Through Depth: Adjacent Causal Injection
for Speculative Decoding", Tencent, https://github.com/Tencent/DSpine) keeps the
DFlash backbone -- a small Qwen3 stack that drafts a whole ``[anchor, MASK, ...]``
block in one forward, conditioned on the target's context hidden states -- and
changes two things:

* in-block attention is **causal** (block position ``t`` sees positions
  ``<= t`` of its own block), and
* after every decoder layer an **adjacent causal injection** writes position
  ``t - 1``'s state into position ``t``. Each position reads a code
  ``z_t = sqrt(C) * normalize(R_l u_t)`` from its post-layer hidden ``u_t``; the
  predecessor code ``pred_t = z_{t-1}`` (the anchor token's fixed codebook code
  for ``t = 1``) is gated and projected back into the residual stream:
  ``h_t = u_t + rms(u_t) * W_l (sigmoid(G_q z_t + G_p pred_t + b) * A pred_t)``.
  Position 0 (the anchor) is never written.

Because attention is causal, a message written at layer ``l`` reaches the next
position at layer ``l + 1``: the left-to-right dependency unfolds over depth
while every position is still drafted in parallel.

Decoding additionally redoes the **last** layer's write with the exact code of
the token chosen for the previous position ("exact-sender refinement") and
rescores each position's top-k backbone candidates with it, walking the block
left to right (:meth:`Qwen3DSpineDraftModel.refine`).

The injection parameters are named and laid out exactly as the released
checkpoints (e.g. ``tencent/Qwen3-8B-DSpine-b16``) store them under
``selected_prefix_bridge.*``, so a trained draft's state dict is what SGLang's
``--speculative-algorithm DSPINE`` loader expects, with no conversion step. The
training objective lives in
``nemo_automodel.components.speculative.dflash.dspine_core``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor
from transformers import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel

# The architecture name the DSpine serving loaders accept (SGLang's DSPINE plugin
# takes ``DSpineDraftModel`` or the plain ``DFlashDraftModel``).
DSPINE_ARCHITECTURE = "DSpineDraftModel"


def _dflash_config(config: PretrainedConfig) -> dict:
    """Return the draft's ``dflash_config`` block (an empty dict when absent)."""
    return getattr(config, "dflash_config", None) or {}


class _ZeroTerminalProjection(nn.Module):
    """The released layout's terminal projection, which DSpine keeps disabled.

    The serving loader requires ``selected_prefix_bridge.out.weight`` to be
    present and all zero (a terminal-disabled bridge). It never affects the
    forward, so it is a buffer rather than a trainable parameter.
    """

    def __init__(self, rank: int, code_size: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.zeros(code_size, rank))


class DSpineCodeBridge(nn.Module):
    """Adjacent causal injection between the draft's decoder layers.

    Attribute names and shapes follow the released ``selected_prefix_bridge.*``
    checkpoint layout (``C = code_size``, ``R = rank``, ``H = hidden``,
    ``L = layers``):

    * ``early_read_weights`` [L-1, C, H] and ``read_weight`` [C, H]: the per-layer
      readouts ``R_l`` (layers ``0..L-2`` and the last layer), initialized from
      the target head's principal directions by :meth:`initialize_transfer_space`;
    * ``feature_write_weights`` [L, H, R]: the per-layer writes ``W_l``,
      zero-initialized so every injection starts as the identity;
    * ``prev`` (``A``), ``gate`` (``G_q`` with bias ``b``) and
      ``predecessor_gate`` (``G_p``, zero-initialized): shared across layers;
    * ``codebook`` [V, C]: fixed unit-norm token codes derived from the target
      head; never trained;
    * ``out.weight`` and two version scalars that the serving loader requires.
    """

    def __init__(self, *, hidden_size: int, num_layers: int, vocab_size: int, code_size: int, rank: int) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"DSpine needs at least 2 draft layers, got num_hidden_layers={num_layers}.")
        if not 0 < code_size <= min(hidden_size, vocab_size - 1):
            raise ValueError(
                f"DSpine code_size must be in [1, min(hidden_size, vocab_size - 1)] = "
                f"[1, {min(hidden_size, vocab_size - 1)}], got {code_size}."
            )
        if rank < 1:
            raise ValueError(f"DSpine rank must be >= 1, got {rank}.")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.code_size = code_size
        self.rank = rank
        self.code_scale = math.sqrt(code_size)
        self.early_read_weights = nn.Parameter(torch.empty(num_layers - 1, code_size, hidden_size))
        self.read_weight = nn.Parameter(torch.empty(code_size, hidden_size))
        self.feature_write_weights = nn.Parameter(torch.empty(num_layers, hidden_size, rank))
        self.prev = nn.Linear(code_size, rank, bias=False)
        self.gate = nn.Linear(code_size, rank, bias=True)
        self.predecessor_gate = nn.Linear(code_size, rank, bias=False)
        self.out = _ZeroTerminalProjection(rank, code_size)
        self.register_buffer("codebook", torch.zeros(vocab_size, code_size))
        self.register_buffer("independent_write_version", torch.ones((), dtype=torch.int64))
        self.register_buffer("terminal_disabled_version", torch.ones((), dtype=torch.int64))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Put the bridge on its identity start.

        The writes and the predecessor gate start at zero (so the freshly built
        draft is numerically a causal DFlash), the gate bias at zero, and the
        shared message/gate projections at PyTorch's default ``Linear`` init. The
        readouts get a placeholder ``Linear``-style init here; training replaces
        them, together with the codebook, via :meth:`initialize_transfer_space`.
        """
        for weight in (*self.early_read_weights, self.read_weight):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.feature_write_weights.zero_()
        self.prev.reset_parameters()
        self.gate.reset_parameters()
        self.gate.bias.zero_()
        self.predecessor_gate.weight.zero_()

    def layer_read_weight(self, layer: int) -> torch.Tensor:
        """Return layer ``layer``'s readout ``R_l``, a Tensor of shape [code, hidden]."""
        return self.read_weight if layer == self.num_layers - 1 else self.early_read_weights[layer]

    def feature(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        """Read each position's code ``sqrt(C) * normalize(R_l h)``.

        Args:
            hidden: Tensor of shape [..., hidden], with arbitrary leading dimensions.
            layer: Index of the draft layer whose readout to use.

        Returns:
            Tensor of shape [..., code] in ``hidden.dtype``; each row has norm
            ``sqrt(code)`` (zero for an all-zero input row).
        """
        projected = F.linear(hidden, self.layer_read_weight(layer))
        return (self.code_scale * F.normalize(projected.float(), dim=-1)).to(hidden.dtype)

    def token_codes(self, token_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Return the scaled codebook codes ``sqrt(C) * E[token_ids]``.

        Args:
            token_ids: Long tensor of shape [...], with arbitrary leading dimensions.
            dtype: Output dtype.

        Returns:
            Tensor of shape [..., code].
        """
        return (self.code_scale * F.embedding(token_ids, self.codebook)).to(dtype)

    def write(
        self, hidden: torch.Tensor, receiver: torch.Tensor, predecessor: torch.Tensor, layer: int
    ) -> torch.Tensor:
        """Add the gated predecessor message to the receiving positions.

        ``h + rms(h) * W_l (sigmoid(G_q receiver + G_p predecessor + b) * A predecessor)``,
        where ``rms(h) = ||h|| / sqrt(hidden)`` (no epsilon, no learned weight).

        Args:
            hidden: Tensor of shape [..., hidden]; the receiving positions'
                post-layer residual stream.
            receiver: Tensor of shape [..., code]; the receivers' own codes.
            predecessor: Tensor of shape [..., code]; the codes written into them.
            layer: Index of the draft layer whose write ``W_l`` to use.

        Returns:
            Tensor of shape [..., hidden]; a new tensor (inputs are not mutated).
        """
        predecessor = predecessor.to(receiver.dtype)
        gate = torch.sigmoid(self.gate(receiver) + self.predecessor_gate(predecessor))
        delta = F.linear(gate * self.prev(predecessor), self.feature_write_weights[layer])
        scale = (torch.linalg.vector_norm(hidden.float(), dim=-1, keepdim=True) / math.sqrt(hidden.shape[-1])).to(
            hidden.dtype
        )
        return hidden + scale * delta

    def forward(
        self,
        hidden: torch.Tensor,
        layer: int,
        block_token_ids: torch.Tensor,
        replace_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inject each position's predecessor code after decoder layer ``layer``.

        Args:
            hidden: Tensor of shape [batch, blocks, block_size, hidden]; the layer
                output for every draft block.
            layer: Index of the decoder layer just applied.
            block_token_ids: Long tensor of shape [batch, blocks, block_size];
                ``[..., 0]`` is each block's anchor token, whose codebook code is
                position 1's predecessor. ``[..., t]`` for ``t >= 1`` is only read
                for teacher-forced blocks (the true token at block position ``t``).
            replace_mask: Optional bool tensor of shape [batch, blocks]; blocks
                whose predecessor codes are replaced by the codebook codes of the
                true previous tokens (the training curriculum). ``None`` disables it.

        Returns:
            Tensor of shape [batch, blocks, block_size, hidden]; position 0 is
            ``hidden[..., 0, :]`` unchanged.
        """
        features = self.feature(hidden, layer)
        anchor = self.token_codes(block_token_ids[..., :1], features.dtype)
        predecessors = torch.cat((anchor, features[..., 1:-1, :]), dim=-2)
        if replace_mask is not None:
            teacher = self.token_codes(block_token_ids[..., :-1], features.dtype)
            predecessors = torch.where(replace_mask[..., None, None], teacher, predecessors)
        successors = self.write(hidden[..., 1:, :], features[..., 1:, :], predecessors, layer)
        return torch.cat((hidden[..., :1, :], successors), dim=-2)

    @torch.no_grad()
    def initialize_transfer_space(self, head_weight: torch.Tensor, chunk_size: int = 4096) -> None:
        """Build the fixed codebook and the readouts from the target's output head.

        Follows SpecForge's ``initialize_transfer_space``: l2-normalize the head
        rows, centre them, take the top-``code_size`` eigenvectors of their
        covariance, whiten the projected rows by the eigenvalues and l2-normalize
        again. Every readout is set to the (transposed) principal directions.
        Computed in float32 and then cast to the buffers' dtype.

        Args:
            head_weight: Tensor of shape [vocab, hidden]; the frozen target's full
                (unsharded) output-projection weight.
            chunk_size: Head rows processed per step, bounding the float32
                working set.

        Raises:
            ValueError: If the head shape does not match the draft, ``chunk_size``
                is not positive, or the head has fewer than ``code_size`` positive
                principal directions.
        """
        vocab_size = self.codebook.shape[0]
        if tuple(head_weight.shape) != (vocab_size, self.hidden_size):
            raise ValueError(
                f"DSpine transfer space needs the target head's full [vocab, hidden] = "
                f"[{vocab_size}, {self.hidden_size}] weight, got {tuple(head_weight.shape)}."
            )
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
        covariance = torch.zeros(self.hidden_size, self.hidden_size, device=head_weight.device, dtype=torch.float32)
        total = covariance.new_zeros(self.hidden_size)
        for rows in head_weight.split(chunk_size):
            normalized = F.normalize(rows.float(), dim=-1)
            total.add_(normalized.sum(dim=0))
            covariance.add_(normalized.T @ normalized)
        mean = total / vocab_size
        covariance.sub_(vocab_size * torch.outer(mean, mean))
        values, directions = torch.linalg.eigh(covariance)
        values = values[-self.code_size :]
        directions = directions[:, -self.code_size :]
        # Standard numerical-rank tolerance: eigenvalues of a rank-deficient head's
        # covariance come out as float noise, not exact zeros, and whitening by
        # them would amplify that noise.
        rank_tolerance = values[-1] * self.hidden_size * torch.finfo(torch.float32).eps
        if not torch.isfinite(values).all() or (values <= rank_tolerance).any():
            raise ValueError(
                f"The target head has fewer than code_size={self.code_size} positive principal directions."
            )
        inv_sqrt_values = values.rsqrt()
        for start in range(0, vocab_size, chunk_size):
            rows = head_weight[start : start + chunk_size]
            codes = F.normalize(((F.normalize(rows.float(), dim=-1) - mean) @ directions) * inv_sqrt_values, dim=-1)
            self.codebook[start : start + rows.shape[0]].copy_(codes.to(device=self.codebook.device))
        readout = directions.T.to(device=self.read_weight.device)
        self.read_weight.copy_(readout)
        self.early_read_weights.copy_(readout.expand_as(self.early_read_weights))


@dataclass
class DSpineDraftOutput:
    """Outputs of one DSpine draft forward.

    Attributes:
        hidden_states: Tensor of shape [batch, draft, hidden]; the final-normed
            draft hidden states, ready for the target's output head. ``draft`` is
            ``blocks * block_size``.
        pre_injection: Tensor of shape [batch, draft, hidden]; the last layer's
            output *before* its injection, which :meth:`Qwen3DSpineDraftModel.refine`
            rewrites with an exact predecessor code.
        layer_features: Tensor of shape [layers, batch, draft, code]; each layer's
            codes read from its *post*-injection hidden states.
    """

    hidden_states: torch.Tensor
    pre_injection: torch.Tensor
    layer_features: torch.Tensor


def _mark_transfer_space_loaded(module: "Qwen3DSpineDraftModel", incompatible_keys) -> None:
    """``load_state_dict`` post-hook: a load that restored the codebook makes the transfer space ready."""
    if not any(key.endswith("selected_prefix_bridge.codebook") for key in incompatible_keys.missing_keys):
        module.transfer_space_ready = True


class Qwen3DSpineDraftModel(Qwen3DFlashDraftModel):
    """DSpine draft: the DFlash Qwen3 stack with causal blocks and adjacent causal injection."""

    def __init__(self, config: Qwen3Config) -> None:
        cfg = _dflash_config(config)
        if cfg.get("draft_block_causal") is not True:
            raise ValueError(
                "DSpine drafts with causal in-block attention; dflash_config.draft_block_causal must be true."
            )
        if getattr(config, "is_causal", None) is False:
            raise ValueError("DSpine drafts with causal in-block attention; the draft config sets is_causal=false.")
        # Stamp what the serving loaders read: causal draft attention and the
        # DSpine architecture name.
        config.is_causal = True
        config.architectures = [DSPINE_ARCHITECTURE]
        super().__init__(config)
        if self.projector_type is not None:
            raise ValueError(
                f"DSpine has no Domino correction head; projector_type must be unset, got {self.projector_type!r}."
            )
        if self.block_size < 2:
            raise ValueError(f"DSpine block_size must hold an anchor and at least one draft, got {self.block_size}.")
        # Exact-sender refinement scores candidates with raw head-row dot products,
        # as the serving kernels do, so the target's logits must be the plain head.
        if (
            self.output_multiplier != 1.0
            or self.final_logit_softcapping is not None
            or self.input_embedding_scale != 1.0
        ):
            raise ValueError(
                "DSpine requires the plain target head and embedding: output_multiplier=1, "
                "final_logit_softcapping unset, and input_embedding_scale=1."
            )

        num_layers = config.num_hidden_layers
        if cfg.get("selected_prefix_feature_writes") is not True:
            raise ValueError("DSpine requires dflash_config.selected_prefix_feature_writes = true.")
        if cfg.get("selected_prefix_feature_write_mode") != "independent":
            raise ValueError("DSpine requires dflash_config.selected_prefix_feature_write_mode = 'independent'.")
        if cfg.get("selected_prefix_terminal", True):
            raise ValueError(
                "DSpine requires a terminal-disabled bridge: dflash_config.selected_prefix_terminal = false."
            )
        feature_layers = [int(i) for i in cfg.get("selected_prefix_feature_layers", [])]
        if feature_layers != list(range(1, num_layers + 1)):
            raise ValueError(
                f"DSpine writes after every draft layer: dflash_config.selected_prefix_feature_layers must be "
                f"{list(range(1, num_layers + 1))}, got {feature_layers}."
            )
        teacher_layers = [int(i) for i in cfg.get("selected_prefix_feature_teacher_layers", [])]
        if any(not 1 <= i <= num_layers for i in teacher_layers):
            raise ValueError(
                f"dflash_config.selected_prefix_feature_teacher_layers must be 1-based layer ids in "
                f"[1, {num_layers}], got {teacher_layers}."
            )
        # 0-based layers whose predecessor codes the training curriculum may replace.
        self.teacher_forcing_layers = frozenset(i - 1 for i in teacher_layers)
        bridge_cfg = cfg.get("selected_prefix_code_bridge") or {}
        refine_cfg = cfg.get("selected_prefix_last_write_refine") or {}
        if refine_cfg.get("kind") != "exact_sender" or int(refine_cfg.get("version", 1)) != 1:
            raise ValueError(
                "DSpine requires dflash_config.selected_prefix_last_write_refine = "
                f"{{kind: 'exact_sender', version: 1, head_topk}}, got {refine_cfg!r}."
            )
        head_topk = int(refine_cfg["head_topk"])
        # The serving kernels only support power-of-two widths in [2, 64].
        if head_topk & (head_topk - 1) or not 2 <= head_topk <= min(64, config.vocab_size):
            raise ValueError(f"DSpine head_topk must be a power of two in [2, 64], got {head_topk}.")
        self.head_topk = head_topk

        self.selected_prefix_bridge = DSpineCodeBridge(
            hidden_size=config.hidden_size,
            num_layers=num_layers,
            vocab_size=config.vocab_size,
            code_size=int(bridge_cfg.get("code_size", 1024)),
            rank=int(bridge_cfg.get("rank", 512)),
        )
        # Version markers the serving loader checks for the causal draft and the
        # exact-sender refinement.
        self.register_buffer("draft_block_causal_version", torch.ones((), dtype=torch.int64))
        self.register_buffer("last_write_refine_version", torch.ones((), dtype=torch.int64))
        # Whether the codebook and readouts hold real values: set by
        # ``initialize_transfer_space`` or by loading a state dict that carries the
        # codebook. Not a buffer -- the serving loader rejects unknown tensors.
        self.transfer_space_ready = False
        self.register_load_state_dict_post_hook(_mark_transfer_space_loaded)

    def initialize_transfer_space(self, head_weight: torch.Tensor) -> None:
        """Build the bridge's fixed codebook and readouts from the target head.

        Must run once on a freshly built draft before training; a checkpoint
        restore afterwards overwrites both with the trained values.

        Args:
            head_weight: Tensor of shape [vocab, hidden]; the frozen target's
                output-projection weight. A vocab-sharded ``DTensor`` (a
                tensor-parallel target) is gathered to the full weight first,
                which is a collective over its mesh.
        """
        if isinstance(head_weight, DTensor):
            head_weight = head_weight.full_tensor()
        self.selected_prefix_bridge.initialize_transfer_space(head_weight)
        self.transfer_space_ready = True

    @torch.no_grad()
    def broadcast_transfer_space(self, src: int = 0, group: dist.ProcessGroup | None = None) -> None:
        """Copy rank ``src``'s codebook and readouts to every rank of ``group``.

        Each rank derives the transfer space from its own copy of the target head,
        and an eigendecomposition's eigenvector signs are not guaranteed to agree
        across devices. The codebook is a buffer, which the draft's DDP wrapper
        does not broadcast, so align the codebook and the readouts derived with it
        explicitly. A collective: every rank of ``group`` must call it.

        Args:
            src: Global rank whose transfer space is kept.
            group: Process group to broadcast over; ``None`` is the default group.
        """
        bridge = self.selected_prefix_bridge
        for tensor in (bridge.codebook, bridge.read_weight, bridge.early_read_weights):
            # Runs at setup, before any forward, so no autograd graph observes the write.
            dist.broadcast(tensor.detach(), src=src, group=group)

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        block_token_ids: torch.Tensor,
        replace_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> DSpineDraftOutput:
        """Run the DSpine draft over ``[context | blocks]`` with an injection after every layer.

        Args:
            position_ids: Long tensor of shape [batch, context + draft].
            attention_mask: The causal block mask over [batch, 1, draft,
                context + draft] (a dense additive tensor or a flex
                ``BlockMask``); required, since the in-block causality lives in it.
            noise_embedding: Tensor of shape [batch, draft, hidden]; the embedded
                ``[anchor, MASK, ...]`` blocks laid end to end, ``draft`` being
                ``blocks * block_size``.
            target_hidden: Tensor of shape [batch, context, layers * hidden]; the
                concatenated target-model context features.
            block_token_ids: Long tensor of shape [batch, blocks, block_size]; see
                :meth:`DSpineCodeBridge.forward`.
            replace_mask: Optional bool tensor of shape [batch, blocks]; blocks
                teacher-forced in ``dflash_config.selected_prefix_feature_teacher_layers``.
            past_key_values: Unsupported; must be ``None``.
            use_cache: Unsupported; must be ``False``.
            **kwargs: Forwarded to the attention implementation.

        Returns:
            DSpineDraftOutput for the whole draft.

        Raises:
            ValueError: On a KV cache request, a missing mask, or block shapes that
                disagree with ``block_size``.
        """
        if use_cache or past_key_values is not None:
            raise ValueError("The DSpine training forward has no draft KV cache; serve DSpine with SGLang's DSPINE.")
        if attention_mask is None:
            raise ValueError("DSpine requires an explicit causal block attention mask.")
        batch, blocks, size = block_token_ids.shape
        if size != self.block_size or tuple(noise_embedding.shape[:2]) != (batch, blocks * size):
            raise ValueError(
                f"DSpine block_token_ids {tuple(block_token_ids.shape)} and noise_embedding "
                f"{tuple(noise_embedding.shape)} must describe [batch, blocks, block_size={self.block_size}]."
            )
        hidden = noise_embedding
        context = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden, position_ids)
        layer_features = []
        pre_injection = hidden
        for index, layer in enumerate(self.layers):
            hidden = layer(
                hidden_states=hidden,
                target_hidden=context,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            pre_injection = hidden
            injected = self.selected_prefix_bridge(
                hidden.view(batch, blocks, size, -1),
                index,
                block_token_ids,
                replace_mask if index in self.teacher_forcing_layers else None,
            )
            layer_features.append(self.selected_prefix_bridge.feature(injected, index).view(batch, blocks * size, -1))
            hidden = injected.view(batch, blocks * size, -1)
        return DSpineDraftOutput(
            hidden_states=self.norm(hidden),
            pre_injection=pre_injection,
            layer_features=torch.stack(layer_features),
        )

    def refine(self, pre_injection: torch.Tensor, predecessor_ids: torch.Tensor) -> torch.Tensor:
        """Redo the last layer's write with an exact predecessor token (exact-sender refinement).

        Args:
            pre_injection: Tensor of shape [..., hidden]; last-layer outputs before
                their injection (see :class:`DSpineDraftOutput`), with arbitrary
                leading dimensions.
            predecessor_ids: Long tensor of shape [...] (the leading dimensions of
                ``pre_injection``); the token each row's predecessor committed.

        Returns:
            Tensor of shape [..., hidden]; the final-normed refined hidden states.
        """
        bridge = self.selected_prefix_bridge
        last = len(self.layers) - 1
        receiver = bridge.feature(pre_injection, last)
        predecessor = bridge.token_codes(predecessor_ids, receiver.dtype)
        return self.norm(bridge.write(pre_injection, receiver, predecessor, last))

    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int] | None,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> torch.LongTensor:
        """DSpine has no in-repo decode loop; serve with SGLang's ``--speculative-algorithm DSPINE``.

        Args:
            target: The frozen verifier.
            input_ids: Long tensor of shape [1, prompt].
            max_new_tokens: Maximum number of tokens to generate.
            stop_token_ids: Token ids that end generation, or ``None``.
            temperature: Sampling temperature.
            top_p: Nucleus mass to keep.
            top_k: Candidates to keep.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "DSpine drafts are trained here and served by SGLang's DSPINE speculative algorithm "
            "(https://github.com/Tencent/DSpine); there is no in-repo spec_generate."
        )
