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
#
# Parts of this file are adapted from Megatron-LM (https://github.com/NVIDIA/Megatron-LM), file
# megatron/core/transformer/experimental_attention_variant/dsa.py at revision
# f78809f6d9c88f8638ea7822966f35057a63fd5c. The copyright notice of that file and the notice of the LICENSE file
# of that repository, which applies to all of its files unless otherwise noted, are retained below.
#
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""ERNIE 5.5 dynamic sparse attention (DSA).

A DSA layer attends, for every query, to a causal local window of the query's document and to the remote keys
of the same document that an indexer selects. This module holds the layer and its parts:

* ``apply_indexer_rope`` and ``apply_main_rope``: the two rotary channel layouts of a DSA layer.
* ``rotate_indexer_activation``: the scaled Hadamard rotation of the indexer query and key.
* ``Ernie5_5Indexer``: the projections that produce the indexer query, key and per-head weights.
* ``ernie5_5_index_scores`` and ``select_window_and_remote``: the index scores and the dense key selection of the
  eager path.
* ``latent_mqa_core``: the eager attention core over a dense [sequence, sequence] selection.
* ``merge_remote_and_window`` and ``cudnn_dsa_core``: the sparse key indices and the attention core of the cuDNN path.
* ``Ernie5_5DSAFP32Parameters`` and ``Ernie5_5DSASparseFP32Parameters``: the FP32 sink and virtual-head parameters,
  each with the core that consumes them.
* ``Ernie5_5DSAAttention``: the layer. ``backend.attn`` selects the eager path for short sequences or the cuDNN path
  for one unpadded row that holds one document; both paths share all parameters.
* ``freeze_ernie5_5_indexer_params``: disables gradients for the indexer parameters.

Provenance: the projection layout, the indexer precision contract and the operation order follow two sources.

* The ERNIE 5.5 DSA-layer attention of Megatron-Bridge (Apache License 2.0) at revision
  ``7c55987a027084e87e36277c9711e87bc52a2116``:
  https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/7c55987a027084e87e36277c9711e87bc52a2116/src/megatron/bridge/models/ernie/modeling_ernie55/latent_mqa_attention.py
* The DSA of Megatron-LM (3-clause BSD terms of the repository ``LICENSE`` file; its copyright notices, conditions
  and disclaimer are retained in the comment block above) at revision
  ``f78809f6d9c88f8638ea7822966f35057a63fd5c``:
  https://github.com/NVIDIA/Megatron-LM/blob/f78809f6d9c88f8638ea7822966f35057a63fd5c/megatron/core/transformer/experimental_attention_variant/dsa.py
"""

import math
from collections.abc import Callable
from dataclasses import replace

import torch
from torch import nn

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.cudnn_sparse_attention import cudnn_sparse_attention, plan_sparse_indices
from nemo_automodel.components.models.ernie5_5.layers import DENSE_ATTENTION_BACKENDS, virtual_head_mix
from nemo_automodel.components.models.glm_moe_dsa.kernels.cudnn_dsa import (
    cudnn_indexer_topk,
    prepare_cudnn_dsa_packed_metadata,
)
from nemo_automodel.shared.import_utils import safe_import

# Optional CUDA kernel of the Hadamard transform. Without it the same arithmetic runs in PyTorch.
_HAS_FAST_HADAMARD, _fast_hadamard_transform = safe_import("fast_hadamard_transform")


def apply_indexer_rope(x: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    """Rotate the leading channels of every head in the half-split rotary layout.

    The first ``rope`` channels of a head are rotated, pairing channel ``i`` with channel ``i + rope / 2``; the
    remaining channels pass through. The cosine and sine are computed in the dtype of ``frequencies`` and then cast
    to the dtype of ``x``.

    Args:
        x: Tensor of shape [batch, sequence, heads, channels].
        frequencies: Rotation angles in radians, of shape [batch, sequence, rope], on the device of ``x``. ``rope`` is
            the number of rotated channels: positive, even and at most ``channels``. Entries ``i`` and
            ``i + rope / 2`` are the angle of one channel pair and are expected to be equal; this is not checked.

    Returns:
        Newly allocated tensor with the shape and dtype of ``x``: the first ``rope`` channels rotated, the other
        channels copied.

    Raises:
        ValueError: If ``x`` is not rank 4, ``frequencies`` is not rank 3, their batch and sequence dimensions differ,
            or ``rope`` is zero, odd or larger than ``channels``.
    """
    if x.ndim != 4 or frequencies.ndim != 3 or frequencies.shape[:2] != x.shape[:2]:
        raise ValueError("Expected x [B,S,H,D] and frequencies [B,S,R]")
    size = frequencies.shape[-1]
    if size <= 0 or size % 2 or size > x.shape[-1]:
        raise ValueError("RoPE width must be positive, even, and no wider than the head")
    pe, rest = x[..., :size], x[..., size:]
    first, second = pe.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    cos = frequencies.cos().to(x.dtype).unsqueeze(-2)
    sin = frequencies.sin().to(x.dtype).unsqueeze(-2)
    return torch.cat((pe * cos + rotated * sin, rest), dim=-1)


def apply_main_rope(x: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent channel pairs and return them in the half-split layout.

    The input pairs channel ``2i`` with channel ``2i + 1``. The output holds the rotated even channels in its first
    half and the rotated odd channels in its second half, which is the layout of ``apply_indexer_rope``.

    Args:
        x: Tensor of shape [batch, sequence, heads, rope], with ``rope`` even. Every channel is rotated.
        frequencies: Rotation angles in radians, of shape [batch, sequence, rope], on the device of ``x``; see
            ``apply_indexer_rope``.

    Returns:
        Newly allocated tensor with the shape and dtype of ``x``, in the half-split layout.

    Raises:
        ValueError: If ``x`` is not rank 4, ``frequencies`` is not rank 3, their final dimensions differ or are
            odd or zero, or their batch and sequence dimensions differ.
    """
    if x.ndim != 4 or frequencies.ndim != 3 or x.shape[-1] != frequencies.shape[-1] or x.shape[-1] % 2:
        raise ValueError("Expected [B,S,H,R] and matching even-width frequencies [B,S,R]")
    half_split = torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1)
    return apply_indexer_rope(half_split, frequencies)


def rotate_indexer_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply the scaled Hadamard rotation of the DSA indexer.

    Precision contract: the input, of any floating-point dtype, is rounded to BF16; the butterfly and the scaling by
    ``channels ** -0.5`` run in FP32; the result is rounded to BF16 once. On a CUDA device the
    ``fast_hadamard_transform`` kernel computes this when the package is installed. Without the package, and on
    every other device, the same arithmetic runs in PyTorch, so both paths return the same bits.

    Args:
        x: Floating-point tensor of shape [..., channels], with arbitrary leading dimensions; ``channels`` is a
            power of two.

    Returns:
        BF16 tensor of shape [..., channels] on the device of ``x``.

    Raises:
        ValueError: If ``channels`` is not a power of two.
    """
    channels = x.shape[-1]
    if channels < 1 or channels & (channels - 1):
        raise ValueError(f"The indexer rotation requires a power-of-two channel count; got {channels}")
    # First rounding: the kernel takes BF16 input, so the PyTorch path rounds an FP32 input the same way.
    x = x.to(torch.bfloat16)
    scale = channels**-0.5
    if x.is_cuda and _HAS_FAST_HADAMARD:
        return _fast_hadamard_transform.hadamard_transform(x, scale=scale)
    rotated = x.float().unsqueeze(-1)
    for _ in range(channels.bit_length() - 1):
        rotated = torch.cat(
            (rotated[..., ::2, :] + rotated[..., 1::2, :], rotated[..., ::2, :] - rotated[..., 1::2, :]), -1
        )
    return (rotated.squeeze(-2) * scale).to(torch.bfloat16)


def freeze_ernie5_5_indexer_params(model: nn.Module) -> None:
    """Set ``requires_grad`` to False on every parameter of every ``Ernie5_5Indexer`` in a model.

    The indexer only selects keys and receives no gradient from the attention output, so its parameters are kept
    out of the trainable set. The call changes the flags once and installs no hook: code that enables gradients
    afterwards, such as a trainability policy or ``model.requires_grad_(True)``, must call this function again.

    Args:
        model: Model, wrapper or pipeline stage that may contain ERNIE 5.5 DSA indexers. Parameters outside an
            ``Ernie5_5Indexer`` are left unchanged.
    """
    for module in model.modules():
        if isinstance(module, Ernie5_5Indexer):
            module.requires_grad_(False)


class Ernie5_5Indexer(nn.Module):
    """Project hidden states and the query latent to the indexer query, key and per-head weights.

    The outputs feed the index scores, which only decide which remote keys the attention reads. Both inputs are
    detached, so no gradient reaches the rest of the model through the indexer.

    Parameters: ``wq_b.weight`` of shape [heads * head_dim, q_rank], ``wk.weight`` of shape [head_dim, hidden] and
    ``weights_proj.weight`` of shape [heads, hidden] in the configured dtype, and ``k_norm.weight`` and
    ``k_norm.bias`` of shape [head_dim], which must stay FP32.
    """

    def __init__(
        self,
        hidden_size: int,
        q_rank: int,
        num_heads: int,
        head_dim: int,
        *,
        dtype: torch.dtype,
        norm_eps: float = 1e-6,
    ) -> None:
        """Create the three bias-free projections and the FP32 key LayerNorm.

        Args:
            hidden_size: Width of the input hidden states.
            q_rank: Width of the input query latent states.
            num_heads: Number of indexer heads.
            head_dim: Channels in each indexer head; a power of two.
            dtype: Floating-point dtype of the linear projections.
            norm_eps: Positive finite epsilon for the key LayerNorm.

        Raises:
            ValueError: If a dimension is not positive, ``head_dim`` is not a power of two (the activation
                rotation requires it) or ``norm_eps`` is not a positive finite number.
        """
        super().__init__()
        if min(hidden_size, q_rank, num_heads, head_dim) <= 0 or not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("Indexer dimensions and finite normalization epsilon must be positive")
        if head_dim & (head_dim - 1):
            raise ValueError(f"Indexer head_dim must be a power of two; got {head_dim}")
        self.num_heads, self.head_dim = num_heads, head_dim
        self.wq_b = nn.Linear(q_rank, num_heads * head_dim, bias=False, dtype=dtype)
        self.wk = nn.Linear(hidden_size, head_dim, bias=False, dtype=dtype)
        self.k_norm = nn.LayerNorm(head_dim, eps=norm_eps, dtype=torch.float32)
        self.weights_proj = nn.Linear(hidden_size, num_heads, bias=False, dtype=dtype)

    def forward(
        self,
        hidden: torch.Tensor,
        q_latent: torch.Tensor,
        *,
        rope: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project and rotate the indexer inputs without propagating gradients into the backbone.

        Args:
            hidden: Tensor of shape [batch, sequence, hidden]. It is detached before the projections.
            q_latent: Tensor of shape [batch, sequence, q_rank]. It is detached before the projection.
            rope: Position rotation. It is called with the query of shape [batch, sequence, heads, head_dim] and with
                the key of shape [batch, sequence, 1, head_dim], and must return a tensor of the same shape and dtype.
                Each result is then rotated by ``rotate_indexer_activation``.

        Returns:
            A tuple of the rotated query of shape [batch, sequence, heads, head_dim], the rotated key of shape
            [batch, sequence, head_dim] and the head weights of shape [batch, sequence, heads], all BF16. The key
            is normalized in FP32 with autocast disabled and cast back to the dtype of its projection before the
            position rotation; ``rotate_indexer_activation`` rounds the query and the key to BF16. The head
            weights are rounded to BF16 and then scaled by ``heads ** -0.5 * head_dim ** -0.5``.

        Raises:
            RuntimeError: If the weight or the bias of the key LayerNorm is not FP32.
        """
        if self.k_norm.weight.dtype != torch.float32 or self.k_norm.bias.dtype != torch.float32:
            raise RuntimeError("ERNIE DSA indexer key LayerNorm parameters must remain FP32")
        hidden, q_latent = hidden.detach(), q_latent.detach()
        q = self.wq_b(q_latent).unflatten(-1, (self.num_heads, self.head_dim))
        q = rotate_indexer_activation(rope(q))
        k = self.wk(hidden)
        with torch.autocast(device_type=k.device.type, enabled=False):
            k = self.k_norm(k.float()).to(k.dtype)
        k = rotate_indexer_activation(rope(k.unsqueeze(-2))).squeeze(-2)
        weights = self.weights_proj(hidden).to(torch.bfloat16) * self.num_heads**-0.5 * self.head_dim**-0.5
        return q, k, weights

    def init_weights(self, init_std: float = 0.02) -> None:
        """Initialize the projections and the key norm, including after meta-device materialization.

        Args:
            init_std: Positive finite standard deviation of the projection weights.

        Raises:
            ValueError: If ``init_std`` is not a positive finite number.
        """
        if not math.isfinite(init_std) or init_std <= 0:
            raise ValueError("init_std must be finite and positive")
        for linear in (self.wq_b, self.wk, self.weights_proj):
            nn.init.normal_(linear.weight, mean=0.0, std=init_std)
        self.k_norm.reset_parameters()


def ernie5_5_index_scores(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Score every key for every query with the ERNIE scaling order.

    The score of a query and a key is the sum over heads of ``relu(<q_head, k>) * weight_head``. The products and
    the sum are computed in FP32 with autocast disabled. The weights are multiplied by ``sqrt(channels)`` in their
    own dtype before the sum, and the sum is divided by ``sqrt(channels)`` afterwards; with BF16 weights this
    rounding order affects which keys a top-k selects.

    The scores are not masked. The caller applies the causal and document limits and selects the top-k, as
    ``select_window_and_remote`` does.

    Args:
        q: Indexer query of shape [batch, queries, heads, channels].
        k: Indexer key of shape [batch, keys, channels], shared by all heads.
        weights: Per-head weights of shape [batch, queries, heads].

    Returns:
        FP32 scores of shape [batch, queries, keys].

    Raises:
        ValueError: If ``q`` is not rank 4, ``k`` is not rank 3, ``weights`` is not [batch, queries, heads], or the
            batch or channel dimensions of ``q`` and ``k`` differ.
    """
    if q.ndim != 4 or k.ndim != 3 or weights.shape != q.shape[:3]:
        raise ValueError("Expected q [B,Q,H,D], k [B,K,D] and weights [B,Q,H]")
    if k.shape[0] != q.shape[0] or k.shape[-1] != q.shape[-1]:
        raise ValueError("Q and K batch and head dimensions must match")
    scale = q.shape[-1] ** 0.5
    with torch.autocast(device_type=q.device.type, enabled=False):
        dot = torch.einsum("bqhd,bkd->bqhk", q.float(), k.float()).relu()
        return (dot * (weights * scale).float().unsqueeze(-1)).sum(dim=2) / scale


@torch.no_grad()
def select_window_and_remote(
    scores: torch.Tensor, document_ids: torch.Tensor, *, window: int, remote_topk: int
) -> torch.Tensor:
    """Select, for every query, a causal local window and the top-scoring remote keys of its document.

    A key is eligible for a query when both are real tokens of the same document and the key is not after the
    query. The local window is the query and the ``window - 1`` eligible keys before it. The other eligible keys are
    remote, and the ``remote_topk`` highest-scoring of them are added. The function runs without gradient tracking.

    Args:
        scores: Finite floating tensor of shape [batch, sequence, sequence], indexed [batch, query, key].
        document_ids: Integer tensor of shape [batch, sequence] on the device of ``scores``. Tokens of one document
            carry the same id, and a negative id marks padding. The caller must give every contiguous document of a
            row its own id: tokens that share an id are treated as one document.
        window: Positive number of local keys per query, the query included.
        remote_topk: Non-negative number of remote keys per query. Zero selects the local window only.

    Returns:
        Boolean tensor of shape [batch, sequence, sequence], indexed [batch, query, key], that is True for the
        selected keys. The rows of padding queries are all False.

    Raises:
        ValueError: If ``scores`` is not [batch, sequence, sequence], ``document_ids`` is not [batch, sequence],
            ``window`` is not positive, ``remote_topk`` is negative or a score is not finite.
    """
    if scores.ndim != 3 or scores.shape[1] != scores.shape[2] or document_ids.shape != scores.shape[:2]:
        raise ValueError("Expected scores [batch, sequence, sequence] and matching document IDs")
    if window < 1 or remote_topk < 0 or not torch.isfinite(scores).all():
        raise ValueError("Scores must be finite, window positive and remote_topk non-negative")
    length = scores.shape[1]
    positions = torch.arange(length, device=scores.device)
    query, key = positions[:, None], positions[None, :]
    valid = document_ids >= 0
    same = document_ids[:, :, None] == document_ids[:, None, :]
    allowed = same & valid[:, :, None] & valid[:, None, :] & (key <= query)
    local = allowed & (key > query - window)
    remote = allowed & ~local
    count = min(remote_topk, length)
    if count == 0:
        return local
    top = scores.masked_fill(~remote, -torch.inf).topk(count, dim=-1)
    selected = torch.zeros_like(allowed)
    selected.scatter_(-1, top.indices, top.values.isfinite())
    return local | selected


def latent_mqa_core(
    query: torch.Tensor,
    latent_kv: torch.Tensor,
    scores: torch.Tensor,
    document_ids: torch.Tensor,
    sink: torch.Tensor,
    value_up: torch.Tensor,
    gate_logits: torch.Tensor,
    vha_u: torch.Tensor,
    vha_v: torch.Tensor,
    *,
    window: int,
    remote_topk: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Run the eager DSA core: latent multi-query attention with a sink, value up-projection, head mix and gate.

    Every head attends with its own query to one latent key/value per token, over the keys that
    ``select_window_and_remote`` selects. A per-head sink logit takes part in the softmax and its probability is
    discarded, so a query without a selected key (padding) produces zeros. The attended latent is projected to the
    value channels of each head, mixed across heads with ``virtual_head_mix`` and multiplied by the sigmoid of the
    gate logits.

    The logits, the softmax and the value up-projection are computed in FP32 with autocast disabled, whatever the
    input dtypes; the head mix and the gate run in the dtype of ``query``. The result is not guaranteed to equal the
    result of ``cudnn_dsa_core`` bit for bit.

    Args:
        query: Tensor of shape [batch, sequence, heads, latent + rope]: the query of each head in the key space,
            ``latent`` content channels followed by ``rope`` rotary channels.
        latent_kv: Tensor of shape [batch, sequence, latent + rope], the key shared by all heads. Its first
            ``latent`` channels are also the value.
        scores: Finite index scores of shape [batch, sequence, sequence], indexed [batch, query, key].
        document_ids: Integer tensor of shape [batch, sequence]; see ``select_window_and_remote``.
        sink: Per-head sink logits of shape [heads].
        value_up: Value up-projection of shape [heads, value_dim, latent].
        gate_logits: Tensor of shape [batch, sequence, heads * value_dim].
        vha_u: Virtual-head input factor of shape [heads, rank].
        vha_v: Virtual-head output factor of shape [heads, rank].
        window: Positive number of local keys per query.
        remote_topk: Non-negative number of remote keys per query.
        softmax_scale: Factor applied to the query-key products.

    Returns:
        Tensor of shape [batch, sequence, heads * value_dim] in the dtype of ``query``, with the value channels of
        the heads concatenated in head order.

    Raises:
        ValueError: If a rank or a dimension disagrees with the shapes above, or ``select_window_and_remote``
            rejects the scores, the document ids, ``window`` or ``remote_topk``.
    """
    if query.ndim != 4 or latent_kv.ndim != 3 or value_up.ndim != 3:
        raise ValueError("Expected rank-4 query and rank-3 latent KV and value projection")
    batch, sequence, heads, qdim = query.shape
    if latent_kv.shape != (batch, sequence, qdim) or value_up.shape[0] != heads or sink.shape != (heads,):
        raise ValueError("Query, latent KV, sink and value projection dimensions do not agree")
    value_dim, latent = value_up.shape[1:]
    if latent > qdim or gate_logits.shape != (batch, sequence, heads * value_dim):
        raise ValueError("Invalid latent width or output gate dimensions")
    selected = select_window_and_remote(scores, document_ids, window=window, remote_topk=remote_topk)
    if selected.shape != (batch, sequence, sequence):
        raise ValueError("Index scores and document IDs must match query batch and sequence")
    with torch.autocast(device_type=query.device.type, enabled=False):
        logits = torch.einsum("bqhd,bkd->bhqk", query.float(), latent_kv.float()) * softmax_scale
        logits = logits.masked_fill(~selected[:, None], -torch.inf)
        sink_column = sink.float().view(1, heads, 1, 1).expand(batch, heads, sequence, 1)
        probabilities = torch.cat((logits, sink_column), dim=-1).softmax(dim=-1)[..., :-1]
        latent_output = torch.einsum("bhqk,bkc->bqhc", probabilities, latent_kv[..., :latent].float())
        value_output = torch.einsum("bqhc,hvc->bqhv", latent_output, value_up.float())
        mixed = virtual_head_mix(value_output.to(query.dtype), vha_u, vha_v)
        return mixed.reshape(batch, sequence, heads * value_dim) * gate_logits.sigmoid().to(query.dtype)


def merge_remote_and_window(
    remote_indices: torch.Tensor,
    document_starts: torch.Tensor,
    causal_lengths: torch.Tensor,
    query_valid: torch.Tensor,
    *,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine the remote indices of every query with its local causal window.

    Queries and keys are the tokens of one unpadded row, and an index is the offset of a token in that row. The
    remote part of the result keeps the input order and replaces with -1 every index that is absent, belongs to
    another document, or lies inside or after the local window. The local part lists the last ``window`` offsets up
    to the query in ascending order, with -1 for positions before the start of the document.

    All inputs must describe contiguous real documents. Their values are not checked for consistency, and
    context-parallel and padded-storage metadata are not validated.

    Args:
        remote_indices: Int32 tensor of shape [queries, remote_topk] with row offsets of remote keys, or -1 for
            absent entries.
        document_starts: Int32 tensor of shape [queries] with the row offset of the first token of each query's
            document.
        causal_lengths: Int32 tensor of shape [queries] counting the tokens from the start of the document through
            the query.
        query_valid: Boolean tensor of shape [queries]. A False entry gives a row of -1.
        window: Positive number of local causal positions per query.

    Returns:
        The int32 indices of shape [queries, remote_topk + window], remote part first, with invalid entries set to
        -1, and the int32 length of the remote range of each query, ``max(causal_length - window, 0)``, of shape
        [queries]. The indices are neither sorted nor compacted; a consumer whose kernel requires that must do it.

    Raises:
        ValueError: If ``window`` is not a positive integer, ``remote_indices`` is not an int32 matrix, a metadata
            tensor does not have shape [queries], the starts or the lengths are not int32, the validity is not
            boolean, or the tensors are on different devices.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    if remote_indices.ndim != 2 or remote_indices.dtype != torch.int32:
        raise ValueError("remote_indices must be int32 [queries, topk]")
    n = remote_indices.shape[0]
    if document_starts.shape != (n,) or causal_lengths.shape != (n,) or query_valid.shape != (n,):
        raise ValueError("Expected per-query metadata matching the remote indices")
    if document_starts.dtype != torch.int32 or causal_lengths.dtype != torch.int32 or query_valid.dtype != torch.bool:
        raise ValueError("Expected int32 starts/lengths and bool validity")
    if any(t.device != remote_indices.device for t in (document_starts, causal_lengths, query_valid)):
        raise ValueError("All indices and metadata must share a device")
    remote_lengths = (causal_lengths - window).clamp_min(0)
    remote_end = document_starts + remote_lengths
    valid = query_valid.unsqueeze(1)
    good_remote = valid & (remote_indices >= document_starts.unsqueeze(1)) & (remote_indices < remote_end.unsqueeze(1))
    remote = remote_indices.masked_fill(~good_remote, -1)
    offsets = torch.arange(window, device=remote.device, dtype=torch.int32)
    local = document_starts[:, None] + causal_lengths[:, None] - window + offsets
    good_local = valid & (local >= document_starts[:, None]) & (local < (document_starts + causal_lengths)[:, None])
    local = local.masked_fill(~good_local, -1)
    return torch.cat((remote, local), dim=1), remote_lengths


def cudnn_dsa_core(
    query: torch.Tensor,
    latent_kv: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_weights: torch.Tensor,
    sink: torch.Tensor,
    value_up: torch.Tensor,
    gate_logits: torch.Tensor,
    vha_u: torch.Tensor,
    vha_v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    max_seqlen: int,
    window: int,
    remote_topk: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Run the cuDNN DSA core on the tokens of one unpadded row that holds one document.

    ``cudnn_indexer_topk`` selects the remote keys of every query, ``merge_remote_and_window`` adds the local
    window, and ``cudnn_sparse_attention`` attends over the union with the per-head sink. The attended latent is
    then projected to the value channels of each head in FP32, mixed across heads and gated in the dtype of
    ``query``, as in ``latent_mqa_core``. The indexer inputs are detached, so the selection receives no gradient.

    Queries and keys are the same ``tokens`` rows. Context parallelism is not supported, and neither is a row
    that holds several documents: ``cu_seqlens`` with more than two entries raises ``NotImplementedError``. Before
    the first kernel is called, ``prepare_cudnn_dsa_packed_metadata`` validates the rest of the document metadata:
    it raises ``TypeError`` unless ``cu_seqlens`` is int32, and ``ValueError`` unless ``cu_seqlens`` is
    one-dimensional, starts at zero and ends at ``tokens`` and ``max_seqlen`` equals ``tokens``.

    All tensors must be on one CUDA device. The two kernels accept BF16 inputs with ``latent + rope`` equal to 512
    or 576, ``index_dim`` equal to 128, 32 or 64 indexer heads and ``remote_topk`` of at most 2048, and raise
    otherwise.

    Args:
        query: BF16 tensor of shape [tokens, heads, latent + rope].
        latent_kv: BF16 tensor of shape [tokens, latent + rope], the key shared by all heads. Its first ``latent``
            channels are also the value.
        index_q: BF16 indexer query of shape [tokens, index_heads, index_dim].
        index_k: BF16 indexer key of shape [tokens, index_dim].
        index_weights: FP32 or BF16 indexer head weights of shape [tokens, index_heads], as ``Ernie5_5Indexer``
            returns them. They are multiplied by ``sqrt(index_dim)`` in their own dtype before the kernel.
        sink: FP32 per-head sink logits of shape [heads].
        value_up: Value up-projection of shape [heads, value_dim, latent].
        gate_logits: Tensor of shape [tokens, heads * value_dim].
        vha_u: Virtual-head input factor of shape [heads, rank].
        vha_v: Virtual-head output factor of shape [heads, rank].
        cu_seqlens: Int32 tensor ``[0, tokens]`` of shape [2]: the cumulative document lengths of a row that holds
            one document.
        max_seqlen: Length of the document, equal to ``tokens``.
        window: Positive number of local causal keys per query.
        remote_topk: Positive number of remote keys the indexer selects per query.
        softmax_scale: Factor applied to the query-key products.

    Returns:
        Core output of shape [tokens, heads * value_dim] in the dtype of ``query``.

    Raises:
        ValueError: If ``tokens`` is zero, the shapes of the query, the latent key/value, the value projection or
            the gate logits disagree, ``latent`` exceeds ``latent + rope``, or ``window`` or ``remote_topk`` is not
            positive.
        NotImplementedError: If ``cu_seqlens`` has more than two entries, that is, the row holds several documents.
    """
    tokens, heads, qdim = query.shape
    if tokens == 0 or latent_kv.shape != (tokens, qdim) or value_up.shape[0] != heads:
        raise ValueError("Incompatible query, latent KV or value projection")
    value_dim, latent = value_up.shape[1:]
    if gate_logits.shape != (tokens, heads * value_dim) or latent > qdim:
        raise ValueError("Incompatible output gate or latent dimensions")
    if window < 1 or remote_topk < 1:
        raise ValueError("The cuDNN DSA core requires a positive window and a positive remote top-k")
    if cu_seqlens.ndim == 1 and cu_seqlens.shape[0] > 2:
        raise NotImplementedError(
            "The cuDNN DSA core does not support a row that holds several documents; "
            f"cu_seqlens must have two entries, got {cu_seqlens.shape[0]}"
        )
    metadata = prepare_cudnn_dsa_packed_metadata(cu_seqlens, tokens, max_seqlen=max_seqlen)
    remote_metadata = replace(metadata, causal_lengths=(metadata.causal_lengths - window).clamp_min(0))
    remote = cudnn_indexer_topk(
        index_q.detach(),
        index_k.detach(),
        index_weights.detach() * (index_q.shape[-1] ** 0.5),
        cu_seqlens,
        remote_topk,
        packed_metadata=remote_metadata,
    ).squeeze(1)
    union, _ = merge_remote_and_window(
        remote,
        metadata.starts,
        metadata.causal_lengths,
        metadata.query_valid,
        window=window,
    )
    indices, counts = plan_sparse_indices(union, tokens)
    attended = cudnn_sparse_attention(
        query,
        latent_kv[:, None],
        indices[:, None],
        softmax_scale,
        topk_length=counts,
        all_rows_nonempty=True,
        attn_sink=sink,
    )
    with torch.autocast(device_type=query.device.type, enabled=False):
        values = torch.einsum("qhc,hvc->qhv", attended[..., :latent].float(), value_up.float())
        mixed = virtual_head_mix(values.to(query.dtype), vha_u, vha_v)
        return mixed.reshape(tokens, heads * value_dim) * gate_logits.sigmoid().to(query.dtype)


class Ernie5_5DSAFP32Parameters(nn.Module):
    """Hold the FP32 sink and virtual-head parameters of a DSA layer and run the eager core with them.

    The three parameters form one module so that a dtype-aware FSDP wrap can treat them as one FP32 unit.

    Parameters, all FP32: ``softmax_offset`` of shape [heads], the per-head sink logits, initialized to zeros;
    ``vha_postmix_U`` of shape [heads, rank], allocated without initialization and filled by the owning layer;
    ``vha_postmix_V`` of shape [heads, rank], initialized to zeros.
    """

    def __init__(self, num_heads: int, vha_rank: int) -> None:
        """Allocate the sink logits and the two virtual-head factors.

        Args:
            num_heads: Number of attention heads.
            vha_rank: Rank of the virtual-head mixing factors.
        """
        super().__init__()
        self.softmax_offset = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
        self.vha_postmix_U = nn.Parameter(torch.empty(num_heads, vha_rank, dtype=torch.float32))
        self.vha_postmix_V = nn.Parameter(torch.zeros(num_heads, vha_rank, dtype=torch.float32))

    def forward(
        self,
        query: torch.Tensor,
        latent_kv: torch.Tensor,
        scores: torch.Tensor,
        document_ids: torch.Tensor,
        value_up: torch.Tensor,
        gate_logits: torch.Tensor,
        *,
        window: int,
        remote_topk: int,
        softmax_scale: float,
    ) -> torch.Tensor:
        """Run ``latent_mqa_core`` with the parameters of this module.

        The core runs inside this module's forward because returning the parameters themselves would allow an FSDP
        post-forward reshard to free their storage before the caller uses them. ``latent_mqa_core`` validates the
        inputs and raises ``ValueError`` for shapes that disagree.

        Args:
            query: Tensor of shape [batch, sequence, heads, latent + rope].
            latent_kv: Tensor of shape [batch, sequence, latent + rope] shared by all heads.
            scores: Finite index scores of shape [batch, sequence, sequence], indexed [batch, query, key].
            document_ids: Integer tensor of shape [batch, sequence]; negative ids are padding.
            value_up: Value up-projection of shape [heads, value_dim, latent].
            gate_logits: Tensor of shape [batch, sequence, heads * value_dim].
            window: Positive number of local causal keys per query.
            remote_topk: Non-negative number of remote keys per query.
            softmax_scale: Factor applied to the query-key products.

        Returns:
            Core output of shape [batch, sequence, heads * value_dim] in the dtype of ``query``.

        Raises:
            RuntimeError: If one of the three parameters of this module is not FP32.
        """
        if any(p.dtype != torch.float32 for p in (self.softmax_offset, self.vha_postmix_U, self.vha_postmix_V)):
            raise RuntimeError("ERNIE DSA sink and VHA parameters must remain FP32")
        return latent_mqa_core(
            query,
            latent_kv,
            scores,
            document_ids,
            self.softmax_offset,
            value_up,
            gate_logits,
            self.vha_postmix_U,
            self.vha_postmix_V,
            window=window,
            remote_topk=remote_topk,
            softmax_scale=softmax_scale,
        )


class Ernie5_5DSASparseFP32Parameters(Ernie5_5DSAFP32Parameters):
    """Hold the same FP32 parameters as the parent class and run the cuDNN core with them.

    The core runs inside this module for the reason given in ``Ernie5_5DSAFP32Parameters.forward``.
    """

    def forward(
        self,
        query: torch.Tensor,
        latent_kv: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        index_weights: torch.Tensor,
        value_up: torch.Tensor,
        gate_logits: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        max_seqlen: int,
        window: int,
        remote_topk: int,
        softmax_scale: float,
    ) -> torch.Tensor:
        """Run ``cudnn_dsa_core`` with the parameters of this module.

        ``cudnn_dsa_core`` validates the inputs: it raises ``ValueError`` for shapes that disagree and
        ``NotImplementedError`` for a row that holds several documents.

        Args:
            query: BF16 tensor of shape [tokens, heads, latent + rope].
            latent_kv: BF16 tensor of shape [tokens, latent + rope] shared by all heads.
            index_q: BF16 indexer query of shape [tokens, index_heads, index_dim].
            index_k: BF16 indexer key of shape [tokens, index_dim].
            index_weights: Indexer head weights of shape [tokens, index_heads].
            value_up: Value up-projection of shape [heads, value_dim, latent].
            gate_logits: Tensor of shape [tokens, heads * value_dim].
            cu_seqlens: Int32 tensor ``[0, tokens]`` of shape [2]: the cumulative document lengths of a row that
                holds one document.
            max_seqlen: Length of the document, equal to ``tokens``.
            window: Positive number of local causal keys per query.
            remote_topk: Positive number of remote keys the indexer selects per query.
            softmax_scale: Factor applied to the query-key products.

        Returns:
            Core output of shape [tokens, heads * value_dim] in the dtype of ``query``.

        Raises:
            RuntimeError: If one of the three parameters of this module is not FP32.
        """
        if any(p.dtype != torch.float32 for p in (self.softmax_offset, self.vha_postmix_U, self.vha_postmix_V)):
            raise RuntimeError("ERNIE DSA sink and VHA parameters must remain FP32")
        return cudnn_dsa_core(
            query,
            latent_kv,
            index_q,
            index_k,
            index_weights,
            self.softmax_offset,
            value_up,
            gate_logits,
            self.vha_postmix_U,
            self.vha_postmix_V,
            cu_seqlens,
            max_seqlen=max_seqlen,
            window=window,
            remote_topk=remote_topk,
            softmax_scale=softmax_scale,
        )


class Ernie5_5DSAAttention(nn.Module):
    """ERNIE 5.5 dynamic sparse attention (DSA) layer.

    Latent multi-query attention over a causal local window plus the remote keys
    an indexer selects, followed by the value up-projection, virtual-head mixing,
    the query gate and the output projection.

    The attention backend selects one of two paths that share all parameters:

    - ``backend.attn == "cudnn"``: cuDNN top-k selection and sparse attention for
      one unpadded row that holds one document; a row with several documents is
      rejected. The forward runs with autocast disabled, and the kernels require
      BF16 inputs and the released latent and indexer dimensions.
    - ``backend.attn`` in ``("eager", "torch", "sdpa")``, or no backend: eager
      PyTorch attention over a dense [sequence, sequence] mask, limited to
      ``max_eager_sequence`` tokens. The three names select this same path; no
      scaled-dot-product kernel is called.

    Every other value of ``backend.attn`` is rejected.

    Neither path supports context parallelism or a KV cache.

    The indexer only selects keys. Its projections receive no gradient from the
    attention output, and this layer computes no indexer loss, so the indexer is
    not trained here; ``freeze_ernie5_5_indexer_params`` keeps it frozen. This
    differs from the reference training recipe, which trains the indexer with a
    KL term against the attention distribution.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_rank: int,
        kv_rank: int,
        nope_dim: int,
        rope_dim: int,
        value_dim: int,
        index_heads: int,
        index_dim: int,
        vha_rank: int,
        *,
        window: int,
        remote_topk: int,
        dtype: torch.dtype,
        rope_theta: float = 8_000_000.0,
        norm_eps: float = 1e-6,
        max_eager_sequence: int = 512,
        backend: BackendConfig | None = None,
    ) -> None:
        """Build the projections, the indexer and the FP32 parameter holder.

        Args:
            hidden_size: Width of the hidden states.
            num_heads: Number of attention heads.
            q_rank: Rank of the query latent.
            kv_rank: Rank of the shared key/value latent.
            nope_dim: Channels per head without rotary position.
            rope_dim: Rotary channels per head; even and at most ``index_dim``.
            value_dim: Value channels per head.
            index_heads: Number of indexer heads.
            index_dim: Channels per indexer head; a power of two.
            vha_rank: Rank of the virtual-head mixing factors.
            window: Number of local causal keys per query.
            remote_topk: Number of remote keys the indexer selects per query. The eager path accepts zero, which
                selects the local window only; the cuDNN path requires at least one.
            dtype: Floating-point dtype of the projections.
            rope_theta: Rotary base.
            norm_eps: Epsilon of the latent norms and the indexer key norm.
            max_eager_sequence: Longest sequence the eager path accepts.
            backend: Backend selection; only ``attn`` is read. None selects the eager path.

        Raises:
            ValueError: If a dimension, the window, the top-k or the rotary base is invalid, ``index_dim`` is not
                a power of two, or ``remote_topk`` is zero with ``backend.attn == "cudnn"``.
            NotImplementedError: If ``backend.attn`` is neither ``"cudnn"`` nor one of the dense backends.
        """
        super().__init__()
        if (
            min(
                hidden_size, num_heads, q_rank, kv_rank, nope_dim, rope_dim, value_dim, index_heads, index_dim, vha_rank
            )
            <= 0
        ):
            raise ValueError("All attention dimensions must be positive")
        if rope_dim % 2 or rope_dim > index_dim or window < 1 or remote_topk < 0 or rope_theta <= 0:
            raise ValueError("Invalid rotary dimension, window, top-k, or rotary base")
        if max_eager_sequence < 1 or norm_eps <= 0:
            raise ValueError("max_eager_sequence and norm_eps must be positive")
        attn = backend.attn if backend is not None else "eager"
        if attn != "cudnn" and attn not in DENSE_ATTENTION_BACKENDS:
            raise NotImplementedError(
                f"ERNIE DSA supports attn='cudnn' and the dense backends {DENSE_ATTENTION_BACKENDS}; got attn={attn!r}"
            )
        if attn == "cudnn" and remote_topk < 1:
            raise ValueError(f"The cuDNN DSA path requires a positive remote top-k; got remote_topk={remote_topk}")
        self.uses_cudnn = attn == "cudnn"
        self.num_heads, self.kv_rank = num_heads, kv_rank
        self.nope_dim, self.rope_dim, self.value_dim = nope_dim, rope_dim, value_dim
        self.window, self.remote_topk, self.rope_theta = window, remote_topk, rope_theta
        self.max_eager_sequence = max_eager_sequence
        self.softmax_scale = (nope_dim + rope_dim) ** -0.5

        self.q_a_proj = nn.Linear(hidden_size, q_rank, bias=False, dtype=dtype)
        self.q_a_layernorm = nn.RMSNorm(q_rank, eps=norm_eps, dtype=dtype)
        self.q_b_proj = nn.Linear(q_rank, num_heads * (nope_dim + rope_dim), bias=False, dtype=dtype)
        self.kv_a_proj_with_mqa = nn.Linear(hidden_size, kv_rank + rope_dim, bias=False, dtype=dtype)
        self.kv_a_layernorm = nn.RMSNorm(kv_rank, eps=norm_eps, dtype=dtype)
        # Separate K/V parameters preserve the reference's Muon group layout.
        self.k_b_proj = nn.Parameter(torch.empty(num_heads * kv_rank, nope_dim, dtype=dtype))
        self.v_b_proj = nn.Parameter(torch.empty(num_heads * value_dim, kv_rank, dtype=dtype))
        self.indexer = Ernie5_5Indexer(hidden_size, q_rank, index_heads, index_dim, dtype=dtype, norm_eps=norm_eps)
        holder = Ernie5_5DSASparseFP32Parameters if self.uses_cudnn else Ernie5_5DSAFP32Parameters
        self.fp32_params = holder(num_heads, vha_rank)
        self.gate_proj = nn.Linear(q_rank, num_heads * value_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(num_heads * value_dim, hidden_size, bias=False, dtype=dtype)
        nn.init.normal_(self.k_b_proj, std=0.02)
        nn.init.normal_(self.v_b_proj, std=0.02)
        nn.init.normal_(self.vha_postmix_U, std=0.02)

    @property
    def softmax_offset(self) -> torch.Tensor:
        """FP32 per-head sink logits of shape [heads], owned by ``fp32_params``."""
        return self.fp32_params.softmax_offset

    @property
    def vha_postmix_U(self) -> torch.Tensor:
        """FP32 virtual-head input factor of shape [heads, vha_rank], owned by ``fp32_params``."""
        return self.fp32_params.vha_postmix_U

    @property
    def vha_postmix_V(self) -> torch.Tensor:
        """FP32 virtual-head output factor of shape [heads, vha_rank], owned by ``fp32_params``."""
        return self.fp32_params.vha_postmix_V

    def init_weights(self, init_std: float = 0.02) -> None:
        """Initialize all parameters, including after meta-device materialization.

        The projections, the two separate K/V parameters and ``vha_postmix_U`` are drawn from a normal
        distribution, the norms are reset, and ``vha_postmix_V`` and the sink logits are set to zero.

        Args:
            init_std: Positive finite standard deviation of the normal draws.

        Raises:
            ValueError: If ``init_std`` is not a positive finite number. No parameter is changed in that case.
        """
        if not math.isfinite(init_std) or init_std <= 0:
            raise ValueError("init_std must be finite and positive")
        for linear in (self.q_a_proj, self.q_b_proj, self.kv_a_proj_with_mqa, self.gate_proj, self.o_proj):
            nn.init.normal_(linear.weight, mean=0.0, std=init_std)
        self.q_a_layernorm.reset_parameters()
        self.kv_a_layernorm.reset_parameters()
        self.indexer.init_weights(init_std)
        nn.init.normal_(self.k_b_proj, mean=0.0, std=init_std)
        nn.init.normal_(self.v_b_proj, mean=0.0, std=init_std)
        nn.init.normal_(self.vha_postmix_U, mean=0.0, std=init_std)
        nn.init.zeros_(self.vha_postmix_V)
        nn.init.zeros_(self.softmax_offset)

    def _project(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states to the attention and indexer inputs.

        Args:
            hidden: Tensor of shape [batch, sequence, hidden].
            position_ids: Integer tensor of shape [batch, sequence].

        Returns:
            The query latent [batch, sequence, q_rank], the attention query
            [batch, sequence, heads, kv_rank + rope], the shared latent key/value
            [batch, sequence, kv_rank + rope], and the BF16 indexer Q
            [batch, sequence, index_heads, index_dim], K [batch, sequence, index_dim]
            and head weights [batch, sequence, index_heads].
        """
        inverse = self.rope_theta ** (
            -torch.arange(0, self.rope_dim, 2, device=hidden.device, dtype=torch.float32) / self.rope_dim
        )
        angles = position_ids.to(torch.float32).unsqueeze(-1) * inverse
        frequencies = torch.cat((angles, angles), dim=-1)
        q_latent = self.q_a_layernorm(self.q_a_proj(hidden))
        q = self.q_b_proj(q_latent).unflatten(-1, (self.num_heads, self.nope_dim + self.rope_dim))
        q_nope, q_rope = q.split((self.nope_dim, self.rope_dim), dim=-1)
        kv, k_rope = self.kv_a_proj_with_mqa(hidden).split((self.kv_rank, self.rope_dim), dim=-1)
        kv = self.kv_a_layernorm(kv)
        k_weight = self.k_b_proj.reshape(self.num_heads, self.kv_rank, self.nope_dim)
        absorbed = torch.einsum("bqhd,hcd->bqhc", q_nope, k_weight)
        query = torch.cat((absorbed, apply_main_rope(q_rope, frequencies)), dim=-1)
        latent_kv = torch.cat((kv, apply_main_rope(k_rope.unsqueeze(-2), frequencies).squeeze(-2)), dim=-1)
        index_q, index_k, index_weights = self.indexer(
            hidden, q_latent, rope=lambda value: apply_indexer_rope(value, frequencies)
        )
        return q_latent, query, latent_kv, index_q, index_k, index_weights

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        document_ids: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Return the attention output.

        The eager path takes ``document_ids``; the cuDNN path takes ``cu_seqlens``
        and ``max_seqlen``. Passing the metadata of the other path raises. On the
        cuDNN path a row that holds several documents is rejected here, before the
        projections run; ``cudnn_dsa_core`` describes the remaining checks of
        ``cu_seqlens`` and ``max_seqlen``.

        Args:
            hidden: Tensor of shape [batch, sequence, hidden]. The cuDNN path requires batch 1.
            position_ids: Integer tensor of shape [batch, sequence]; positions restart for each document.
            document_ids: Eager path only. Integer tensor of shape [batch, sequence] that
                identifies each contiguous document of a row; negative ids are padding.
            cu_seqlens: cuDNN path only. Int32 tensor ``[0, sequence]`` of shape [2]: the
                cumulative document lengths of an unpadded row that holds one document.
            max_seqlen: cuDNN path only. Length of the document, equal to ``sequence``.

        Returns:
            Output of shape [batch, sequence, hidden], and None in the place where
            other attention layers return attention weights.

        Raises:
            ValueError: If shapes disagree, the metadata does not match the path, or
                the eager path receives no token or more than ``max_eager_sequence`` tokens.
            NotImplementedError: If ``cu_seqlens`` has more than two entries, that is,
                the row holds several documents.
        """
        if self.uses_cudnn:
            if cu_seqlens is None or max_seqlen is None or document_ids is not None:
                raise ValueError("The cuDNN DSA path requires cu_seqlens and max_seqlen and no document_ids")
            if cu_seqlens.ndim == 1 and cu_seqlens.shape[0] > 2:
                raise NotImplementedError(
                    "The cuDNN DSA path does not support a row that holds several documents; "
                    f"cu_seqlens must have two entries, got {cu_seqlens.shape[0]}"
                )
            # The cuDNN kernels require BF16 inputs: keep the tested operation dtypes under an outer autocast.
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                output = self._forward_cudnn(hidden, position_ids, cu_seqlens, max_seqlen=max_seqlen)
            return output, None
        if document_ids is None or cu_seqlens is not None or max_seqlen is not None:
            raise ValueError("The eager DSA path requires document_ids and no cu_seqlens or max_seqlen")
        return self._forward_eager(hidden, document_ids, position_ids), None

    def _forward_eager(
        self,
        hidden: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the eager path over a dense [sequence, sequence] selection.

        Args:
            hidden: Tensor of shape [batch, sequence, hidden], with at most ``max_eager_sequence`` tokens per row.
            document_ids: Integer tensor of shape [batch, sequence] that identifies each contiguous document of a
                row; negative ids are padding.
            position_ids: Integer tensor of shape [batch, sequence]; positions restart for each document.

        Returns:
            Output of shape [batch, sequence, hidden].

        Raises:
            ValueError: If ``hidden`` is not rank 3, the ids do not have shape [batch, sequence], or the sequence
                is empty or longer than ``max_eager_sequence``.
        """
        if hidden.ndim != 3 or document_ids.shape != hidden.shape[:2] or position_ids.shape != hidden.shape[:2]:
            raise ValueError("Expected hidden [B,S,H] and matching document and position IDs [B,S]")
        batch, sequence, _ = hidden.shape
        if sequence < 1 or sequence > self.max_eager_sequence:
            raise ValueError("Sequence length must be between 1 and max_eager_sequence on the eager DSA path")
        q_latent, query, latent_kv, index_q, index_k, index_weights = self._project(hidden, position_ids)
        scores = ernie5_5_index_scores(index_q, index_k, index_weights)
        core = self.fp32_params(
            query,
            latent_kv,
            scores,
            document_ids,
            self.v_b_proj.reshape(self.num_heads, self.value_dim, self.kv_rank),
            self.gate_proj(q_latent),
            window=self.window,
            remote_topk=self.remote_topk,
            softmax_scale=self.softmax_scale,
        )
        return self.o_proj(core)

    def _forward_cudnn(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Run the cuDNN path on one unpadded row that holds one document.

        Args:
            hidden: Tensor of shape [1, sequence, hidden]: one row without padding.
            position_ids: Integer tensor of shape [1, sequence] with the position of each token in the document.
            cu_seqlens: Int32 tensor ``[0, sequence]`` of shape [2]: the cumulative document lengths of the row.
            max_seqlen: Length of the document, equal to ``sequence``.

        Returns:
            Output of shape [1, sequence, hidden].

        Raises:
            ValueError: If ``hidden`` is not [1, sequence, hidden] or ``position_ids`` is not [1, sequence].
        """
        if hidden.ndim != 3 or hidden.shape[0] != 1 or position_ids.shape != hidden.shape[:2]:
            raise ValueError("Expected one unpadded packed row and matching positions")
        q_latent, query, latent_kv, index_q, index_k, index_weights = self._project(hidden, position_ids)
        core = self.fp32_params(
            query[0],
            latent_kv[0],
            index_q[0],
            index_k[0],
            index_weights[0],
            self.v_b_proj.reshape(self.num_heads, self.value_dim, self.kv_rank),
            self.gate_proj(q_latent)[0],
            cu_seqlens,
            max_seqlen=max_seqlen,
            window=self.window,
            remote_topk=self.remote_topk,
            softmax_scale=self.softmax_scale,
        )
        return self.o_proj(core[None])
