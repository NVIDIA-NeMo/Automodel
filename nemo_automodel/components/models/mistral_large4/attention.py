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

"""Floating DSA/MLA baseline with explicit, differentiable per-forward KV sharing."""

from dataclasses import dataclass

import torch
import torch.distributed._functional_collectives as funcol
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import functional as F

from nemo_automodel.components.models.common import BackendConfig, initialize_rms_norm_module
from nemo_automodel.components.models.glm_moe_dsa.optimized_kernels import (
    is_dsa_kernel_available,
    tilelang_sparse_attention,
)
from nemo_automodel.components.models.mistral_large4.config import MistralLarge4Config
from nemo_automodel.components.models.mistral_large4.context_parallel import gather_tokens
from nemo_automodel.components.models.mistral_large4.rope_utils import (
    FP32LayerNorm,
    SSMax,
    apply_interleaved_rope,
)


def sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    value_weight: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Run sparse MLA on local packed queries and gathered latent KV.

    Args:
        q: Contiguous CUDA BF16 [local_tokens, heads, 576] absorbed queries.
            The last 64 coordinates are rotary keys, or zeros in NoPE layers.
        kv: CUDA BF16 [global_tokens, 576] latent/rotary KV in CP-rank order.
        indices: Integer [local_tokens, selected_keys] global KV coordinates;
            -1 denotes invalid slots. Caller enforces document/causal boundaries.
        value_weight: BF16 [heads, value_dim, 512] differentiable KV-B value weight.
        scale: Original Mistral head-dimension attention scale.

    Returns:
        BF16 [local_tokens, heads, value_dim]. Empty query rows are zero and
        retain autograd dependencies, including the CP collective on KV.
        Inputs are never mutated. No dense query-by-global-key mask is built.
    """
    if not is_dsa_kernel_available("sparse_attn"):
        raise RuntimeError("Mistral TileLang sparse MLA requires the TileLang kernel dependencies")
    if q.device.type != "cuda" or q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise ValueError("Mistral TileLang sparse MLA requires CUDA BF16 queries and KV")
    if q.shape[-1] != 576 or kv.shape[-1] != 576 or value_weight.shape[-1] != 512:
        raise ValueError("Mistral TileLang sparse MLA requires latent dimension 512 and rotary tail 64")
    # External CUDA kernels read raw pointers and do not trigger PyTorch tensor
    # dispatch. Complete and unwrap functional all-gather before passing KV to
    # TileLang; wait_tensor preserves the collective autograd edge.
    kv = funcol.wait_tensor(kv)
    # The kernel consumes selected KV only; its block width is 64. Keep
    # real top-k choices unchanged and fill extra slots with the -1 sentinel.
    selected = F.pad(indices.to(torch.int32), (0, (-indices.shape[-1]) % 64), value=-1)
    rows = (selected >= 0).any(-1).nonzero(as_tuple=True)[0]
    # Do not send all-masked queries to the sparse softmax (0/0). Keep the KV
    # autograd edge even when this CP rank owns no valid queries, so every peer
    # executes the same backward collectives and returns remote-key gradients.
    zero = (q.sum() + kv.sum() + value_weight.sum()) * 0.0
    output = q.new_zeros((q.shape[0], q.shape[1], value_weight.shape[1])) + zero
    if rows.numel():
        values = tilelang_sparse_attention(
            q.index_select(0, rows).contiguous(),
            kv[:, None].contiguous(),
            selected.index_select(0, rows)[:, None].contiguous(),
            value_weight,
            scale,
        )
        output = output.index_copy(0, rows, values)
    return output


@dataclass(frozen=True)
class SequenceLayout:
    """Local token metadata plus optional CP-global keys, all flattened in rank order."""

    positions: torch.Tensor
    sequence_ids: torch.Tensor
    valid: torch.Tensor
    cp_mesh: DeviceMesh | None = None
    key_sequence_ids: torch.Tensor | None = None
    key_valid: torch.Tensor | None = None

    def allowed(self, start: int, end: int) -> torch.Tensor:
        """Return an independent bool [query_tokens, key_tokens] causal document mask."""
        key_doc = self.sequence_ids if self.key_sequence_ids is None else self.key_sequence_ids
        key_valid = self.valid if self.key_valid is None else self.key_valid
        rows = torch.arange(key_valid.numel(), device=self.valid.device)
        offset = 0 if self.cp_mesh is None else self.cp_mesh.get_local_rank() * self.valid.numel()
        queries = torch.arange(start, end, device=self.valid.device) + offset
        return (
            (self.sequence_ids[start:end, None] == key_doc[None, :])
            & (rows[None, :] <= queries[:, None])
            & key_valid[None, :]
            & self.valid[start:end, None]
        )


class MistralLarge4Indexer(nn.Module):
    """Frozen hard DSA selector; an indexer training objective has not been specified."""

    def __init__(self, config: MistralLarge4Config, *, layer_idx: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = config
        self.use_rope = not config.indexer_nope_layer_flags[layer_idx]
        self.q_input_norm = nn.LayerNorm(config.hidden_size, eps=config.indexer_norm_eps, dtype=dtype)
        self.wq_b = nn.Linear(config.hidden_size, config.index_n_heads * config.index_head_dim, bias=False, dtype=dtype)
        self.wk = nn.Linear(config.hidden_size, config.index_head_dim, bias=False, dtype=dtype)
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False, dtype=dtype)
        self.k_norm = FP32LayerNorm(config.index_head_dim, eps=config.indexer_norm_eps, dtype=torch.float32)
        self.ssmax = SSMax(config.index_n_heads, dtype) if config.indexer_ssmax_scaling else None
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        tables: tuple[torch.Tensor, torch.Tensor],
        layout: SequenceLayout,
    ) -> torch.Tensor:
        """Select valid causal keys with stable ascending-key tie breaking.

        Args:
            x: Floating tensor [tokens, hidden].
            tables: FP32 cosine/sine pair, each [tokens, rotary_dim / 2].
            layout: Local positions/document IDs/validity [tokens], optional global CP key metadata.

        Returns:
            Independent int64 indices [tokens, min(topk, tokens)], with -1 for
            missing keys. Selection is discrete and carries no LM-loss gradient.
            This floating baseline does not round queries/keys to serving FP8.
        """
        if any(p.requires_grad for p in self.parameters()):
            raise ValueError("DSA indexer must remain frozen until its training objective is specified")
        c = self.config
        q = self.wq_b(self.q_input_norm(x)).unflatten(-1, (c.index_n_heads, c.index_head_dim))
        k = self.k_norm(self.wk(x))
        if self.ssmax is not None:
            q = self.ssmax(q, (layout.positions.float() + 1).log())
        if self.use_rope:
            d = c.qk_rope_head_dim
            q = torch.cat((apply_interleaved_rope(q[..., :d], tables), q[..., d:]), -1)
            k = torch.cat((apply_interleaved_rope(k[..., :d], tables), k[..., d:]), -1)
        weights = self.weights_proj(x).float() * (c.index_head_dim**-0.5 * c.index_n_heads**-0.5)
        k = gather_tokens(k, layout.cp_mesh, differentiable=False)
        selected = []
        # Query chunking bounds temporary scores; no persistent sequence-squared cache.
        for start in range(0, x.shape[0], 128):
            end = min(start + 128, x.shape[0])
            scores = torch.einsum("thd,sd->ths", q[start:end].float(), k.float()).relu()
            scores = (scores * weights[start:end, :, None]).sum(1)
            scores = scores.masked_fill(~layout.allowed(start, end), -torch.inf)
            values, indices = scores.sort(dim=-1, descending=True, stable=True)
            width = min(c.index_topk, k.shape[0])
            selected.append(indices[:, :width].masked_fill(~values[:, :width].isfinite(), -1))
        return torch.cat(selected)


class MistralLarge4Attention(nn.Module):
    """Direct-query MLA; KV consumers retain their source's autograd graph."""

    def __init__(
        self, config: MistralLarge4Config, backend: BackendConfig, *, layer_idx: int, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attn_backend = backend.attn
        self.use_rope = not config.nope_layer_flags[layer_idx]
        self.shares_kv = config.kv_source_layers[layer_idx] != layer_idx
        self.nope_dim = config.qk_nope_head_dim if self.use_rope else config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * config.head_dim, bias=False, dtype=dtype
        )
        if not self.shares_kv:
            self.kv_a_proj_with_mqa = nn.Linear(
                config.hidden_size,
                config.kv_lora_rank + (config.qk_rope_head_dim if self.use_rope else 0),
                bias=False,
                dtype=dtype,
            )
            self.kv_a_layernorm = initialize_rms_norm_module(
                backend.rms_norm,
                config.kv_lora_rank,
                eps=config.rms_norm_eps,
                dtype=dtype,
            )
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            config.num_attention_heads * (self.nope_dim + config.v_head_dim),
            bias=False,
            dtype=dtype,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.v_head_dim, config.hidden_size, bias=False, dtype=dtype
        )
        self.ssmax = SSMax(config.num_attention_heads, dtype) if config.ssmax_scaling else None
        self.indexer = (
            MistralLarge4Indexer(config, layer_idx=layer_idx, dtype=dtype)
            if config.index_sharing[layer_idx] == "full"
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        tables: tuple[torch.Tensor, torch.Tensor],
        layout: SequenceLayout,
        shared_kv: tuple[torch.Tensor, torch.Tensor] | None,
        indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Run selected sparse MLA or the expanded-KV SDPA reference (including LoRA).

        Args:
            x: Floating tensor [tokens, hidden].
            tables: FP32 cosine/sine pair, each [tokens, rotary_dim / 2].
            layout: Local positions/document IDs/validity [tokens], optional global CP key metadata.
            shared_kv: Optional source latent [tokens, latent_dim] and rotated
                keys [tokens, rotary_dim], or [tokens, 0] for NoPE. Never detached.
            indices: Optional preceding full indexer's int64 selected keys
                [tokens, min(topk, tokens)], with -1 padding.

        Returns:
            Output [tokens, hidden], latent/rotary pair in shared_kv layouts,
            and selected keys in indices layout. KV and indices may alias inputs;
            none of the inputs are modified. KV sharing stays local to each CP shard.
        """
        c = self.config
        q = self.q_proj(x).unflatten(-1, (c.num_attention_heads, c.head_dim))
        if self.shares_kv:
            if shared_kv is None:
                raise ValueError(f"Layer {self.layer_idx} is missing source KV")
            latent, k_pe = shared_kv
        else:
            kv = self.kv_a_proj_with_mqa(x)
            latent = self.kv_a_layernorm(kv[..., : c.kv_lora_rank])
            k_pe = (
                apply_interleaved_rope(kv[..., c.kv_lora_rank :], tables)
                if self.use_rope
                else x.new_empty((x.shape[0], 0))
            )
        if self.use_rope:
            q = torch.cat((q[..., : self.nope_dim], apply_interleaved_rope(q[..., self.nope_dim :], tables)), -1)
        if self.ssmax is not None:
            q = self.ssmax(q, (layout.positions.float() + 1).log())
        if self.indexer is not None:
            indices = self.indexer(x, tables, layout)
        if indices is None:
            raise ValueError("IndexShare requires preceding full-layer indices")
        if self.attn_backend == "tilelang":
            out = self._sparse_forward(q, latent, k_pe, indices, layout)
            return self.o_proj(out.flatten(-2)), (latent, k_pe), indices
        # The SDPA reference invokes the projection module, including KV-B LoRA.
        kv = self.kv_b_proj(latent).unflatten(-1, (c.num_attention_heads, self.nope_dim + c.v_head_dim))
        k, v = kv.split((self.nope_dim, c.v_head_dim), -1)
        if self.use_rope:
            k = torch.cat((k, k_pe[:, None].expand(-1, c.num_attention_heads, -1)), -1)
        # Local KV stays in the sharing cache; transport only expanded attention
        # keys/values. Remote query gradients return to their producing CP rank.
        transported = gather_tokens(torch.cat((k, v), -1), layout.cp_mesh, differentiable=True)
        k, v = transported.split((c.head_dim, c.v_head_dim), -1)
        outputs = []
        for start in range(0, x.shape[0], 128):
            end = min(start + 128, x.shape[0])
            chosen = indices[start:end]
            # scatter_add correctly handles multiple -1 slots clamped to zero.
            mask = torch.zeros((end - start, k.shape[0]), device=x.device, dtype=torch.int32)
            mask.scatter_add_(1, chosen.clamp_min(0), (chosen >= 0).to(torch.int32))
            allowed = mask.bool() & layout.allowed(start, end)
            out = F.scaled_dot_product_attention(
                q[start:end].transpose(0, 1).float(),
                k.transpose(0, 1).float(),
                v.transpose(0, 1).float(),
                attn_mask=allowed[None],
                dropout_p=0.0,
                scale=c.head_dim**-0.5,
            )
            outputs.append(out.transpose(0, 1).to(x.dtype))
        return self.o_proj(torch.cat(outputs).flatten(-2)), (latent, k_pe), indices

    def _sparse_forward(
        self,
        q: torch.Tensor,
        latent: torch.Tensor,
        k_pe: torch.Tensor,
        indices: torch.Tensor,
        layout: SequenceLayout,
    ) -> torch.Tensor:
        """Absorb KV-B into queries/outputs and gather compressed KV for sparse MLA.

        Args:
            q: Local [tokens, heads, head_dim] rotated/scaled queries.
            latent: Local [tokens, 512] normalized KV, possibly shared from another layer.
            k_pe: Local [tokens, 64] rotated keys, or [tokens, 0] for NoPE.
            indices: Integer [tokens, selected_keys] global KV indices with -1 padding.
            layout: Local/global document, position and validity metadata as documented
                by SequenceLayout. Global keys use CP-rank concatenation order.

        Returns:
            Local [tokens, heads, value_dim] attention outputs. KV-B effective weights
            stay differentiable, retaining nonzero LoRA A/B gradients. NoPE uses a
            zero rotary tail so the same 512+64 kernel applies exactly.
        """
        c = self.config
        materialize = getattr(self.kv_b_proj, "materialize_effective_weight", None)
        weight = materialize() if materialize is not None else self.kv_b_proj.weight
        weight = weight.view(c.num_attention_heads, self.nope_dim + c.v_head_dim, c.kv_lora_rank)
        key_weight, value_weight = weight.split((self.nope_dim, c.v_head_dim), dim=1)
        q_absorbed = torch.einsum("thd,hdc->thc", q[..., : self.nope_dim], key_weight.to(q.dtype))
        q_tail = q[..., self.nope_dim :] if self.use_rope else q.new_zeros((*q.shape[:-1], 64))
        key_tail = k_pe if self.use_rope else latent.new_zeros((latent.shape[0], 64))
        kv = gather_tokens(torch.cat((latent, key_tail), -1), layout.cp_mesh, differentiable=True)
        key_docs = layout.sequence_ids if layout.key_sequence_ids is None else layout.key_sequence_ids
        key_valid = layout.valid if layout.key_valid is None else layout.key_valid
        safe = indices.clamp(0, kv.shape[0] - 1)
        offset = 0 if layout.cp_mesh is None else layout.cp_mesh.get_local_rank() * q.shape[0]
        query_indices = torch.arange(q.shape[0], device=q.device) + offset
        allowed = (
            (indices >= 0)
            & (indices < kv.shape[0])
            & (indices <= query_indices[:, None])
            & (layout.sequence_ids[:, None] == key_docs[safe])
            & layout.valid[:, None]
            & key_valid[safe]
        )
        return sparse_mla(
            torch.cat((q_absorbed, q_tail), -1).contiguous(),
            kv,
            indices.masked_fill(~allowed, -1),
            value_weight.to(q.dtype),
            scale=c.head_dim**-0.5,
        )
