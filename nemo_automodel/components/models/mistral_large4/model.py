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

"""Mistral Large 4 text model with expert, pipeline and context parallelism."""

from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    shard_sequence_for_cp_contiguous,
)
from nemo_automodel.components.models.common import (
    BackendConfig,
    HFCheckpointingMixin,
    compute_lm_head_logits,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import Float32RMSNorm, cast_model_to_dtype
from nemo_automodel.components.models.mistral_large4.attention import (
    MistralLarge4Attention,
    SequenceLayout,
)
from nemo_automodel.components.models.mistral_large4.config import MistralLarge4Config
from nemo_automodel.components.models.mistral_large4.context_parallel import (
    CPBatch,
    gather_tokens,
    prepare_cp_inputs,
    shifted_cp_loss,
)
from nemo_automodel.components.models.mistral_large4.parallelization import MistralLarge4Parallelizer
from nemo_automodel.components.models.mistral_large4.rope_utils import MistralLarge4Rotary, SSMax
from nemo_automodel.components.models.mistral_large4.state_dict_adapter import MistralLarge4StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MLP, Gate, MoE
from nemo_automodel.shared.utils import dtype_from_str


@dataclass
class MistralLarge4LayerState:
    """Per-forward state; never retained on a module or detached.

    Attributes:
        tables: FP32 cosine/sine tensors, each [tokens, rotary_dim / 2].
        shared_kv: Source-layer IDs mapped to latent [tokens, latent_dim] and
            rotated key [tokens, rotary_dim] tensors; NoPE keys have width zero.
        indices: Optional int64 [tokens, min(index_topk, tokens)] selected keys.
    """

    tables: tuple[torch.Tensor, torch.Tensor]
    shared_kv: dict[int, tuple[torch.Tensor, torch.Tensor]]
    indices: torch.Tensor | None = None


class MistralLarge4PipelineMixin:
    """Use the existing PP hooks and tuple transport without replicated parameters."""

    _pp_keep_self_forward: bool = True
    _pp_return_hidden_states_supported: bool = True
    _pp_return_hidden_states: bool = False
    config: MistralLarge4Config
    model: nn.Module
    lm_head: nn.Linear | None
    cp_mesh: DeviceMesh | None

    def _is_pipeline_parallel_stage(self) -> bool:
        """Identify trimmed stages from their actual module ownership."""
        return self.model.embed_tokens is None or self.lm_head is None

    def _pipeline_layer_ids(self) -> list[int]:
        """Return original, contiguous decoder indices after the shared splitter."""
        layers = self.model.layers
        ids = [int(i) for i in layers] if isinstance(layers, nn.ModuleDict) else list(range(len(layers)))
        if not ids or ids != list(range(ids[0], ids[-1] + 1)):
            raise ValueError("Mistral Large 4 PP stages require nonempty contiguous decoder layers")
        if (ids[0] == 0) != (self.model.embed_tokens is not None):
            raise ValueError("Mistral Large 4 PP stage zero must exclusively own token embeddings")
        if (ids[-1] == self.config.num_hidden_layers - 1) != (self.lm_head is not None):
            raise ValueError("Mistral Large 4 PP final decoder stage must exclusively own lm_head")
        if (ids[0] == 0) != (self.model.rotary_emb is not None):
            raise ValueError("Mistral Large 4 PP stage zero must exclusively own learnable LeRoPE")
        if (self.lm_head is not None) != (self.model.norm is not None):
            raise ValueError("Mistral Large 4 PP final decoder stage must exclusively own the final norm")
        return ids

    def _pipeline_live_sources(self, boundary: int) -> list[int]:
        """Select produced KV sources still needed at or after a decoder boundary."""
        return sorted({source for source in self.config.kv_source_layers[boundary:] if source < boundary})

    def customize_pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        *,
        layers_prefix: str,
        text_model: nn.Module | None = None,
    ) -> list[list[str]]:
        """Assign the single learnable LeRoPE module to stage zero.

        Args:
            module_names_per_stage: Generated module FQNs in forward stage order.
            layers_prefix: Decoder parent prefix, required to be ``model.``.
            text_model: Resolved backbone supplied by the generic splitter.

        Returns:
            Independent FQN lists with LeRoPE exclusively on the first stage.
        """
        del text_model
        if layers_prefix != "model.":
            raise ValueError("Mistral Large 4 PP expects decoder layers under model.layers")
        stages = [[name for name in names if name != "model.rotary_emb"] for names in module_names_per_stage]
        stages[0].append("model.rotary_emb")
        return stages

    def _pipeline_boundary_metas(
        self, boundary: int, *, batch: int, sequence: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        """Describe residual, rotary, layout and live sharing tensors at one cut.

        Returns:
            Meta tensors: residual [batch, sequence, hidden] in compute dtype;
            FP32 tables [batch, sequence, rotary_dim / 2, 2] (cosine then sine);
            int64 layout [batch, sequence, 3] (position, document, validity);
            one [batch, sequence, latent_dim + rotary_width] compute-dtype tensor
            per live KV source, in increasing source order; and, only if the
            next layer reuses selection, int64 indices
            [batch, sequence, min(index_topk, CP * batch * sequence)]. Sequence
            is CP-local; selected indices address rank-ordered global keys.
            NoPE width is zero.
        """
        c = self.config
        tensors = [
            torch.empty(batch, sequence, c.hidden_size, device="meta", dtype=dtype),
            torch.empty(batch, sequence, c.qk_rope_head_dim // 2, 2, device="meta", dtype=torch.float32),
            torch.empty(batch, sequence, 3, device="meta", dtype=torch.long),
        ]
        for source in self._pipeline_live_sources(boundary):
            width = c.kv_lora_rank + (0 if c.nope_layer_flags[source] else c.qk_rope_head_dim)
            tensors.append(torch.empty(batch, sequence, width, device="meta", dtype=dtype))
        if c.index_sharing[boundary] == "shared":
            cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1
            tensors.append(
                torch.empty(
                    batch, sequence, min(c.index_topk, cp_size * batch * sequence), device="meta", dtype=torch.long
                )
            )
        return tuple(tensors)

    def get_pipeline_stage_metas(
        self, *, is_first: bool, microbatch_size: int, seq_len: int, dtype: torch.dtype
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Provide exact tuple metadata to the existing PyTorch PP schedule.

        Args:
            is_first: Whether this is the first logical stage.
            microbatch_size: Number of documents in one pipeline microbatch.
            seq_len: Global nonpacked sequence length before CP padding.
            dtype: Body compute dtype; rotary transport always remains FP32.

        Returns:
            Input/output meta tuples using ``_pipeline_boundary_metas``' layouts.
            Stage zero instead takes int64 IDs [batch, sequence]. The final
            stage emits FP32 capped logits [batch, local_sequence, vocab], or
            native compute-dtype hidden states [batch, local_sequence, hidden]
            for fused CE. All boundary tensors are CP-local; stage zero alone
            receives global IDs. CP pads sequence to a multiple of 2 * CP.
        """
        ids = self._pipeline_layer_ids()
        if is_first != (ids[0] == 0):
            raise ValueError("Mistral Large 4 PP stage metadata disagrees with embedding ownership")
        cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1
        local_length = (seq_len + (-seq_len) % (2 * cp_size)) // cp_size if cp_size > 1 else seq_len
        shape = (microbatch_size, local_length)
        inputs = (
            (torch.empty(microbatch_size, seq_len, device="meta", dtype=torch.long),)
            if is_first
            else self._pipeline_boundary_metas(ids[0], batch=microbatch_size, sequence=local_length, dtype=dtype)
        )
        if self.lm_head is None:
            outputs = self._pipeline_boundary_metas(
                ids[-1] + 1, batch=microbatch_size, sequence=local_length, dtype=dtype
            )
        else:
            width = self.config.hidden_size if self._pp_return_hidden_states else self.config.vocab_size
            outputs = (
                torch.empty(
                    *shape, width, device="meta", dtype=dtype if self._pp_return_hidden_states else torch.float32
                ),
            )
        return inputs, outputs

    def _receive_pipeline_state(
        self, hidden: torch.Tensor, tensors: tuple[torch.Tensor, ...]
    ) -> tuple[SequenceLayout, MistralLarge4LayerState]:
        """Decode the prior stage's tensors without detaching differentiable state.

        Args:
            hidden: Compute-dtype residual [batch, sequence, hidden].
            tensors: Remaining boundary tensors in ``_pipeline_boundary_metas``
                order; no tensor aliases are mutated.

        Returns:
            Layout with flattened [tokens] position/document/validity fields,
            and state with FP32 tables [tokens, rotary_dim / 2], source KV pairs
            [tokens, latent_dim] / [tokens, rotary_width], and optional integer
            selection [tokens, min(index_topk, CP * tokens)]. Tokens are CP-local;
            key document/validity metadata is [CP * tokens] in CP-rank order.
            Views alias received tensors; gathered metadata owns new storage.
        """
        boundary = self._pipeline_layer_ids()[0]
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.hidden_size:
            raise ValueError("Mistral Large 4 PP residual must have shape [batch, sequence, hidden]")
        metas = self._pipeline_boundary_metas(
            boundary, batch=hidden.shape[0], sequence=hidden.shape[1], dtype=hidden.dtype
        )
        if len(tensors) != len(metas) - 1:
            raise ValueError("Mistral Large 4 PP received an incomplete rotary/layout/KV/IndexShare state")
        for tensor, meta in zip(tensors, metas[1:], strict=True):
            if tensor.shape != meta.shape or tensor.dtype != meta.dtype:
                raise ValueError(
                    f"Mistral Large 4 PP state expected {meta.shape}/{meta.dtype}, got {tensor.shape}/{tensor.dtype}"
                )
        tables = tensors[0].reshape(-1, self.config.qk_rope_head_dim // 2, 2).unbind(-1)
        positions, documents, valid = tensors[1].reshape(-1, 3).unbind(-1)
        layout = SequenceLayout(
            positions,
            documents,
            valid.bool(),
            self.cp_mesh,
            gather_tokens(documents, self.cp_mesh, differentiable=False),
            gather_tokens(valid.bool(), self.cp_mesh, differentiable=False),
        )
        sources = self._pipeline_live_sources(boundary)
        cache = {}
        for source, tensor in zip(sources, tensors[2 : 2 + len(sources)], strict=True):
            flat = tensor.flatten(0, 1)
            cache[source] = (flat[:, : self.config.kv_lora_rank], flat[:, self.config.kv_lora_rank :])
        indices = tensors[-1].flatten(0, 1) if self.config.index_sharing[boundary] == "shared" else None
        return layout, MistralLarge4LayerState(tables, cache, indices)

    def _finish_pipeline_stage(
        self, hidden: torch.Tensor, layout: SequenceLayout, state: MistralLarge4LayerState
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Emit capped final logits or fresh tensors for the next stage.

        Args:
            hidden: Compute-dtype residual [batch, sequence, hidden].
            layout: Flattened integer positions/documents and bool validity [tokens].
            state: FP32 cosine/sine [tokens, rotary_dim / 2], live KV pairs
                [tokens, latent_dim] / [tokens, rotary_width], and selected
                int64 keys [tokens, min(index_topk, tokens)].

        Returns:
            Final FP32 logits [batch, sequence, vocab], or compute-dtype hidden
            states [batch, sequence, hidden] for fused CE. Non-final stages emit
            the tuple documented by ``_pipeline_boundary_metas``; transported
            state owns fresh storage and remains connected to autograd.
        """
        if self.lm_head is not None:
            if self._pp_return_hidden_states:
                return hidden
            logits = compute_lm_head_logits(self.lm_head, hidden).logits
            cap = self.config.final_logit_softcapping
            return cap * torch.tanh(logits.float() / cap)
        boundary = self._pipeline_layer_ids()[-1] + 1
        batch, sequence = hidden.shape[:2]
        tensors = [
            hidden,
            torch.stack(state.tables, -1).reshape(batch, sequence, -1, 2),
            torch.stack((layout.positions, layout.sequence_ids, layout.valid.long()), -1).reshape(batch, sequence, 3),
        ]
        for source in self._pipeline_live_sources(boundary):
            tensors.append(torch.cat(state.shared_kv[source], -1).reshape(batch, sequence, -1))
        if self.config.index_sharing[boundary] == "shared":
            if state.indices is None:
                raise ValueError("Mistral Large 4 PP cannot forward missing IndexShare selections")
            tensors.append(state.indices.reshape(batch, sequence, -1).clone())
        return tuple(tensors)


def _mistral_swiglu(projections: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Preserve Mistral's BF16 SiLU and product rounding before the down GEMM.

    Args:
        projections: BF16 or FP32 tensor [routed_tokens, 2*intermediate], gate
            then up. May be noncontiguous. No input is modified.
        weights: All-one activation weights [routed_tokens, 1]; actual FP32
            routing weights are applied by the expert backend after down.

    Returns:
        Independent tensor [routed_tokens, intermediate] in projections.dtype.
        SiLU and multiplication each round to the model compute dtype, matching
        the released individual-expert MLP rather than a fused FP32 activation.
    """
    del weights
    gate, up = projections.chunk(2, dim=-1)
    return F.silu(gate) * up


class MistralLarge4Block(nn.Module):
    """Pre-normalized MLA followed by dense or routed/shared SwiGLU."""

    def __init__(
        self, config: MistralLarge4Config, backend: BackendConfig, *, layer_idx: int, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.self_attn = MistralLarge4Attention(config, backend, layer_idx=layer_idx, dtype=dtype)
        self.input_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        is_moe = layer_idx >= config.first_k_dense_replace
        self.moe_config = (
            MoEConfig(
                n_routed_experts=config.n_routed_experts,
                n_shared_experts=config.n_shared_experts,
                n_activated_experts=config.num_experts_per_tok,
                n_expert_groups=1,
                n_limited_groups=1,
                train_gate=True,
                gate_bias_update_factor=0.0,
                aux_loss_coeff=0.0,
                score_func="softmax",
                route_scale=config.routed_scaling_factor,
                dim=config.hidden_size,
                inter_dim=config.intermediate_size,
                moe_inter_dim=config.moe_intermediate_size,
                norm_topk_prob=True,
                # Selected softmax gives exactly one for top-1, as required by
                # Mistral's normalized routing. For top-k retain full-softmax rounding.
                softmax_before_topk=config.num_experts_per_tok > 1,
                router_weights_fp32=True,
                gate_dtype=torch.float32,
                apply_router_weight_after_down=True,
                dtype=dtype,
            )
            if is_moe
            else None
        )
        self.mlp = (
            MoE(self.moe_config, replace(backend, gate_precision=torch.float32))
            if is_moe
            else MLP(config.hidden_size, config.intermediate_size, backend.linear, dtype=dtype)
        )
        # The released MLP rounds SiLU and its product to the compute dtype.
        # Retain only this numerical specialization; routing, parameter layout,
        # token exchange, weighted combine and sharding use the shared MoE.
        if is_moe:
            if isinstance(self.mlp.experts, GroupedExperts):
                self.mlp.experts.expert_activation_grouped = _mistral_swiglu
            elif isinstance(self.mlp.experts, GroupedExpertsDeepEP):
                self.mlp.experts.expert_activation = _mistral_swiglu

    def forward(
        self,
        x: torch.Tensor,
        tables: tuple[torch.Tensor, torch.Tensor],
        layout: SequenceLayout,
        shared_kv: tuple[torch.Tensor, torch.Tensor] | None,
        indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Run one block with immutable sharing inputs, safe for recomputation.

        Args:
            x: Unsharded [tokens, hidden] activations.
            tables: FP32 cosine/sine pair [tokens, rotary_dim / 2].
            layout: Positions, document IDs and validity [tokens].
            shared_kv: Optional latent [tokens, latent_dim], rotated keys
                [tokens, rotary_dim] (zero final dimension for NoPE).
            indices: Optional preceding int64 [tokens, min(topk, tokens)] selection.

        Returns:
            Hidden [tokens, hidden], KV pair and selected indices in input layouts.
            Shared KV/indices may alias inputs and are never modified.
        """
        update, kv, selected = self.self_attn(self.input_layernorm(x), tables, layout, shared_kv, indices)
        x = x + update
        return x + self.mlp(self.post_attention_layernorm(x)), kv, selected


class MistralLarge4Model(nn.Module):
    """Text backbone; each forward owns the lifetime of its shared source activations."""

    def __init__(self, config: MistralLarge4Config, backend: BackendConfig, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.rotary_emb = MistralLarge4Rotary(config)
        self.layers = nn.ModuleList(
            [MistralLarge4Block(config, backend, layer_idx=i, dtype=dtype) for i in range(config.num_hidden_layers)]
        )
        self.norm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self._last_consumers = {
            source: max(i for i, s in enumerate(config.kv_source_layers) if s == source)
            for layer, source in enumerate(config.kv_source_layers)
            if layer != source
        }

    def forward(
        self, hidden: torch.Tensor, layout: SequenceLayout, state: MistralLarge4LayerState | None = None
    ) -> tuple[torch.Tensor, MistralLarge4LayerState]:
        """Run owned layers with explicit, per-microbatch shared state.

        Args:
            hidden: Unsharded compute-dtype activations [tokens, hidden].
            layout: Positions, document IDs and validity, each [tokens].
            state: Optional received FP32 tables [tokens, rotary_dim / 2], KV
                pairs [tokens, latent_dim] / [tokens, rotary_width], and int64
                selections [tokens, min(index_topk, tokens)]. Never mutated.

        Returns:
            Activations [tokens, hidden], normalized only on the final stage,
            and state with the same tensor layouts as the input. KV/tables may
            alias received inputs; the returned cache mapping is independent.
        """
        tables = self.rotary_emb(layout.positions) if state is None else state.tables
        cache = {} if state is None else dict(state.shared_kv)
        indices = None if state is None else state.indices
        blocks = self.layers.items() if isinstance(self.layers, nn.ModuleDict) else enumerate(self.layers)
        for key, block in blocks:
            i = int(key)
            source = self.config.kv_source_layers[i]
            hidden, kv, indices = block(hidden, tables, layout, cache.get(source), indices)
            if i in self._last_consumers:
                cache[i] = kv
            if source != i and self._last_consumers[source] == i:
                del cache[source]
        if self.norm is not None:
            hidden = self.norm(hidden)
        return hidden, MistralLarge4LayerState(tables, cache, indices)


class MistralLarge4ForCausalLM(MistralLarge4PipelineMixin, HFCheckpointingMixin, PreTrainedModel, MoEFSDPSyncMixin):
    """Text model; source FP8 weights are reconstructed on load.

    Training runs in floating point: serving FP8 indexer/KV-cache execution and
    DSpark drafting are not implemented. DSA parameters remain frozen pending a
    training specification.
    """

    parallelizer = MistralLarge4Parallelizer()
    parallel_layer_groups = {"language": ("model.layers",)}
    packed_mask_type = "document_ids"

    config_class = MistralLarge4Config
    base_model_prefix = "model"
    _supports_sdpa = True
    _owns_cp_attention = True  # Both supported backends own their CP KV transport.
    supports_thd = True
    _packed_cp_attn_backends = ("tilelang",)
    _no_split_modules = ["MistralLarge4Block"]
    _keep_in_fp32_modules_strict = ["rotary_emb", "indexer.k_norm", "mlp.gate.weight"]
    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY

    @dataclass(frozen=True)
    class ModelCapabilities:
        """SDPA and TileLang CP support BSHD; packed THD CP requires TileLang."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True
        supports_thd: bool = True

    def __init__(self, config: MistralLarge4Config, backend: BackendConfig | None = None) -> None:
        reject_unsupported_tie_word_embeddings(type(self), config)
        config.validate_architecture()
        super().__init__(config)
        self.backend = backend or BackendConfig(
            attn="sdpa",
            linear="torch",
            rms_norm="torch_fp32",
            rope="torch",
            rope_fusion=False,
            experts="torch",
            dispatcher="torch",
        )
        if (
            self.backend.attn not in ("sdpa", "tilelang")
            or self.backend.linear != "torch"
            or self.backend.rms_norm != "torch_fp32"
            or self.backend.rope != "torch"
            or self.backend.rope_fusion
            or (self.backend.experts, self.backend.dispatcher)
            not in (("torch", "torch"), ("torch_mm", "torch"), ("torch_mm", "hybridep"))
            or self.backend.te_fp8 is not None
        ):
            raise ValueError(
                "Large 4 requires sdpa or tilelang, torch linear/rope, torch_fp32 RMSNorm, torch/torch or torch_mm with torch/hybridep experts/dispatcher, and no FP8 compute or RoPE fusion"
            )
        dtype = dtype_from_str(config.dtype, torch.float32)
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Large 4 baseline supports FP32 or BF16 storage")
        if self.backend.attn == "tilelang" and (
            dtype != torch.bfloat16 or config.kv_lora_rank != 512 or config.qk_rope_head_dim != 64
        ):
            raise ValueError("Mistral TileLang requires BF16, kv_lora_rank=512 and qk_rope_head_dim=64")
        if self.backend.dispatcher == "hybridep" and dtype != torch.bfloat16:
            raise ValueError("Mistral HybridEP requires BF16 model storage and compute")
        self.backend = replace(self.backend, gate_precision=torch.float32)
        self.model = MistralLarge4Model(config, self.backend, dtype=dtype)
        self.moe_config = next((block.moe_config for block in self.model.layers if block.moe_config is not None), None)
        self.cp_mesh: DeviceMesh | None = None
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = MistralLarge4StateDictAdapter(
                dtype=dtype,
                grouped_experts=True,
                n_routed_experts=config.n_routed_experts,
            )

    @classmethod
    def from_config(
        cls, config: MistralLarge4Config, backend: BackendConfig | None = None
    ) -> "MistralLarge4ForCausalLM":
        """Construct the registered model through AutoModel's existing interface."""
        return cls(config, backend)

    @torch.no_grad()
    def initialize_weights(self, buffer_device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        """Initialize materialized parameters in their resident storage dtype.

        Initialize in place without temporarily promoting the entire pretrained
        base to FP32. Protected modules own separate FP32
        FSDP units and must never pass through the broad body-dtype cast.

        Args:
            buffer_device: Device on which the caller materialized parameters.
            dtype: Optional final body storage dtype; sensitive modules remain FP32.
        """
        del buffer_device
        for module in self.modules():
            if isinstance(module, (GroupedExperts, GroupedExpertsDeepEP)):
                module.init_weights(module.gate_and_up_projs.device, init_std=0.02)
            elif isinstance(module, Gate):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
                nn.init.normal_(module.weight, std=0.02)
                if isinstance(module, (nn.Linear, nn.Conv2d)) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.RMSNorm, Float32RMSNorm)):
                nn.init.ones_(module.weight)
                if isinstance(module, nn.LayerNorm) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, MistralLarge4Rotary):
                nn.init.zeros_(module.alpha)
            elif isinstance(module, SSMax):
                nn.init.zeros_(module.s)
                nn.init.ones_(module.b)
        if dtype is not None:
            cast_model_to_dtype(self, dtype, skip_modules=("rotary_emb", "k_norm", "gate"))

    def prepare_model_inputs_for_cp(self, batch: CPBatch, *, num_chunks: int = 1) -> dict[str, ContextParallelSharder]:
        """Prepare CP-local targets and positions with the model-owned sharder.

        Args:
            batch: Global IDs/masks [batch, sequence] or embeddings
                [batch, sequence, hidden], with optional complete image payloads.
                THD requires IDs [microbatches, packed_sequence] plus
                seq_lens/seq_lens_padded [microbatches, documents].
            num_chunks: Number of PP microbatches along the batch dimension.

        Returns:
            Sharder mapping using prepare_cp_inputs' tensor contract. PP keeps
            global primary inputs and shards them in stage zero's forward.
        """
        if self.cp_mesh is None and batch.get("qkv_format") != "thd":
            return {}
        return prepare_cp_inputs(batch, num_chunks=num_chunks, shard_primary=not self._is_pipeline_parallel_stage())

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the untied input embedding."""
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Linear:
        """Return the untied language head."""
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *pipeline_tensors: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        qkv_format: str = "bshd",
        use_cache: bool = False,
        _mistral_packed_document_ids: torch.Tensor | None = None,
    ) -> CausalLMOutputWithPast | torch.Tensor | tuple[torch.Tensor, ...]:
        """Run causal or document-packed text with capped logits and optional shifted CE.

        Args:
            input_ids: Integer [batch, sequence], or packed [tokens] with thd
                format. Under CP, text-only PP1 takes pre-sharded local IDs;
                stage zero of PP takes global IDs and shards after embedding.
            pipeline_tensors: Non-first PP boundary tensors in the layouts
                documented by ``_pipeline_boundary_metas``. input_ids then holds
                compute-dtype residual [batch, sequence, hidden].
            inputs_embeds: Floating [batch, sequence, hidden] or packed
                [tokens, hidden], replacing IDs with the same CP ownership.
            position_ids: Nonnegative integer positions [batch, local_sequence]
                under CP, otherwise matching the input token axes.
            attention_mask: Binary keep mask or nonnegative packed document IDs,
                matching input token axes. CP stage zero receives a global mask
                and shards it alongside embeddings; later stages use transported layout.
            padding_mask: Optional bool padding mask; true excludes a token.
                CP sharder supplies [batch, local_sequence], including CP padding.
            labels: Optional integer targets, same token shape; -100 ignored. This
                HF-style path shifts labels once, excluding padding/document edges.
            logits_to_keep: Zero for all, positive tail length, or int64 [kept_positions].
                Labels require all positions. Index selection applies to the sequence axis.
            output_hidden_states: Return final [batch, sequence, hidden] activations.
            cu_seqlens: SDPA THD requires integer offsets [documents + 1]
                strictly partitioning flattened storage. Prepared THD receives padded
                global offsets [1, documents + 1] from its sharder; its local
                document IDs and padding mask preserve inter-document padding.
            seq_lens: Optional real document lengths [batch, documents], with
                -1000 padding, retained by the packed sharder for document-count validation.
            qkv_format: bshd or thd (packed output has a leading batch dimension of one).
            use_cache: Must be false; incremental generation caches are not supported.
            _mistral_packed_document_ids: Model-prepared integer [batch, local_sequence]
                document identity for prepared THD. Each microbatch contains one packed row.

        Returns:
            CausalLMOutputWithPast with capped logits [batch, kept_sequence, vocab],
            optional scalar loss and final hidden states [batch, sequence, hidden].
            PP instead returns the tensor/tuple contract from
            ``_finish_pipeline_stage`` and requires an external loss. Packed
            PP inputs must pass through the model-owned sharder.
            The hidden states are uncapped; external fused loss must set
            logit_softcapping to config.final_logit_softcapping. Chunked CE is rejected
            by the existing loss implementation for softcapped models.
        """
        if use_cache or qkv_format not in ("bshd", "thd"):
            raise ValueError("Large 4 baseline supports bshd/thd full-sequence execution without a KV cache")
        prepared_packed = _mistral_packed_document_ids is not None
        if prepared_packed and qkv_format != "thd":
            raise ValueError("Prepared packed document IDs require THD format")
        if prepared_packed and labels is not None:
            raise ValueError("Prepared THD uses already-shifted labels through the external training loss")
        is_stage = self._is_pipeline_parallel_stage()
        if is_stage:
            self._pipeline_layer_ids()
            if not prepared_packed and (qkv_format != "bshd" or cu_seqlens is not None):
                raise ValueError("Mistral Large 4 PP requires BSHD or model-prepared packed THD")
            if labels is not None or not isinstance(logits_to_keep, int) or logits_to_keep != 0:
                raise ValueError("Mistral Large 4 PP requires full outputs and labels in the pipeline loss")
            if self.model.embed_tokens is None:
                if input_ids is None or inputs_embeds is not None:
                    raise ValueError("Non-first Mistral Large 4 PP stages require positional residual and state")
                layout, state = self._receive_pipeline_state(input_ids, pipeline_tensors)
                batch, length = input_ids.shape[:2]
                hidden, state = self.model(input_ids.flatten(0, 1), layout, state)
                return self._finish_pipeline_stage(hidden.reshape(batch, length, -1), layout, state)
        if self.cp_mesh is not None and ((qkv_format != "bshd" and not prepared_packed) or position_ids is None):
            raise ValueError("Large 4 CP requires model-prepared BSHD/THD shards with global position_ids")
        if pipeline_tensors:
            raise ValueError("Only non-first Mistral Large 4 PP stages accept propagated state")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids and inputs_embeds")
        hidden = self.model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if is_stage and self.cp_mesh is not None:
            # Stage zero owns global text/image embeddings; later stages returned
            # above with already-local residuals and must never shard them again.
            hidden, _, _ = shard_sequence_for_cp_contiguous(self.cp_mesh, hidden)
            if attention_mask is not None:
                attention_mask, _, _ = shard_sequence_for_cp_contiguous(self.cp_mesh, attention_mask)
        if hidden.ndim not in (2, 3) or hidden.shape[-1] != self.config.hidden_size:
            raise ValueError("Expected [batch, sequence, hidden] or [tokens, hidden] embeddings")
        if hidden.ndim == 2 and qkv_format != "thd":
            raise ValueError("Flattened input requires thd format and document boundaries")
        batch, length = hidden.shape[:2] if hidden.ndim == 3 else (1, hidden.shape[0])
        count, device = batch * length, hidden.device
        if count == 0:
            raise ValueError("Empty input sequences are unsupported")
        rows = torch.arange(count, device=device)
        if prepared_packed:
            if batch != 1 or _mistral_packed_document_ids.shape != hidden.shape[:-1]:
                raise ValueError("Mistral packed THD requires one packed row and matching CP-local document IDs")
            doc = _mistral_packed_document_ids.reshape(-1).to(device=device, dtype=torch.long)
            if seq_lens is not None and bool((doc >= (seq_lens >= 0).sum()).any()):
                raise ValueError("Mistral packed THD document IDs exceed the packed document count")
        elif qkv_format == "thd":
            if (
                batch != 1
                or cu_seqlens is None
                or cu_seqlens.ndim != 1
                or cu_seqlens.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("THD requires batch=1 and one-dimensional integer cu_seqlens")
            cu = cu_seqlens.to(device=device, dtype=torch.long)
            if cu.numel() < 2 or int(cu[0]) != 0 or int(cu[-1]) != count or bool((cu[1:] <= cu[:-1]).any()):
                raise ValueError("cu_seqlens must strictly partition token storage from zero to token count")
        else:
            if cu_seqlens is not None:
                raise ValueError("cu_seqlens requires thd format")
            cu = torch.arange(0, count + 1, length, device=device)
        if not prepared_packed:
            doc = torch.bucketize(rows, cu[1:], right=True)
        valid = torch.ones(count, dtype=torch.bool, device=device)
        if attention_mask is not None:
            if attention_mask.shape != hidden.shape[:-1] or not bool((attention_mask >= 0).all()):
                raise ValueError("attention_mask must contain nonnegative document IDs matching input token axes")
            mask = attention_mask.reshape(-1).to(device=device)
            if not bool((mask == mask.long()).all()):
                raise ValueError("attention_mask must contain integer document IDs")
            valid = mask.bool()
            if qkv_format == "bshd":
                # Shared packing uses 1..N document IDs per row, zero for padding.
                doc = doc * (mask.max().long() + 1) + mask.long()
        if padding_mask is not None:
            if padding_mask.shape != hidden.shape[:-1] or padding_mask.dtype != torch.bool:
                raise ValueError("padding_mask must be boolean and match input token axes")
            valid = valid & ~padding_mask.reshape(-1).to(device)
        if position_ids is None:
            # Left/right padding does not consume document-relative positions.
            cumulative = valid.long().cumsum(0)
            starts = torch.cat((torch.ones(1, dtype=torch.bool, device=device), doc[1:] != doc[:-1]))
            first = torch.where(starts, rows, 0).cummax(0).values
            prior = F.pad(cumulative, (1, 0))[first]
            positions = (cumulative - prior - 1).clamp_min(0)
        else:
            if (
                position_ids.shape != hidden.shape[:-1]
                or position_ids.dtype not in (torch.int32, torch.int64)
                or bool((position_ids < 0).any())
            ):
                raise ValueError("position_ids must be nonnegative integers matching input token axes")
            positions = position_ids.reshape(-1).to(device)
        layout = SequenceLayout(
            positions,
            doc,
            valid,
            self.cp_mesh,
            gather_tokens(doc, self.cp_mesh, differentiable=False),
            gather_tokens(valid, self.cp_mesh, differentiable=False),
        )
        hidden, state = self.model(hidden.reshape(count, -1), layout)
        hidden = hidden.reshape(batch, length, -1)
        if is_stage:
            return self._finish_pipeline_stage(hidden, layout, state)
        if isinstance(logits_to_keep, int) and logits_to_keep < 0:
            raise ValueError("logits_to_keep must be nonnegative")
        if labels is not None and (not isinstance(logits_to_keep, int) or logits_to_keep != 0):
            raise ValueError("Labels require logits for all positions")
        output = compute_lm_head_logits(self.lm_head, hidden, logits_to_keep, output_hidden_states=output_hidden_states)
        cap = self.config.final_logit_softcapping
        output.logits = cap * torch.tanh(output.logits.float() / cap)
        if labels is not None:
            if labels.shape != (input_ids.shape if input_ids is not None else inputs_embeds.shape[:-1]):
                raise ValueError("labels must match input token axes")
            if self.cp_mesh is not None:
                output.loss = shifted_cp_loss(output.logits, labels.reshape(batch, length), valid, self.cp_mesh)
                return output
            targets = labels.reshape(batch, length)[:, 1:].clone()
            edge = (
                valid.reshape(batch, length)[:, :-1]
                & valid.reshape(batch, length)[:, 1:]
                & (doc.reshape(batch, length)[:, :-1] == doc.reshape(batch, length)[:, 1:])
            )
            targets.masked_fill_(~edge, -100)
            loss = F.cross_entropy(
                output.logits[:, :-1].reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            output.loss = loss / (targets != -100).sum().clamp_min(1)
        return output
