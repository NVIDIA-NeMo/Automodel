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

"""MiniMax M3 (mixed sparse/dense MoE) text backbone.

Stage 1 implements ``MiniMaxM3TextModel`` and the standalone
``MiniMaxM3SparseForCausalLM`` so the language path can be parity-tested against
the sglang reference before the vision tower / VLM wrapper (Stage 3) embeds the
text model as ``language_model``.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.distributed.cp_sharder import (
    ContextParallelismSharder,
    round_robin_local_indices,
    shard_batch_aux_only,
    shard_sequence_for_cp,
)
from nemo_automodel.components.models.common import (
    BackendConfig,
    get_rope_config,
    initialize_linear_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding, position_ids_to_freqs_cis
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLConfig, MiniMaxM3VLTextConfig
from nemo_automodel.components.models.minimax_m3_vl.layers import Block, MiniMaxM3RMSNorm
from nemo_automodel.components.models.minimax_m3_vl.mtp import MiniMaxM3MTP
from nemo_automodel.components.models.minimax_m3_vl.state_dict_adapter import (
    MiniMaxM3StateDictAdapter,
    MiniMaxM3VLStateDictAdapter,
)
from nemo_automodel.components.models.minimax_m3_vl.vision_encoder import MiniMaxM3VisionModel
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


@dataclass
class MiniMaxM3CausalLMOutput:
    """Forward output carrying the primary logits and optional per-depth MTP logits."""

    logits: torch.Tensor
    mtp_per_depth_logits: list[torch.Tensor] | None = None


def build_moe_config(config: Any, dtype: torch.dtype) -> MoEConfig:
    """Build the routed-expert ``MoEConfig`` for the M3 backbone.

    Shared experts are handled in :class:`~...layers.Block` (SwiGLU-OAI), so
    ``n_shared_experts`` is 0 here. Routed experts use the ``swigluoai``
    activation ``gate * sigmoid(alpha * gate) * (up + 1)`` over the concatenated
    grouped gate/up projection produced by ``MoESplitExpertsStateDictMixin``.
    """
    return MoEConfig(
        dim=config.hidden_size,
        inter_dim=config.intermediate_size,
        moe_inter_dim=config.intermediate_size,
        n_routed_experts=config.num_local_experts,
        n_shared_experts=0,
        n_activated_experts=config.num_experts_per_tok,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=1e-3,
        score_func="sigmoid" if str(getattr(config, "scoring_func", "sigmoid")).lower() != "softmax" else "softmax",
        route_scale=float(getattr(config, "routed_scaling_factor", 1.0)),
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        router_bias=False,
        expert_bias=False,
        expert_activation="swigluoai",
        activation_alpha=float(getattr(config, "swiglu_alpha", 1.702)),
        activation_limit=float(getattr(config, "swiglu_limit", 7.0)),
        softmax_before_topk=False,
        force_e_score_correction_bias=bool(getattr(config, "use_routing_bias", True)),
        dtype=dtype,
    )


class MiniMaxM3TextModel(nn.Module):
    """Embedding + decoder stack + final norm for the M3 text backbone."""

    def __init__(
        self,
        config: Any,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
    ):
        super().__init__()
        self.backend = backend
        # MiniMax M3 routes experts in fp32 (sglang hardcodes the router gate to
        # fp32: projection + sigmoid + correction-bias add). Forcing it here avoids
        # bf16 top-k drift -> different expert selection -> different logits. Set
        # before the decoder/MTP blocks (which build the MoE gates) are constructed.
        if self.backend.gate_precision is None:
            self.backend.gate_precision = torch.float32
        self.config = config
        self.config.num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", None))

        dtype = get_dtype(getattr(config, "torch_dtype", "bfloat16"), torch.bfloat16)
        self.moe_config = moe_config or build_moe_config(config, dtype)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.num_hidden_layers):
            self.layers[str(layer_id)] = Block(layer_id, config, self.moe_config, backend)

        gemma = getattr(config, "use_gemma_norm", False)
        self.norm = MiniMaxM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma=gemma)

        self.max_seq_len = config.max_position_embeddings
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        if not hasattr(config, "rope_parameters") or config.rope_parameters is None:
            rotary_dim = getattr(config, "rotary_dim", self.head_dim)
            config.rope_parameters = {
                "rope_theta": getattr(config, "rope_theta", 5000000.0),
                "rope_type": "default",
                "partial_rotary_factor": rotary_dim / self.head_dim,
            }

        base, rope_scaling, partial_rotary_factor = get_rope_config(config)
        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            base=base,
            dtype=torch.float32,
            initial_context_length=rope_scaling.get("original_max_position_embeddings", 4096),
            scaling_factor=rope_scaling.get("factor", 1.0),
            ntk_alpha=rope_scaling.get("beta_slow", 1.0),
            ntk_beta=rope_scaling.get("beta_fast", 32.0),
            partial_rotary_factor=partial_rotary_factor,
            device=torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"),
        )

        # Multi-token prediction (DeepSeek-V3 style); shares the main lm_head.
        num_mtp = int(getattr(config, "num_mtp_modules", 0) or 0)
        self.mtp = MiniMaxM3MTP(config, self.moe_config, backend, num_mtp) if num_mtp > 0 else None

    def make_freqs_cis(self, position_ids: torch.Tensor, **attn_kwargs: Any) -> torch.Tensor:
        return position_ids_to_freqs_cis(
            self.rotary_emb,
            position_ids,
            qkv_format=attn_kwargs.get("qkv_format", "bshd"),
            for_fused_rope=self.backend.rope_fusion,
            cp_size=attn_kwargs.get("cp_size", 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        # Pipeline stages after the first receive the previous stage's hidden
        # states in the input_ids slot (a float tensor) with embed_tokens=None.
        if inputs_embeds is None and input_ids is not None and torch.is_floating_point(input_ids):
            inputs_embeds = input_ids
            input_ids = None
        h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(0, h.shape[1], device=h.device).unsqueeze(0).expand(h.shape[0], -1)

        freqs_cis = self.make_freqs_cis(position_ids, **attn_kwargs)

        for layer in self.layers.values():
            h = layer(
                x=h,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                # Forwarded so CP-aware sparse attention can derive per-document
                # boundaries (position_ids reset to 0 per packed document) for
                # block-diagonal masking; ignored/popped by the eager path.
                position_ids=position_ids,
                **attn_kwargs,
            )

        # norm is None on non-final pipeline stages.
        return self.norm(h) if self.norm is not None else h

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        # embed_tokens / norm / layers can be None under meta-device + PP/sharded
        # init (the framework calls init_weights tolerating absent modules), so
        # guard each (matching minimax_m2 / step3p7).
        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()
            self.rotary_emb.device = buffer_device
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)
        if self.mtp is not None:
            self.mtp.init_weights(buffer_device)

    def mtp_logits(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        lm_head: nn.Module,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> list[torch.Tensor]:
        """Per-depth MTP logits from the final hidden states (shares ``lm_head``)."""
        if position_ids is None:
            position_ids = (
                torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
                .unsqueeze(0)
                .expand(hidden_states.shape[0], -1)
            )
        freqs_cis = self.make_freqs_cis(position_ids, **attn_kwargs)
        return self.mtp(
            hidden_states,
            input_ids=input_ids,
            embed_fn=self.embed_tokens,
            lm_head=lm_head,
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **attn_kwargs,
        )


class MiniMaxM3SparseForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Standalone M3 text backbone for causal LM (Stage 1 parity target)."""

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY

    _keep_in_fp32_modules_strict = ["mlp.gate.e_score_correction_bias"]

    # The state-dict adapter loads every tensor from the checkpoint, so skip HF
    # random init on load (also avoids DTensor-collective hangs under sharding/PP).
    _skip_init_weights_on_load = True

    @classmethod
    def from_config(
        cls, config: Any, moe_config: MoEConfig | None = None, backend: BackendConfig | None = None, **kwargs
    ):
        return cls(config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        config = MiniMaxM3VLTextConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        reject_unsupported_tie_word_embeddings(type(self), config)
        self.backend = backend or BackendConfig()
        self.model = MiniMaxM3TextModel(config, backend=self.backend, moe_config=moe_config)
        self.lm_head = initialize_linear_module(self.backend.linear, config.hidden_size, config.vocab_size, bias=False)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = MiniMaxM3StateDictAdapter(
                self.config,
                self.model.moe_config,
                self.backend,
                dtype=get_dtype(getattr(config, "torch_dtype", "bfloat16"), torch.bfloat16),
            )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if attn_kwargs.get("qkv_format") == "thd":
            input_ids, position_ids, padding_mask, attn_kwargs = squeeze_input_for_thd(
                input_ids, position_ids, padding_mask, attn_kwargs
            )
            attention_mask = None

        hidden = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **attn_kwargs,
        )
        logits = self.lm_head(hidden) if self.lm_head else hidden
        if attn_kwargs.get("qkv_format") == "thd":
            return logits.unsqueeze(0)
        if self.model.mtp is not None and self.training and input_ids is not None:
            mtp_logits = self.model.mtp_logits(
                hidden,
                input_ids,
                self.lm_head,
                position_ids=position_ids,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                **attn_kwargs,
            )
            return MiniMaxM3CausalLMOutput(logits=logits, mtp_per_depth_logits=mtp_logits)
        return logits

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        buffer_device = buffer_device or torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        with buffer_device:
            self.model.init_weights(buffer_device=buffer_device)
            final_out_std = self.config.hidden_size**-0.5
            if self.lm_head is not None:
                nn.init.trunc_normal_(
                    self.lm_head.weight, mean=0.0, std=final_out_std, a=-3 * final_out_std, b=3 * final_out_std
                )
        cast_model_to_dtype(self, dtype)
        with buffer_device:
            self.model.rotary_emb.device = buffer_device


class MiniMaxM3SparseForConditionalGeneration(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """MiniMax M3 VL: CLIP-style vision tower + projector/merger + M3 text backbone.

    Vision features (``vision_tower(pixel_values, grid_thw)``) are spliced into
    the text embeddings at ``image_token_index`` / ``video_token_index``
    positions, then run through the (sparse/dense MoE) language model + lm_head.
    """

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY

    # Pipeline-parallel routing: keep this VLM's own forward (which splices vision
    # features) instead of letting patch_hf_model_for_pp swap in the generic
    # CausalLM forward (which would drop pixel_values). MTP per-depth outputs are
    # logits (shared lm_head); rotary buffers stay fp32. "rotary_emb" covers the text
    # rope; "inv_freq" additionally pins the vision tower's rotary buffer
    # (vision_encoder.py) fp32 — the bf16 cast would otherwise round it and degrade
    # vision RoPE (see llama/rope_utils.py).
    _keep_in_fp32_modules = ["rotary_emb", "inv_freq"]
    _keep_in_fp32_modules_strict = ["mlp.gate.e_score_correction_bias"]
    _pp_keep_self_forward: bool = True
    mtp_outputs_are_logits = True
    # Opt into context parallelism on the SDPA attention backend (M3's block-sparse DSA
    # bias is an explicit additive mask that only SDPA accepts, not TE). Dense layers use
    # the standard CP path (mask-strip hook + is_causal); sparse layers require the
    # CP-aware indexer attention for a correct global-sequence bias.
    _supports_cp_sdpa = True
    # The state-dict adapter fully populates every tensor from the checkpoint
    # (MXFP8 -> bf16), so skip HF random init on load. This also avoids the
    # stage-divergent DTensor collectives in initialize_weights() under sharding/PP.
    _skip_init_weights_on_load = True
    # CP submesh, installed by the MoE parallelizer's apply_cp when context
    # parallelism is active; None (default) means the forward embeds and shards
    # nothing for CP. See prepare_model_inputs_for_cp / forward.
    cp_mesh = None

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class.

        Mirrors the MiniMax M2 backbone: PP + EP are validated; TP is unsupported
        by the custom MoE parallelizer. Context parallelism is supported via the
        CP-aware block-sparse DSA attention (dense layers use the standard CP
        path; sparse layers gather K/V and rebuild the global block-sparse mask
        using FlexAttention).
        """

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True

    @classmethod
    def from_config(
        cls, config: Any, moe_config: MoEConfig | None = None, backend: BackendConfig | None = None, **kwargs
    ):
        return cls(config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        config = MiniMaxM3VLConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        reject_unsupported_tie_word_embeddings(type(self), config)
        text_config = config.text_config
        self.backend = backend or BackendConfig()
        self.model = MiniMaxM3TextModel(text_config, backend=self.backend, moe_config=moe_config)
        self.lm_head = initialize_linear_module(
            self.backend.linear, text_config.hidden_size, text_config.vocab_size, bias=False
        )
        self.vision_tower = MiniMaxM3VisionModel(
            config.vision_config,
            text_config.hidden_size,
            config.projector_hidden_size,
            projector_hidden_act=config.projector_hidden_act,
            multimodal_projector_bias=config.multimodal_projector_bias,
            patch_merge_bias=getattr(config, "patch_merge_bias", config.multimodal_projector_bias),
        )
        self.image_token_index = config.image_token_index
        self.video_token_index = config.video_token_index
        self.vocab_size = text_config.vocab_size
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = MiniMaxM3VLStateDictAdapter(
                config,
                self.model.moe_config,
                self.backend,
                dtype=get_dtype(getattr(text_config, "torch_dtype", "bfloat16"), torch.bfloat16),
            )

    @property
    def language_model(self):
        return self.model

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def customize_pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        *,
        layers_prefix: str,
        text_model: nn.Module | None = None,
    ) -> list[list[str]]:
        """Rewrite auto-generated pipeline FQNs to M3's real module paths.

        M3's text stack lives directly under ``self.model`` and the vision tower
        is a top-level sibling (``vision_tower``). The framework, seeing the
        ``language_model`` property, derives a nested ``model.language_model.``
        prefix for the text modules and a ``model.`` prefix for the multimodal
        encoders. Map both back to M3's actual paths so per-stage module nulling
        keeps/drops the correct submodules.
        """
        if getattr(self.model, "mtp", None) is not None:
            raise NotImplementedError(
                "MiniMax M3 VL does not support MTP modules under pipeline parallelism yet; "
                "set text_config.num_mtp_modules=0 for pp_size>1 runs."
            )
        from nemo_automodel.components.distributed.pipelining.hf_utils import MULTIMODAL_SUFFIXES

        text_prefix = "model."  # M3's text stack lives directly under self.model
        fixed: list[list[str]] = []
        for stage in module_names_per_stage:
            names: list[str] = []
            for name in stage:
                if layers_prefix != text_prefix and name.startswith(layers_prefix):
                    names.append(text_prefix + name[len(layers_prefix) :])
                elif name.startswith(text_prefix) and name[len(text_prefix) :] in MULTIMODAL_SUFFIXES:
                    names.append(name[len(text_prefix) :])
                else:
                    names.append(name)
            fixed.append(names)
        return fixed

    def _is_pipeline_parallel_stage(self) -> bool:
        """True when this is a partial pipeline stage (some text modules nulled)."""
        if self.lm_head is None:
            return True
        if getattr(self.model, "embed_tokens", None) is None:
            return True
        try:
            return len(self.model.layers) != int(self.config.text_config.num_hidden_layers)
        except (TypeError, AttributeError):
            return False

    def get_pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Per-stage input/output meta tensors for the PP schedule's shape inference.

        First stage consumes the FULL token ids ``[mb, seq]``; later stages
        consume hidden states. The final stage (owning ``lm_head``) emits logits;
        earlier stages emit hidden states.

        Under context parallelism the first stage embeds the full sequence and
        shards it to this rank's round-robin chunk pair inside forward
        (see :func:`shard_sequence_for_cp`), so every stage output and every
        later-stage input carries the LOCAL (padded-to-``2*cp`` then ``//cp``)
        sequence length while the first stage's input stays full-length. At
        ``cp_size == 1`` the lengths coincide and the layout is symmetric.
        """
        text_config = self.config.text_config
        hidden_size = text_config.hidden_size
        vocab_size = text_config.vocab_size

        cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1
        local_seq_len = seq_len
        if cp_size > 1:
            padded_seq_len = seq_len + (-seq_len) % (2 * cp_size)
            local_seq_len = padded_seq_len // cp_size

        def meta(*shape: int) -> torch.Tensor:
            return torch.empty(*shape, device="meta", dtype=dtype)

        # Inter-stage tensors (hidden states) carry the model/activation dtype the
        # framework passes in. token ids are always long.
        if is_first:
            inputs_meta = (torch.empty(microbatch_size, seq_len, device="meta", dtype=torch.long),)
        else:
            inputs_meta = (meta(microbatch_size, local_seq_len, hidden_size),)

        if self.lm_head is not None:
            # Logits follow lm_head's own param dtype, which may diverge from the
            # model dtype if lm_head is ever kept in fp32 (_keep_in_fp32_modules);
            # deriving it here keeps the schedule's output buffer correctly sized.
            head_dtype = getattr(getattr(self.lm_head, "weight", None), "dtype", dtype)
            outputs_meta = (torch.empty(microbatch_size, local_seq_len, vocab_size, device="meta", dtype=head_dtype),)
        else:
            outputs_meta = (meta(microbatch_size, local_seq_len, hidden_size),)
        return inputs_meta, outputs_meta

    @staticmethod
    def _to_grid_list(grid_thw) -> list[list[int]]:
        if isinstance(grid_thw, torch.Tensor):
            return grid_thw.detach().cpu().to(torch.int64).tolist()
        return [list(map(int, g)) for g in grid_thw]

    def _splice_multimodal(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        grid_thw,
        token_index: int,
    ) -> torch.Tensor:
        # The vision tower's bidirectional patch attention is not CP-sharded; when
        # this embed+splice runs in-forward under an active CP ring context it must
        # suspend the ring dispatcher, or torch's load-balanced ring SDPA rejects
        # the non-causal attention. No-op when CP is inactive.
        from nemo_automodel.components.distributed.cp_utils import cp_dispatcher_suspended  # noqa: PLC0415

        with cp_dispatcher_suspended(self.cp_mesh):
            features = self.vision_tower(pixel_values, self._to_grid_list(grid_thw))
        mask = input_ids == token_index
        expected = int(mask.sum().item())
        if features.shape[0] != expected:
            raise ValueError(
                f"MiniMax M3 VL: got {features.shape[0]} vision tokens for {expected} placeholder positions "
                f"(token_index={token_index})."
            )
        inputs_embeds[mask] = features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        return inputs_embeds

    def _embed_and_splice(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw=None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw=None,
    ) -> torch.Tensor:
        """Embed token ids and splice vision/video features into the embeddings.

        Shared by ``forward`` and ``prepare_model_inputs_for_cp`` so the splice logic
        lives in one place.
        """
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None or pixel_values_videos is not None:
            inputs_embeds = inputs_embeds.clone()
        if pixel_values is not None:
            inputs_embeds = self._splice_multimodal(
                inputs_embeds, input_ids, pixel_values, image_grid_thw, self.image_token_index
            )
        if pixel_values_videos is not None:
            inputs_embeds = self._splice_multimodal(
                inputs_embeds, input_ids, pixel_values_videos, video_grid_thw, self.video_token_index
            )
        return inputs_embeds

    def prepare_model_inputs_for_cp(
        self,
        batch: dict[str, Any],
        *,
        num_chunks: int = 1,
    ) -> dict[str, Any]:
        """Return a sharder-only CP backend; embed + splice + shard happen in forward.

        Context-parallel embedding, vision splice, and sequence sharding now run
        inside ``forward`` per microbatch (Megatron-style): the returned
        :class:`ContextParallelismSharder` round-robin-shards only the no-grad
        aux streams (labels/position_ids/loss_mask/padding_mask) via
        :func:`shard_batch_aux_only` and leaves ``input_ids`` and the multimodal
        inputs full-length for the forward. The forward then embeds+splices the
        full sequence and calls :func:`shard_sequence_for_cp` on the result, so
        the embeddings and vision tower are trainable under CP and the PP×CP
        shared pre-embed graph no longer exists. Nothing is consumed here.
        Defining this method is the opt-in signal the recipe checks
        (``hasattr(model, "prepare_model_inputs_for_cp")``).

        Args:
            batch: The full-sequence batch; left intact (nothing consumed).
            num_chunks: Accepted for hook-signature parity; unused (round-robin CP).
        """
        del batch, num_chunks
        return {
            "cp_sharder": ContextParallelismSharder(
                shard_batch=shard_batch_aux_only,
                local_token_global_indices=round_robin_local_indices,
            )
        }

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw=None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw=None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        is_pp_stage = self._is_pipeline_parallel_stage()

        # Authoritative MTP-under-PP guard: keyed on the config (which survives the
        # splitter nulling the mtp module) and is_pp_stage, so it fires for both the
        # auto-generated split (also caught earlier in customize_pipeline_stage_modules)
        # and a manually supplied module_fqns_per_model_part that bypasses that hook.
        if is_pp_stage and int(getattr(self.config.text_config, "num_mtp_modules", 0) or 0) > 0:
            raise NotImplementedError(
                "MiniMax M3 VL does not support MTP modules under pipeline parallelism yet; "
                "set text_config.num_mtp_modules=0 for pp_size>1 runs."
            )

        # Pipeline stage 0 does not receive media in the batch: the VLM-PP collate
        # strips pixel_values/grids and stage_vlm_media_for_pp attaches them here as
        # per-microbatch chunks. Pull this microbatch's media off the cursor before
        # embedding so vision features still get spliced (mirrors KimiVL/Step3p7).
        # `_vlm_image_grid_hws_chunks` holds M3's image_grid_thw values (the PP media
        # prep stores whatever grid the model emits under that key), so no reshape.
        chunks = getattr(self, "_vlm_pixel_values_chunks", None)
        if (
            pixel_values is None
            and pixel_values_videos is None
            and input_ids is not None
            and not torch.is_floating_point(input_ids)
            and (chunks is not None or getattr(self, "_vlm_pixel_values_videos_chunks", None) is not None)
        ):
            chunk_idx = getattr(self, "_vlm_chunk_idx", 0)
            consumed = False
            if chunks is not None and (input_ids == self.image_token_index).any() and chunk_idx < len(chunks):
                pixel_values = chunks[chunk_idx]
                image_grid_thw = self._vlm_image_grid_hws_chunks[chunk_idx]
                consumed = True
            video_chunks = getattr(self, "_vlm_pixel_values_videos_chunks", None)
            if (
                video_chunks is not None
                and (input_ids == self.video_token_index).any()
                and chunk_idx < len(video_chunks)
            ):
                pixel_values_videos = video_chunks[chunk_idx]
                video_grid_thw = self._vlm_video_grid_thw_chunks[chunk_idx]
                consumed = True
            if consumed:
                self._vlm_chunk_idx = chunk_idx + 1

        cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1

        # Media under CP×PP rides the same per-microbatch side channel as cp1×PP:
        # stage_vlm_media_for_pp stashed grid-aware pixel chunks (pulled just above),
        # and the embed + vision splice below runs on this microbatch's FULL sequence
        # before shard_sequence_for_cp shards it. So the CP shard composes with the
        # media staging without changing the stage metas (the first-stage input is
        # still input_ids [mb, S]; media never enters the stage tensor stream).

        # Pipeline stages after the first receive the previous stage's hidden
        # states in the input_ids slot (a float tensor); route them straight to
        # the text model (no embedding / vision splicing on non-first stages).
        if inputs_embeds is None and input_ids is not None and torch.is_floating_point(input_ids):
            inputs_embeds = input_ids
            input_ids = None

        if inputs_embeds is None:
            inputs_embeds = self._embed_and_splice(
                input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
            # Per-microbatch CP: keep this rank's round-robin chunk pair of the
            # freshly embedded full sequence. The aux streams and the ring-SDPA
            # context were sharded to the same layout by shard_batch_aux_only, and
            # the shard is differentiable so gradients reach the embeddings and
            # vision tower.
            if cp_size > 1:
                inputs_embeds, _, _ = shard_sequence_for_cp(self.cp_mesh, inputs_embeds, seq_dim=1)

        hidden = self.model(
            None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        # lm_head is None on non-final pipeline stages -> forward hidden states.
        logits = self.lm_head(hidden) if self.lm_head is not None else hidden

        # Pipeline stages return a plain tensor; the schedule wires each stage's
        # output to the next stage's input and only the final stage owns lm_head.
        if is_pp_stage:
            return logits

        if self.model.mtp is not None and self.training and input_ids is not None:
            mtp_logits = self.model.mtp_logits(
                hidden, input_ids, self.lm_head, position_ids=position_ids, attention_mask=attention_mask, **kwargs
            )
            return MiniMaxM3CausalLMOutput(logits=logits, mtp_per_depth_logits=mtp_logits)
        return logits

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        buffer_device = buffer_device or torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        with buffer_device:
            self.model.init_weights(buffer_device=buffer_device)
            if self.lm_head is not None:
                final_out_std = self.config.text_config.hidden_size**-0.5
                nn.init.trunc_normal_(
                    self.lm_head.weight, mean=0.0, std=final_out_std, a=-3 * final_out_std, b=3 * final_out_std
                )
        cast_model_to_dtype(self, dtype)
        with buffer_device:
            self.model.rotary_emb.device = buffer_device


ModelClass = MiniMaxM3SparseForConditionalGeneration
