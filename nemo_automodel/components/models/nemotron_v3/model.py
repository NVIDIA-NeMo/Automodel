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

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common import (
    BackendConfig,
    HFCheckpointingMixin,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Block
from nemo_automodel.components.models.nemotron_v3.mtp import (
    _resolve_block_types_per_sublayer,
    build_mtp_config_from_hf,
    build_nemotron_v3_mtp,
)
from nemo_automodel.components.models.nemotron_v3.state_dict_adapter import NemotronV3StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


@dataclass
class NemotronHCausalLMOutputWithPast(CausalLMOutputWithPast):
    """``CausalLMOutputWithPast`` plus declared MTP fields.

    The MTP per-depth hidden states and scaling factor must be regular
    dataclass fields (rather than dynamically-set attributes) so they survive
    output-restructuring layers like FSDP2's mixed-precision output cast,
    which rebuild ``ModelOutput`` instances from declared fields only.
    """

    mtp_per_depth_h: Optional[list[torch.Tensor]] = None
    mtp_loss_scaling_factor: Optional[float] = None


class NemotronV3Model(nn.Module):
    """NemotronV3 base model (without LM head).

    This is a hybrid architecture with Mamba2, Attention, MLP, and MoE layers.
    """

    def __init__(
        self,
        config,
        backend: BackendConfig | None = None,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict | None = None,
    ):
        """Initialize NemotronV3Model.

        Args:
            config: NemotronH config with model parameters
            backend: Backend configuration for MoE and other components
            moe_config: MoE configuration (optional, will create default if None)
            moe_overrides: Optional dict of overrides to apply to the default MoE config
        """
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        if moe_config is not None and moe_overrides is not None:
            raise ValueError("Cannot pass both moe_config and moe_overrides; use one or the other.")
        moe_defaults = dict(
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=1,  # NemotronV3 has 1 shared expert
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=config.n_group,
            n_limited_groups=config.topk_group,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,  # No aux loss for NemotronV3
            score_func="sigmoid",  # NemotronV3 uses sigmoid scoring
            route_scale=config.routed_scaling_factor,
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,  # For shared expert
            moe_inter_dim=config.moe_intermediate_size,  # For routed experts
            norm_topk_prob=config.norm_topk_prob,
            router_bias=False,
            expert_bias=config.mlp_bias,
            expert_activation="relu2",  # NemotronV3 uses ReLU² activation
            dtype=config.torch_dtype,
            shared_expert_gate=False,
            shared_expert_inter_dim=config.moe_shared_expert_intermediate_size,
            shared_expert_activation="relu2",  # Use ReLU² for shared experts
            force_e_score_correction_bias=True,  # NemotronV3 checkpoint has this buffer
            moe_latent_size=getattr(config, "moe_latent_size", None),
        )
        if moe_overrides:
            moe_defaults.update(moe_overrides)
        self.moe_config = moe_config or MoEConfig(**moe_defaults)

        # Embeddings
        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)

        # Transformer layers (hybrid: mamba, attention, mlp, moe)
        self.layers = nn.ModuleDict()
        for idx in range(config.num_hidden_layers):
            self.layers[str(idx)] = NemotronV3Block(
                config, layer_idx=idx, moe_config=self.moe_config, backend=self.backend
            )

        # Final norm
        self.norm = initialize_rms_norm_module(
            self.backend.rms_norm,
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        causal_mask_mapping: dict[str, torch.Tensor] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the model.  Supports BSHD ``[B, S, H]`` and THD ``[T, H]``.

        Pipeline-parallel awareness: when ``self.embed_tokens is None`` (non-first
        PP stage), ``input_ids`` is interpreted as the upstream hidden-state
        tensor and routed through the ``inputs_embeds`` branch. When
        ``self.norm is None`` (non-last PP stage), the final norm is skipped.
        """
        # Get embeddings (PP-aware: a trimmed mid-stage has embed_tokens=None
        # and receives the prior stage's hidden states as input_ids).
        if inputs_embeds is None:
            if getattr(self, "embed_tokens", None) is not None:
                if input_ids is None:
                    raise ValueError("input_ids must be provided if inputs_embeds is not provided")
                hidden_states = self.embed_tokens(input_ids)
            else:
                if input_ids is None or input_ids.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise ValueError("Non-first PP stage expects an upstream hidden-state tensor")
                hidden_states = input_ids
        else:
            hidden_states = inputs_embeds

        # When qkv_format="thd" is explicitly requested AND we're using TE
        # attention (which natively supports THD), squeeze to 2D [T, H] so
        # attention layers pick the THD branch. Other attention backends
        # (sdpa, flex) only support 4D BSHD, so leave hidden_states as
        # [B, S, H] and rely on cu_seqlens for mamba's seq_idx alone.
        _attn_impl = getattr(getattr(self, "backend", None), "attn", None)
        squeezed_for_thd = False
        if (
            kwargs.get("qkv_format") == "thd"
            and _attn_impl == "te"
            and hidden_states.dim() == 3
            and hidden_states.shape[0] == 1
        ):
            hidden_states = hidden_states.squeeze(0)
            squeezed_for_thd = True

        is_thd = hidden_states.dim() == 2

        # ``seq_idx`` is what Mamba consumes for per-token segment boundaries.
        # Under neat packing it's derived in the outer ``NemotronHForCausalLM``
        # forward from ``_packed_seq_ids`` and threaded down via kwargs.
        # Non-THD path (no THD collator) doesn't emit ``cu_seqlens`` — recover
        # it from a 2D indexed attention_mask so mamba's seq_idx derivation
        # has something to work with. Skipped when attention_mask is the neat
        # 4D bool mask (seq_idx is already built upstream from _packed_seq_ids).
        if (
            "cu_seqlens" not in kwargs
            and attention_mask is not None
            and attention_mask.dim() == 2
            and attention_mask.dtype != torch.bool
        ):
            seq_lens = attention_mask.sum(dim=-1).to(torch.int32)
            kwargs["cu_seqlens"] = F.pad(seq_lens.cumsum(0).to(torch.int32), (1, 0))

        # Normalize cu_seqlens / cu_seqlens_padded / max_seqlen to 1D / 0D.
        # The THD collator emits per-microbatch rows stacked as
        # ``[num_microbatches, max_K]`` (cu_seqlens / _padded) and
        # ``[num_microbatches]`` (max_seqlen), with ``-1000`` right-padding on
        # cu_seqlens rows so they're rectangular across chunks with differing
        # sub-seq counts (see ``thd_utils.split_batch_into_thd_chunks.pad_and_stack``).
        # PyTorch PP then ``tensor_split``-chunks along dim 0, so each stage
        # forward receives ``[1, max_K]`` (cu_seqlens) and ``[1]`` (max_seqlen).
        # Squeeze to 1D / 0D, strip any sentinel right-pad, and clone to a
        # fresh contiguous tensor so the values survive the TE backward (PP
        # may free view-backed kwarg storage). Otherwise the kernel sees
        # ``-1000`` boundaries and either writes OOB (TE cuDNN-fused-attn-bwd
        # on a 576-token "phantom" sub-seq) or computes garbage seq_idx
        # (mamba).
        _SEQLEN_SENTINEL = -1000
        for _k in ("cu_seqlens", "cu_seqlens_padded"):
            _v = kwargs.get(_k)
            if isinstance(_v, torch.Tensor) and _v.dim() == 2 and _v.shape[0] == 1:
                _v = _v.squeeze(0)
            if isinstance(_v, torch.Tensor) and _v.dim() == 1:
                # Strip sentinel right-pad if present (cheap no-op when absent).
                if (_v == _SEQLEN_SENTINEL).any():
                    _v = _v[_v != _SEQLEN_SENTINEL]
                kwargs[_k] = _v.contiguous().clone()
        # max_seqlen may arrive as [1] tensor; flatten to a scalar tensor.
        _ms = kwargs.get("max_seqlen")
        if isinstance(_ms, torch.Tensor) and _ms.dim() >= 1 and _ms.numel() == 1:
            kwargs["max_seqlen"] = _ms.flatten()[0].clone()

        # Get 4D causal mask for attention layers (from precomputed masks).
        causal_mask = causal_mask_mapping.get("full_attention") if causal_mask_mapping is not None else None

        # Detect neat-packed SDPA path: ``seq_idx`` was already built upstream
        # from ``_packed_seq_ids`` and attention_mask is a 4D block-causal
        # bool. Under this path Mamba relies on seq_idx alone (no 2D padding
        # mask is meaningful), and the 4D mask is the attention mask.
        _neat_packed = "seq_idx" in kwargs and attention_mask is not None and attention_mask.dim() == 4

        # Apply transformer layers
        for layer in self.layers.values():
            # Pass appropriate mask based on layer type
            if is_thd:
                mask = None
            elif _neat_packed:
                # Mamba uses seq_idx for segment boundaries; pass no mask.
                # Attention gets the 4D block-causal mask directly.
                mask = attention_mask if layer.block_type == "attention" else None
            elif layer.block_type == "attention":
                # Attention layers use 4D causal mask; fall back to 2D attention_mask
                # when causal_mask is None (e.g. during TE+CP training where CP split
                # removes the precomputed 4D mask) so TE can use padding_causal mode.
                mask = causal_mask if causal_mask is not None else attention_mask
            elif layer.block_type == "mamba":
                # Mamba layers use 2D padding mask during prefill, None during decode
                mask = None if (past_key_values is not None and past_key_values.has_previous_state) else attention_mask
            else:
                # MLP/MoE layers don't use mask
                mask = None

            hidden_states = layer(
                hidden_states,
                attention_mask=mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        # Final norm (skipped on non-last PP stages where the splitter trims
        # the norm module to None).
        if getattr(self, "norm", None) is not None:
            hidden_states = self.norm(hidden_states)

        # Restore batch dimension if we squeezed for THD
        if squeezed_for_thd:
            hidden_states = hidden_states.unsqueeze(0)

        return hidden_states

    @torch.no_grad()
    def initialize_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize model weights according to NemotronV3 spec.

        After PP splitting, ``embed_tokens`` may be ``None`` on non-first
        stages and ``norm`` may be ``None`` on non-last stages; guard each.

        Args:
            buffer_device: Device to use for buffer initialization
        """
        with buffer_device:
            if getattr(self, "embed_tokens", None) is not None:
                nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=self.config.initializer_range)
            if getattr(self, "norm", None) is not None:
                self.norm.reset_parameters()

        # Initialize all layers via delegation
        for block in self.layers.values():
            block.init_weights(buffer_device=buffer_device)


class NemotronHForCausalLM(HFCheckpointingMixin, GenerationMixin, nn.Module, MoEFSDPSyncMixin):
    """NemotronV3 model with language modeling head.

    Supports ``.generate()`` from ``transformers.generation.GenerationMixin`` with O(1)
    per-step KV caching for attention layers and recurrent state caching for Mamba2 layers.
    """

    # Prevent GenerationMixin from creating a DynamicCache: the hybrid Mamba2/Attention
    # architecture uses its own NemotronHybridCache.
    _is_stateful: bool = True
    main_input_name: str = "input_ids"

    # Keep our own PP-aware forward (with hybrid mask routing, THD squeeze,
    # mamba kwargs, MTP variadic propagation). Skips ``patch_hf_model_for_pp``
    # at split time — see hf_utils.py:591-600.
    _pp_keep_self_forward: bool = True

    @classmethod
    def from_config(
        cls,
        config,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        """Create model from config.

        Args:
            config: NemotronH config
            backend: Backend configuration
            **kwargs: Additional arguments

        Returns:
            NemotronHForCausalLM instance
        """
        return cls(config, backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        """Load pretrained model.

        Args:
            pretrained_model_name_or_path: Path or name of pretrained model
            *model_args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            NemotronHForCausalLM instance
        """
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config,
        backend: BackendConfig | None = None,
        *,
        mtp_loss_scaling_factor: float = 0.1,
        num_nextn_predict_layers: int | None = None,
        mtp_use_repeated_layer: bool = False,
        **kwargs,
    ):
        """Initialize NemotronV3ForCausalLM.

        Args:
            config: NemotronH config.
            backend: Backend configuration.
            mtp_loss_scaling_factor: Auxiliary-loss weight for the MTP head
                (default ``0.1``). Programmatic override only — not exposed
                as a YAML knob to keep recipe configs auto-detected.
            num_nextn_predict_layers: Optional override for the HF config's
                ``num_nextn_predict_layers`` field (i.e. the MTP forward
                iteration count). When ``None``, the value from ``config`` is
                used. Set explicitly when the trained model used weight-tied
                MTP (``mtp_use_repeated_layer=True``) and the HF export only
                retains the physical depth count.
            mtp_use_repeated_layer: When ``True``, build a single physical
                MTP depth and reuse it across all iterations. Mirrors
                Megatron's ``--mtp-use-repeated-layer``. Defaults to ``False``.
            **kwargs: Additional arguments. Recognized keys:
                ``moe_config``, ``moe_overrides``.
        """
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()

        # Base model
        moe_overrides = kwargs.pop("moe_overrides", None)
        moe_config = kwargs.pop("moe_config", None)
        self.model = NemotronV3Model(
            config,
            backend=self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )
        self.output_hidden_states = config.to_dict().get("output_hidden_states", False)

        # LM head
        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.lm_head = initialize_linear_module(
            self.backend.linear,
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=dtype,
        )

        # self.mtp is None when num_nextn_predict_layers is absent or 0.
        self.mtp_config = build_mtp_config_from_hf(
            config,
            loss_scaling_factor=mtp_loss_scaling_factor,
            num_nextn_predict_layers=num_nextn_predict_layers,
            use_repeated_layer=mtp_use_repeated_layer,
        )
        if self.mtp_config.enabled:
            # Resolve block types from either ``mtp_hybrid_override_pattern``
            # (symbol-string form) or ``mtp_layers_block_type`` (list form).
            # The list flows through here so build_nemotron_v3_mtp doesn't
            # need to re-parse the (possibly sentinel) MTPConfig.layer_pattern.
            block_types = _resolve_block_types_per_sublayer(config)
            self.mtp = build_nemotron_v3_mtp(
                config,
                mtp_config=self.mtp_config,
                backend=self.backend,
                moe_config=self.model.moe_config,
                dtype=dtype,
                block_types=block_types,
            )
        else:
            self.mtp = None

        # Create state_dict_adapter if enabled (needed to convert HF checkpoints)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = NemotronV3StateDictAdapter(
                config=config,
                moe_config=self.model.moe_config,
                backend=self.backend,
                dtype=dtype,
            )

        # Required by GenerationMixin.generate()
        self.generation_config = GenerationConfig()

    @property
    def device(self) -> torch.device:
        """Return the device of the first model parameter (required by GenerationMixin)."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the first model parameter (used by cache construction)."""
        return next(self.parameters()).dtype

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _is_pipeline_parallel_stage(self) -> bool:
        """True when this module instance has been trimmed to a PP stage subset.

        Detection mirrors ``DeepseekV4ForCausalLM._is_pipeline_parallel_stage``:
        any of (a) ``lm_head`` is None, (b) inner ``embed_tokens`` is None,
        (c) ``model.layers`` count diverges from ``config.num_hidden_layers``
        is sufficient — the PP splitter nulls these attributes when trimming.

        The checks use ``hasattr`` to distinguish "splitter nulled the
        attribute" (attribute present, value is None) from "caller replaced
        ``self.model`` with a stub that doesn't declare the attribute"
        (attribute absent). Tests that swap in stub inner modules should not
        be misclassified as PP stages.
        """
        if self.lm_head is None:
            return True
        if hasattr(self.model, "embed_tokens") and self.model.embed_tokens is None:
            return True
        if hasattr(self.model, "layers"):
            try:
                return len(self.model.layers) != int(self.config.num_hidden_layers)
            except TypeError:
                return False
        return False

    def _build_mtp_embed_inputs_for_pp(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Build the per-depth rolled-token embeddings on the first PP stage.

        The first PP stage owns ``embed_tokens`` and is the only rank that can
        produce the future-token embeddings consumed by the MTP head on the
        final stage. The tuple flows alongside ``hidden_states`` through every
        intermediate stage as additional positional outputs (see ``forward``).

        Args:
            input_ids: Token ids ``[B, S]`` (int).

        Returns:
            Tuple of length ``self.mtp_config.num_layers`` containing
            ``[B, S, hidden]`` embeddings for depths 1..D (i.e. for predicting
            tokens shifted left by 1..D positions).
        """
        if getattr(self.model, "embed_tokens", None) is None:
            raise ValueError("First PP stage must own embed_tokens to build MTP embeddings")
        if input_ids.dtype not in (torch.int32, torch.int64, torch.long):
            raise ValueError("First PP stage must receive token ids to build MTP embeddings")

        from nemo_automodel.components.models.common.mtp import roll_tensor  # noqa: PLC0415

        cur_input_ids = input_ids
        embeds: list[torch.Tensor] = []
        for _ in range(self.mtp_config.num_layers):
            cur_input_ids = roll_tensor(cur_input_ids, shifts=-1, dim=-1)
            embeds.append(self.model.embed_tokens(cur_input_ids))
        return tuple(embeds)

    def customize_pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        *,
        layers_prefix: str,
        text_model: nn.Module | None = None,
    ) -> list[list[str]]:
        """Pin the MTP head to the last PP stage's FQN list.

        Called by ``split_model_into_stages`` (functional.py:494-502) after the
        default per-stage FQN auto-generation. The auto-generator includes
        ``embed_tokens`` on the first stage and ``norm``/``lm_head`` on the
        last stage but doesn't know about ``model.mtp``; this hook appends it.
        """
        del layers_prefix, text_model  # unused — no per-stage rotary to replicate
        stage_modules = [list(m) for m in module_names_per_stage]
        if self.mtp is not None and stage_modules:
            last = stage_modules[-1]
            if "mtp" not in last:
                last.append("mtp")
        return stage_modules

    def get_pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Return analytical (inputs_meta, outputs_meta) for a PP stage.

        Inter-stage tensors are plain ``[B, S, H]`` (no HC stream). With MTP
        enabled, every transfer carries ``1 + D`` tensors so the variadic
        forward signature is exercised on every microbatch.
        """
        hidden_shape = (microbatch_size, seq_len, self.config.hidden_size)
        mtp_depth = int(getattr(self.mtp_config, "num_layers", 0) or 0)

        def meta(shape: tuple[int, ...], d: torch.dtype = dtype) -> torch.Tensor:
            return torch.empty(*shape, device="meta", dtype=d)

        def append_mtp(primary: torch.Tensor) -> tuple[torch.Tensor, ...]:
            if mtp_depth == 0:
                return (primary,)
            return (primary, *(meta(hidden_shape) for _ in range(mtp_depth)))

        if is_first:
            inputs_meta: tuple[torch.Tensor, ...] = (
                torch.empty(microbatch_size, seq_len, device="meta", dtype=torch.long),
            )
        else:
            inputs_meta = append_mtp(meta(hidden_shape))

        if self.lm_head is not None:
            primary_out = meta((microbatch_size, seq_len, self.config.vocab_size))
        else:
            primary_out = meta(hidden_shape)
        outputs_meta = append_mtp(primary_out)
        # Last stage emits an extra [B, S] int32 ``seq_idx`` tail when MTP is
        # enabled. The model's last-stage forward stashes the per-microbatch
        # sub-sequence id tensor here so the loss fn can mask MTP label rolls
        # at sub-sequence boundaries. Using the output tuple (rather than a
        # side-channel queue on the loss fn) keeps the seq_idx bonded to its
        # microbatch by the PP runtime itself — schedule-agnostic.
        if self.lm_head is not None and mtp_depth > 0:
            outputs_meta = (*outputs_meta, meta((microbatch_size, seq_len), d=torch.int32))
        return inputs_meta, outputs_meta

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        *mtp_embed_inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask_mapping: Optional[dict[str, torch.Tensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Forward pass with optional loss computation.

        Supports both BSHD format (``input_ids`` shape ``[B, S]``) and THD format
        (``input_ids`` shape ``[T]`` after ``squeeze_input_for_thd``).  When
        ``kwargs["qkv_format"] == "thd"``, inputs are squeezed to THD before the
        base-model forward and logits are unsqueezed back to ``[1, T, V]`` on exit.

        Pipeline-parallel awareness: when run as a PP stage, ``input_ids`` is
        the upstream stage's hidden-state tensor on non-first stages, and
        ``*mtp_embed_inputs`` carries ``num_nextn_predict_layers`` future-token
        embeddings produced by the first stage. Non-final stages return
        ``(hidden_states, *mtp_embed_inputs)`` (tuple of arity ``1 + D``); the
        final stage returns ``(logits, *mtp_per_depth_h)``. The single-rank
        (no-PP) path returns :class:`NemotronHCausalLMOutputWithPast` unchanged.
        """
        is_pp_stage = self._is_pipeline_parallel_stage()
        is_first_stage = getattr(self.model, "embed_tokens", None) is not None
        has_lm_head = self.lm_head is not None
        mtp_depth = int(getattr(self.mtp_config, "num_layers", 0) or 0)
        pp_mtp_enabled = is_pp_stage and self.mtp_config.enabled

        # Neat-packed SDPA path: ``_packed_seq_ids`` is the indexed [B,S]
        # 1-based per-token sub-sequence id (0 = pad). Convert once here to
        # Mamba's ``seq_idx`` (int32; equality classes are what matter, so
        # the 1-based form works directly) and drop the original key so
        # downstream layers / the inner model see only seq_idx. Both the
        # backbone forward and MTP sublayer chain will consume seq_idx from
        # kwargs / mtp_kwargs.  We also derive a 2D padding_mask from
        # ``_packed_seq_ids`` (1 = real token, 0 = pad) because Mamba's
        # mixer multiplies ``hidden_states * attention_mask.unsqueeze(-1)``
        # which only works for a 2D mask, and the neat collater puts a 4D
        # block-causal bool mask under ``attention_mask`` (for SDPA).
        _packed_seq_ids = kwargs.pop("_packed_seq_ids", None)
        if isinstance(_packed_seq_ids, torch.Tensor) and "seq_idx" not in kwargs:
            kwargs["seq_idx"] = _packed_seq_ids.to(torch.int32).contiguous()
            if padding_mask is None:
                padding_mask = _packed_seq_ids == 0

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )

        # Stash the pre-squeeze ``[B, S]`` int input_ids so the MTP embedding
        # tuple can be built AFTER the inner forward — FSDP2 requires the
        # root module's forward (``self.model``) to run before any of its
        # children's forwards, otherwise the lazy-init check at
        # ``_fsdp_state.py:_lazy_init`` raises. We build the tuple post-
        # inner-forward but use the pre-squeeze shape so the emitted tensors
        # match the ``[B, S, H]`` contract from ``get_pipeline_stage_metas``.
        pre_squeeze_input_ids = (
            input_ids
            if (
                pp_mtp_enabled
                and is_first_stage
                and not has_lm_head
                and not mtp_embed_inputs
                and input_ids is not None
                and input_ids.dtype in (torch.int32, torch.int64, torch.long)
            )
            else None
        )

        # THD format covers both the data layout (qkv_format=='thd') and
        # whether we should squeeze to ``[T, H]`` for the attention backend.
        # Squeezing only helps TE (which natively supports THD); SDPA/flex
        # expect 4D BSHD. Keep ``is_thd`` true whenever qkv_format='thd' so
        # the post-forward unsqueeze still restores the batch dim for the
        # 2D-returning inner model variants — only gate the *squeeze* on
        # the backend.
        _attn_impl = getattr(getattr(self, "backend", None), "attn", None)
        is_thd = kwargs.get("qkv_format") == "thd"
        squeeze_for_thd = is_thd and _attn_impl == "te"
        if squeeze_for_thd and is_first_stage:
            input_ids, position_ids, padding_mask, kwargs = squeeze_input_for_thd(
                input_ids, position_ids, padding_mask, kwargs
            )
            attention_mask = None
            causal_mask_mapping = None

        # MoE module consumes a padding mask derived from attention_mask. On
        # non-first stages where the trainer's collate-built attention_mask
        # arrives as a chunked kwarg, derive it here too.
        if padding_mask is None and attention_mask is not None and attention_mask.dim() == 2:
            padding_mask = attention_mask.bool().logical_not()

        # Forward through base model. On non-first stages, the hidden-state
        # tensor arrives in ``input_ids`` (the PP positional input slot) and
        # the inner model routes it through its ``inputs_embeds`` branch.
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            causal_mask_mapping=causal_mask_mapping,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

        # Now that the root forward has run, FSDP2 has lazy-initialized the
        # root state and is happy with subsequent child-only calls. Build
        # the per-depth MTP token embeddings from the pre-squeeze ``[B, S]``
        # input_ids so the emitted tensors stay ``[B, S, H]`` (matching the
        # `get_pipeline_stage_metas` contract) regardless of THD squeezing.
        if pre_squeeze_input_ids is not None:
            mtp_embed_inputs = self._build_mtp_embed_inputs_for_pp(pre_squeeze_input_ids)

        # Mark cache as having state after the first forward pass (prefill done)
        if past_key_values is not None:
            past_key_values.has_previous_state = True

        # LM head only exists on the last stage. Non-final stages return the
        # backbone's post-norm/no-norm hidden state.
        if has_lm_head:
            if isinstance(logits_to_keep, int) and logits_to_keep == 0:
                logits = self.lm_head(hidden_states)
            else:
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                if hidden_states.dim() == 2:
                    logits = self.lm_head(hidden_states[slice_indices, :])
                else:
                    logits = self.lm_head(hidden_states[:, slice_indices, :])
        else:
            logits = hidden_states

        loss = None
        if labels is not None and has_lm_head and not is_pp_stage:
            # Non-PP path computes its own loss. Under PP, loss is computed
            # by ``PipelineCausalLMLoss`` from the returned tuple.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        # MTP head: lives on the last PP stage (or on rank 0 in non-PP). On
        # non-final / non-owning stages, emit placeholder empties so the
        # inter-stage tuple arity (``1 + D``) is constant across all stages.
        # Final stage in eval mode also emits placeholders (matches DSV4).
        mtp_per_depth_h: list[torch.Tensor] | None = None
        if self.mtp is not None and self.training:
            mtp_attention_mask = (
                causal_mask_mapping.get("full_attention") if causal_mask_mapping is not None else attention_mask
            )
            # Thread the THD-packing context from the outer kwargs (built by
            # the recipe's THD batch processor) into the MTP sublayer call
            # chain. Without this, the MTP attention sublayer would auto-
            # detect ``qkv_format='bshd'`` from the 3D hidden_states shape and
            # treat the packed sequence as one contiguous causal block,
            # bleeding attention across sequence boundaries.
            mtp_kwargs = {
                "position_ids": position_ids,
                "attention_mask": mtp_attention_mask,
            }
            for _k in (
                "qkv_format",
                "cu_seqlens",
                "cu_seqlens_padded",
                "max_seqlen",
                "max_seqlen_q",
                "max_seqlen_kv",
                "seq_idx",
                # CP context: backbone attention receives cp_rank/cp_size via
                # an attention forward pre-hook (set by
                # ``attach_context_parallel_hooks``). MTP sublayer attention
                # reuses the same NemotronV3Attention class but goes through
                # a different call site, so we must forward CP context
                # explicitly. Without these TE's CP-aware flash-varlen path
                # has wrong indices and produces illegal-memory-access.
                "cp_rank",
                "cp_size",
            ):
                if _k in kwargs:
                    _v = kwargs[_k]
                    # TE THD path expects 1D ``[num_seqs+1]`` cu_seqlens. The
                    # THD collator emits stacked ``[num_microbatches, K]``
                    # rows that PP ``tensor_split``s along dim 0, so each
                    # per-microbatch row reaches us as ``[1, K]``. The inner
                    # backbone forward already normalizes for its own use,
                    # but we re-normalize for the MTP call chain.
                    if _k in ("cu_seqlens", "cu_seqlens_padded", "seq_idx") and isinstance(_v, torch.Tensor):
                        if _v.dim() == 2 and _v.shape[0] == 1:
                            _v = _v.squeeze(0)
                        if _v.dim() == 1:
                            # Strip THD-collator sentinels (-1000) before they
                            # reach the MTP sublayer's TE attention. Without
                            # this, the kernel sees pairs like (1024, -1000)
                            # which it interprets as a (-2024)-length sub-seq
                            # and OOB-writes during backward. Matches the
                            # backbone filter in NemotronV3Model.forward.
                            if (_v == -1000).any():
                                _v = _v[_v != -1000]
                            _v = _v.contiguous().clone()
                    mtp_kwargs[_k] = _v
            # Squeeze to 2D ``[T, H]`` when THD-packed so the attention layer
            # selects its THD branch.  Also squeeze the propagated MTP embed
            # tensors for the same reason.
            mtp_hidden = hidden_states
            mtp_embeds_for_call = tuple(mtp_embed_inputs) if mtp_embed_inputs else ()
            if is_thd:
                if mtp_hidden.dim() == 3 and mtp_hidden.shape[0] == 1:
                    mtp_hidden = mtp_hidden.squeeze(0)
                mtp_embeds_for_call = tuple(
                    e.squeeze(0) if (e.dim() == 3 and e.shape[0] == 1) else e for e in mtp_embeds_for_call
                )

            if mtp_embeds_for_call:
                # Final PP stage: embeddings were produced upstream.
                mtp_per_depth_h = self.mtp(
                    hidden_states=mtp_hidden,
                    embed_inputs=mtp_embeds_for_call,
                    **mtp_kwargs,
                )
            else:
                # Non-PP single-rank: roll input_ids locally.
                mtp_per_depth_h = self.mtp(
                    hidden_states=mtp_hidden,
                    input_ids=input_ids,
                    embed_fn=self.model.embed_tokens,
                    **mtp_kwargs,
                )
            # If hidden_states was squeezed for MTP and downstream consumers
            # expect 3D, restore. (Currently no downstream consumer expects
            # it; per-depth outputs are returned as their own list.)
            if is_thd and mtp_per_depth_h is not None:
                mtp_per_depth_h = [h.unsqueeze(0) if h.dim() == 2 else h for h in mtp_per_depth_h]
        elif pp_mtp_enabled and has_lm_head:
            # Final stage in eval, or MTP-enabled config without self.mtp on
            # this rank — keep tuple arity by emitting empties.
            mtp_per_depth_h = [hidden_states.new_empty(hidden_states.shape) for _ in range(mtp_depth)]

        # Restore the batch dim for THD only when the inner forward returned
        # 2D logits.  When the caller feeds the model via ``inputs_embeds``
        # (shape ``[1, T, H]``), ``NemotronHModel.forward`` squeezes to
        # ``[T, H]`` for the layer stack and unsqueezes back to ``[1, T, H]``
        # before returning (see the ``squeezed_for_thd`` branch); the lm_head
        # then yields ``[1, T, V]`` already and a second unsqueeze here would
        # produce a spurious ``[1, 1, T, V]``.
        if is_thd and logits.dim() == 2:
            logits = logits.unsqueeze(0)

        # PP path: return a tuple.
        #
        # First/middle stages emit ``(hidden_states, *mtp_embed_inputs)`` with
        # arity ``1 + D``. The next stage gets its own per-microbatch
        # ``_packed_seq_ids`` via PP-chunked kwargs, so seq_idx isn't passed
        # between mid-stages.
        #
        # The LAST stage emits ``(logits, *mtp_per_depth_h, seq_idx)`` with
        # arity ``1 + D + 1`` when MTP is enabled. The ``seq_idx`` tail binds
        # the per-microbatch sub-sequence layout to its corresponding loss
        # call — the PP runtime guarantees the output of forward(mb_i) is
        # passed to _compute_loss(mb_i), so the loss fn always sees the right
        # seq_idx for the microbatch it's reducing. Works for every PP
        # schedule, not just 1f1b/interleaved1f1b.
        if is_pp_stage:
            if pp_mtp_enabled:
                if not has_lm_head:
                    return (logits, *mtp_embed_inputs)
                assert mtp_per_depth_h is not None
                # Build the seq_idx tail. Sources in order of preference:
                #   1. ``seq_idx`` already in kwargs (neat-path upstream
                #      conversion from ``_packed_seq_ids``).
                #   2. ``cu_seqlens`` in kwargs (THD/TE pack path); derive
                #      per-token sub-seq id via searchsorted (same as the
                #      mamba layer's internal derivation).
                #   3. No packing → all-1 sentinel; the loss-fn's cross-
                #      sub-seq mask becomes a no-op.
                # Derive [B, S] from logits in all cases.
                if logits.dim() == 3:
                    _B, _S = logits.shape[:2]
                elif logits.dim() == 2:
                    _B, _S = 1, logits.shape[0]
                else:
                    _B, _S = 1, hidden_states.shape[-2]

                _seq_idx_tail = kwargs.get("seq_idx", None)
                if not isinstance(_seq_idx_tail, torch.Tensor):
                    _cu = kwargs.get("cu_seqlens", None)
                    if isinstance(_cu, torch.Tensor):
                        _cu1d = _cu.squeeze(0) if (_cu.dim() == 2 and _cu.shape[0] == 1) else _cu
                        if _cu1d.dim() == 1:
                            _positions = torch.arange(_S, device=_cu1d.device)
                            _seq_idx_1d = torch.searchsorted(_cu1d[1:].contiguous(), _positions).to(torch.int32)
                            _seq_idx_tail = _seq_idx_1d.unsqueeze(0).expand(_B, _S).contiguous()
                if not isinstance(_seq_idx_tail, torch.Tensor):
                    _seq_idx_tail = torch.ones((_B, _S), dtype=torch.int32, device=logits.device)
                else:
                    if _seq_idx_tail.dim() == 1:
                        _seq_idx_tail = _seq_idx_tail.unsqueeze(0)
                    if _seq_idx_tail.dtype != torch.int32:
                        _seq_idx_tail = _seq_idx_tail.to(torch.int32)
                return (logits, *mtp_per_depth_h, _seq_idx_tail)
            return logits

        # Non-PP path: dataclass return for compatibility with existing
        # consumers and the recipe's MTP loss reader.
        return NemotronHCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=(hidden_states,) if output_hidden_states else None,
            attentions=None,
            mtp_per_depth_h=mtp_per_depth_h,
            mtp_loss_scaling_factor=(self.mtp_config.loss_scaling_factor if mtp_per_depth_h is not None else None),
        )

    @staticmethod
    def _make_causal_mask(
        query_len: int,
        kv_len: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a 4D SDPA-compatible causal mask.

        Prefill (query_len == kv_len): standard lower-triangular causal mask.
        Decode (query_len == 1): all-zeros row allowing attention to all cached positions.
        """
        if query_len == 1:
            # Decode: attend to all positions
            return torch.zeros(batch_size, 1, 1, kv_len, dtype=dtype, device=device)
        # Prefill: lower-triangular causal mask
        mask = torch.zeros(batch_size, 1, query_len, kv_len, dtype=dtype, device=device)
        mask.masked_fill_(
            torch.triu(torch.ones(query_len, kv_len, device=device), diagonal=1).bool(),
            float("-inf"),
        )
        return mask

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        **kwargs,
    ) -> dict:
        """Prepare model inputs for each generation step.

        On the first call (prefill), creates a :class:`NemotronHybridCache` and
        forwards the full prompt.  On subsequent calls (decode), only the newly
        generated token is forwarded.

        Args:
            input_ids: Accumulated token ids [batch_size, current_seq_len].
            attention_mask: Padding mask [batch_size, current_seq_len].
            inputs_embeds: Pre-computed embeddings for the first step (optional).
            past_key_values: NemotronHybridCache from the previous step (None on first call).
            cache_position: Token position indices.
            use_cache: Whether to use caching (default True).
            **kwargs: Remaining model kwargs.

        Returns:
            Dict of keyword arguments to pass to :meth:`forward`.
        """
        from nemo_automodel.components.models.nemotron_v3.cache import NemotronHybridCache

        batch_size = input_ids.shape[0]

        # Create cache on first call, or replace non-NemotronHybridCache
        # (transformers v5.5+ GenerationMixin may pre-create a DynamicCache)
        if past_key_values is None or not isinstance(past_key_values, NemotronHybridCache):
            past_key_values = NemotronHybridCache(self.config, batch_size, self.dtype, self.device)
            # First call: cache_position covers the full prompt
            if cache_position is None:
                prompt_len = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
                cache_position = torch.arange(prompt_len, device=input_ids.device)

        # After prefill, send only the new token
        if past_key_values.has_previous_state:
            input_ids = input_ids[:, -1:]
            if cache_position is None:
                kv_len = past_key_values.get_seq_length()
                cache_position = torch.tensor([kv_len], device=input_ids.device)
            elif cache_position.ndim == 1 and cache_position.numel() > 1:
                # GenerationMixin may forward the full prompt positions on decode
                # even though only the last token is being decoded. Nemotron-v3's
                # Mamba cache update expects a single decode position here.
                cache_position = cache_position[-1:]

        # On the first step, prefer inputs_embeds when available
        if inputs_embeds is not None and not past_key_values.has_previous_state:
            model_inputs = {"input_ids": None, "inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        # Build causal mask for attention layers
        query_len = (
            input_ids.shape[1] if model_inputs["inputs_embeds"] is None else model_inputs["inputs_embeds"].shape[1]
        )
        kv_len = past_key_values.get_seq_length() + query_len
        causal_mask = self._make_causal_mask(query_len, kv_len, batch_size, self.dtype, self.device)

        model_inputs["causal_mask_mapping"] = {"full_attention": causal_mask}
        model_inputs["past_key_values"] = past_key_values
        model_inputs["cache_position"] = cache_position
        model_inputs["use_cache"] = use_cache
        model_inputs["attention_mask"] = attention_mask
        return model_inputs

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize model weights.

        PP-aware: skips ``lm_head`` and ``mtp`` initialization when those have
        been trimmed to ``None`` on a non-owning stage. ``self.model`` itself
        also internally guards ``embed_tokens`` and ``norm``.

        Args:
            buffer_device: Device to use for buffer initialization
            dtype: Target dtype for model weights
        """
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            self.model.initialize_weights(buffer_device=buffer_device)
            if self.lm_head is not None:
                nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.initializer_range)
            if self.mtp is not None:
                for sublayer in self.mtp.layers:
                    sublayer.init_weights(buffer_device=buffer_device)

        cast_model_to_dtype(self, dtype)


ModelClass = NemotronHForCausalLM
