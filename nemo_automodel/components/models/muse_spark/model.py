# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native AutoModel implementation of the Muse Spark Open Omni model."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    ShardLayout,
    round_robin_local_indices,
    shard_batch_aux_only,
    shard_sequence_for_cp_round_robin,
)
from nemo_automodel.components.distributed.context_parallel.utils import cp_dispatcher_suspended
from nemo_automodel.components.models.common import BackendConfig, compute_lm_head_logits, initialize_linear_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.muse_spark.audio import MuseSparkAudioEncoder
from nemo_automodel.components.models.muse_spark.config import MuseSpark1xOmniConfig
from nemo_automodel.components.models.muse_spark.configuration_muse_spark import MuseSparkConfig
from nemo_automodel.components.models.muse_spark.hf_state_dict_adapter import MuseSparkHFStateDictAdapter
from nemo_automodel.components.models.muse_spark.layers import (
    MuseSparkDecoderLayer,
    MuseSparkRMSNorm,
    MuseSparkRotaryEmbedding,
    build_muse_spark_moe_config,
    calculate_gain_center,
    next_dtype,
)
from nemo_automodel.components.models.muse_spark.state_dict_adapter import MuseSparkStateDictAdapter
from nemo_automodel.components.models.muse_spark.vision import MuseSparkVisionModel
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.shared.import_utils import safe_import


def _residual_dtype(config: MuseSpark1xOmniConfig) -> torch.dtype:
    """Resolve the checkpoint residual-stream dtype."""
    return {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[config.residual_dtype]


def _select_cp_positions(
    position_ids: torch.Tensor | None,
    *,
    full_seq_len: int,
    local_indices: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Select the local round-robin positions from a BSHD position stream.

    Args:
        position_ids: Full ``[B,S]`` or already-local ``[B,S/cp]`` positions.
        full_seq_len: Unpadded global sequence length.
        local_indices: Global positions retained by this CP rank, shape
            ``[padded_sequence/cp]``.
        batch_size: BSHD batch size.
        device: Device on which default positions are constructed.

    Returns:
        Local position ids with shape ``[B,padded_sequence/cp]``.
    """
    if position_ids is None:
        positions = torch.arange(full_seq_len, device=device, dtype=torch.long)
        position_ids = positions.unsqueeze(0).expand(batch_size, -1)
    sequence_axis = 2 if position_ids.ndim == 3 else 1
    if position_ids.shape[sequence_axis] == local_indices.numel():
        return position_ids
    if position_ids.shape[sequence_axis] != full_seq_len:
        raise ValueError(
            "Muse Spark Open CP position_ids must have global or local sequence length; "
            f"got {tuple(position_ids.shape)}, global={full_seq_len}, local={local_indices.numel()}"
        )
    return position_ids.index_select(sequence_axis, local_indices.to(position_ids.device)).contiguous()


def _causal_mask(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    window: int | None,
) -> torch.Tensor:
    """Build an SDPA allow-mask for BSHD hidden states.

    Args:
        hidden_states: Input activations shaped ``[B,S,H]``.
        attention_mask: Optional padding mask ``[B,S]`` or ready SDPA mask
            broadcastable to ``[B,heads,S,S]``.
        window: Number of causal key positions visible to each query, or None
            for full causal attention.

    Returns:
        Boolean SDPA mask shaped ``[B,1,S,S]`` (or the caller-provided
        higher-rank mask).
    """
    if attention_mask is not None and attention_mask.ndim >= 3:
        return attention_mask
    batch, sequence = hidden_states.shape[:2]
    allowed = torch.ones((sequence, sequence), dtype=torch.bool, device=hidden_states.device).tril()
    if window is not None:
        allowed = allowed.triu(diagonal=-(window - 1))
    allowed = allowed.view(1, 1, sequence, sequence).expand(batch, -1, -1, -1)
    if attention_mask is not None:
        key_is_valid = attention_mask.to(device=hidden_states.device, dtype=torch.bool).view(batch, 1, 1, sequence)
        allowed = allowed & key_is_valid
    return allowed


class MuseSparkTextModel(nn.Module):
    """Muse Spark Open decoder backbone with an FP32 residual stream."""

    def __init__(
        self,
        config: MuseSpark1xOmniConfig,
        backend: BackendConfig,
        moe_config: MoEConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.backend = backend
        self.moe_config = moe_config or build_muse_spark_moe_config(config, next_dtype(config))
        dtype = next_dtype(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.embed_norm = (
            MuseSparkRMSNorm(
                config.hidden_size,
                config.rms_norm_eps,
                dtype=dtype,
                has_weight=False,
                compile_norm=backend.compile_norm,
            )
            if config.normalize_tok_embeddings
            else None
        )
        self.m_emb = float(config.metap.get("m_emb", 1.0)) if config.metap.get("use_metap", False) else 1.0
        self.rotary_emb = MuseSparkRotaryEmbedding(config.head_dim, config.rope_theta)
        self.layers = nn.ModuleList(
            [
                MuseSparkDecoderLayer(config, layer_idx, self.moe_config, backend)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = MuseSparkRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            compile_norm=backend.compile_norm,
            has_weight=config.norm_affine,
            zero_centered_gamma=config.norm_zero_centered_gamma,
            gain_center=calculate_gain_center(config.output_norm_gain_center_type, -1),
        )

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed ``[B,S]`` or packed ``[T]`` ids and apply the parameterless norm."""
        hidden_states = self.embed_tokens(input_ids)
        if self.embed_norm is not None:
            hidden_states = self.embed_norm(hidden_states)
        return hidden_states * self.m_emb

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        """Run BSHD or packed THD decoder activations.

        Args:
            input_ids: Local token ids [batch, sequence] or packed [tokens].
            inputs_embeds: Local embeddings [batch, sequence, hidden] or [tokens, hidden].
            position_ids: Local positions [batch, sequence] or packed [tokens].
            attention_mask: Token validity [batch, sequence] (True/1 means valid),
                or a prepared attention mask broadcastable to [batch, heads, sequence, sequence].
            padding_mask: Local token padding [batch, sequence] or [tokens], with
                True meaning padding. Takes precedence over attention_mask.
            output_hidden_states: Whether to return each layer's residual states.
            **kwargs: Attention backend metadata, including packed sequence boundaries.

        Returns:
            Decoder output with last_hidden_state [batch, sequence, hidden] or
            [tokens, hidden], and optionally a tuple of residuals in the same layout.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        is_thd = kwargs.get("qkv_format") == "thd"
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        expected_rank = 2 if is_thd else 3
        if inputs_embeds.ndim != expected_rank:
            raise ValueError(
                f"Muse Spark Open {'THD' if is_thd else 'BSHD'} inputs must have rank {expected_rank}, "
                f"got {tuple(inputs_embeds.shape)}"
            )
        sequence = inputs_embeds.shape[-2]
        if position_ids is None:
            position_ids = torch.arange(sequence, device=inputs_embeds.device, dtype=torch.long)
            if not is_thd:
                position_ids = position_ids.unsqueeze(0).expand(inputs_embeds.shape[0], -1)
        if is_thd and position_ids.ndim > 1:
            position_ids = position_ids.squeeze(0)

        # The attention backend consumes token validity separately from MoE.
        # Derive this before converting a 2-D mask to a causal attention mask;
        # higher-rank attention masks and packed metadata are not token masks.
        if not is_thd and padding_mask is None and attention_mask is not None and attention_mask.ndim == 2:
            padding_mask = ~attention_mask.bool()

        stream = inputs_embeds.to(_residual_dtype(self.config))
        branch_states = inputs_embeds.to(next_dtype(self.config))
        position_embeddings = self.rotary_emb(branch_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        full_mask = sliding_mask = None
        if not is_thd and self.backend.attn != "te":
            full_mask = _causal_mask(branch_states, attention_mask, None)
            if self.config.sliding_window:
                sliding_mask = _causal_mask(branch_states, attention_mask, self.config.sliding_window)

        layer_iter = self.layers.values() if isinstance(self.layers, nn.ModuleDict) else self.layers
        for layer in layer_iter:
            if output_hidden_states:
                all_hidden_states += (stream,)
            layer_mask = None
            if not is_thd:
                if self.backend.attn == "te":
                    layer_mask = attention_mask
                else:
                    layer_mask = sliding_mask if layer.layer_type == "sliding_attention" else full_mask
            stream = layer(
                stream,
                position_embeddings,
                layer_mask,
                padding_mask=padding_mask,
                **kwargs,
            )

        if self.norm is not None:
            norm_input = stream if self.config.use_fp32_for_output else stream.to(next_dtype(self.config))
            hidden_states = self.norm(norm_input)
        else:
            # A non-final pipeline stage must preserve the checkpoint's FP32
            # residual stream across the P2P boundary.  The next stage casts
            # only each attention/FFN branch to the model compute dtype.
            hidden_states = stream
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class MuseSpark1xOmniForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Native Muse Spark Open text, image/video, and waveform-audio model."""

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    # The outer forward owns multimodal embedding/splicing and the checkpoint's
    # FP32 residual stream, so the generic Hugging Face PP patch must not replace
    # it during stage construction.
    _pp_keep_self_forward: bool = True
    _pp_return_hidden_states_supported: bool = True
    _keep_in_fp32_modules = ["rotary_emb"]
    _keep_in_fp32_modules_strict = [
        "feed_forward.gate.weight",
        "feed_forward.gate.e_score_correction_bias",
    ]
    _packed_cp_attn_backends = ("te", "magi")
    supports_thd = True

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Parallel and packed-layout capabilities of the native model."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True
        supports_thd: bool = True

    @classmethod
    def from_config(
        cls,
        config: MuseSpark1xOmniConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ) -> "MuseSpark1xOmniForCausalLM":
        """Construct the native model from a resolved checkpoint config."""
        del kwargs
        return cls(config, moe_config=moe_config, backend=backend)

    def __init__(
        self,
        config: MuseSpark1xOmniConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
    ) -> None:
        super().__init__()
        reject_unsupported_tie_word_embeddings(type(self), config)
        self.config = config
        self.backend = backend or BackendConfig()
        self.model = MuseSparkTextModel(config, self.backend, moe_config)
        dtype = next_dtype(config)
        self.vision_model = MuseSparkVisionModel(config)
        self.vision_projection = initialize_linear_module(
            self.backend.linear, self.vision_model.output_size, config.hidden_size, bias=False, dtype=dtype
        )
        audio = config.audio_perception_modality_args
        self.audio_encoder = (
            MuseSparkAudioEncoder(
                config.hidden_size,
                dtype,
                patch_size=int(audio.get("patch_size", 1920)),
                kernel_size=int(audio.get("kernel_size", 7)),
                patchify_sizes=audio.get("patchify_sizes", (16, 6, 5, 4)),
                hidden_sizes=audio.get("hidden_sizes", (256, 512, 1024, 2048)),
                layers_per_stage=audio.get("layers_per_stage", (2, 2, 2, 2)),
            )
            if config.use_audio_perception
            else None
        )
        self.lm_head = initialize_linear_module(
            self.backend.linear, config.hidden_size, config.vocab_size, bias=False, dtype=dtype
        )
        self.cp_mesh = None
        if self.backend.enable_hf_state_dict_adapter:
            adapter_cls = (
                MuseSparkHFStateDictAdapter if isinstance(config, MuseSparkConfig) else MuseSparkStateDictAdapter
            )
            self.state_dict_adapter = adapter_cls(
                config,
                self.model.moe_config,
                self.backend,
                dtype=dtype,
            )

    def get_input_embeddings(self) -> nn.Module:
        """Return the text token embedding module."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Replace the text token embedding module."""
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        """Return the untied language-model head."""
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        """Replace the untied language-model head."""
        self.lm_head = value

    def customize_pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        *,
        layers_prefix: str,
        text_model: nn.Module | None = None,
    ) -> list[list[str]]:
        """Map generated PP FQNs to Muse Spark Open's real multimodal layout.

        Args:
            module_names_per_stage: Auto-generated module FQNs for each stage.
            layers_prefix: Prefix used for the text decoder, normally ``model.``.
            text_model: Resolved text module supplied by the PP splitter; unused.

        Returns:
            Stage module FQNs with the top-level vision/audio towers and
            projector on stage zero, plus the token embedding norm there.
        """
        del text_model
        from nemo_automodel.components.distributed.pipelining.hf_utils import (  # noqa: PLC0415
            MULTIMODAL_SUFFIXES,
        )

        fixed: list[list[str]] = []
        text_prefix = "model."
        for stage_idx, stage in enumerate(module_names_per_stage):
            names: list[str] = []
            for name in stage:
                if layers_prefix != text_prefix and name.startswith(layers_prefix):
                    name = text_prefix + name[len(layers_prefix) :]
                if name.startswith(text_prefix) and name[len(text_prefix) :] in MULTIMODAL_SUFFIXES:
                    name = name[len(text_prefix) :]
                names.append(name)
            if stage_idx == 0:
                # generate_hf_model_fqn_per_model_part does not know about the
                # token embedding norm or the checkpoint's projection name.
                names.extend(("model.embed_norm", "vision_projection"))
            fixed.append(list(dict.fromkeys(names)))
        return fixed

    def _is_pipeline_parallel_stage(self) -> bool:
        """Return whether this module is a partial pipeline stage."""
        if self.lm_head is None or getattr(self.model, "embed_tokens", None) is None:
            return True
        try:
            return len(self.model.layers) != int(self.config.num_hidden_layers)
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
        """Build exact PP input/output metadata without a probe forward.

        Args:
            is_first: Whether this is the first physical pipeline stage.
            microbatch_size: Number of samples in one pipeline microbatch.
            seq_len: Global token length presented to stage zero, for both
                ordinary BSHD and packed THD batches.
            dtype: Framework-selected compute dtype for final logits or fused
                loss inputs. Intermediate PP activations follow the model's
                residual dtype, independently of the parameter/compute dtype.
                FSDP must preserve these natural model output dtypes.

        Returns:
            A pair ``(inputs_meta, outputs_meta)``.  Token ids have layout
            ``[B,S]``; stage activations are ``[B,S,H]``; final logits are
            ``[B,S,V]``.  Packed THD computation is restored to a leading
            singleton batch dimension at stage boundaries. Stage zero selects
            its CP-local packed tokens before embedding; later stages receive
            already-local activations. No per-batch mode is needed to size them.
        """
        residual_dtype = _residual_dtype(self.config)
        output_dtype = (
            residual_dtype if self.lm_head is None else (torch.float32 if self.config.use_fp32_for_output else dtype)
        )
        cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1
        local_seq_len = seq_len
        if cp_size > 1:
            padded_seq_len = seq_len + (-seq_len) % (2 * cp_size)
            local_seq_len = padded_seq_len // cp_size

        if is_first:
            inputs_meta = (torch.empty(microbatch_size, seq_len, device="meta", dtype=torch.long),)
        else:
            inputs_meta = (
                torch.empty(
                    microbatch_size,
                    local_seq_len,
                    self.config.hidden_size,
                    device="meta",
                    dtype=residual_dtype,
                ),
            )

        if self.lm_head is None or getattr(self, "_pp_return_hidden_states", False) is True:
            outputs_meta = (
                torch.empty(
                    microbatch_size,
                    local_seq_len,
                    self.config.hidden_size,
                    device="meta",
                    dtype=output_dtype,
                ),
            )
        else:
            outputs_meta = (
                torch.empty(
                    microbatch_size,
                    local_seq_len,
                    self.config.vocab_size,
                    device="meta",
                    dtype=output_dtype,
                ),
            )
        return inputs_meta, outputs_meta

    def _normalized_media(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the checkpoint's parameterless embedding norm to ``[tokens,H]`` media."""
        if self.model.embed_norm is not None:
            features = self.model.embed_norm(features)
        return features * self.model.m_emb

    def _vision_features(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Encode flattened image/video patches into normalized ``[tokens,H]`` features."""
        with cp_dispatcher_suspended(self.cp_mesh):
            features = self.vision_model(pixel_values, grid_thw).last_hidden_state
            features = self.vision_projection(features.to(self.vision_projection.weight.dtype))
        return self._normalized_media(features)

    def _audio_features(self, audio_values: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Encode waveform values into normalized ``[tokens,H]`` features."""
        if self.audio_encoder is None:
            raise RuntimeError("Muse Spark Open audio inputs require the configured waveform encoder")
        waveforms = list(audio_values) if isinstance(audio_values, (list, tuple)) else [audio_values]
        parameter = next(self.audio_encoder.parameters())
        waveforms = [torch.as_tensor(value, device=parameter.device, dtype=parameter.dtype) for value in waveforms]
        with cp_dispatcher_suspended(self.cp_mesh):
            features = torch.cat(self.audio_encoder(waveforms), dim=0)
        return self._normalized_media(features)

    @staticmethod
    def _local_media_features(
        features: torch.Tensor,
        local_mask: torch.Tensor,
        global_mask: torch.Tensor | None,
        local_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """Select global media features for one packed CP rank.

        Args:
            features: Global feature stream shaped ``[global_media_tokens,H]``.
            local_mask: Boolean mask over local packed tokens, shape ``[T_local]``.
            global_mask: Boolean mask over the unsharded packed stream, shape
                ``[T_global]``; required together with ``local_indices``.
            local_indices: Global indices for every local token, shape
                ``[T_local]``.

        Returns:
            Feature rows corresponding to true entries in ``local_mask``, in
            local token order, shape ``[local_media_tokens,H]``.
        """
        if global_mask is None and local_indices is None:
            return features
        if global_mask is None or local_indices is None:
            raise ValueError("Packed Muse Spark Open CP media requires both a global mask and local-token indices")
        global_mask = global_mask.reshape(-1).to(device=local_mask.device, dtype=torch.bool)
        local_indices = local_indices.reshape(-1).to(device=local_mask.device, dtype=torch.long)
        if int(global_mask.sum().item()) != features.shape[0]:
            raise ValueError("Muse Spark Open global media placeholders do not match encoded features")
        if local_indices.numel() != local_mask.numel():
            raise ValueError("Muse Spark Open local packed indices do not align with the local token stream")
        feature_index = global_mask.to(torch.long).cumsum(0) - 1
        selected = feature_index.index_select(0, local_indices)[local_mask]
        if selected.numel() and int(selected.min().item()) < 0:
            raise ValueError("Muse Spark Open CP selected a non-media global token as a media feature")
        return features.index_select(0, selected.to(features.device))

    def _splice_features(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        token_id: int,
        features: torch.Tensor,
        *,
        global_mask: torch.Tensor | None,
        local_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """Splice ``[media_tokens,H]`` features into BSHD or THD token embeddings."""
        mask = input_ids == token_id
        selected = self._local_media_features(features, mask.reshape(-1), global_mask, local_indices)
        if int(mask.sum().item()) != selected.shape[0]:
            raise ValueError(
                f"Muse Spark Open produced {selected.shape[0]} features for {int(mask.sum().item())} "
                f"placeholder tokens (token id {token_id})"
            )
        hidden_states[mask] = selected.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return hidden_states

    def _embed_and_splice(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        audio_values: list[torch.Tensor] | torch.Tensor | None,
        local_indices: torch.Tensor | None,
        global_image_mask: torch.Tensor | None,
        global_video_mask: torch.Tensor | None,
        global_audio_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build normalized text embeddings and splice all supplied modalities."""
        hidden_states = self.model.embed(input_ids)
        if pixel_values is None and pixel_values_videos is None and audio_values is None:
            return hidden_states
        hidden_states = hidden_states.clone()
        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("pixel_values requires image_grid_thw")
            hidden_states = self._splice_features(
                hidden_states,
                input_ids,
                self.config.patch_token_id,
                self._vision_features(pixel_values, image_grid_thw),
                global_mask=global_image_mask,
                local_indices=local_indices,
            )
        if pixel_values_videos is not None:
            if video_grid_thw is None:
                raise ValueError("pixel_values_videos requires video_grid_thw")
            hidden_states = self._splice_features(
                hidden_states,
                input_ids,
                self.config.video_token_id,
                self._vision_features(pixel_values_videos, video_grid_thw),
                global_mask=global_video_mask,
                local_indices=local_indices,
            )
        if audio_values is not None:
            hidden_states = self._splice_features(
                hidden_states,
                input_ids,
                self.config.audio_token_id,
                self._audio_features(audio_values),
                global_mask=global_audio_mask,
                local_indices=local_indices,
            )
        return hidden_states

    def _validate_thd_documents(self, batch: dict[str, Any]) -> int:
        """Validate packed document lengths and return the real maximum length."""
        sequence_lengths = batch.get("seq_lens")
        if not isinstance(sequence_lengths, torch.Tensor):
            raise ValueError("Packed Muse Spark Open context parallelism requires per-document seq_lens")
        real_lengths = sequence_lengths[sequence_lengths > 0]
        maximum = int(real_lengths.max().item()) if real_lengths.numel() else 0
        return maximum

    def prepare_model_inputs_for_cp(self, batch: dict[str, Any], *, num_chunks: int = 1) -> dict[str, Any]:
        """Select BSHD round-robin or packed TE THD context-parallel preparation.

        Args:
            batch: Token tensors of shape [batch, sequence], packed real/padded
                lengths of shape [batch, documents], and optional media payloads.
            num_chunks: Number of independently scheduled PP microbatches.

        Returns:
            Model-owned sharder and metadata. Global media masks have shape
            [batch, sequence]; chunk-local token indices have shape
            [chunks, local_tokens] after multi-chunk sharding. PP input ids
            retain [chunks, global_tokens], while labels and positions are local.
            For BSHD, derived padding_mask [batch, sequence] is global here;
            the auxiliary sharder makes it CP-local before PP microbatching.
        """
        if self.backend.attn == "magi":
            from nemo_automodel.components.models.muse_spark.magi_cp import prepare_magi_inputs

            return prepare_magi_inputs(self, batch, num_chunks=num_chunks)
        if batch.get("qkv_format") == "thd":
            maximum = self._validate_thd_documents(batch)
            thd_num_chunks = int(num_chunks)
            pp_stage = self._is_pipeline_parallel_stage()
            prepared: dict[str, Any] = {"_muse_spark_thd_max_real_seqlen": maximum}
            input_ids = batch.get("input_ids")
            if input_ids is None:
                return prepared
            # Pipeline VLM collation moves raw media into pre-chunked storage
            # before CP preparation.  Placeholder ownership still has to be
            # captured from the full token stream so stage zero can select its
            # local image/video rows after the THD partition.
            pp_media = batch.get("_vlm_pp_media_chunks")
            pp_media = pp_media if isinstance(pp_media, dict) else {}
            if batch.get("pixel_values") is not None or pp_media.get("pixel_values") is not None:
                prepared["_muse_spark_global_image_mask"] = input_ids == self.config.patch_token_id
            if batch.get("pixel_values_videos") is not None or pp_media.get("pixel_values_videos") is not None:
                prepared["_muse_spark_global_video_mask"] = input_ids == self.config.video_token_id
            if batch.get("audio_values") is not None:
                prepared["_muse_spark_global_audio_mask"] = input_ids == self.config.audio_token_id

            # Keep media indexing model-owned. The shared sharder exposes a
            # single-stream index map; for PP chunks, reconstruct each map from
            # the padded document boundaries returned with that chunk.
            def shard_muse_spark_thd(
                cp_mesh,
                tp_mesh,
                local_batch,
                *,
                loss_mask=None,
                padding_token_id: int = 0,
            ):
                """Shard packed tokens and retain Muse's media coordinate maps.

                Args:
                    cp_mesh: Context-parallel mesh, or None for unsharded inputs.
                    tp_mesh: Unused tensor-parallel mesh.
                    local_batch: Token tensors of shape [batch, sequence],
                        packed lengths of shape [batch, documents], and media.
                    loss_mask: Unused optional tensor of shape [batch, sequence].
                    padding_token_id: Token used for padding.

                Returns:
                    Context factory, sharded batch, and optional single-stream
                    layout. PP input_ids retain [chunks, global_tokens];
                    targets, positions, and indices have [chunks, local_tokens].
                    Without PP, input_ids and positions are flat [local_tokens],
                    while targets retain [1, local_tokens] to match the head.
                    Each index is relative
                    to the full padded token stream of its own chunk.
                """
                del tp_mesh, loss_mask
                from nemo_automodel.components.distributed.context_parallel.utils import (  # noqa: PLC0415
                    make_cp_batch_for_te,
                )

                source_ids = local_batch.get("input_ids")
                row_shape = tuple(source_ids.shape[:2]) if source_ids is not None and source_ids.ndim >= 2 else None
                sharded, local_indices = make_cp_batch_for_te(
                    cp_mesh,
                    local_batch,
                    padding_token_id=padding_token_id,
                    qkv_format="thd",
                    num_chunks=thd_num_chunks,
                    seq_lens_padding_value=-1000,
                    return_local_indices=True,
                )
                # The head restores [batch, tokens, hidden] for fused losses.
                # Match its batch axis on targets without changing TE's flat
                # input/position tensors or the local token coordinate map.
                for key in ("labels", "loss_mask"):
                    value = sharded.get(key)
                    if isinstance(value, torch.Tensor) and value.ndim == 1:
                        sharded[key] = value.unsqueeze(0)
                if thd_num_chunks > 1:
                    cp_size = cp_mesh.size() if cp_mesh is not None else 1
                    local_tokens = sharded["input_ids"].shape[-1]
                    if cp_size > 1:
                        has_tex, tex = safe_import("transformer_engine_torch")
                        if not has_tex:
                            raise ImportError("Muse Spark Open packed CP requires transformer_engine_torch")
                        # The shared TE sharder returns padded document boundaries
                        # in cu_seqlens. Remove only the per-row metadata sentinel;
                        # using real lengths would misalign media after padding.
                        local_indices = torch.stack(
                            [
                                tex.thd_get_partitioned_indices(
                                    cu[cu != -1000],
                                    local_tokens * cp_size,
                                    cp_size,
                                    sharded["cp_rank"],
                                )
                                for cu in sharded["cu_seqlens"]
                            ]
                        )
                    else:
                        local_indices = torch.arange(
                            local_tokens, device=sharded["input_ids"].device, dtype=torch.long
                        ).expand(thd_num_chunks, -1)
                if pp_stage:
                    # Match ordinary VLM PP: stage zero receives the full token
                    # stream, while labels, positions, and stage activations are
                    # CP-local. THD conversion only flattens existing padded rows,
                    # so the original ids preserve every chunk's coordinate space.
                    # forward() selects local ids before embedding.
                    sharded["input_ids"] = source_ids.reshape(thd_num_chunks, -1).to(torch.long).contiguous()
                if local_indices.ndim == 2:
                    # PyTorch PP schedules the leading chunk dimension as
                    # independent microbatches. Keep the corresponding local
                    # coordinate row and the pre-chunked media payload aligned
                    # on that same axis; a step-wide ShardLayout is
                    # intentionally unavailable because the chunks are
                    # separate token spaces.
                    sharded["_thd_local_indices"] = local_indices
                    for key in (
                        "_muse_spark_thd_max_real_seqlen",
                        "_muse_spark_global_image_mask",
                        "_muse_spark_global_video_mask",
                        "_muse_spark_global_audio_mask",
                    ):
                        if key in local_batch:
                            sharded[key] = local_batch[key]
                    if "_vlm_pp_media_chunks" in local_batch:
                        sharded["_vlm_pp_media_chunks"] = local_batch["_vlm_pp_media_chunks"]
                    return contextlib.nullcontext, sharded, None
                if pp_stage:
                    # Keep a singleton batch axis on single-microbatch local
                    # targets and positions for the PP scheduler. The token ids
                    # already have their full [1, global_tokens] boundary shape.
                    for key in ("input_ids", "labels", "position_ids", "padding_mask", "loss_mask"):
                        value = sharded.get(key)
                        if isinstance(value, torch.Tensor) and value.ndim == 1:
                            sharded[key] = value.unsqueeze(0)
                sharded["_thd_local_indices"] = local_indices
                layout = ShardLayout(
                    local_token_global_indices=local_indices,
                    padded_seq_len=row_shape[0] * row_shape[1] if row_shape is not None else None,
                    input_row_shape=row_shape,
                )
                return contextlib.nullcontext, sharded, layout

            prepared["cp_sharder"] = ContextParallelSharder(shard_batch=shard_muse_spark_thd)
            return prepared
        prepared = {
            "cp_sharder": ContextParallelSharder(
                shard_batch=shard_batch_aux_only,
                local_token_global_indices=round_robin_local_indices,
            )
        }
        # The auxiliary sharder removes attention_mask before forward. Preserve
        # padding here so it follows the same CP shard and PP chunk as labels.
        attention_mask = batch.get("attention_mask")
        if batch.get("padding_mask") is None and attention_mask is not None and attention_mask.ndim == 2:
            prepared["padding_mask"] = ~attention_mask.bool()
        return prepared

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        audio_values: list[torch.Tensor] | torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor, ...] | torch.Tensor:
        """Run causal language modeling for BSHD or packed THD multimodal inputs.

        Args:
            input_ids: Token ids [batch, sequence] or packed [tokens]. Under PP,
                stage zero receives global ids and later stages receive CP-local
                floating-point activations [batch, local_sequence, hidden] here.
            inputs_embeds: Optional embeddings [batch, sequence, hidden] or packed
                [tokens, hidden]; already local when received from a PP stage.
            position_ids: Local positions [batch, local_sequence] or [local_tokens].
            attention_mask: Optional attention mask broadcastable to [batch, heads,
                sequence, sequence].
            labels: Optional local targets [batch, local_sequence] or [local_tokens].
            pixel_values: Image patches [patches, patch_features].
            image_grid_thw: Image patch grids [images, 3] in temporal/height/width order.
            pixel_values_videos: Video patches [patches, patch_features].
            video_grid_thw: Video patch grids [videos, 3].
            audio_values: Mono waveforms [samples] or [samples, 1], supplied as
                one tensor or a list of variable-length tensors.
            logits_to_keep: Final token count, or indices [kept_tokens]; zero
                selects every position.
            kwargs: Attention/packing metadata and model-owned media coordinates.
                _thd_local_indices [local_tokens] selects from the current
                microbatch's global padded stream; global media masks cover that
                full stream. PP may add a leading singleton microbatch dimension.

        Returns:
            Causal LM output with logits [batch, local_sequence, vocab] and hidden
            states [batch, local_sequence, hidden], plus optional scalar loss.
            Packed outputs restore a singleton batch axis. Intermediate PP
            stages place hidden states in the logits field.
            When the recipe requests PP hidden-state output, the final stage
            instead returns scaled hidden states [batch, local_sequence, hidden]
            without invoking the head. The fused loss must apply the configured
            output soft cap after projection. Scaling hidden states and scaling
            logits are algebraically equivalent but can round differently in
            BF16; the ordinary logits path retains post-projection scaling.
        """
        return_pipeline_hidden_states = getattr(self, "_pp_return_hidden_states", False) is True
        if return_pipeline_hidden_states and labels is not None:
            raise ValueError("PP hidden-state output requires labels to be passed to the pipeline loss")

        # The PP schedule passes the preceding stage's floating-point residual
        # stream in the positional ``input_ids`` slot.  It must bypass token
        # embedding and all modality work on non-first stages.
        received_pipeline_residual = (
            inputs_embeds is None and input_ids is not None and torch.is_floating_point(input_ids)
        )
        if received_pipeline_residual:
            inputs_embeds = input_ids
            input_ids = None

        magi_microbatch = kwargs.pop("_muse_spark_magi_microbatch", None)
        if magi_microbatch is not None:
            kwargs["_muse_spark_magi_plan"] = kwargs["_muse_spark_magi_plan"].select(magi_microbatch)

        is_thd = kwargs.get("qkv_format") == "thd"
        local_indices = kwargs.pop("_thd_local_indices", None)
        if is_thd and input_ids is not None and self._is_pipeline_parallel_stage():
            if local_indices is None:
                raise ValueError("Packed Muse Spark Open PP inputs require local-token indices from CP preparation")
            input_ids = input_ids.reshape(-1).index_select(0, local_indices.reshape(-1).to(torch.long))
        global_image_mask = kwargs.pop("_muse_spark_global_image_mask", None)
        global_video_mask = kwargs.pop("_muse_spark_global_video_mask", None)
        global_audio_mask = kwargs.pop("_muse_spark_global_audio_mask", None)

        # VLM media is removed from the PP tensor stream and staged on the
        # first model part per microbatch.  The cursor indexes *global*
        # microbatches, whereas input_ids is already CP-local here.  Therefore
        # every CP rank must advance over an empty/local-token-free media chunk
        # as well; gating the cursor on local placeholder presence desynchronizes
        # it as soon as image tokens land on only one CP shard.
        image_chunks = getattr(self, "_vlm_pixel_values_chunks", None)
        video_chunks = getattr(self, "_vlm_pixel_values_videos_chunks", None)
        if (
            input_ids is not None
            and not torch.is_floating_point(input_ids)
            and (image_chunks is not None or video_chunks is not None)
        ):
            chunk_idx = getattr(self, "_vlm_chunk_idx", 0) or 0
            consumed = False
            if image_chunks is not None and chunk_idx < len(image_chunks):
                consumed = True
                if pixel_values is None:
                    image_chunk = image_chunks[chunk_idx]
                    if image_chunk.numel():
                        pixel_values = image_chunk
                        image_grid_thw = self._vlm_image_grid_hws_chunks[chunk_idx]
                        if image_grid_thw.shape[-1] == 2:
                            temporal = torch.ones(
                                image_grid_thw.shape[0],
                                1,
                                dtype=image_grid_thw.dtype,
                                device=image_grid_thw.device,
                            )
                            image_grid_thw = torch.cat((temporal, image_grid_thw), dim=-1)
                    elif (global_image_mask is not None and bool(global_image_mask.any().item())) or bool(
                        (input_ids == self.config.patch_token_id).any().item()
                    ):
                        raise ValueError(
                            f"Muse Spark Open PP image chunk {chunk_idx} is empty but its global microbatch "
                            "contains image placeholders"
                        )
            if video_chunks is not None and chunk_idx < len(video_chunks):
                consumed = True
                if pixel_values_videos is None:
                    video_chunk = video_chunks[chunk_idx]
                    if video_chunk.numel():
                        pixel_values_videos = video_chunk
                        video_grid_thw = self._vlm_video_grid_thw_chunks[chunk_idx]
                    elif (global_video_mask is not None and bool(global_video_mask.any().item())) or bool(
                        (input_ids == self.config.video_token_id).any().item()
                    ):
                        raise ValueError(
                            f"Muse Spark Open PP video chunk {chunk_idx} is empty but its global microbatch "
                            "contains video placeholders"
                        )
            if consumed:
                self._vlm_chunk_idx = chunk_idx + 1

        if is_thd:
            if input_ids is not None and input_ids.ndim > 1:
                input_ids = input_ids.squeeze(0)
            if inputs_embeds is not None and inputs_embeds.ndim > 2:
                inputs_embeds = inputs_embeds.squeeze(0)
            if position_ids is not None and position_ids.ndim > 1:
                position_ids = position_ids.squeeze(0)
            if labels is not None and labels.ndim > 1:
                labels = labels.squeeze(0)
            for key, value in tuple(kwargs.items()):
                if not isinstance(value, torch.Tensor):
                    continue
                if key == "max_seqlen":
                    kwargs[key] = int(value.item())
                    continue
                if value.ndim > 1:
                    value = value.squeeze(0)
                if key in ("cu_seqlens", "cu_seqlens_padded"):
                    value = value[value != -1000].contiguous()
                kwargs[key] = value
            attention_mask = None

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Muse Spark Open requires input_ids or inputs_embeds")
            inputs_embeds = self._embed_and_splice(
                input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                audio_values=audio_values,
                local_indices=local_indices,
                global_image_mask=global_image_mask,
                global_video_mask=global_video_mask,
                global_audio_mask=global_audio_mask,
            )

        # Stage zero embeds the full BSHD token stream and shards it once for CP.
        # Every later PP stage receives that already-local residual through the
        # positional ``input_ids`` slot. Sharding that floating-point residual a
        # second time shortens S/cp to S/cp^2 and deadlocks PP against its S/cp
        # communication metadata.
        if not is_thd and not received_pipeline_residual and self.cp_mesh is not None and self.cp_mesh.size() > 1:
            full_sequence = inputs_embeds.shape[1]
            inputs_embeds, cp_indices, _ = shard_sequence_for_cp_round_robin(self.cp_mesh, inputs_embeds, seq_dim=1)
            position_ids = _select_cp_positions(
                position_ids,
                full_seq_len=full_sequence,
                local_indices=cp_indices,
                batch_size=inputs_embeds.shape[0],
                device=inputs_embeds.device,
            )
            attention_mask = None

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=kwargs.pop("padding_mask", None),
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if return_pipeline_hidden_states:
            loss_hidden_states = hidden_states * self.config.output_multiplier
            if is_thd and loss_hidden_states.ndim == 2:
                loss_hidden_states = loss_hidden_states.unsqueeze(0)
            return loss_hidden_states

        projected = compute_lm_head_logits(
            self.lm_head,
            hidden_states,
            logits_to_keep,
            is_thd=is_thd,
            fp32_lm_head=self.config.use_fp32_for_output,
        )
        if self.lm_head is None:
            # No output scaling or soft cap may touch the residual stream
            # passed between non-final PP stages.
            logits = projected.logits
        else:
            logits = projected.logits * self.config.output_multiplier
            if self.config.output_soft_cap_temp is not None:
                cap = float(self.config.output_soft_cap_temp)
                logits = torch.tanh(logits / cap) * cap

        loss = None
        if labels is not None:
            if not (isinstance(logits_to_keep, int) and logits_to_keep == 0):
                raise ValueError("labels require logits_to_keep=0")
            if is_thd and labels.ndim == 1:
                labels = labels.unsqueeze(0)
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, logits.shape[-1]),
                labels[..., 1:].contiguous().view(-1),
                ignore_index=-100,
            )

        loss_hidden_states = hidden_states * self.config.output_multiplier
        if is_thd and loss_hidden_states.ndim == 2:
            loss_hidden_states = loss_hidden_states.unsqueeze(0)
        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=loss_hidden_states,
            attentions=None,
        )
        return result if return_dict else result.to_tuple()

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize every parameter for checkpoint-free construction."""
        buffer_device = buffer_device or next(self.parameters()).device
        linear_types: tuple[type[nn.Module], ...] = (nn.Linear,)
        if self.backend.linear == "te":
            available, te = safe_import("transformer_engine.pytorch")
            if not available:
                raise ImportError("Muse Spark Open linear='te' requires Transformer Engine")
            linear_types += (te.Linear,)
        with buffer_device:
            self.model.rotary_emb.reset_parameters()
            for module in self.modules():
                if isinstance(module, (*linear_types, nn.Embedding)):
                    nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                    if isinstance(module, linear_types) and module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, MuseSparkRMSNorm):
                    module.reset_parameters()
            layer_iter = (
                self.model.layers.values() if isinstance(self.model.layers, nn.ModuleDict) else self.model.layers
            )
            for layer in layer_iter:
                nn.init.zeros_(layer.add_attn.gate)
                nn.init.zeros_(layer.add_ffn.gate)
                layer.feed_forward.experts.init_weights(buffer_device, self.config.initializer_range)
                if hasattr(layer.feed_forward.gate, "weight"):
                    nn.init.normal_(layer.feed_forward.gate.weight, std=self.config.initializer_range)
                correction = getattr(layer.feed_forward.gate, "e_score_correction_bias", None)
                if correction is not None:
                    correction.zero_()
            if self.vision_model is not None:
                self.vision_model.init_weights(buffer_device, self.config.initializer_range)
            if self.audio_encoder is not None:
                self.audio_encoder.init_weights(buffer_device, self.config.initializer_range)
        # Routers own separate FP32 FSDP units. Keep their local backing tensors
        # out of the broad BF16 cast: restoring DTensor.data afterwards changes
        # its advertised dtype without restoring the shard's storage dtype.
        cast_model_to_dtype(self, dtype, skip_modules=("gate",))

    def update_moe_gate_bias(self) -> None:
        """Keep the checkpoint-trained, non-updating router correction bias fixed."""
        return


ModelClass = MuseSpark1xOmniForCausalLM

__all__ = ["MuseSpark1xOmniForCausalLM", "MuseSparkTextModel", "ModelClass"]
