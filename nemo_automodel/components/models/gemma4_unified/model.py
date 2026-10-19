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

"""NeMo wrapper for the Hugging Face Gemma4 Unified model."""

from dataclasses import dataclass
from typing import Any

import torch
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedCausalLMOutputWithPast,
    Gemma4UnifiedTextAttention,
)
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedForConditionalGeneration as HFGemma4UnifiedForConditionalGeneration,
)

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    contiguous_local_indices,
    shard_sequence_for_cp_contiguous,
)
from nemo_automodel.components.models.common import compute_lm_head_logits
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.gemma4_moe.cp_attention import (
    attach_gemma4_cp_ring_attention,
    gemma4_vision_group_ids,
)
from nemo_automodel.components.models.gemma4_moe.cp_batch import make_contiguous_aux_only_shard_cp_batch_and_ctx
from nemo_automodel.components.models.gemma4_unified.state_dict_adapter import (
    Gemma4UnifiedStateDictAdapter,
)


class Gemma4UnifiedForConditionalGeneration(HFCheckpointingMixin, HFGemma4UnifiedForConditionalGeneration):
    """Gemma4 Unified with NeMo checkpoint key conversion and model-owned context parallelism.

    The text stack has the same attention layout as dense Gemma4 (sliding-window and full layers,
    vision-bidirectional sliding layers), so context parallelism reuses Gemma4's contiguous
    per-rank shards and its p2p ring attention. Without CP the forward is Hugging Face's.
    """

    tie_word_embeddings_support: TieSupport = TieSupport.TIED_ONLY
    # Owns CP batch sharding and the per-layer ring attention transport.
    _owns_cp_attention = True
    # CP submesh, recorded by _cp_shard_batch_aux_only when the recipe first shards a batch;
    # None means the forward runs without CP.
    cp_mesh = None

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Context parallelism is model-owned; no TP/PP/EP implementations are provided."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = False
        supports_ep: bool = False

    def __init__(self, config) -> None:
        reject_unsupported_tie_word_embeddings(type(self), config)
        super().__init__(config)
        self.state_dict_adapter = Gemma4UnifiedStateDictAdapter()
        for module in self.modules():
            if isinstance(module, Gemma4UnifiedTextAttention):
                attach_gemma4_cp_ring_attention(module)

    def tie_weights(self, *_args: object, **_kwargs: object) -> None:
        """Tie the output head to Gemma4 Unified's text input embedding."""
        self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def setup_cp_attention(self, cp_mesh) -> None:
        """Install the ring attention on every text self-attention module (idempotent).

        Args:
            cp_mesh: The context-parallel device submesh.

        Raises:
            ValueError: If the text model does not use SDPA attention, which the ring replaces.
        """
        if getattr(self, "_cp_enabled", False):
            return
        attn_implementation = self.config.get_text_config()._attn_implementation
        if attn_implementation != "sdpa":
            raise ValueError(
                "Gemma4 Unified context parallelism runs its ring attention in place of SDPA; load the model "
                f"with attn_implementation='sdpa' (got {attn_implementation!r})."
            )
        self._cp_enabled = True
        for module in self.modules():
            if module is not self and callable(getattr(module, "setup_cp_attention", None)):
                module.setup_cp_attention(cp_mesh)

    def _cp_shard_batch_aux_only(self, cp_mesh, tp_mesh, batch, *, loss_mask=None, padding_token_id=0):
        """Shard the no-grad batch streams contiguously and record ``cp_mesh`` for the forward.

        ``labels``, ``position_ids``, ``loss_mask``, ``padding_mask`` and the synthesized
        ``_packed_seq_ids`` document map are sharded here; ``input_ids`` and ``pixel_values``
        stay full-length, since the forward embeds and splices images on the full sequence.
        """
        self.setup_cp_attention(cp_mesh)
        self.cp_mesh = cp_mesh
        return make_contiguous_aux_only_shard_cp_batch_and_ctx(
            cp_mesh,
            tp_mesh,
            batch,
            loss_mask=loss_mask,
            padding_token_id=padding_token_id,
        )

    def prepare_model_inputs_for_cp(self, batch: dict[str, Any], *, num_chunks: int = 1) -> dict[str, Any]:
        """Return the model-owned contiguous CP sharder; the batch itself is left intact.

        Args:
            batch: The full-sequence batch.
            num_chunks: Accepted for hook-signature parity; contiguous CP has one chunk.

        Returns:
            ``{"cp_sharder": ContextParallelSharder}`` for the CP dispatch.

        Raises:
            ValueError: If the batch has no ``input_ids``.
        """
        del num_chunks
        if batch.get("input_ids") is None:
            raise ValueError("prepare_model_inputs_for_cp requires input_ids.")
        return {
            "cp_sharder": ContextParallelSharder(
                shard_batch=self._cp_shard_batch_aux_only,
                local_token_global_indices=contiguous_local_indices,
            )
        }

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
        video_position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        mm_token_type_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Gemma4UnifiedCausalLMOutputWithPast:
        """Hugging Face's forward, or the context-parallel forward once CP is set up.

        Image blocks marked by ``mm_token_type_ids`` attend bidirectionally on sliding-window layers
        only, as in Hugging Face's generation masks for this model and in dense Gemma4. Hugging
        Face's own training forward also applies that overlay to the full-attention layers, so
        without CP the forward passes it the generation masks; CP and non-CP training then match
        inference.

        Under CP the forward embeds the full ``input_ids`` and splices image features as Hugging
        Face does, builds the vision-bidirectional block ids on the full sequence, then keeps this
        rank's contiguous slice; the aux-only sharder already sliced ``position_ids``,
        ``padding_mask`` and ``_packed_seq_ids`` to the same layout, and every text self-attention
        module runs the ring.

        Args:
            input_ids: Token ids ``[batch, sequence]``; full-length under CP.
            pixel_values: Raw image patches for the vision embedder.
            pixel_values_videos: Video patches; not supported under CP.
            input_features: Audio frames; not supported under CP.
            attention_mask: Attention mask; under CP the ring builds its own masks.
            input_features_mask: Audio frame mask.
            position_ids: Position ids; under CP this rank's slice ``[batch, local_sequence]``.
            image_position_ids: Patch positions ``[images, patches, 2]`` (``-1`` marks padding).
            video_position_ids: Video patch positions.
            past_key_values: KV cache; unused under CP.
            mm_token_type_ids: Modality ids ``[batch, sequence]``; image blocks attend bidirectionally on
                sliding-window layers when set.
            inputs_embeds: Input embeddings; not supported under CP.
            labels: Labels for the Hugging Face loss; under CP the recipe computes the loss instead.
            use_cache: Whether to return a KV cache; off under CP.
            logits_to_keep: Integer suffix length or index tensor selecting the returned positions.
            mm_encoder_outputs: Precomputed multimodal encoder outputs; not supported under CP.
            **kwargs: Remaining keyword arguments. Under CP, ``padding_mask`` (True = pad) and
                ``_packed_seq_ids`` (0 = pad) carry this rank's slices to the ring.

        Returns:
            Model outputs; under CP, logits for this rank's sequence shard ``[batch, local_sequence, vocab]``.

        Raises:
            ValueError: Under CP, for inputs the context-parallel forward does not support.
        """
        if not getattr(self, "_cp_enabled", False):
            text_config = self.config.get_text_config()
            if (
                mm_token_type_ids is not None
                and text_config.use_bidirectional_attention == "vision"
                and not isinstance(attention_mask, dict)
                and past_key_values is None
            ):
                # The mask builder reads only the batch/sequence shape, dtype and device.
                shape_like = input_ids if inputs_embeds is None else inputs_embeds[..., 0]
                attention_mask = self.create_masks_for_generate(
                    self.config,
                    shape_like.new_empty((*shape_like.shape[:2], 0), dtype=self.dtype),
                    attention_mask,
                    None,
                    position_ids,
                    mm_token_type_ids=mm_token_type_ids,
                )
            return super().forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                input_features=input_features,
                attention_mask=attention_mask,
                input_features_mask=input_features_mask,
                position_ids=position_ids,
                image_position_ids=image_position_ids,
                video_position_ids=video_position_ids,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                mm_encoder_outputs=mm_encoder_outputs,
                **kwargs,
            )
        if input_ids is None or inputs_embeds is not None or mm_encoder_outputs is not None:
            raise ValueError(
                "Gemma4 Unified context parallelism requires input_ids, without inputs_embeds or mm_encoder_outputs."
            )
        if pixel_values_videos is not None or input_features is not None:
            raise ValueError("Gemma4 Unified context parallelism supports text and image inputs, not video or audio.")
        padding_mask = kwargs.pop("padding_mask", None)
        packed_seq_ids = kwargs.pop("_packed_seq_ids", None)

        image_mask, video_mask, audio_mask = self.model.get_placeholder_mask(input_ids)
        text_config = self.config.get_text_config()
        llm_input_ids = input_ids.masked_fill(image_mask | video_mask | audio_mask, text_config.pad_token_id)
        inputs_embeds = self.model.get_input_embeddings()(llm_input_ids)
        if pixel_values is not None:
            image_features = self.model.get_image_features(pixel_values, image_position_ids, return_dict=True)
            image_features = torch.cat(image_features.pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(inputs_embeds), image_features
            )

        cp_mesh = self.cp_mesh
        has_vision_tokens = mm_token_type_ids is not None and text_config.use_bidirectional_attention == "vision"
        mm_local = vision_group_local = None
        if has_vision_tokens:
            mm_local = shard_sequence_for_cp_contiguous(cp_mesh, mm_token_type_ids, seq_dim=1, pad_value=0)[0]
            vision_group_local = shard_sequence_for_cp_contiguous(
                cp_mesh, gemma4_vision_group_ids(mm_token_type_ids), seq_dim=1, pad_value=-1
            )[0]
        inputs_embeds = shard_sequence_for_cp_contiguous(cp_mesh, inputs_embeds, seq_dim=1)[0]

        # Hugging Face's decoder layers do not hand this metadata to self_attn, so it is stashed on
        # each ring-hooked module; it is overwritten every forward and kept for the activation
        # checkpoint recompute in backward.
        cp_meta = {
            "mm_token_type_ids": mm_local,
            "padding_mask": padding_mask,
            "_packed_seq_ids": packed_seq_ids,
            "_gemma4_vision_group_ids": vision_group_local,
            "_gemma4_has_vision_tokens": has_vision_tokens,
        }
        for module in self.modules():
            if getattr(module, "_cp_uses_attention_hook", False):
                module._cp_dense_metadata = cp_meta

        text_outputs = self.model.language_model(
            inputs_embeds=inputs_embeds, position_ids=position_ids, use_cache=False, **kwargs
        )
        logits = compute_lm_head_logits(self.lm_head, text_outputs.last_hidden_state, logits_to_keep).logits
        if (final_logit_softcapping := text_config.final_logit_softcapping) is not None:
            logits = torch.tanh(logits / final_logit_softcapping) * final_logit_softcapping
        return Gemma4UnifiedCausalLMOutputWithPast(logits=logits, hidden_states=text_outputs.hidden_states)
