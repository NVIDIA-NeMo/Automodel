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

"""DeepSeek V4.1 text and image backbone for AutoModel training.

Forward contract (reference ``Transformer.forward`` of ``inference/model.py``):

1. Embed tokens and expand the hidden state into ``hc_mult`` residual streams.
2. Run the 40 blocks with single-pass mHC: each block receives the input mix
   produced by the previous block's FFN site (a one-hot mix reads stream 0 at
   the start).  Engram modules write into the streams before their layer.
3. Collapse the streams with the last FFN-site mix, apply the final RMSNorm and
   the fp32 ``lm_head``.

Cross-layer CSA2 state (shared compressed KV, index keys, Top-K indices and the
hierarchical candidate pool) lives in per-layer snapshots of
:class:`~nemo_automodel.components.models.deepseek_v41.attention.DeepseekV41AttentionState`.
Snapshots share tensors and preserve the state needed for activation recomputation.

The optional vision tower inserts projected image patches and learned image
delimiters into the text sequence. Text and image batches use full sequences
with two-dimensional token layouts. DSpark draft layers (``mtp.*``) are built
separately by :mod:`nemo_automodel.components.models.deepseek_v41.dspark` so
their objective cannot backpropagate into this backbone. Inference-time KV
caching and SWA bounded replay remain out of scope.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    contiguous_local_indices,
)
from nemo_automodel.components.distributed.cp_vision_frame_shard import maybe_distribute_visual
from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module, initialize_rms_norm_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import (
    _has_dtensor_params,
    cast_model_to_dtype,
    compute_lm_head_logits,
)
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4VisionGate
from nemo_automodel.components.models.deepseek_v41.attention import (
    DeepseekV41Attention,
    DeepseekV41AttentionState,
    csa2_selection_replay,
    use_triton_rope,
    validate_attention_inputs,
)
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config, DeepseekV41TextConfig
from nemo_automodel.components.models.deepseek_v41.cp import gather_sequence, shard_cp_batch
from nemo_automodel.components.models.deepseek_v41.engram import (
    DeepseekV41Engram,
    DeepseekV41NgramHash,
    compile_engram_gate,
)
from nemo_automodel.components.models.deepseek_v41.fsdp import PARALLELIZER
from nemo_automodel.components.models.deepseek_v41.layers import (
    DeepseekV41HyperConnection,
    DeepseekV41RMSNorm,
    compile_hc_cores,
    compile_norm_core,
)
from nemo_automodel.components.models.deepseek_v41.packing import packed_layout
from nemo_automodel.components.models.deepseek_v41.processing import (
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
    image_inputs_from_batch,
)
from nemo_automodel.components.models.deepseek_v41.quantization import compile_cache_quantization
from nemo_automodel.components.models.deepseek_v41.state_dict_adapter import DeepseekV41StateDictAdapter
from nemo_automodel.components.models.deepseek_v41.vision import (
    DeepseekV41VisionAligner,
    DeepseekV41VisionTransformer,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.shared.utils import dtype_from_str


class DeepseekV41Block(nn.Module):
    """CSA2 and MoE sublayers with the single-pass mHC coefficient handoff."""

    def __init__(
        self,
        config: DeepseekV41TextConfig,
        layer_idx: int,
        backend: BackendConfig,
        moe_config: MoEConfig,
        *,
        engram_process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        dtype = dtype_from_str(config.dtype, torch.bfloat16)
        self.attn = DeepseekV41Attention(config, layer_idx, backend)
        self.ffn = MoE(moe_config, backend)
        # V4.1 uses exactly V4's modality-aware score routing with hash routing
        # disabled. The shared MoE keeps ownership of gate, experts and dispatch.
        # As in DeepSeek-V4, BackendConfig.fake_balanced_gate keeps the
        # FakeBalancedGate that MoE built (benchmark forced balance).
        if not backend.fake_balanced_gate:
            self.ffn.gate = DeepseekV4VisionGate(
                DeepseekV4Config(vocab_size=config.vocab_size),
                moe_config,
                gate_precision=torch.float32,
                hash_routing=False,
            )
        norm = (
            partial(initialize_rms_norm_module, "te", device=self.attn.wq_a.weight.device)
            if backend.rms_norm == "te"
            else DeepseekV41RMSNorm
        )
        self.attn_norm = norm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.ffn_norm = norm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        sinkhorn_backend = "tilelang" if backend.attn in ("tilelang", "cudnn") else "torch"
        self.attn_hc = DeepseekV41HyperConnection(config, sinkhorn_backend=sinkhorn_backend)
        self.ffn_hc = DeepseekV41HyperConnection(config, sinkhorn_backend=sinkhorn_backend)
        self.engram = (
            DeepseekV41Engram(config, layer_idx, backend, process_group=engram_process_group)
            if layer_idx in config.engram_layer_ids
            else None
        )

    @property
    def self_attn(self) -> DeepseekV41Attention:
        """Expose the attention module to the shared CP parallelizer."""
        return self.attn

    @property
    def mlp(self) -> MoE:
        """Expose the shared parallelizer's MoE interface without duplicate registration."""
        return self.ffn

    def nemo_checkpoint_context_fn(
        self,
        context_fn: Callable[[], tuple[AbstractContextManager, AbstractContextManager]] | None,
    ) -> Callable[[], tuple[AbstractContextManager, AbstractContextManager]] | None:
        """Extend this block's activation-checkpoint contexts with CSA2 selection replay.

        The MoE parallelizer calls this model-owned hook when it wraps the block in a checkpoint
        wrapper. With ``reuse_sparse_selection_on_recompute`` the attention layer records the frozen
        indexer's selection and the sparse-index plan during the checkpoint forward and takes them back
        during the recompute instead of recomputing them; with the knob off the incoming contexts are
        returned unchanged.

        Args:
            context_fn: The parallelizer's ``torch.utils.checkpoint`` context factory, or ``None``.

        Returns:
            The (possibly wrapped) context factory.
        """
        if not self.attn.reuse_selection_on_recompute:
            return context_fn
        return csa2_selection_replay.checkpoint_context_fn(context_fn)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pre_mix: torch.Tensor,
        state: DeepseekV41AttentionState,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        engram_hash_ids: torch.Tensor | None = None,
        cp_group: dist.ProcessGroup | None = None,
        packed_seq_ids: torch.Tensor | None = None,
        inputs_validated: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepseekV41AttentionState]:
        """Execute one block while retaining differentiable shared KV ownership.

        Args:
            hidden_states: Tensor of shape [batch, sequence, streams, hidden].
            pre_mix: FP32 tensor of shape [batch, sequence, streams].
            state: Shared CSA2 tensors with layouts documented by
                DeepseekV41AttentionState; no tensor is mutated.
            position_ids: Integer tensor of shape [batch, sequence].
            attention_mask: Optional binary tensor of shape [batch, sequence].
            image_mask: Optional boolean tensor of shape [batch, sequence].
            engram_hash_ids: Optional logical memory rows [batch, local_sequence, hash_heads].
            cp_group: CP group; all token axes are local and CSA2 keys are global.
            packed_seq_ids: Optional document IDs [batch, local_sequence], zero for padding.

        Returns:
            Updated streams [batch, sequence, streams, hidden], the next pre-mix
            [batch, sequence, streams], and the new immutable CSA2 state.
        """
        if pre_mix.dtype != torch.float32:
            raise TypeError(
                "Single-pass mHC requires FP32 carried coefficients. Configure the FSDP mixed precision policy "
                "with cast_forward_inputs=False and output_dtype=None."
            )
        if self.engram is not None:
            if engram_hash_ids is None:
                raise ValueError("An Engram block requires engram_hash_ids")
            token_mask = None if image_mask is None else ~image_mask
            if attention_mask is not None:
                token_mask = attention_mask.bool() if token_mask is None else token_mask & attention_mask.bool()
            hidden_states = self.engram(hidden_states, engram_hash_ids, token_mask=token_mask)
        attn_mix = self.attn_hc(hidden_states)
        collapsed = self.attn_hc.collapse(hidden_states, pre_mix)
        attended = self.attn(
            self.attn_norm(collapsed),
            position_ids=position_ids,
            state=state,
            attention_mask=attention_mask,
            cp_group=cp_group,
            packed_seq_ids=packed_seq_ids,
            inputs_validated=inputs_validated,
        )
        hidden_states = self.attn_hc.expand(attended.hidden_states, hidden_states, attn_mix)
        ffn_mix = self.ffn_hc(hidden_states)
        collapsed = self.ffn_hc.collapse(hidden_states, attn_mix.pre)
        gate = self.ffn.gate
        if isinstance(gate, DeepseekV4VisionGate):
            vision_types = None if image_mask is None else image_mask.to(torch.int32) - 1
            gate.set_routing_context(None, vision_types)
        padding_mask = None if attention_mask is None else ~attention_mask.bool()
        output = self.ffn(self.ffn_norm(collapsed), padding_mask)
        hidden_states = self.ffn_hc.expand(output, hidden_states, ffn_mix)
        return hidden_states, ffn_mix.pre, attended.state


class DeepseekV41Model(nn.Module):
    """DeepSeek V4.1 decoder stack: embeddings, hyper-connected blocks, final norm."""

    def __init__(
        self,
        config: DeepseekV41Config,
        backend: BackendConfig,
        moe_config: MoEConfig,
        *,
        tokenizer: PreTrainedTokenizerFast | None = None,
        engram_process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        self.vision_config = config.vision_config
        self.image_token_id = config.image_token_id
        top_config = config
        config = config.text_config
        self.config = config
        self.moe_config = moe_config
        if backend.compile_hc:
            compile_hc_cores()
        if backend.compile_norm and backend.rms_norm != "te":
            compile_norm_core()
        if config.compile_cache_quant:
            compile_cache_quantization()
        if config.compile_engram:
            compile_engram_gate()
        dtype = dtype_from_str(config.dtype, torch.bfloat16)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        active_engram = any(i < config.num_hidden_layers for i in config.engram_layer_ids)
        if active_engram and tokenizer is None:
            raise ValueError("DeepSeek V4.1 Engram requires its original fast tokenizer for compressed N-gram hashing")
        self.engram_hash = DeepseekV41NgramHash(config, tokenizer) if active_engram else None
        self.layers = nn.ModuleDict(
            {
                str(i): DeepseekV41Block(config, i, backend, moe_config, engram_process_group=engram_process_group)
                for i in range(config.num_hidden_layers)
            }
        )
        norm = (
            partial(initialize_rms_norm_module, "te", device=self.embed_tokens.weight.device)
            if backend.rms_norm == "te"
            else DeepseekV41RMSNorm
        )
        self.norm = norm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

        self.vision = None
        self.aligner = None
        if self.vision_config.num_hidden_layers > 0:
            self.vision = DeepseekV41VisionTransformer(top_config)
            self.aligner = DeepseekV41VisionAligner(top_config)
            for name in ("image_start", "image_end", "image_newline"):
                parameter = nn.Parameter(torch.empty(config.hidden_size, dtype=dtype))
                nn.init.normal_(parameter, std=config.initializer_range)
                self.register_parameter(name, parameter)

    def _image_embeddings(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_hws: torch.Tensor | None,
        vision_token_types: torch.Tensor,
        *,
        cp_group: dist.ProcessGroup | None = None,
        packed_seq_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse globally ordered images into the local contiguous language shard.

        Args:
            input_ids: Integer local token IDs [batch, local_sequence].
            pixel_values: Global patches [patches, 3, patch_size, patch_size],
                replicated within CP; None for a text-only batch.
            image_grid_hws: Global image grids [images, 2], in batch-row order.
            vision_token_types: Local markers [batch, local_sequence]; -1 for
                text, 0/1/2/3 for image start/content/newline/end.
            cp_group: CP-only group, or None. Complete spans are validated after
                gathering marker metadata, so a shard may start inside an image.
            packed_seq_ids: Local document IDs [batch, local_sequence], zero
                for padding. An image must lie wholly inside one document.

        Returns:
            Local fused embeddings [batch, local_sequence, hidden]. Vision
            blocks and the projector each execute once, even on text-only
            ranks, preserving FSDP collective order. Frame-sharder backward
            sums contributions from all CP language shards.
        """
        if self.vision is None:
            raise ValueError("pixel_values requires an enabled DeepSeek V4.1 vision encoder")
        if vision_token_types.shape != input_ids.shape:
            raise ValueError("vision_token_types must match input_ids [batch, local_sequence]")
        # Validate IDs after gathering too: every CP rank sees the same failure
        # before any rank enters a visual collective.
        types = gather_sequence(vision_token_types, cp_group)
        ids = gather_sequence(input_ids, cp_group)
        if torch.any(ids[types >= 0] != self.image_token_id):
            raise ValueError("Every image-span token must use the checkpoint's image_token_id")
        if pixel_values is None:
            if image_grid_hws is not None or torch.any(types >= 0):
                raise ValueError("Image spans require pixel_values")
            images = ()
        else:
            if image_grid_hws is None:
                raise ValueError("pixel_values requires image_grid_hws")
            images = image_inputs_from_batch(
                pixel_values, image_grid_hws, types, downsample_ratio=self.vision_config.downsample_ratio
            )
        if packed_seq_ids is not None:
            documents = gather_sequence(packed_seq_ids, cp_group).cpu()
            for item in images:
                span_docs = documents[item.batch_index, item.start : item.start + item.types.numel()]
                if torch.any(span_docs <= 0) or torch.any(span_docs != span_docs[0]):
                    raise ValueError("An image span cannot cross a packed document boundary or padding")

        embedded = self.embed_tokens(input_ids)
        if not images:
            # All DP ranks must call the same trainable FSDP units. This dummy
            # never enters language tokens and receives exactly zero gradient.
            size = self.vision_config.patch_size
            pixel_values = embedded.new_zeros(1, 3, size, size)
            image_grid_hws = torch.ones(1, 2, dtype=torch.long, device=embedded.device)
        patches = pixel_values.to(device=embedded.device, dtype=self.vision.patch_embed.proj.weight.dtype)
        grids = image_grid_hws.to(embedded.device)
        grid_thw = torch.cat([torch.ones_like(grids[:, :1]), grids], dim=1)
        encoded = maybe_distribute_visual(self.vision, patches.flatten(1), grid_thw).pooler_output
        start = 0 if cp_group is None else dist.get_rank(cp_group) * input_ids.shape[1]
        end = start + input_ids.shape[1]
        # Project only images contributing content to this language shard. An
        # image straddling a shard boundary is projected as a whole on its
        # consumers, preserving spatial-merge cells; only local token uses
        # contribute gradients. Each FSDP-owned projector is still called once.
        selected_patches = []
        selected_grids = []
        destination_indices: list[int] = []
        feature_indices: list[int] = []
        patch_offset = feature_offset = 0
        ratio = self.vision_config.downsample_ratio
        for item in images:
            patch_end = patch_offset + item.n_vit_h * item.n_vit_w
            height = (item.n_vit_h + ratio - 1) // ratio
            width = (item.n_vit_w + ratio - 1) // ratio
            # Plan integer placements on the host from already validated grids.
            # Hundreds of images must not cause one GPU synchronization each.
            keep = []
            if item.start < end and item.start + item.types.numel() > start:
                for row in range(height):
                    for col in range(width):
                        position = item.start + 1 + row * (width + 1) + col
                        if start <= position < end:
                            keep.append((position, row * width + col))
            if keep:
                selected_patches.append(encoded[patch_offset:patch_end])
                selected_grids.append((item.n_vit_h, item.n_vit_w))
                destination_indices.extend(item.batch_index * input_ids.shape[1] + pos - start for pos, _ in keep)
                feature_indices.extend(feature_offset + index for _, index in keep)
                feature_offset += height * width
            patch_offset = patch_end
        if selected_patches:
            local_encoded = encoded if len(selected_patches) == len(images) else torch.cat(selected_patches)
            local_grids = torch.tensor(selected_grids, dtype=torch.long, device=embedded.device)
        else:
            # Link the dummy to the gather even when no local image content is
            # used, so every rank participates in its backward collective.
            local_encoded = encoded[:1] * 0
            local_grids = torch.ones(1, 2, dtype=torch.long, device=embedded.device)
        features = self.aligner(local_encoded, image_grid_hws=local_grids).to(embedded.dtype)
        # Keep the visual graph alive even on a shard with no image tokens.
        embedded = embedded + features.reshape(-1)[:1].sum() * 0
        if destination_indices:
            destinations = torch.tensor(destination_indices, dtype=torch.long, device=embedded.device)
            indices = torch.tensor(feature_indices, dtype=torch.long, device=embedded.device)
            embedded = embedded.flatten(0, 1).index_copy(0, destinations, features[indices]).view_as(embedded)
        else:
            embedded = embedded.clone()
        for marker, value in (
            (IMAGE_START, self.image_start),
            (IMAGE_END, self.image_end),
            (IMAGE_NEW_LINE, self.image_newline),
        ):
            embedded[vision_token_types == marker] = value.to(embedded.dtype)
        return embedded

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_hws: torch.Tensor | None = None,
        vision_token_types: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_residual_streams: bool = False,
        cp_group: dist.ProcessGroup | None = None,
        packed_seq_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        """Compute all positions without inference-only prefill shortcuts.

        Args:
            input_ids: Integer tensor of shape [batch, sequence], also used for Engram.
            position_ids: Optional integer tensor of shape [batch, sequence].
            attention_mask: Optional binary right-padding tensor [batch, sequence].
            image_mask: Optional boolean image-span tensor [batch, sequence].
            pixel_values: Optional image patches [all_patches, 3, patch_size, patch_size].
            image_grid_hws: Optional patch grids [images, 2].
            vision_token_types: Optional image/text markers [batch, sequence].
            inputs_embeds: Optional projected multimodal embeddings [batch, sequence, hidden].
            output_residual_streams: Whether to retain the residual streams before each block
                (parity and debugging; not the recipe interface).
            packed_seq_ids: Optional document IDs [batch, local_sequence], zero for padding.
            cp_group: CP group; token axes above are local sequence shards.
                Engram hashes use global token history, with only local rows looked up.

        Returns:
            Final normalized hidden states [batch, sequence, hidden] and, under
            ``output_residual_streams``, the per-block residual streams
            [batch, sequence, streams, hidden] captured before each block; else None.
        """
        # Fuse within this FSDP owner's forward so learned image delimiters
        # are unsharded before they are read.
        if self.vision is not None and (inputs_embeds is None or pixel_values is not None):
            if vision_token_types is None:
                if pixel_values is not None or image_grid_hws is not None:
                    raise ValueError("Image inputs require vision_token_types")
                vision_token_types = torch.full_like(input_ids, -1)
            inputs_embeds = self._image_embeddings(
                input_ids,
                pixel_values,
                image_grid_hws,
                vision_token_types,
                cp_group=cp_group,
                packed_seq_ids=packed_seq_ids,
            )
            image_mask = vision_token_types >= 0
        elif (
            pixel_values is not None
            or image_grid_hws is not None
            or (vision_token_types is not None and torch.any(vision_token_types >= 0))
        ):
            raise ValueError("Image inputs require an enabled vision encoder")
        if position_ids is None:
            start = 0 if cp_group is None else dist.get_rank(cp_group) * input_ids.shape[1]
            position_ids = (torch.arange(input_ids.shape[1], device=input_ids.device) + start).expand_as(input_ids)
        embedded = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        hidden_states = embedded.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)
        pre_mix = torch.zeros(*input_ids.shape, self.config.hc_mult, device=embedded.device, dtype=torch.float32)
        pre_mix[..., 0] = 1
        token_mask = None if image_mask is None else ~image_mask
        if attention_mask is not None:
            token_mask = attention_mask.bool() if token_mask is None else token_mask & attention_mask.bool()
        hashes = None
        if self.engram_hash is not None:
            full_ids = gather_sequence(input_ids, cp_group)
            full_mask = None if token_mask is None else gather_sequence(token_mask, cp_group)
            full_seq_ids = None if packed_seq_ids is None else gather_sequence(packed_seq_ids, cp_group)
            hashes = self.engram_hash(full_ids, token_mask=full_mask, sequence_ids=full_seq_ids)
            start = 0 if cp_group is None else dist.get_rank(cp_group) * input_ids.shape[1]
            hashes = hashes[:, start : start + input_ids.shape[1]]
        # The host-synchronising input checks run once here instead of in every attention layer (and again in
        # every checkpoint recompute): with 40 layers that was 80+ launch-queue drains per step.
        validate_attention_inputs(
            position_ids, attention_mask, packed_seq_ids, sequence=input_ids.shape[1], cp_group=cp_group
        )
        state = DeepseekV41AttentionState()
        captured = [] if output_residual_streams else None
        for layer in self.layers.values():
            if captured is not None:
                captured.append(hidden_states)
            hidden_states, pre_mix, state = layer(
                hidden_states,
                pre_mix,
                state,
                position_ids=position_ids,
                attention_mask=attention_mask,
                image_mask=image_mask,
                engram_hash_ids=None if layer.engram is None else hashes[:, :, layer.engram.layer_hash_index],
                cp_group=cp_group,
                packed_seq_ids=packed_seq_ids,
                inputs_validated=True,
            )
        hidden_states = DeepseekV41HyperConnection.collapse(hidden_states, pre_mix)
        return self.norm(hidden_states), None if captured is None else tuple(captured)


class DeepseekV41ForCausalLM(HFCheckpointingMixin, PreTrainedModel, MoEFSDPSyncMixin):
    """DeepSeek V4.1 causal LM with optional vision and an fp32 ``lm_head``.

    ``engram_process_group`` explicitly selects contiguous row owners for the
    Engram tables. Distributed models default to WORLD, including a one-rank
    WORLD. Without distributed initialization, tables remain local. FSDP's
    shard mesh must match the owner group exactly.
    """

    _owns_cp_attention: bool = True
    config_class: type[DeepseekV41Config] = DeepseekV41Config
    base_model_prefix: str = "model"
    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    # Reference-sensitive tensors that must stay fp32 regardless of the outer cast policy.
    _keep_in_fp32_modules_strict = [
        "attn_hc",
        "ffn_hc",
        "attn.sinks_param",
        "attn.compressor.wgate",
        "attn.compressor.wkv",
        "e_score_correction_bias",
        "bias_vl",
        "lm_head",
        "vision.norm",
        "norm1.weight",
        "norm2.weight",
    ]
    # fp32 holders whose forward ignores its input: one FSDP unit across all blocks (see
    # parallelizer_utils.fully_shard_across_blocks) — per-block units could only be reduce-scattered one by one
    # in FSDP2's root post-backward callback after the whole backward.
    _fsdp_shard_across_blocks = ["attn.sinks_param"]
    _packed_cp_attn_backends = ("tilelang", "cudnn")

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = False
        supports_ep: bool = True
        supports_thd: bool = True
        supports_cp_vision_frame_sharding: bool = True

    @classmethod
    def from_config(cls, config: DeepseekV41Config, **kwargs: Any) -> DeepseekV41ForCausalLM:
        """Construct using the NeMo registry's configuration entry point."""
        return cls(config, **kwargs)

    def __init__(
        self,
        config: DeepseekV41Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        *,
        tokenizer: PreTrainedTokenizerFast | None = None,
        revision: str | None = None,
        engram_process_group: dist.ProcessGroup | None = None,
    ) -> None:
        reject_unsupported_tie_word_embeddings(type(self), config)
        super().__init__(config)
        self.cp_mesh: DeviceMesh | None = None
        text = config.text_config
        ratios = text.compress_ratios[: text.num_hidden_layers]
        self._packed_alignment = math.lcm(*(r for r in ratios if r))
        self.backend = backend or BackendConfig(
            attn="tilelang", linear="torch", rms_norm="torch_fp32", experts="torch_mm", dispatcher="hybridep"
        )
        if self.backend.rope == "triton":
            # Process-wide dispatch of the attention module's rotation (same once-per-process pattern as the
            # compiled cores): modules built before this call pick it up too. Default "torch" leaves eager.
            use_triton_rope(True)
        dtype = dtype_from_str(text.dtype, torch.bfloat16)
        if engram_process_group is None and dist.is_available() and dist.is_initialized():
            engram_process_group = dist.group.WORLD
        if tokenizer is None and any(i < text.num_hidden_layers for i in text.engram_layer_ids):
            tokenizer = config.build_tokenizer(revision=revision)
        moe_config = moe_config or MoEConfig(
            dim=text.hidden_size,
            inter_dim=text.moe_intermediate_size,
            moe_inter_dim=text.moe_intermediate_size,
            n_routed_experts=text.n_routed_experts,
            n_shared_experts=text.n_shared_experts,
            n_activated_experts=text.num_experts_per_tok,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sqrtsoftplus",
            route_scale=text.routed_scaling_factor,
            norm_topk_prob=text.norm_topk_prob,
            router_weights_fp32=True,
            force_e_score_correction_bias=True,
            swiglu_limit=text.swiglu_limit,
            dtype=dtype,
        )
        self.model = DeepseekV41Model(
            config, self.backend, moe_config, tokenizer=tokenizer, engram_process_group=engram_process_group
        )
        self.lm_head = initialize_linear_module(
            self.backend.linear, text.hidden_size, text.vocab_size, bias=False, dtype=torch.float32
        )
        self.moe_config = moe_config
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = DeepseekV41StateDictAdapter(config, moe_config, self.backend, dtype=dtype)

    def prepare_model_inputs_for_cp(self, batch: dict[str, Any], *, num_chunks: int = 1) -> dict[str, Any]:
        """Select contiguous text sharding with complete compression groups.

        Args:
            batch: Full text tensors of shape [batch, global_sequence]. The hook
                leaves tensors intact; sharding happens through the returned strategy.
            num_chunks: Framework chunk count; this model uses one chunk.

        Returns:
            Model-owned CP sharder. Its local token tensors have shape
            [batch, padded_global_sequence / cp_size], in global position order.
        """
        ratios = self.config.text_config.compress_ratios[: self.config.text_config.num_hidden_layers]
        return {
            "cp_sharder": ContextParallelSharder(
                shard_batch=partial(
                    shard_cp_batch,
                    pad_multiple=math.lcm(*(r for r in ratios if r)),
                    packed_alignment=self._packed_alignment,
                    sync_packed_length=self.backend.dispatcher == "hybridep",
                ),
                local_token_global_indices=contiguous_local_indices,
            )
        }

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the untied token embedding module."""
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        """Return the independent vocabulary projection."""
        return self.lm_head

    def get_dspark_target_feature_modules(self, layer_ids: list[int]) -> tuple[nn.Module, ...]:
        """Return modules whose inputs are the released DSpark target features.

        The reference implementation captures the residual streams immediately
        before attention in each selected layer, after that layer's optional
        Engram update. The attention hyper-connection is the first module to
        consume those streams, so its forward input is the exact capture point.

        Args:
            layer_ids: Strictly increasing decoder-layer indices in
                ``[0, num_hidden_layers)``.

        Returns:
            Modules ordered like ``layer_ids``. Each receives a tensor of shape
            [batch, sequence, streams, hidden] as its first forward argument.

        Raises:
            ValueError: If the indices are duplicated, unsorted, or outside the
                active decoder depth.
        """
        if layer_ids != sorted(set(layer_ids)):
            raise ValueError("DSpark target layer IDs must be strictly increasing")
        if any(
            type(layer_id) is not int or layer_id < 0 or layer_id >= len(self.model.layers) for layer_id in layer_ids
        ):
            raise ValueError(
                f"DSpark target layer IDs must be integers in [0, {len(self.model.layers)}), got {layer_ids}"
            )
        return tuple(self.model.layers[str(layer_id)].attn_hc for layer_id in layer_ids)

    def _nemo_prepare_model_owned_dtensors(self, fsdp_mesh: DeviceMesh) -> set[nn.Parameter]:
        """Register owner table DTensors before FSDP records ignored parameters.

        Args:
            fsdp_mesh: One-dimensional shard mesh whose ranks and ordering must
                match the Engram owner group.

        Returns:
            Exact registered parameter identities to exclude from FSDP. Each
            has global shape [padded_rows, head_dim] and placement Shard(0);
            local storage has shape [padded_rows / owner_world_size, head_dim].
        """
        parameters: set[nn.Parameter] = set()
        for layer in self.model.layers.values():
            if layer.engram is None:
                continue
            parameters.add(layer.engram.embed.parallelize_weight(fsdp_mesh))
        return parameters

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        *,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_hws: torch.Tensor | None = None,
        vision_token_types: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool = False,
        output_residual_streams: bool = False,
        cp_group: dist.ProcessGroup | None = None,
        packed_seq_ids: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        seq_lens_padded: torch.Tensor | None = None,
        qkv_format: str | None = None,
    ) -> CausalLMOutputWithPast:
        """Compute full-vocabulary logits or hidden states for the training loss.

        Args:
            input_ids: Integer tensor of shape [batch, sequence].
            attention_mask: Optional binary right-padding tensor [batch, sequence].
            position_ids: Optional integer tensor of shape [batch, sequence].
            labels: Optional targets of shape [batch, sequence], with -100 ignored.
            pixel_values: Optional image patches [all_patches, 3, patch_size, patch_size].
            image_grid_hws: Optional patch-grid sizes [images, 2].
            vision_token_types: Optional image/text markers [batch, sequence].
            logits_to_keep: Number of final positions, or integer position indices [kept].
            output_hidden_states: Attach the final normalized hidden states
                [batch, sequence, hidden], in the backbone's activation dtype, as
                every other model does; the recipes' linear losses project them
                with ``lm_head.weight``.
            output_residual_streams: Attach the per-layer residual streams instead, a
                tuple of [batch, sequence, streams, hidden] captured before each block,
                for parity checks and debugging. Exclusive with ``output_hidden_states``.
            seq_lens: Optional real document lengths [batch, documents] for packed text.
            seq_lens_padded: Optional physical document spans [batch, documents].
            qkv_format: Optional "thd" marker from packed_sequence_thd_collater.
            packed_seq_ids: Prepared document IDs [batch, local_sequence], zero for padding.
                The model-owned CP sharder supplies these after aligning and sharding a pack.
            cp_group: CP group; input/output token axes contain only the local
                contiguous shard. The recipe computes loss from globally shifted labels.

        Returns:
            CausalLMOutputWithPast containing logits [batch, kept_sequence, vocab],
            optional scalar loss, and ``hidden_states``: the final hidden states
            [batch, sequence, hidden] under ``output_hidden_states``, the per-block
            residual streams under ``output_residual_streams``, else None. No
            inference KV cache.
        """
        if cp_group is None and self.cp_mesh is not None:
            cp_group = self.cp_mesh.get_group()
        if cp_group is not None and dist.get_world_size(cp_group) > 1:
            if labels is not None:
                raise ValueError("CP training requires the recipe loss with labels shifted before sequence sharding")
        if packed_seq_ids is not None:
            if labels is not None:
                raise ValueError("Prepared packed input requires the recipe loss with independently shifted labels")
            if position_ids is None or attention_mask is None:
                raise ValueError(
                    "Prepared packed input requires document-local position_ids and a binary attention_mask"
                )
            if packed_seq_ids.shape != input_ids.shape or packed_seq_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("packed_seq_ids must be an integer tensor with shape [batch, local_sequence]")
        layout = None
        if seq_lens is not None:
            if packed_seq_ids is not None:
                raise ValueError("Provide either seq_lens or prepared packed_seq_ids")
            if cp_group is not None and dist.get_world_size(cp_group) > 1:
                raise ValueError("Packed CP input must pass through the model-owned ContextParallelSharder")
            layout = packed_layout(
                seq_lens,
                seq_lens_padded=seq_lens_padded,
                input_shape=tuple(input_ids.shape),
                alignment=self._packed_alignment,
                minimum_length=input_ids.shape[1],
            )
            input_ids = layout.pack(input_ids, fill=self.config.text_config.pad_token_id or 0)
            if vision_token_types is not None:
                vision_token_types = layout.pack(vision_token_types, fill=-1)
            position_ids = layout.position_ids
            packed_seq_ids = layout.sequence_ids
            attention_mask = packed_seq_ids > 0
        elif seq_lens_padded is not None or (qkv_format == "thd" and packed_seq_ids is None):
            raise ValueError("Packed text requires seq_lens")
        if qkv_format not in (None, "thd"):
            raise ValueError("DeepSeek V4.1 qkv_format must be omitted or 'thd'")
        if output_hidden_states and output_residual_streams:
            raise ValueError(
                "output_hidden_states returns the final hidden states and output_residual_streams the per-layer "
                "residual streams; request one of them"
            )
        hidden, captured = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_hws=image_grid_hws,
            vision_token_types=vision_token_types,
            output_residual_streams=output_residual_streams,
            cp_group=cp_group,
            packed_seq_ids=packed_seq_ids,
        )
        if layout is not None:
            hidden = layout.restore(hidden)
            captured = None if captured is None else tuple(layout.restore(values) for values in captured)
        projected = compute_lm_head_logits(
            self.lm_head, hidden, logits_to_keep, output_hidden_states=output_hidden_states
        )
        loss = None
        if labels is not None:
            if projected.logits is None or projected.logits.shape[:2] != labels.shape:
                raise ValueError("labels require logits for every input position")
            targets = labels[:, 1:]
            if layout is not None:
                original_ids = layout.restore(layout.sequence_ids)
                same_document = (original_ids[:, 1:] > 0) & (original_ids[:, 1:] == original_ids[:, :-1])
                targets = targets.masked_fill(~same_document, -100)
            loss = F.cross_entropy(
                projected.logits[:, :-1].float().reshape(-1, self.config.text_config.vocab_size),
                targets.reshape(-1),
                reduction="sum" if layout is not None else "mean",
            )
            if layout is not None:
                loss = loss / (targets != -100).sum().clamp_min(1)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=projected.logits,
            hidden_states=captured if output_residual_streams else projected.hidden_states,
        )

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        """Initialize every trainable backbone weight after meta materialization."""
        if buffer_device is None:
            buffer_device = self.model.embed_tokens.weight.device
        std = self.config.text_config.initializer_range
        nn.init.normal_(self.model.embed_tokens.weight, std=std)
        nn.init.normal_(self.lm_head.weight, std=std)
        if self.model.vision is not None:
            self.model.vision.init_weights(std)
            self.model.aligner.init_weights(std)
            for parameter in (self.model.image_start, self.model.image_end, self.model.image_newline):
                nn.init.normal_(parameter, std=std)
        if self.model.engram_hash is not None:
            self.model.engram_hash.init_weights()
        for layer in self.model.layers.values():
            layer.ffn.init_weights(buffer_device, init_std=std)
            layer.attn_hc.reset_parameters(std)
            layer.ffn_hc.reset_parameters(std)
            nn.init.ones_(layer.attn_norm.weight)
            nn.init.ones_(layer.ffn_norm.weight)
            if isinstance(layer.ffn.gate, DeepseekV4VisionGate):
                layer.ffn.gate.bias_vl.zero_()
            layer.attn.reset_parameters(std)
            if layer.engram is not None:
                layer.engram.init_weights()
        nn.init.ones_(self.model.norm.weight)
        # As in DeepSeek V4, construction fixes storage dtypes before sharding.
        # Casting again here would round strict FP32 DTensor storage.
        if not _has_dtensor_params(self):
            cast_model_to_dtype(self, dtype)
        for layer in self.model.layers.values():
            if layer.engram is not None:
                layer.engram.embed.mark_sharding_contract()


DeepseekV41ForCausalLM.parallelizer = PARALLELIZER
ModelClass = DeepseekV41ForCausalLM
