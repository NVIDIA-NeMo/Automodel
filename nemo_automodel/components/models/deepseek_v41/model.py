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
:class:`~nemo_automodel.components.models.deepseek_v41.layers.DeepseekV41SharedState`.
Snapshots share tensors and preserve the state needed for activation recomputation.

The optional vision tower inserts projected image patches and learned image
delimiters into the text sequence. Text and image batches use full sequences
with two-dimensional token layouts. DSpark draft layers (``mtp.*``),
inference-time KV caching, and SWA bounded replay remain out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import (
    cast_model_to_dtype,
    compute_lm_head_logits,
)
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config, DeepseekV41TextConfig
from nemo_automodel.components.models.deepseek_v41.engram import DeepseekV41NgramHash
from nemo_automodel.components.models.deepseek_v41.layers import (
    DeepseekV41Block,
    DeepseekV41HyperConnection,
    DeepseekV41RMSNorm,
    DeepseekV41RotaryEmbedding,
    DeepseekV41SharedState,
    build_window_topk_indices,
    make_identity_pre_mix,
)
from nemo_automodel.components.models.deepseek_v41.processing import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
    image_inputs_from_batch,
)
from nemo_automodel.components.models.deepseek_v41.state_dict_adapter import DeepseekV41StateDictAdapter
from nemo_automodel.components.models.deepseek_v41.vision import (
    DeepseekV41VisionAligner,
    DeepseekV41VisionTransformer,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class DeepseekV41Model(nn.Module):
    """DeepSeek V4.1 decoder stack: embeddings, hyper-connected blocks, final norm."""

    def __init__(
        self,
        config: DeepseekV41TextConfig,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        tokenizer: PreTrainedTokenizerFast | None = None,
        engram_process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.config = config

        model_dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        moe_defaults = dict(
            dim=config.hidden_size,
            inter_dim=config.moe_intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=config.n_shared_experts,
            n_activated_experts=config.num_experts_per_tok,
            # noaux_tc routing without group limits.
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            force_e_score_correction_bias=True,
            score_func="sqrtsoftplus",
            router_weights_fp32=True,
            combine_in_fp32=True,
            route_scale=config.routed_scaling_factor,
            aux_loss_coeff=0,
            norm_topk_prob=config.norm_topk_prob,
            dtype=model_dtype,
            # Routed and shared experts use clamped SwiGLU in fp32 (reference ``Expert.forward``).
            swiglu_limit=float(config.swiglu_limit),
        )
        self.moe_config = moe_config or MoEConfig(**moe_defaults)
        if not self.moe_config.combine_in_fp32:
            raise ValueError("DeepSeek V4.1 requires MoEConfig.combine_in_fp32=True for released expert arithmetic")

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=model_dtype)
        active_engram = any(i < config.num_hidden_layers for i in config.engram_layer_ids)
        if active_engram and tokenizer is None:
            raise ValueError("DeepSeek V4.1 Engram requires its original fast tokenizer for compressed N-gram hashing")
        self.engram_hash = DeepseekV41NgramHash(config, tokenizer) if active_engram else None
        self.layers = nn.ModuleDict()
        for layer_id in range(config.num_hidden_layers):
            self.layers[str(layer_id)] = DeepseekV41Block(
                layer_id,
                config,
                self.moe_config,
                backend,
                engram_process_group=engram_process_group,
            )
        self.norm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=model_dtype)

        # Base rope (no YaRN) for pure sliding-window layers, compress rope (YaRN)
        # for every CSA2 layer and for the pooled latents (reference ``Attention.__init__``).
        partial_rotary_factor = float(config.qk_rope_head_dim) / float(config.head_dim)
        self.rotary_emb = DeepseekV41RotaryEmbedding(
            rope_theta=float(config.rope_theta),
            head_dim=int(config.head_dim),
            partial_rotary_factor=partial_rotary_factor,
            rope_scaling=None,
        )
        self.rotary_emb_compress = DeepseekV41RotaryEmbedding(
            rope_theta=float(config.compress_rope_theta),
            head_dim=int(config.head_dim),
            partial_rotary_factor=partial_rotary_factor,
            rope_scaling=getattr(config, "rope_scaling", None),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        """Compute full sequences and optional per-block residual streams.

        Args:
            input_ids: Integer token IDs [batch, sequence], also used for Engram.
            inputs_embeds: Optional projected image/text embeddings [batch, sequence, hidden].
            position_ids: Optional contiguous zero-based positions [batch, sequence].
            attention_mask: Optional binary right-padding mask [batch, sequence].
            image_mask: Optional boolean image-span mask [batch, sequence].
            output_hidden_states: Whether to retain streams before each block.

        Returns:
            Final normalized hidden states [batch, sequence, hidden] and optional
            per-block streams [batch, sequence, streams, hidden].
        """
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        inputs_embeds = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        seq_ids = torch.ones(batch, seq_len, dtype=torch.long, device=device)
        padding_mask = None if attention_mask is None else ~attention_mask.bool()
        if padding_mask is not None:
            seq_ids = seq_ids.masked_fill(padding_mask, 0)
        vision_token_types = None if image_mask is None else image_mask.to(torch.int32) - 1

        engram_hash_ids = None
        engram_mask = None
        if self.engram_hash is not None:
            engram_mask = seq_ids > 0
            if vision_token_types is not None:
                engram_mask = engram_mask & (vision_token_types < 0)
            engram_hash_ids = self.engram_hash(input_ids, position_ids=position_ids, token_mask=engram_mask)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        position_embeddings_compress = self.rotary_emb_compress(inputs_embeds, position_ids)

        h = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        pre_mix = make_identity_pre_mix(h, self.config.hc_mult)
        state = DeepseekV41SharedState(window_topk_idxs=build_window_topk_indices(seq_ids, self.config.sliding_window))
        moe_padding_mask = padding_mask.to(device) if padding_mask is not None else None

        captured = [] if output_hidden_states else None
        for layer in self.layers.values():
            if captured is not None:
                captured.append(h)
            layer_hash_ids = None
            if engram_hash_ids is not None and layer.engram is not None:
                layer_hash_ids = engram_hash_ids[:, :, layer.engram.layer_hash_index, :]
            h, pre_mix, state = layer(
                h,
                pre_mix,
                padding_mask=moe_padding_mask,
                attention_mask=attention_mask,
                engram_hash_ids=layer_hash_ids,
                engram_mask=engram_mask,
                vision_token_types=vision_token_types,
                position_embeddings=position_embeddings,
                position_embeddings_compress=position_embeddings_compress,
                rotary_compress=self.rotary_emb_compress,
                position_ids=position_ids,
                seq_ids=seq_ids,
                state=state,
            )

        h = DeepseekV41HyperConnection.collapse(h, pre_mix)
        return self.norm(h), None if captured is None else tuple(captured)

    def update_moe_gate_bias(self) -> None:
        with torch.no_grad():
            for block in self.layers.values():
                if isinstance(block.ffn, MoE) and self.moe_config.gate_bias_update_factor > 0:
                    block.ffn.gate.update_bias()

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            buffer_device = self.embed_tokens.weight.device
        init_std = float(self.config.initializer_range)
        if self.engram_hash is not None:
            self.engram_hash.init_weights()
        with buffer_device:
            nn.init.normal_(self.embed_tokens.weight, std=init_std)
            self.norm.reset_parameters()
        for layer in self.layers.values():
            layer.init_weights(buffer_device=buffer_device, init_std=init_std)


class DeepseekV41ForCausalLM(HFCheckpointingMixin, PreTrainedModel, MoEFSDPSyncMixin):
    """DeepSeek V4.1 causal LM with optional vision and an fp32 ``lm_head``.

    ``engram_process_group`` explicitly selects contiguous row owners for the
    Engram tables. Distributed models default to WORLD, including a one-rank
    WORLD. Without distributed initialization, tables remain local. FSDP's
    shard mesh must match the owner group exactly.
    """

    config_class: type[DeepseekV41Config] = DeepseekV41Config
    base_model_prefix: str = "model"
    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    # Reference-sensitive tensors that must stay fp32 regardless of the outer cast policy.
    _keep_in_fp32_modules_strict = [
        "attn_hc.fn",
        "attn_hc.base",
        "attn_hc.scale",
        "ffn_hc.fn",
        "ffn_hc.base",
        "ffn_hc.scale",
        "attn.sinks_param",
        # Compressor weights stay fp32 in storage. Projection uses the incoming
        # activation dtype via _InputDtypeLinear; pooling remains fp32.
        "attn.compressor.wkv",
        "attn.compressor.wgate",
        "e_score_correction_bias",
        "bias_vl",
        "vision.norm",
        "norm1.weight",
        "norm2.weight",
        "lm_head",
        "rotary_emb",
    ]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = True
        supports_thd: bool = False

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
        engram_process_group: dist.ProcessGroup | None = None,
    ) -> None:
        reject_unsupported_tie_word_embeddings(type(self), config)
        super().__init__(config)
        text = config.text_config
        self.backend = backend or BackendConfig(
            attn="tilelang", linear="torch", rms_norm="torch_fp32", experts="torch_linear", dispatcher="hybridep"
        )
        if engram_process_group is None and dist.is_available() and dist.is_initialized():
            engram_process_group = dist.group.WORLD
        if tokenizer is None and any(i < text.num_hidden_layers for i in text.engram_layer_ids):
            tokenizer = config.build_tokenizer()
        self.model = DeepseekV41Model(
            text,
            backend=self.backend,
            moe_config=moe_config,
            tokenizer=tokenizer,
            engram_process_group=engram_process_group,
        )
        self.model.vision = None
        self.model.aligner = None
        if config.vision_config.num_hidden_layers > 0:
            self.model.vision = DeepseekV41VisionTransformer(config)
            self.model.aligner = DeepseekV41VisionAligner(config)
            for name in ("image_start", "image_end", "image_newline"):
                parameter = nn.Parameter(torch.empty(text.hidden_size, dtype=get_dtype(text.torch_dtype)))
                nn.init.normal_(parameter, std=text.initializer_range)
                self.model.register_parameter(name, parameter)
        self.lm_head = initialize_linear_module(
            self.backend.linear, text.hidden_size, text.vocab_size, bias=False, dtype=torch.float32
        )
        self.moe_config = self.model.moe_config
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = DeepseekV41StateDictAdapter(
                self.config,
                self.model.moe_config,
                self.backend,
                dtype=get_dtype(text.torch_dtype, torch.bfloat16),
            )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def _image_embeddings(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_hws: torch.Tensor,
        vision_token_types: torch.Tensor,
    ) -> torch.Tensor:
        """Insert image features and learned delimiters into fresh token embeddings.

        Args:
            input_ids: Integer tensor of shape [batch, sequence].
            pixel_values: Tensor of shape [all_patches, 3, patch_size, patch_size].
            image_grid_hws: Integer tensor of shape [images, 2], containing patch grids.
            vision_token_types: Integer tensor of shape [batch, sequence], with
                -1 for text and 0/1/2/3 for image start/content/newline/end.

        Returns:
            Tensor of shape [batch, sequence, hidden], retaining text and image
            gradients. The supplied input tensors are not modified.
        """
        if self.model.vision is None:
            raise ValueError("pixel_values requires an enabled DeepSeek V4.1 vision encoder")
        if vision_token_types.shape != input_ids.shape:
            raise ValueError("vision_token_types must match input_ids [batch, sequence]")
        if torch.any(input_ids[vision_token_types >= 0] != self.config.image_token_id):
            raise ValueError("Every image-span token must use the checkpoint's image_token_id")
        images = image_inputs_from_batch(
            pixel_values,
            image_grid_hws,
            vision_token_types,
            downsample_ratio=self.config.vision_config.downsample_ratio,
        )
        embedded = self.model.embed_tokens(input_ids)
        for item in images:
            patches = item.patches.to(device=embedded.device, dtype=self.model.vision.patch_embed.proj.weight.dtype)
            features = self.model.vision(patches, item.n_vit_h, item.n_vit_w)
            features = self.model.aligner(features, item.n_vit_h, item.n_vit_w).to(embedded.dtype)
            types = item.types.to(embedded.device)
            if (types == IMAGE).sum() != features.shape[0]:
                raise ValueError("Image token count does not match the downsampled vision grid")
            span = embedded.new_empty(types.shape[0], embedded.shape[-1])
            span[types == IMAGE] = features
            span[types == IMAGE_START] = self.model.image_start.to(embedded.dtype)
            span[types == IMAGE_END] = self.model.image_end.to(embedded.dtype)
            span[types == IMAGE_NEW_LINE] = self.model.image_newline.to(embedded.dtype)
            flat_indices = item.batch_index * input_ids.shape[1] + torch.arange(
                item.start, item.start + types.shape[0], device=embedded.device
            )
            embedded = embedded.flatten(0, 1).index_copy(0, flat_indices, span).view_as(embedded)
        return embedded

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
        return_hidden_states: bool = False,
        output_hidden_states: bool = False,
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
            return_hidden_states: Return final hidden states for the recipe's loss.
            output_hidden_states: Capture residual streams for numerical comparisons.

        Returns:
            CausalLMOutputWithPast containing logits [batch, kept_sequence, vocab],
            optional scalar loss, and requested hidden tensors. No inference KV cache.
        """
        inputs_embeds = None
        image_mask = None
        if pixel_values is not None:
            if image_grid_hws is None or vision_token_types is None:
                raise ValueError("pixel_values requires image_grid_hws and vision_token_types")
            inputs_embeds = self._image_embeddings(input_ids, pixel_values, image_grid_hws, vision_token_types)
            image_mask = vision_token_types >= 0
        elif image_grid_hws is not None or (vision_token_types is not None and torch.any(vision_token_types >= 0)):
            raise ValueError("Image spans require pixel_values; image placeholders cannot be trained as ordinary text")
        hidden, captured = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            output_hidden_states=output_hidden_states,
        )
        projected = compute_lm_head_logits(
            self.lm_head, hidden, logits_to_keep, output_hidden_states=return_hidden_states
        )
        loss = None
        if labels is not None:
            if projected.logits is None or projected.logits.shape[:2] != labels.shape:
                raise ValueError("labels require logits for every input position")
            loss = F.cross_entropy(
                projected.logits[:, :-1].float().reshape(-1, self.config.text_config.vocab_size),
                labels[:, 1:].reshape(-1),
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=projected.logits,
            hidden_states=captured if output_hidden_states else projected.hidden_states,
        )

    def update_moe_gate_bias(self) -> None:
        self.model.update_moe_gate_bias()

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        """Initialize on the model's device and honor dtype after sharding."""
        if buffer_device is None:
            buffer_device = self.model.embed_tokens.weight.device
        with buffer_device:
            self.model.init_weights(buffer_device=buffer_device)
            if self.model.vision is not None:
                self.model.vision.init_weights(self.config.text_config.initializer_range)
                self.model.aligner.init_weights(self.config.text_config.initializer_range)
                for name in ("image_start", "image_end", "image_newline"):
                    nn.init.normal_(getattr(self.model, name), std=self.config.text_config.initializer_range)
            nn.init.normal_(self.lm_head.weight, std=self.config.text_config.initializer_range)
        cast_model_to_dtype(self, dtype)
        for layer in self.model.layers.values():
            if layer.engram is not None:
                layer.engram.embed.mark_sharding_contract()


ModelClass = DeepseekV41ForCausalLM
