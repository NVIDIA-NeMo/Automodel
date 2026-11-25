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

"""Hy4-flash decoder stack for AutoModel training (text backbone; the vision path is added in ``vision.py``).

Forward contract (transformers reference ``HYV4FlashTextModel``):

1. Embed tokens and replicate them into ``hc_mult`` residual streams ``[batch, sequence, hc_mult, hidden]``.
2. Each of the 42 blocks: optional Engram update of the streams, then
   ``attn_hc`` -> ``input_layernorm`` -> gated MLA -> ``post_self_attn_layernorm`` -> scatter to the streams,
   then ``ffn_hc`` -> ``post_attention_layernorm`` -> dense MLP / latent MoE -> ``post_mlp_layernorm`` -> scatter.
3. ``hc_head`` collapses the streams, then the final RMSNorm and the fp32 ``lm_head``.

Batch rows (or packed documents) are flattened into one token axis for attention; documents never attend to each
other. Context parallelism, PP and vision are layered on this contract in sibling modules.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import partial
from typing import TypedDict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    contiguous_local_indices,
)
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.lm_head import LMHead, slice_logits_to_keep
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import _has_dtensor_params, cast_model_to_dtype
from nemo_automodel.components.models.deepseek_v41.cp import gather_sequence, shard_cp_batch
from nemo_automodel.components.models.deepseek_v41.packing import packed_layout
from nemo_automodel.components.models.hy_v4_flash.attention import HY4FlashAttention, HY4FlashDocLayout, HY4FlashRope
from nemo_automodel.components.models.hy_v4_flash.config import HY4FlashConfig, HY4FlashTextConfig
from nemo_automodel.components.models.hy_v4_flash.engram import HY4FlashEngramLayer, build_ngram_tokens
from nemo_automodel.components.models.hy_v4_flash.layers import (
    HY4FlashHyperConnection,
    HY4FlashHyperHead,
    HY4FlashRMSNorm,
    inverse_frequencies,
    mrope_cos_sin,
)
from nemo_automodel.components.models.hy_v4_flash.vision import HY4FlashVisionTower
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MLP, MoE


def build_moe_config(text: HY4FlashTextConfig, dtype: torch.dtype) -> MoEConfig:
    """MoE settings of the latent-space sigmoid router (all routed experts run at ``routed_expert_hidden_size``)."""
    return MoEConfig(
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
        score_func="sigmoid",
        route_scale=text.routed_scaling_factor,
        norm_topk_prob=text.norm_topk_prob,
        router_weights_fp32=True,
        gate_dtype=torch.float32,
        force_e_score_correction_bias=True,
        swiglu_limit=text.swiglu_limit,
        moe_latent_size=text.routed_expert_hidden_size,
        dtype=dtype,
    )


class HY4FlashCPInputs(TypedDict):
    """Model-owned context-parallel inputs handed to the training recipe."""

    cp_sharder: ContextParallelSharder


@dataclass(frozen=True)
class HY4FlashBatch:
    """Flattened-token view of one forward pass, shared by all layers.

    Attributes:
        layout: Document boundaries over the flattened ``batch * sequence`` token axis.
        ropes: Rotary tables per attention type over the flattened token axis.
        ngram_tokens: Integer tensor [batch, sequence, max_ngram_size] or ``None`` without Engram layers.
        engram_mask: Bool tensor [batch, sequence] or ``None``.
        padding_mask: Bool tensor [batch, sequence], True at padding, or ``None``.
    """

    layout: HY4FlashDocLayout
    ropes: dict[str, HY4FlashRope]
    ngram_tokens: torch.Tensor | None
    engram_mask: torch.Tensor | None
    padding_mask: torch.Tensor | None


def vision_position_ids(start: int, grid_thw: tuple[int, int, int], merge: int) -> torch.Tensor:
    """3-D positions of one image/video: every merged row has ``w + 1`` tokens (the last is the newline).

    All three axes of a frame start at the same offset and each frame starts a new block of ``max(h, w + 1)`` positions.

    Returns:
        Integer tensor of shape [3, t * h * (w + 1)] (time, height, width), offset by ``start``.
    """
    t, h, w = grid_thw[0], grid_thw[1] // merge, grid_thw[2] // merge
    time = torch.arange(t) * max(h, w + 1)
    rows = torch.arange(h)
    cols = torch.arange(w + 1)
    grid_t, grid_h, grid_w = torch.meshgrid(time, rows, cols, indexing="ij")
    return torch.stack((grid_t, grid_h + grid_t, grid_w + grid_t)).reshape(3, -1) + start


def compute_mrope_position_ids(
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    merge: int,
) -> torch.Tensor:
    """M-RoPE positions of mixed text / image rows (the reference ``get_rope_index``).

    Text advances all three axes by one per token; an image block takes :func:`vision_position_ids` and the next
    text token continues at ``max + 1``. Padding positions get zero.

    Args:
        mm_token_type_ids: Integer tensor of shape [batch, sequence], 0 for text and 1 for image tokens.
        image_grid_thw: Integer tensor of shape [images, 3] in row-major order of appearance, or ``None``.
        attention_mask: Optional tensor of shape [batch, sequence], 1 for real tokens.
        merge: Spatial merge size.

    Returns:
        Integer tensor of shape [3, batch, sequence].
    """
    rows, seq = mm_token_type_ids.shape
    positions = torch.zeros(3, rows, seq, dtype=torch.long)
    grids = iter([] if image_grid_thw is None else image_grid_thw.tolist())
    types = mm_token_type_ids.cpu()
    mask = None if attention_mask is None else attention_mask.bool().cpu()
    for row in range(rows):
        row_types = types[row] if mask is None else types[row][mask[row]]
        pieces, current, index = [], 0, 0
        for key, group in itertools.groupby(row_types.tolist()):
            count = len(list(group))
            if key == 0:
                pieces.append(torch.arange(count).expand(3, -1) + current)
                current += count
            elif key == 1:
                block = vision_position_ids(current, tuple(next(grids)), merge)
                if block.shape[1] != count:
                    raise ValueError(
                        f"image block of {count} tokens does not match its grid ({block.shape[1]} expected)"
                    )
                pieces.append(block)
                current = int(block.max()) + 1
            else:
                raise ValueError(f"unsupported mm_token_type_id {key}; only text (0) and image (1) are supported")
            index += count
        flat = torch.cat(pieces, dim=1)
        if mask is None:
            positions[:, row] = flat
        else:
            positions[:, row, mask[row]] = flat
    return positions.to(mm_token_type_ids.device)


class HY4FlashBlock(nn.Module):
    """One sandwich-normed, hyper-connected decoder layer."""

    def __init__(
        self,
        config: HY4FlashTextConfig,
        layer_idx: int,
        backend: BackendConfig,
        moe_config: MoEConfig,
        *,
        engram_process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        dtype = torch.bfloat16
        eps = config.rms_norm_eps
        hidden = config.hidden_size
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = HY4FlashAttention(config, layer_idx, backend)
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = MoE(moe_config, backend)
            # The reference shared expert is a plain SwiGLU; the shared MoE would clamp it like the routed experts.
            self.mlp.shared_experts = MLP(
                hidden,
                config.moe_intermediate_size * config.n_shared_experts,
                backend.linear,
                dtype=dtype,
            )
        else:
            self.mlp = MLP(hidden, config.intermediate_size, backend.linear, dtype=dtype)
        self.input_layernorm = HY4FlashRMSNorm(hidden, eps, dtype)
        self.post_self_attn_layernorm = HY4FlashRMSNorm(hidden, eps, dtype)
        self.post_attention_layernorm = HY4FlashRMSNorm(hidden, eps, dtype)
        self.post_mlp_layernorm = HY4FlashRMSNorm(hidden, eps, dtype)
        self.attn_hc = HY4FlashHyperConnection(config)
        self.ffn_hc = HY4FlashHyperConnection(config)
        self.engram_layer = (
            HY4FlashEngramLayer(config, layer_idx, process_group=engram_process_group, dtype=dtype)
            if layer_idx in config.engram_layer_indices
            else None
        )

    def forward(self, streams: torch.Tensor, batch: HY4FlashBatch) -> torch.Tensor:
        """Run the layer.

        Args:
            streams: Tensor of shape [batch, sequence, hc_mult, hidden].
            batch: Flattened-token view shared by all layers.

        Returns:
            Tensor of shape [batch, sequence, hc_mult, hidden].
        """
        rows, seq, _, hidden = streams.shape
        if self.engram_layer is not None:
            streams = streams + self.engram_layer(streams, batch.ngram_tokens, batch.engram_mask)
        post, x = self.attn_hc(streams)
        x = self.input_layernorm(x)
        x = self.self_attn(x.reshape(1, rows * seq, hidden), layout=batch.layout, rope=batch.ropes[self.attention_type])
        x = self.post_self_attn_layernorm(x.view(rows, seq, hidden))
        streams = self.attn_hc.expand(post, x, streams)
        post, x = self.ffn_hc(streams)
        x = self.post_attention_layernorm(x)
        x = self.mlp(x, batch.padding_mask) if isinstance(self.mlp, MoE) else self.mlp(x)
        x = self.post_mlp_layernorm(x)
        return self.ffn_hc.expand(post, x, streams)


class HY4FlashModel(nn.Module):
    """Embeddings, hyper-connected blocks and the output collapse."""

    def __init__(
        self,
        config: HY4FlashTextConfig,
        backend: BackendConfig,
        moe_config: MoEConfig,
        *,
        engram_process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        dtype = torch.bfloat16
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            HY4FlashBlock(config, i, backend, moe_config, engram_process_group=engram_process_group)
            for i in range(config.num_hidden_layers)
        )
        self.hc_head = HY4FlashHyperHead(config)
        self.norm = HY4FlashRMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        # Rotary frequencies are recomputed from the config in fp32 on every forward (never stored as a bf16 buffer).
        self._inv_freq_theta = {t: p["rope_theta"] for t, p in config.layer_rope_parameters.items()}

    def prepare_batch(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        packed_seq_ids: torch.Tensor | None,
        image_token_mask: torch.Tensor | None,
        cp_group: dist.ProcessGroup | None = None,
    ) -> HY4FlashBatch:
        """Build the flattened-token metadata shared by every layer.

        Args:
            input_ids: Integer tensor of shape [batch, sequence].
            position_ids: Integer tensor of shape [3, batch, sequence] (time, height, width).
            attention_mask: Optional bool/int tensor of shape [batch, sequence], 1 for real tokens.
            packed_seq_ids: Optional integer tensor of shape [batch, sequence] with document ids (0 = padding).
            image_token_mask: Optional bool tensor of shape [batch, sequence], True at image placeholder tokens.
            cp_group: Context-parallel group; all token tensors above then hold this rank's contiguous shard
                (batch must be 1) and the returned document layout / rotary tables cover the global sequence.

        Returns:
            The shared batch view.
        """
        cfg = self.config
        rows, seq = input_ids.shape
        valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        cp_size = 1 if cp_group is None else dist.get_world_size(cp_group)
        if packed_seq_ids is not None:
            doc_ids = packed_seq_ids.to(torch.long) * valid
        else:
            doc_ids = torch.arange(1, rows + 1, device=input_ids.device)[:, None].expand(rows, seq) * valid
        if cp_size > 1:
            if rows != 1:
                raise ValueError("context parallelism requires one packed row per rank")
            local_start = dist.get_rank(cp_group) * seq
            flat = gather_sequence(doc_ids, cp_group).reshape(-1)
            flat_positions = gather_sequence(position_ids.permute(1, 2, 0), cp_group).reshape(-1, 3).T
            global_ids = gather_sequence(input_ids, cp_group)
            global_time = flat_positions[0][None]
        else:
            local_start = 0
            flat = doc_ids.reshape(-1)
            flat_positions = position_ids.reshape(3, -1)
            global_ids, global_time = input_ids, position_ids[0]
        change = torch.nonzero(flat[1:] != flat[:-1]).flatten().add(1).tolist()
        bounds = [0, *change, flat.numel()]
        host_ids = flat[bounds[:-1]].tolist()
        doc_ranges = tuple((s, e) for s, e, d in zip(bounds[:-1], bounds[1:], host_ids) if d > 0)
        starts = torch.arange(flat.numel()) + 1  # padding: start after the token, so no key is valid
        ends = starts.clone()
        for start, end in doc_ranges:
            starts[start:end] = start
            ends[start:end] = end
        local = slice(local_start, local_start + rows * seq)
        layout = HY4FlashDocLayout(
            doc_ranges,
            local_start,
            local_start + rows * seq,
            starts[local].to(flat.device),
            ends[local].to(flat.device),
        )
        ropes = {}
        for attention_type, theta in self._inv_freq_theta.items():
            _, rope_dim, _ = cfg.attention_dims(attention_type)
            cos, sin = mrope_cos_sin(
                flat_positions[:, None],
                inverse_frequencies(rope_dim, theta),
                cfg.layer_rope_parameters[attention_type]["mrope_section"],
                torch.float32,
            )
            ropes[attention_type] = HY4FlashRope(cos[0], sin[0])
        ngram = engram_mask = None
        if cfg.engram_layer_ids:
            ngram = build_ngram_tokens(global_ids, global_time, cfg.engram_max_ngram_size, cfg.engram_pad_id)
            ngram = ngram[:, local_start : local_start + seq]
            engram_mask = valid & (input_ids != cfg.engram_pad_id)
            if image_token_mask is not None:
                engram_mask = engram_mask & ~image_token_mask
        return HY4FlashBatch(layout, ropes, ngram, engram_mask, None if attention_mask is None else ~valid)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        packed_seq_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        image_token_mask: torch.Tensor | None = None,
        cp_group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Compute the final hidden states.

        Args:
            input_ids: Integer tensor of shape [batch, sequence] (also feeds Engram).
            position_ids: Integer tensor of shape [3, batch, sequence].
            attention_mask: Optional tensor of shape [batch, sequence], 1 for real tokens.
            packed_seq_ids: Optional integer tensor of shape [batch, sequence] with document ids.
            inputs_embeds: Optional tensor of shape [batch, sequence, hidden] replacing the token embeddings.
            image_token_mask: Optional bool tensor of shape [batch, sequence] of image placeholder tokens.
            cp_group: Optional context-parallel group; tensors then hold the local contiguous shard.

        Returns:
            Tensor of shape [batch, sequence, hidden] after the final norm.
        """
        batch = self.prepare_batch(input_ids, position_ids, attention_mask, packed_seq_ids, image_token_mask, cp_group)
        embedded = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        streams = embedded.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        for layer in self.layers:
            streams = layer(streams, batch)
        return self.norm(self.hc_head(streams))


class HY4FlashForCausalLM(HFCheckpointingMixin, PreTrainedModel, MoEFSDPSyncMixin):
    """Hy4-flash with an fp32 ``lm_head`` (text path)."""

    _owns_cp_attention: bool = True
    config_class = HY4FlashConfig
    base_model_prefix = "model"
    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    _keep_in_fp32_modules_strict = [
        "attn_hc",
        "ffn_hc",
        "hc_head",
        "sinks_param",
        "e_score_correction_bias",
        "key_value_projection",
        "key_gate_projection",
        "compressed_indexer.key_norm",
        "intra_group_ape",
        "lm_head",
    ]
    # Tiny fp32 holders whose forward ignores its input: one FSDP unit across all blocks (see
    # ``parallelizer_utils.fully_shard_across_blocks``).
    _fsdp_shard_across_blocks = ["self_attn.sinks_param"]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = False
        supports_ep: bool = True
        supports_thd: bool = True

    @classmethod
    def from_config(cls, config: HY4FlashConfig, **kwargs: object) -> HY4FlashForCausalLM:
        """Construct using the NeMo registry's configuration entry point."""
        return cls(config, **kwargs)

    def __init__(
        self,
        config: HY4FlashConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        *,
        engram_process_group: dist.ProcessGroup | None = None,
        logits_dtype: torch.dtype | None = torch.float32,
    ) -> None:
        reject_unsupported_tie_word_embeddings(type(self), config)
        super().__init__(config)
        self.cp_mesh: DeviceMesh | None = None
        text = config.text_config
        self.backend = backend or BackendConfig(
            attn="cudnn", linear="torch", rms_norm="torch", experts="torch_mm", dispatcher="hybridep"
        )
        if engram_process_group is None and dist.is_available() and dist.is_initialized():
            engram_process_group = dist.group.WORLD
        moe_config = moe_config or build_moe_config(text, torch.bfloat16)
        self.moe_config = moe_config
        self.model = HY4FlashModel(text, self.backend, moe_config, engram_process_group=engram_process_group)
        self.lm_head = LMHead(text.hidden_size, text.vocab_size, dtype=torch.float32, logits_dtype=logits_dtype)
        vision = config.vision_config
        # ``vit`` is a top-level module, as in the released checkpoint (``vit.*`` keys).
        self.vit = HY4FlashVisionTower(vision).to(torch.bfloat16) if vision.num_hidden_layers > 0 else None
        self.image_token_id = config.image_token_id
        self.spatial_merge_size = vision.spatial_merge_size

    def _embed(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
    ) -> torch.Tensor:
        """Token embeddings with image features scattered into the ``<|image_pad|>`` positions.

        Args:
            input_ids: Integer tensor of shape [batch, sequence].
            pixel_values: Optional tensor of shape [patches, channels * patch_size * patch_size].
            image_grid_thw: Optional integer tensor of shape [images, 3].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        embedded = self.model.embed_tokens(input_ids)
        if self.vit is None:
            return embedded
        if pixel_values is None:
            # Every data-parallel rank must run the sharded tower once; the dummy never reaches a token.
            size = self.config.vision_config.patch_size
            merge = self.spatial_merge_size
            dummy = embedded.new_zeros(merge * merge, 3 * size * size)
            grid = torch.tensor([[1, merge, merge]], device=embedded.device)
            return embedded + 0 * self.vit(dummy, grid).sum()
        features = self.vit(pixel_values.to(embedded.dtype), image_grid_thw.to(embedded.device)).to(embedded.dtype)
        mask = input_ids == self.image_token_id
        if int(mask.sum()) != features.shape[0]:
            raise ValueError(f"{int(mask.sum())} image placeholder tokens but {features.shape[0]} image features")
        return embedded.masked_scatter(mask.unsqueeze(-1), features)

    def prepare_model_inputs_for_cp(self, batch: dict[str, torch.Tensor], *, num_chunks: int = 1) -> HY4FlashCPInputs:
        """Select contiguous packed sharding (documents are never aligned or reordered).

        Args:
            batch: Full tensors of shape [batch, global_sequence] (the hook leaves them intact).
            num_chunks: Framework chunk count; this model uses one chunk.

        Returns:
            The model-owned CP sharder; its local token tensors have shape [1, padded_global_sequence / cp_size].
        """
        return {
            "cp_sharder": ContextParallelSharder(
                shard_batch=partial(
                    shard_cp_batch,
                    pad_multiple=1,
                    packed_alignment=1,
                    sync_packed_length=self.backend.dispatcher == "hybridep",
                ),
                local_token_global_indices=contiguous_local_indices,
            )
        }

    def _nemo_prepare_model_owned_dtensors(self, fsdp_mesh: DeviceMesh) -> set[nn.Parameter]:
        """Register the row-owner-sharded Engram tables as DTensors before FSDP records its ignored parameters.

        Args:
            fsdp_mesh: One-dimensional shard mesh whose ranks and order equal the Engram owner group.

        Returns:
            The table parameters (global shape [padded_rows, head_dim], ``Shard(0)``) that FSDP must not own.
        """
        return {
            layer.engram_layer.multi_head_embedding.embedding.parallelize_weight(fsdp_mesh)
            for layer in self.model.layers
            if layer.engram_layer is not None
        }

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the token embedding."""
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        """Return the independent vocabulary projection."""
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        packed_seq_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool = False,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        cp_group: dist.ProcessGroup | None = None,
        seq_lens: torch.Tensor | None = None,
        seq_lens_padded: torch.Tensor | None = None,
        qkv_format: str | None = None,
    ) -> CausalLMOutputWithPast:
        """Compute logits (and an optional mean token loss).

        Args:
            input_ids: Integer tensor of shape [batch, sequence].
            position_ids: Optional integer tensor of shape [batch, sequence] or [3, batch, sequence].
            attention_mask: Optional tensor of shape [batch, sequence], 1 for real tokens.
            padding_mask: Optional bool tensor of shape [batch, sequence], True at padding (used if no attention_mask).
            packed_seq_ids: Optional integer tensor of shape [batch, sequence] with packed-document ids.
            labels: Optional integer tensor of shape [batch, sequence], -100 ignored.
            loss_mask: Consumed by the recipe loss; accepted so the standard batch reaches the model.
            logits_to_keep: Number of final positions, or integer positions to keep.
            output_hidden_states: Attach the final hidden states [batch, sequence, hidden].
            pixel_values: Optional tensor of shape [patches, channels * patch_size * patch_size].
            image_grid_thw: Optional integer tensor of shape [images, 3] of (time, height, width) patch counts.
            mm_token_type_ids: Optional integer tensor of shape [batch, sequence], 0 text / 1 image; derived from the
                image placeholder id when omitted.
            cp_group: Context-parallel group; ``input_ids`` etc. are then the local contiguous shard prepared by the
                model-owned sharder (``packed_seq_ids`` and document-local ``position_ids`` included).
            seq_lens: Optional integer tensor of shape [batch, documents] of real document lengths of a packed row
                (direct, non-CP packed forward). ``seq_lens_padded`` gives the physical spans and ``qkv_format`` must be
                ``"thd"`` or omitted.

        Returns:
            ``CausalLMOutputWithPast`` with logits [batch, kept_sequence, vocab].
        """
        del loss_mask
        if cp_group is None and self.cp_mesh is not None:
            cp_group = self.cp_mesh.get_group()
        if attention_mask is None and padding_mask is not None:
            attention_mask = ~padding_mask.bool()
        if qkv_format not in (None, "thd"):
            raise ValueError("Hy4-flash qkv_format must be omitted or 'thd'")
        layout = None
        if seq_lens is not None:
            if packed_seq_ids is not None or (cp_group is not None and dist.get_world_size(cp_group) > 1):
                raise ValueError(
                    "seq_lens is for direct packed forwards; CP input must come from the model-owned sharder"
                )
            layout = packed_layout(
                seq_lens,
                seq_lens_padded=seq_lens_padded,
                input_shape=tuple(input_ids.shape),
                alignment=1,
                minimum_length=input_ids.shape[1],
            )
            input_ids = layout.pack(input_ids, fill=self.config.text_config.pad_token_id or 0)
            position_ids = layout.position_ids
            packed_seq_ids = layout.sequence_ids
            attention_mask = packed_seq_ids > 0
        rows, seq = input_ids.shape
        image_mask = input_ids == self.image_token_id
        if position_ids is None:
            if image_grid_thw is not None:
                if mm_token_type_ids is None:
                    mm_token_type_ids = image_mask.long()
                position_ids = compute_mrope_position_ids(
                    mm_token_type_ids, image_grid_thw, attention_mask, self.spatial_merge_size
                )
            else:
                position_ids = torch.arange(seq, device=input_ids.device).expand(rows, seq)
        if position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        hidden = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            packed_seq_ids=packed_seq_ids,
            inputs_embeds=self._embed(input_ids, pixel_values, image_grid_thw),
            image_token_mask=image_mask,
            cp_group=cp_group,
        )
        if layout is not None:
            hidden = layout.restore(hidden)
        logits = self.lm_head(slice_logits_to_keep(hidden, logits_to_keep))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits, hidden_states=hidden if output_hidden_states else None)

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        """Initialise every weight (checkpoint-free path) as the reference does."""
        if buffer_device is None:
            buffer_device = next(self.parameters()).device
        text = self.config.text_config
        std = text.initializer_range
        nn.init.normal_(self.model.embed_tokens.weight, std=std)
        nn.init.normal_(self.lm_head.weight, std=std)
        for layer in self.model.layers:
            if isinstance(layer.mlp, MoE):
                layer.mlp.init_weights(buffer_device, init_std=std)
            else:
                for module in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
                    nn.init.normal_(module.weight, std=std)
            layer.attn_hc.init_weights(std)
            layer.ffn_hc.init_weights(std)
            attn = layer.self_attn
            for module in (
                attn.q_a_proj,
                attn.q_b_proj,
                attn.kv_a_proj_with_mqa,
                attn.kv_b_proj,
                attn.o_proj,
                attn.gate_proj,
            ):
                nn.init.normal_(module.weight, std=std)
            nn.init.constant_(attn.sinks_param.weight, text.learnable_sink_init)
            for norm in (
                layer.input_layernorm,
                layer.post_self_attn_layernorm,
                layer.post_attention_layernorm,
                layer.post_mlp_layernorm,
                attn.q_a_layernorm,
                attn.kv_a_layernorm,
            ):
                nn.init.ones_(norm.weight)
            if attn.compressed_indexer is not None:
                indexer = attn.compressed_indexer
                for module in (
                    indexer.query_projection,
                    indexer.head_weight_projection,
                    indexer.token_key_projection,
                    indexer.key_value_projection,
                    indexer.key_gate_projection,
                ):
                    nn.init.normal_(module.weight, std=std)
                nn.init.normal_(indexer.intra_group_ape.weight, std=std)
                nn.init.ones_(indexer.token_key_norm.weight)
                nn.init.ones_(indexer.key_norm.weight)
            if layer.engram_layer is not None:
                layer.engram_layer.init_weights()
        self.model.hc_head.init_weights(std)
        nn.init.ones_(self.model.norm.weight)
        if self.vit is not None:
            for name, parameter in self.vit.named_parameters():
                if name.endswith("bias"):
                    nn.init.zeros_(parameter)
                elif parameter.ndim > 1:
                    nn.init.normal_(parameter, std=std)
                else:
                    nn.init.ones_(parameter) if "norm" in name else nn.init.normal_(parameter, std=std)
        if not _has_dtensor_params(self):
            cast_model_to_dtype(self, dtype)


HY4FlashForConditionalGeneration = HY4FlashForCausalLM
ModelClass = HY4FlashForCausalLM
