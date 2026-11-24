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

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.lm_head import LMHead, slice_logits_to_keep
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import _has_dtensor_params, cast_model_to_dtype
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
    ) -> HY4FlashBatch:
        """Build the flattened-token metadata shared by every layer.

        Args:
            input_ids: Integer tensor of shape [batch, sequence].
            position_ids: Integer tensor of shape [3, batch, sequence] (time, height, width).
            attention_mask: Optional bool/int tensor of shape [batch, sequence], 1 for real tokens.
            packed_seq_ids: Optional integer tensor of shape [batch, sequence] with document ids (0 = padding).
            image_token_mask: Optional bool tensor of shape [batch, sequence], True at image placeholder tokens.

        Returns:
            The shared batch view.
        """
        cfg = self.config
        rows, seq = input_ids.shape
        valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        if packed_seq_ids is not None:
            doc_ids = packed_seq_ids.to(torch.long) * valid
        else:
            doc_ids = torch.arange(1, rows + 1, device=input_ids.device)[:, None].expand(rows, seq) * valid
        flat = doc_ids.reshape(-1)
        change = torch.nonzero(flat[1:] != flat[:-1]).flatten().add(1).tolist()
        bounds = [0, *change, flat.numel()]
        host_ids = flat[bounds[:-1]].tolist()
        doc_ranges = tuple((s, e) for s, e, d in zip(bounds[:-1], bounds[1:], host_ids) if d > 0)
        layout = HY4FlashDocLayout(doc_ranges, 0, rows * seq)
        flat_positions = position_ids.reshape(3, -1)
        ropes = {}
        for attention_type, theta in self._inv_freq_theta.items():
            _, rope_dim, _ = cfg.attention_dims(attention_type)
            cos, sin = mrope_cos_sin(
                flat_positions[:, None], inverse_frequencies(rope_dim, theta), cfg.layer_rope_parameters[attention_type]["mrope_section"], torch.float32
            )
            ropes[attention_type] = HY4FlashRope(cos[0], sin[0])
        ngram = engram_mask = None
        if cfg.engram_layer_ids:
            ngram = build_ngram_tokens(input_ids, position_ids[0], cfg.engram_max_ngram_size, cfg.engram_pad_id)
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
    ) -> torch.Tensor:
        """Compute the final hidden states.

        Args:
            input_ids: Integer tensor of shape [batch, sequence] (also feeds Engram).
            position_ids: Integer tensor of shape [3, batch, sequence].
            attention_mask: Optional tensor of shape [batch, sequence], 1 for real tokens.
            packed_seq_ids: Optional integer tensor of shape [batch, sequence] with document ids.
            inputs_embeds: Optional tensor of shape [batch, sequence, hidden] replacing the token embeddings.
            image_token_mask: Optional bool tensor of shape [batch, sequence] of image placeholder tokens.

        Returns:
            Tensor of shape [batch, sequence, hidden] after the final norm.
        """
        batch = self.prepare_batch(input_ids, position_ids, attention_mask, packed_seq_ids, image_token_mask)
        embedded = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        streams = embedded.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        for layer in self.layers:
            streams = layer(streams, batch)
        return self.norm(self.hc_head(streams))


class HY4FlashForCausalLM(HFCheckpointingMixin, PreTrainedModel, MoEFSDPSyncMixin):
    """Hy4-flash with an fp32 ``lm_head`` (text path)."""

    config_class = HY4FlashConfig
    base_model_prefix = "model"
    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    _keep_in_fp32_modules_strict = [
        "attn_hc",
        "ffn_hc",
        "hc_head",
        "sinks",
        "e_score_correction_bias",
        "gate.weight",
        "key_value_projection",
        "key_gate_projection",
        "intra_group_ape",
        "lm_head",
    ]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
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

        Returns:
            ``CausalLMOutputWithPast`` with logits [batch, kept_sequence, vocab].
        """
        del loss_mask
        if attention_mask is None and padding_mask is not None:
            attention_mask = ~padding_mask.bool()
        rows, seq = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq, device=input_ids.device).expand(rows, seq)
        if position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        hidden = self.model(
            input_ids, position_ids=position_ids, attention_mask=attention_mask, packed_seq_ids=packed_seq_ids
        )
        logits = self.lm_head(slice_logits_to_keep(hidden, logits_to_keep))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits, hidden_states=hidden if output_hidden_states else None)

    @torch.no_grad()
    def initialize_weights(self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16) -> None:
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
            for module in (attn.q_a_proj, attn.q_b_proj, attn.kv_a_proj_with_mqa, attn.kv_b_proj, attn.o_proj, attn.gate_proj):
                nn.init.normal_(module.weight, std=std)
            nn.init.constant_(attn.sinks, text.learnable_sink_init)
            for norm in (layer.input_layernorm, layer.post_self_attn_layernorm, layer.post_attention_layernorm, layer.post_mlp_layernorm, attn.q_a_layernorm, attn.kv_a_layernorm):
                nn.init.ones_(norm.weight)
            if attn.compressed_indexer is not None:
                indexer = attn.compressed_indexer
                for module in (indexer.query_projection, indexer.head_weight_projection, indexer.token_key_projection, indexer.key_value_projection, indexer.key_gate_projection):
                    nn.init.normal_(module.weight, std=std)
                nn.init.normal_(indexer.intra_group_ape, std=std)
                nn.init.ones_(indexer.token_key_norm.weight)
                nn.init.ones_(indexer.key_norm.weight)
            if layer.engram_layer is not None:
                layer.engram_layer.init_weights()
        self.model.hc_head.init_weights(std)
        nn.init.ones_(self.model.norm.weight)
        if not _has_dtensor_params(self):
            cast_model_to_dtype(self, dtype)


ModelClass = HY4FlashForCausalLM
