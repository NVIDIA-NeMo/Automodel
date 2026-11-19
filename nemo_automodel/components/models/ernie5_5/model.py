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

"""ERNIE 5.5 causal language model for AutoModel training.

Forward contract:

1. Embed the tokens and copy the embedding into ``hc_mult`` FP32 residual streams.
2. Run the decoder blocks. Each block has two hyper-connection sites: attention, which is heavily compressed
   attention (HCA) or dynamic sparse attention (DSA) as ``compress_ratios`` selects per layer, and feed-forward, which
   is a dense SiTU MLP in the first ``first_k_dense_replace`` layers and the latent mixture of experts elsewhere.
3. Contract the streams with the hyper-head, then apply the final RMSNorm, the output projection and the MultiMax
   correction of the logits.

One call processes one document: a single unpadded row whose positions start at zero. Padding, a row whose masks or
positions mark several documents, and a KV cache are rejected; several documents concatenated in one row without such
metadata are not detected. ``forward`` declares no packed-sequence and no context-parallel input, so the framework
refuses THD packing when it builds the data loader, and context parallelism through the capability declaration. The
model returns logits only; the recipe computes the loss.

The MTP block of the released checkpoint is not built: the model accepts ``num_nextn_predict_layers == 0`` only and
does not load the MTP weights. The DSA indexers only select keys, receive no gradient and stay frozen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import _has_dtensor_params, cast_model_to_dtype
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.layers import DeepseekV4RotaryEmbedding, build_causal_padding_mask
from nemo_automodel.components.models.ernie5_5.config import (
    Ernie5_5Config,
    Ernie5_5LayerPlan,
    Ernie5_5ModelPlan,
    decoder_only_config,
    hca_rope_scaling,
    main_rope_theta,
    validate_architecture,
)
from nemo_automodel.components.models.ernie5_5.dsa import Ernie5_5DSAAttention, freeze_ernie5_5_indexer_params
from nemo_automodel.components.models.ernie5_5.hca import Ernie5_5HCAAttention
from nemo_automodel.components.models.ernie5_5.layers import (
    DENSE_ATTENTION_BACKENDS,
    Ernie5_5DenseMLP,
    Ernie5_5HyperConnection,
    Ernie5_5HyperHead,
    Ernie5_5MultiMax,
    Ernie5_5RMSNorm,
)
from nemo_automodel.components.models.ernie5_5.moe import Ernie5_5MoE
from nemo_automodel.components.models.ernie5_5.parallelization import PARALLELIZER
from nemo_automodel.components.models.ernie5_5.state_dict_adapter import Ernie5_5StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.shared.utils import dtype_from_str

# Rank of the virtual-head mixing factors of the released checkpoint. The released config.json does not carry it.
_VHA_RANK = 16


def _hca_attention_config(config: Ernie5_5Config, dtype: torch.dtype) -> DeepseekV4Config:
    """Describe the HCA layers in the configuration type of their DeepSeek V4 base class.

    ``Ernie5_5HCAAttention`` extends ``DeepseekV4Attention``, whose constructor reads a ``DeepseekV4Config``. The
    result is only passed to the HCA layers; the model keeps its own ``Ernie5_5Config``.

    Args:
        config: Decoder-only model configuration.
        dtype: Dtype of the attention projections.

    Returns:
        Configuration with the HCA dimensions of ``config``, no hash layers, no MTP layers and no attention dropout.
    """
    return DeepseekV4Config(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        moe_intermediate_size=config.moe_intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        q_lora_rank=config.q_lora_rank,
        o_lora_rank=config.o_lora_rank,
        o_groups=config.o_groups,
        n_routed_experts=config.n_routed_experts,
        n_shared_experts=config.n_shared_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        max_position_embeddings=config.max_position_embeddings,
        compress_ratios=list(config.compress_ratios),
        sliding_window=config.sliding_window,
        attention_dropout=0.0,
        num_hash_layers=0,
        hc_mult=config.hc_mult,
        num_nextn_predict_layers=0,
        rms_norm_eps=config.rms_norm_eps,
        torch_dtype=str(dtype).removeprefix("torch."),
        rope_theta=main_rope_theta(config),
        compress_rope_theta=config.compress_rope_theta,
    )


@dataclass(frozen=True)
class Ernie5_5AttentionInputs:
    """Position and mask inputs of one forward call, shared by all decoder blocks.

    Every row is one unpadded document that starts at position zero.

    Attributes:
        position_ids: Integer tensor of shape [batch, sequence] with the positions ``0..sequence-1`` of every row.
        document_ids: Integer tensor of shape [batch, sequence], all zeros: the document of every token, read by
            the eager DSA path.
        cu_seqlens: Int32 tensor of shape [2] holding ``[0, sequence]``: the cumulative document lengths of the
            single row, read by the cuDNN DSA path.
        hca_rope: Cosine and sine of the HCA compress rotary embedding, each of shape
            [batch, sequence, qk_rope_head_dim], in the dtype of the token embedding.
        hca_rotary: The rotary module that produced ``hca_rope``; the HCA compressor calls it for the positions of
            the compressed keys.
        hca_mask: Additive causal sliding-window mask of shape [batch, 1, sequence, sequence] in the dtype of the
            token embedding: zero for visible keys, the most negative finite value for hidden keys.
    """

    position_ids: torch.Tensor
    document_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    hca_rope: tuple[torch.Tensor, torch.Tensor]
    hca_rotary: DeepseekV4RotaryEmbedding
    hca_mask: torch.Tensor


class Ernie5_5Block(nn.Module):
    """One decoder layer: attention and feed-forward, each mixed into the residual streams by a hyper-connection.

    The feed-forward child is registered as ``moe`` or ``mlp``, the names the MoE parallelizer looks up, and
    ``forward`` reads that attribute on every call, so a wrapper installed in its place takes effect.
    """

    def __init__(
        self,
        config: Ernie5_5Config,
        layer: Ernie5_5LayerPlan,
        backend: BackendConfig,
        moe_config: MoEConfig,
        hca_config: DeepseekV4Config,
    ) -> None:
        """Build the attention, the feed-forward, the two hyper-connections and the two norms of one layer.

        Args:
            config: Decoder-only model configuration.
            layer: Attention and feed-forward kind of this layer.
            backend: Backend selection of the model. A DSA layer reads ``attn`` from it; HCA and MoE layers receive
                a copy with ``attn="eager"``.
            moe_config: Configuration of the latent MoE; read by MoE layers only.
            hca_config: HCA dimensions in the configuration type of the DeepSeek V4 attention; read by HCA layers
                only.
        """
        super().__init__()
        dtype = dtype_from_str(config.dtype, torch.bfloat16)
        dense_backend = replace(backend, attn="eager")
        self.is_dsa = layer.attention == "dsa"
        self.is_moe = layer.feed_forward == "moe"
        self.hidden_size, self.streams = config.hidden_size, config.hc_mult
        if self.is_dsa:
            self.attention = Ernie5_5DSAAttention(
                config.hidden_size,
                config.hybrid_mla_num_attention_heads,
                config.hybrid_mla_q_lora_rank,
                config.hybrid_mla_kv_lora_rank,
                config.hybrid_mla_qk_nope_head_dim,
                config.hybrid_mla_qk_rope_head_dim,
                config.hybrid_mla_v_head_dim,
                config.index_n_heads,
                config.index_head_dim,
                _VHA_RANK,
                window=config.sliding_window,
                remote_topk=config.index_topk,
                dtype=dtype,
                rope_theta=main_rope_theta(config),
                norm_eps=config.rms_norm_eps,
                backend=backend,
            )
        else:
            self.attention = Ernie5_5HCAAttention(hca_config, layer.layer_index, dense_backend, vha_rank=_VHA_RANK)
        if self.is_moe:
            self.moe = Ernie5_5MoE(
                moe_config,
                dense_backend,
                beta=config.activation_situ_beta,
                linear_beta=config.activation_situ_linear_beta,
                norm_eps=config.rms_norm_eps,
            )
        else:
            self.mlp = Ernie5_5DenseMLP(
                config.hidden_size,
                config.intermediate_size,
                dtype=dtype,
                beta=config.activation_situ_beta,
                linear_beta=config.activation_situ_linear_beta,
            )
        self.attn_hc = Ernie5_5HyperConnection(
            config.hc_mult, config.hidden_size, config.hc_sinkhorn_iters, config.hc_eps, config.rms_norm_eps
        )
        self.ffn_hc = Ernie5_5HyperConnection(
            config.hc_mult, config.hidden_size, config.hc_sinkhorn_iters, config.hc_eps, config.rms_norm_eps
        )
        self.input_layernorm = Ernie5_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = Ernie5_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

    def _attend(self, hidden_states: torch.Tensor, inputs: Ernie5_5AttentionInputs) -> torch.Tensor:
        """Run the attention layer with the inputs of its kind.

        Args:
            hidden_states: Normalized tensor of shape [batch, sequence, hidden] in the model dtype.
            inputs: Position and mask inputs of the forward call.

        Returns:
            Attention output of shape [batch, sequence, hidden].
        """
        if not self.is_dsa:
            output, _ = self.attention(
                hidden_states,
                position_embeddings=inputs.hca_rope,
                attention_mask=inputs.hca_mask,
                position_embeddings_compress=inputs.hca_rope,
                rotary_compress=inputs.hca_rotary,
            )
        elif self.attention.uses_cudnn:
            output, _ = self.attention(
                hidden_states,
                position_ids=inputs.position_ids,
                cu_seqlens=inputs.cu_seqlens,
                max_seqlen=hidden_states.shape[1],
            )
        else:
            output, _ = self.attention(
                hidden_states, position_ids=inputs.position_ids, document_ids=inputs.document_ids
            )
        return output

    def forward(self, hidden_streams: torch.Tensor, inputs: Ernie5_5AttentionInputs) -> torch.Tensor:
        """Apply the attention site and the feed-forward site to the residual streams.

        Args:
            hidden_streams: FP32 tensor of shape [batch, sequence, streams, hidden], where ``streams`` is
                ``hc_mult``.
            inputs: Position and mask inputs of the forward call.

        Returns:
            FP32 tensor of shape [batch, sequence, streams, hidden].

        Raises:
            ValueError: If ``hidden_streams`` does not have that shape or is not FP32.
        """
        if hidden_streams.ndim != 4 or hidden_streams.shape[-2:] != (self.streams, self.hidden_size):
            raise ValueError(
                f"Expected streams [batch, sequence, {self.streams}, {self.hidden_size}]; "
                f"got {tuple(hidden_streams.shape)}"
            )
        if hidden_streams.dtype != torch.float32:
            raise ValueError(f"Decoder residual streams must be FP32; got {hidden_streams.dtype}")
        pre, post, comb = self.attn_hc(hidden_streams)
        normalized = self.input_layernorm(self.attn_hc.collapse(hidden_streams, pre))
        hidden_streams = self.attn_hc.expand(self._attend(normalized, inputs), hidden_streams, post, comb)
        pre, post, comb = self.ffn_hc(hidden_streams)
        normalized = self.post_attention_layernorm(self.ffn_hc.collapse(hidden_streams, pre))
        output = self.moe(normalized) if self.is_moe else self.mlp(normalized)
        return self.ffn_hc.expand(output, hidden_streams, post, comb)

    def init_weights(self, buffer_device: torch.device, init_std: float) -> None:
        """Initialize every parameter of the layer after meta-device materialization.

        Args:
            buffer_device: Device of the materialized parameters, passed to the HCA and MoE initializers.
            init_std: Standard deviation of the weight draws.
        """
        if self.is_dsa:
            self.attention.init_weights(init_std)
        else:
            self.attention.init_weights(buffer_device, init_std)
        if self.is_moe:
            self.moe.init_weights(buffer_device, init_std)
        else:
            self.mlp.init_weights(init_std)
        self.attn_hc.init_weights(init_std)
        self.ffn_hc.init_weights(init_std)
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()


class Ernie5_5Model(nn.Module):
    """ERNIE 5.5 decoder: embedding, hyper-connected blocks, hyper-head, final norm, output head and MultiMax."""

    def __init__(self, config: Ernie5_5Config, backend: BackendConfig, moe_config: MoEConfig) -> None:
        """Build the decoder from a decoder-only configuration.

        Args:
            config: Decoder-only model configuration, as ``decoder_only_config`` returns it.
            backend: Backend selection of the model.
            moe_config: Configuration of the latent MoE layers.
        """
        super().__init__()
        self.config = config
        self.moe_config = moe_config
        # Captured once, like the window of the DSA layers, so that every layer of a model uses the same value.
        self.sliding_window = config.sliding_window
        dtype = dtype_from_str(config.dtype, torch.bfloat16)
        hca_config = _hca_attention_config(config, dtype)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            Ernie5_5Block(config, layer, backend, moe_config, hca_config)
            for layer in Ernie5_5ModelPlan.from_config(config).decoder_layers
        )
        self.hyper_head = Ernie5_5HyperHead(config.hc_mult, config.hidden_size, eps=config.rms_norm_eps)
        self.norm = Ernie5_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)
        self.multimax = Ernie5_5MultiMax(dtype=dtype)

    def attention_inputs(self, input_ids: torch.Tensor) -> Ernie5_5AttentionInputs:
        """Build the position and mask inputs of the attention layers for unpadded single-document rows.

        The HCA rotary module is created here, in FP32 and on the input device, so that the model stores no
        frequency buffer that a dtype cast could round. The mask uses the sliding window captured at
        construction.

        Args:
            input_ids: Integer tensor of shape [batch, sequence]; every row is one unpadded document.

        Returns:
            The inputs shared by all decoder blocks, on the device of ``input_ids``.
        """
        config = self.config
        batch, sequence = input_ids.shape
        device, dtype = input_ids.device, self.embed_tokens.weight.dtype
        positions = torch.arange(sequence, device=device).expand(batch, -1)
        rotary = DeepseekV4RotaryEmbedding(
            rope_theta=config.compress_rope_theta,
            head_dim=config.qk_rope_head_dim,
            partial_rotary_factor=1.0,
            rope_scaling=hca_rope_scaling(config),
            device=device,
        )
        rope = rotary(torch.empty(batch, sequence, 1, device=device, dtype=dtype), positions)
        mask = build_causal_padding_mask(
            None,
            seq_len=sequence,
            dtype=dtype,
            device=device,
            batch_size=batch,
            sliding_window=self.sliding_window,
        )
        return Ernie5_5AttentionInputs(
            position_ids=positions,
            document_ids=torch.zeros_like(input_ids),
            cu_seqlens=torch.tensor([0, sequence], device=device, dtype=torch.int32),
            hca_rope=rope,
            hca_rotary=rotary,
            hca_mask=mask,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute the logits of unpadded single-document rows.

        Args:
            input_ids: Integer tensor of shape [batch, sequence]; every row is one unpadded document that starts
                at position zero.

        Returns:
            Logits of shape [batch, sequence, vocab] in the model dtype.
        """
        inputs = self.attention_inputs(input_ids)
        hidden = self.embed_tokens(input_ids).float()
        streams = hidden.unsqueeze(-2).expand(*hidden.shape[:-1], self.config.hc_mult, -1).contiguous()
        for layer in self.layers:
            streams = layer(streams, inputs)
        return self.multimax(self.lm_head(self.norm(self.hyper_head(streams))))

    def init_weights(self, buffer_device: torch.device, init_std: float) -> None:
        """Initialize every parameter of the decoder after meta-device materialization.

        Args:
            buffer_device: Device of the materialized parameters.
            init_std: Standard deviation of the weight draws.
        """
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=init_std)
        for layer in self.layers:
            layer.init_weights(buffer_device, init_std)
        self.hyper_head.init_weights(init_std)
        self.norm.reset_parameters()
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.multimax.multimax_ranges)
        nn.init.zeros_(self.multimax.multimax_ts)


class Ernie5_5_MoeForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """ERNIE 5.5 causal language model with expert parallelism.

    The model is built from a decoder-only configuration (``num_nextn_predict_layers == 0``); ``self.config`` is a
    normalized copy of the configuration passed in, see ``decoder_only_config``.

    ``backend.attn`` selects the attention path. ``"cudnn"`` runs the DSA layers on the cuDNN sparse-attention
    kernels and the HCA layers on eager attention; it requires ``nvidia-cudnn-frontend[cutedsl]`` and FlashMLA,
    BF16 and the released latent and indexer dimensions, and was validated on SM100-class GPUs only. ``"eager"``,
    ``"torch"`` and ``"sdpa"`` run both kinds of layer on eager PyTorch attention on any device; the eager DSA
    path is limited to 512 tokens. Without a ``backend`` argument the model selects ``"cudnn"``; pass
    ``attn="eager"`` where those kernels are not available.

    The model reads one more ``BackendConfig`` field: ``enable_hf_state_dict_adapter`` attaches the checkpoint
    adapter. Every other field must hold a value the layers implement. ``linear`` is ``"torch"``. ``experts`` and
    ``dispatcher`` are ``"torch"`` and ``"torch"``, or ``"torch_mm"`` and ``"hybridep"``. With the second pair the
    routed experts of a job of several processes exchange their tokens through HybridEP, which is supported for
    BF16 compute only (the parallelizer rejects another compute dtype). The remaining fields, the tuning fields
    of HybridEP among them, keep the ``BackendConfig`` default. Other values are rejected at construction. This
    includes ``enable_fsdp_optimizations``: ``MoEFSDPSyncMixin`` would honour it, but it has
    not been validated with this model.

    The DSA indexers only select keys and are not trained. The constructor freezes their parameters, and a
    forward in training mode raises if an indexer holds a trainable parameter: one that a freeze configuration
    enabled again, or an adapter tensor of a LoRA adapter on an indexer projection.

    LoRA is validated on the plain linear projections that an attention module owns directly (not those of its
    indexer or compressor), on the routed experts, and on the projections of the dense FFN and of the shared
    expert.
    """

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    # Parameters and buffers that are stored and computed in FP32 whatever the model dtype is: the hyper-connection
    # and hyper-head mixers, the attention sinks and virtual-head factors, the HCA compressor projections, the DSA
    # indexer key norm, and the MoE router and latent projections.
    _keep_in_fp32_modules_strict = [
        "attn_hc.",
        "ffn_hc.",
        "attention.fp32_params.",
        "attention.vha.",
        "attention.sinks_param.",
        "attention.compressor.wkv.",
        "attention.compressor.wgate.",
        "attention.compressor.ape_param.",
        "attention.indexer.k_norm.",
        "moe.gate.",
        "moe.fc1_latent_proj.",
        "moe.fc2_latent_proj.",
        "hyper_head.",
    ]
    parallelizer = PARALLELIZER

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = True
        supports_thd: bool = False

    @classmethod
    def from_config(cls, config: Ernie5_5Config, backend: BackendConfig | None = None) -> Ernie5_5_MoeForCausalLM:
        """Construct the model through AutoModel's configuration entry point.

        Args:
            config: Model configuration with ``num_nextn_predict_layers == 0``. It is not modified.
            backend: Backend selection. None selects cuDNN DSA attention (the class docstring states what it
                requires) with the torch expert, dispatcher and linear backends.

        Returns:
            The model, with parameters allocated and not initialized from a checkpoint.
        """
        return cls(config, backend)

    def __init__(self, config: Ernie5_5Config, backend: BackendConfig | None = None) -> None:
        """Validate the configuration and the backend, then build the decoder and the checkpoint adapter.

        Args:
            config: Model configuration with ``num_nextn_predict_layers == 0``. It is not modified.
            backend: Backend selection. None selects cuDNN DSA attention (the class docstring states what it
                requires) with the torch expert, dispatcher and linear backends.

        Raises:
            ValueError: If the configuration declares MTP layers or an unsupported architecture, or the backend
                or the dtype is not supported.
            NotImplementedError: If the configuration ties the embedding to the output head, or selects an
                attention option that is not implemented.
        """
        reject_unsupported_tie_word_embeddings(type(self), config)
        super().__init__()
        self.config = decoder_only_config(config)
        validate_architecture(self.config)
        self.backend = backend or BackendConfig(attn="cudnn", linear="torch", experts="torch", dispatcher="torch")
        dtype = dtype_from_str(self.config.dtype, torch.bfloat16)
        self._validate_backend(dtype)
        moe_config = MoEConfig(
            n_routed_experts=self.config.n_routed_experts,
            n_shared_experts=self.config.n_shared_experts,
            n_activated_experts=self.config.num_experts_per_tok,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=self.config.routed_scaling_factor,
            dim=self.config.hidden_size,
            inter_dim=self.config.moe_intermediate_size,
            moe_inter_dim=self.config.moe_intermediate_size,
            moe_latent_size=self.config.moe_latent_size,
            norm_topk_prob=self.config.norm_topk_prob,
            force_e_score_correction_bias=True,
            dtype=dtype,
            gate_dtype=torch.float32,
            router_weights_fp32=True,
        )
        self.model = Ernie5_5Model(self.config, self.backend, moe_config)
        freeze_ernie5_5_indexer_params(self.model)
        # Fix the storage dtypes before sharding: everything in the model dtype except the strict FP32 names.
        cast_model_to_dtype(self, dtype)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = self._create_state_dict_adapter(self.config)

    def _validate_backend(self, dtype: torch.dtype) -> None:
        """Reject backends and dtypes that the layers do not implement.

        Args:
            dtype: Model dtype taken from the configuration.

        Raises:
            ValueError: If ``backend.attn`` is neither ``"cudnn"`` nor an eager name, another ``BackendConfig``
                field holds a value that the model does not implement, the dtype is neither FP32 nor BF16, or the
                cuDNN path is selected without BF16 and the released latent and indexer dimensions.
        """
        backend, config = self.backend, self.config
        if backend.attn != "cudnn" and backend.attn not in DENSE_ATTENTION_BACKENDS:
            raise ValueError(
                f"ERNIE 5.5 supports backend.attn='cudnn' (cuDNN sparse attention in the DSA layers) and "
                f"{DENSE_ATTENTION_BACKENDS} (eager PyTorch attention, at most 512 tokens in the DSA layers); "
                f"got attn={backend.attn!r}. Set `model.backend.attn` in the recipe."
            )
        # The layers implement the torch linear kernels and one of two routed-expert paths: the torch experts with
        # the torch dispatcher, or the grouped-GEMM experts with the HybridEP dispatcher. They read no tuning field,
        # so apart from `attn` and the adapter switch, both copied below, a backend is supported only when it
        # equals the template of its dispatcher. BackendConfig gains fields with every model family; comparing
        # whole objects keeps the check complete when it does.
        hybridep = backend.dispatcher == "hybridep"
        supported = BackendConfig(
            attn=backend.attn,
            linear="torch",
            experts="torch_mm" if hybridep else "torch",
            dispatcher="hybridep" if hybridep else "torch",
            rope_fusion=False,
            enable_hf_state_dict_adapter=backend.enable_hf_state_dict_adapter,
        )
        if backend != supported:
            given, expected = asdict(backend), asdict(supported)
            names = [name for name, value in given.items() if value != expected[name]]
            different = ", ".join(f"{name}={given[name]!r} (supported: {expected[name]!r})" for name in names)
            pairs = (
                ". `experts` and `dispatcher` must be 'torch' and 'torch', or 'torch_mm' and 'hybridep'"
                if "experts" in names or "dispatcher" in names
                else ""
            )
            raise ValueError(f"ERNIE 5.5 does not support these `model.backend` settings: {different}{pairs}")
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"ERNIE 5.5 supports FP32 and BF16 only; got {dtype}")
        if backend.attn == "cudnn" and (
            dtype != torch.bfloat16
            or config.hybrid_mla_kv_lora_rank + config.hybrid_mla_qk_rope_head_dim != 576
            or config.index_head_dim != 128
            or config.index_n_heads not in (32, 64)
        ):
            raise ValueError(
                "cuDNN DSA attention requires BF16, hybrid_mla_kv_lora_rank + hybrid_mla_qk_rope_head_dim == 576, "
                f"index_head_dim == 128 and index_n_heads in (32, 64); got dtype={dtype}, "
                f"hybrid_mla_kv_lora_rank={config.hybrid_mla_kv_lora_rank}, "
                f"hybrid_mla_qk_rope_head_dim={config.hybrid_mla_qk_rope_head_dim}, "
                f"index_head_dim={config.index_head_dim}, index_n_heads={config.index_n_heads}. "
                "Set `model.backend.attn: eager` to run the DSA layers on eager PyTorch attention instead."
            )

    @staticmethod
    def _create_state_dict_adapter(config: Ernie5_5Config) -> StateDictAdapter:
        """Create the adapter between the native state dict and the released checkpoint layout.

        This is the only place that names the adapter class.

        Args:
            config: Decoder-only model configuration.

        Returns:
            The adapter that the checkpointer uses to load and save Hugging Face format weights.
        """
        return Ernie5_5StateDictAdapter(config)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the untied token embedding."""
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Linear:
        """Return the untied output projection."""
        return self.model.lm_head

    def _validate_single_document(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> None:
        """Reject every input that is not one unpadded document starting at position zero.

        Several documents concatenated in one row without position or mask metadata cannot be detected here; the
        data pipeline must supply one document per row.

        Args:
            input_ids: Integer tensor of shape [1, sequence].
            attention_mask: Optional tensor of shape [1, sequence] with one for every token.
            padding_mask: Optional tensor of shape [1, sequence] with zero (False) for every token.
            position_ids: Optional integer tensor of shape [1, sequence] holding ``0..sequence-1``.

        Raises:
            ValueError: If ``input_ids`` is not a nonempty integer tensor of shape [1, sequence] on the device of
                the model, contains the padding token, a mask marks padding or packed documents, or the positions
                do not count up from zero.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not input_ids.numel():
            raise ValueError(
                f"Expected one nonempty document with input_ids [1, sequence]; got {tuple(input_ids.shape)}"
            )
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"input_ids must use an integer token dtype; got {input_ids.dtype}")
        if input_ids.device != self.model.embed_tokens.weight.device:
            raise ValueError(
                f"input_ids are on {input_ids.device} but the model is on {self.model.embed_tokens.weight.device}"
            )
        pad_token_id = self.config.pad_token_id
        if pad_token_id is not None and bool((input_ids == pad_token_id).any()):
            raise ValueError(f"Padding is unsupported: input_ids contain pad_token_id={pad_token_id}")
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids")
            if not bool((attention_mask == 1).all()):
                raise ValueError("Padding is unsupported: attention_mask must be one for every token")
        if padding_mask is not None:
            if padding_mask.shape != input_ids.shape:
                raise ValueError("padding_mask must match input_ids")
            if bool(padding_mask.bool().any()):
                raise ValueError("Padding is unsupported: padding_mask must be zero for every token")
        if position_ids is not None:
            expected = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.int64)[None]
            if position_ids.dtype not in (torch.int32, torch.int64) or not torch.equal(position_ids, expected):
                raise ValueError("Only contiguous zero-based position_ids are supported: a row must hold one document")

    def _require_frozen_indexers(self) -> None:
        """Reject training while a DSA indexer parameter requires gradients.

        The loss gives the indexers no gradient, and an optimizer entry that never receives one has no state to
        restore when a run resumes. The constructor therefore freezes them, and nothing in the model enables
        them again. A freeze configuration whose unfreeze selectors cover an indexer does, and a LoRA adapter on
        an indexer projection adds trainable tensors of its own to the indexer. Both are rejected here at the
        first training step.

        Raises:
            ValueError: If a parameter of a DSA indexer requires gradients.
        """
        for index, layer in enumerate(self.model.layers):
            if layer.is_dsa and any(parameter.requires_grad for parameter in layer.attention.indexer.parameters()):
                raise ValueError(
                    f"model.layers.{index}.attention.indexer has trainable parameters. The DSA indexer only "
                    "selects keys and is not trained: remove it from the unfreeze selectors of freeze_config and "
                    "from the modules that peft.target_modules selects"
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        *,
        use_cache: bool = False,
    ) -> CausalLMOutputWithPast:
        """Compute the logits of one unpadded document.

        The signature has no ``seq_lens`` and no variadic keyword, so the framework's packed-sequence probe is
        false for this model and THD packing is refused when the data loader is built. A row whose masks or
        positions mark padding or several documents is rejected by the mask and position checks. Several documents
        concatenated in one row without such metadata cannot be detected; the data pipeline must supply one
        document per row.

        Args:
            input_ids: Integer tensor of shape [1, sequence]: one document without padding.
            attention_mask: Optional tensor of shape [1, sequence]; must be one for every token.
            padding_mask: Optional tensor of shape [1, sequence]; must be zero (False) for every token.
            position_ids: Optional integer tensor of shape [1, sequence]; must hold ``0..sequence-1``.
            use_cache: Must be False: the model keeps no KV cache.

        Returns:
            ``CausalLMOutputWithPast`` whose ``logits`` have shape [1, sequence, vocab] in the model dtype. No
            loss, hidden states or cache are returned.

        Raises:
            ValueError: If the input is not one unpadded document starting at position zero, ``use_cache`` is set,
                or the model is in training mode and a DSA indexer parameter requires gradients.
        """
        if use_cache:
            raise ValueError("A KV cache is unsupported: use_cache must be False")
        if self.training:
            self._require_frozen_indexers()
        self._validate_single_document(input_ids, attention_mask, padding_mask, position_ids)
        return CausalLMOutputWithPast(logits=self.model(input_ids))

    @torch.no_grad()
    def initialize_weights(self, buffer_device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        """Initialize every parameter after meta-device materialization.

        Weights are drawn with standard deviation ``config.initializer_range``. The attention sinks, the second
        virtual-head factors, the HCA position embeddings of the compressor, the hyper-connection and hyper-head
        offsets, the router correction bias and the MultiMax parameters start at zero; norms and mixer gains start
        at one; the per-expert routing scales start at ``routed_scaling_factor``.

        Args:
            buffer_device: Device of the materialized parameters; defaults to the device of the token embedding.
            dtype: Model dtype; None selects the dtype of the configuration. Storage dtypes are fixed at
                construction; an unsharded model is cast again, which keeps the strict FP32 names in FP32.
        """
        if buffer_device is None:
            buffer_device = self.model.embed_tokens.weight.device
        if dtype is None:
            dtype = dtype_from_str(self.config.dtype, torch.bfloat16)
        self.model.init_weights(buffer_device, self.config.initializer_range)
        # A sharded model keeps the storage dtypes of construction: casting would round FP32 DTensor storage.
        if not _has_dtensor_params(self):
            cast_model_to_dtype(self, dtype)


ModelClass = Ernie5_5_MoeForCausalLM
