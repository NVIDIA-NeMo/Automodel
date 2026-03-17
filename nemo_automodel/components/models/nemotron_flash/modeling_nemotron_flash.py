# coding=utf-8
# Copyright 2025 NVIDIA Corporation. All rights reserved.

"""PyTorch Nemotron-Flash model."""

import copy
import inspect
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

torch._inductor.config.max_autotune_gemm_backends = ["aten"]

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel

try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:
    pass


from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
from transformers.utils import (
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from nemo_automodel.components.models.common.utils import cast_model_to_dtype

from .configuration_nemotron_flash import NemotronFlashConfig

_flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

import torch._dynamo
from einops import pack, repeat, unpack
from mamba_ssm.utils.generation import InferenceParams

from .delta_net import Cache as fla_cache
from .delta_net import DeltaNet
from .fused_mha_with_cache import fused_mha_interface
from .mamba2 import Mamba2

torch._dynamo.config.suppress_errors = True

from torch.cuda import CUDAGraph

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "NemotronFlashConfig"


class NemotronFlashRMSNorm(nn.Module):
    def __init__(self, hidden_size, learnable_weight=True, eps=1e-6):
        super().__init__()
        if learnable_weight:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.weight = None
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.weight is not None:
            return self.weight * hidden_states.to(input_dtype)
        else:
            return hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config, dim, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.base = base
        self.config = config

        self.rope_type = config.rope_type

        self.factor = 2

        max_position_embeddings = self.config.max_position_embeddings

        if config.rope_type is None or config.rope_type == "default":
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
            )
            self.max_seq_len_cached = max_position_embeddings

        elif config.rope_type == "ntk":
            assert self.config.orig_max_position_embeddings is not None
            orig_max_position_embeddings = self.config.orig_max_position_embeddings

            base = base * (
                (self.factor * max_position_embeddings / orig_max_position_embeddings) - (self.factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))

            self.max_seq_len_cached = orig_max_position_embeddings

        elif config.rope_type == "dynamic_ntk":
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
            )
            self.original_inv_freq = inv_freq
            self.max_seq_len_cached = self.config.orig_max_position_embeddings

        else:
            raise ValueError(f"Not support rope_type: {config.rope_type}")

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """

        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            base = self.base * (
                (self.factor * seq_len / self.config.orig_max_position_embeddings) - (self.factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))

            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.config.orig_max_position_embeddings
            and self.max_seq_len_cached > self.config.orig_max_position_embeddings
        ):  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.config.orig_max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        if self.rope_type == "dynamic_ntk":
            self._dynamic_frequency_update(position_ids, device=x.device)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if q is not None:
        q_embed = (q * cos) + (rotate_half(q) * sin)

    else:
        q_embed = None

    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        k_embed = None
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class AttentionDynamicCache(DynamicCache):
    def __init__(self, config, batch_size, dtype=torch.float16, device=None, layer_type=None):
        self.dtype = dtype

        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx].shape[-1] == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=None) -> int:
        if layer_idx is None:
            max_key_len = max(cache.shape[-2] for cache in self.key_cache)
            return max_key_len

        if self.key_cache[layer_idx].shape[-1] == 0:
            return 0

        return self.key_cache[layer_idx].shape[-2]


# Adapted from transformers.models.mistral.modeling_mistral.MistralAttention
class NemotronFlashAttention(nn.Module):
    def __init__(
        self,
        config: NemotronFlashConfig,
        layer_idx: Optional[int] = None,
        input_hidden_size=None,
        output_hidden_size=None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.attn_hidden_size if config.attn_hidden_size > 0 else config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.kq_head_dim = config.kq_head_dim if config.kq_head_dim > 0 else self.head_dim
        self.v_head_dim = config.v_head_dim if config.v_head_dim > 0 else self.head_dim

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size and self.kq_head_dim == self.head_dim:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size if input_hidden_size is None else input_hidden_size,
            self.num_heads * self.kq_head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size if input_hidden_size is None else input_hidden_size,
            self.num_key_value_heads * self.kq_head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size if input_hidden_size is None else input_hidden_size,
            self.num_key_value_heads * self.v_head_dim,
            bias=False,
        )

        if output_hidden_size is None:
            output_hidden_size = self.hidden_size

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, output_hidden_size, bias=False)

        if self.config.kq_norm == "rms":
            self.k_norm = NemotronFlashRMSNorm(self.kq_head_dim)
            self.q_norm = NemotronFlashRMSNorm(self.kq_head_dim)
        elif self.config.kq_norm == "none":
            self.k_norm = None
            self.q_norm = None
        else:
            raise NotImplementedError(f"Unknown kq_norm: {self.config.kq_norm}")

        if self.config.rope:
            self._init_rope()

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(
            config=self.config,
            dim=self.kq_head_dim,
            base=self.rope_theta,
            device=torch.device("cuda"),
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        use_swa=False,
        query_states=None,
        key_states=None,
        value_states=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        raise NotImplementedError("NemotronFlashAttention is an abstract class. Use one of the subclasses.")


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _resolve_attention_mask(
    attention_mask: Optional[torch.Tensor],
    padding_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if attention_mask is not None or padding_mask is None:
        return attention_mask
    if padding_mask.dtype == torch.bool:
        return padding_mask.logical_not().to(dtype=torch.long)
    return 1 - padding_mask


# Adapted from transformers.models.mistral.modeling_mistral.MistralFlashAttention2
class NemotronFlashFlashAttention2(NemotronFlashAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        use_swa=False,
        query_states=None,
        key_states=None,
        value_states=None,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.kq_head_dim).transpose(1, 2).contiguous()

        if self.q_norm is not None:
            query_states = self.q_norm(query_states)

        if self.config.rope:
            cos, sin = self.rotary_emb(hidden_states, position_ids)
            query_states, _ = apply_rotary_pos_emb(query_states, None, cos, sin)

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.kq_head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.v_head_dim).transpose(1, 2)

        if self.k_norm is not None:
            key_states = self.k_norm(key_states)

        if self.config.rope:
            _, key_states = apply_rotary_pos_emb(None, key_states, cos, sin)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and use_swa
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )

        swa_processed_flag = False
        if past_key_value is not None and use_cache:
            kv_layer_idx = self.layer_idx

            cache_has_contents = past_key_value.get_seq_length(kv_layer_idx) > 0

            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
                and use_swa
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[kv_layer_idx][0]
                past_value = past_key_value[kv_layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                past_key_value.key_cache[kv_layer_idx] = past_key
                past_key_value.value_cache[kv_layer_idx] = past_value

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

                swa_processed_flag = True

            key_states, value_states = past_key_value.update(key_states, value_states, kv_layer_idx)

        key_states_no_repeat = key_states
        value_states_no_repeat = value_states

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)  # (batch, slen, num_heads, head_dim)
        key_states = key_states.transpose(1, 2)  # (batch, slen, num_heads, head_dim)
        value_states = value_states.transpose(1, 2)  # (batch, slen, num_heads, head_dim)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows and not swa_processed_flag,
        )

        v_dim = value_states.shape[-2] * value_states.shape[-1]
        attn_output = attn_output.reshape(-1, q_len, v_dim).contiguous()

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, (key_states_no_repeat, value_states_no_repeat)

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        if (
            not self.training and not type(key_layer) == torch.Tensor
        ):  ## this is for handling Mamba2 with output type <class 'mamba_ssm.ops.triton.layernorm_gated.tTensor'>
            key_layer = torch.tensor(key_layer.clone())
            value_layer = torch.tensor(value_layer.clone())
            query_layer = torch.tensor(query_layer.clone())

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class NemotronFlashSDPAAttention(nn.Module):
    def __init__(self, config, layer_idx: int, reuse_kv=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.sliding_window = self.config.sliding_window if self.layer_idx not in self.config.global_attn_idx else None

        self.rotary_emb = NemotronFlashRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            past_seen_tokens = past_key_value.get_seq_length()
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value, (key_states, value_states)


class NemotronFlashRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()

        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


## Interface to use TRTLLM AutoDeploy attention kernel, which enables CUDA Graph capture
class NemotronFlashFusedMHA(NemotronFlashAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fused_mha_interface = fused_mha_interface

    def init_kv_cache(self, max_batch_size, max_seq_len, page_size=-1):
        if hasattr(self, "k_cache"):
            del self.k_cache
            del self.v_cache

            if hasattr(self, "page_table") and self.page_table is not None:
                del self.page_table

            import gc

            gc.collect()

            torch.cuda.empty_cache()

        if page_size is not None and page_size > 0:
            batch_max_pages = (max_seq_len + page_size - 1) // page_size
            cache_max_pages = (max_batch_size * max_seq_len + page_size - 1) // page_size
            self.k_cache = torch.zeros(cache_max_pages, page_size, self.num_key_value_heads, self.kq_head_dim).to(
                self.q_proj.weight
            )
            self.v_cache = torch.zeros(cache_max_pages, page_size, self.num_key_value_heads, self.v_head_dim).to(
                self.q_proj.weight
            )

            self.page_table = torch.zeros(
                max_batch_size, batch_max_pages, device=self.q_proj.weight.device, dtype=torch.int32
            )
        else:
            self.k_cache = torch.zeros(max_batch_size, max_seq_len, self.num_key_value_heads, self.kq_head_dim).to(
                self.q_proj.weight
            )
            self.v_cache = torch.zeros(max_batch_size, max_seq_len, self.num_key_value_heads, self.v_head_dim).to(
                self.q_proj.weight
            )

            self.page_table = None

        self.max_seq_len = max_seq_len

    def reset_kv_cache(self):
        self.k_cache = self.k_cache.zero_()
        self.v_cache = self.v_cache.zero_()

        if self.page_table is not None:
            self.page_table = self.page_table.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        use_swa=False,
        query_states=None,
        key_states=None,
        value_states=None,
        **kwargs,
    ):
        if not hasattr(self, "k_cache"):
            self.init_kv_cache(max_batch_size=1, max_seq_len=8000)

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.kq_head_dim).transpose(1, 2).contiguous()

        if self.q_norm is not None:
            query_states = self.q_norm(query_states)

        if self.config.rope:
            cos, sin = self.rotary_emb(hidden_states, position_ids)
            query_states, _ = apply_rotary_pos_emb(query_states, None, cos, sin)

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.kq_head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.v_head_dim).transpose(1, 2)

        if self.k_norm is not None:
            key_states = self.k_norm(key_states)

        if self.config.rope:
            _, key_states = apply_rotary_pos_emb(None, key_states, cos, sin)

        key_states_no_repeat = key_states
        value_states_no_repeat = value_states

        query_states = query_states.transpose(1, 2)  # (batch, slen, num_heads, head_dim)
        key_states = key_states.transpose(1, 2)  # (batch, slen, num_kv_heads, head_dim)
        value_states = value_states.transpose(1, 2)  # (batch, slen, num_kv_heads, head_dim)

        if self.k_cache.device != query_states.device:
            self.k_cache = self.k_cache.to(query_states)
            self.v_cache = self.v_cache.to(query_states)

        attn_output = self.fused_mha_interface(
            query_states,
            key_states,
            value_states,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            page_table=self.page_table,
            max_seq_len=self.max_seq_len,
            position_ids=position_ids,
        )

        v_dim = query_states.shape[-2] * value_states.shape[-1]
        attn_output = attn_output.reshape(bsz, q_len, v_dim).contiguous()

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, (key_states_no_repeat, value_states_no_repeat)


JAMBA_ATTENTION_CLASSES = {
    "flash_attention_2": NemotronFlashFlashAttention2,
    "fused_mha": NemotronFlashFusedMHA,
    "sdpa": NemotronFlashSDPAAttention,
}


class NemotronFlashMLP(nn.Module):
    def __init__(self, config: NemotronFlashConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.act_fn_name = config.mlp_hidden_act
        self.act_fn = ACT2FN[self.act_fn_name]

        if config.ffn_expand_ratio is not None:
            self.ffn_dim = int(config.ffn_expand_ratio * config.hidden_size) // 128 * 128
        else:
            self.ffn_dim = config.intermediate_size

        self.hidden_dim = config.hidden_size

        self.layer_idx = layer_idx

        if self.act_fn_name == "silu":
            self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.down_proj = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

    def forward(self, x):
        if self.act_fn_name == "silu":
            output = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        elif self.act_fn_name == "relu2":
            output = self.down_proj(self.act_fn(self.up_proj(x)))
        else:
            raise NotImplementedError(f"No such hidden_act: {self.act_fn_name}")

        return output


class NemotronFlashAttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronFlashConfig,
        layer_idx: int,
    ):
        super().__init__()

        self.config = config

        self.layer_idx = layer_idx

        self.self_attn = JAMBA_ATTENTION_CLASSES[config.attn_implementation](config, layer_idx)

        if self.config.intermediate_size > 0:
            self.ffn = NemotronFlashMLP(config, layer_idx=layer_idx)
            self.pre_ffn_layernorm = NemotronFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.ffn = None
            self.pre_ffn_layernorm = None

        self.input_layernorm = NemotronFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        use_swa=False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        if position_ids is not None and position_ids.shape[1] != hidden_states.shape[1]:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)

        residual = hidden_states

        if self.input_layernorm is not None:
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value, current_kv = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            use_swa=use_swa,
        )

        hidden_states = residual + hidden_states

        if self.ffn is not None:
            residual = hidden_states
            if self.pre_ffn_layernorm is not None:
                hidden_states = self.pre_ffn_layernorm(hidden_states)
            hidden_states = self.ffn(hidden_states)

            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (current_kv,)

        return outputs


class FFNDecoderLayer(nn.Module):
    def __init__(self, config: NemotronFlashConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.ffn = NemotronFlashMLP(config, layer_idx=layer_idx)
        self.pre_ffn_layernorm = NemotronFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        use_swa=False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states
        if self.pre_ffn_layernorm is not None:
            hidden_states = self.pre_ffn_layernorm(hidden_states)
        hidden_states = self.ffn(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (None,)

        if use_cache:
            outputs += (None,)

        return outputs


class NemotronFlashMambaDecoderLayer(nn.Module):
    def __init__(self, config: NemotronFlashConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.mamba = Mamba2(config=config, layer_idx=layer_idx)

        self.intermediate_size = config.intermediate_size
        if self.intermediate_size > 0:
            self.ffn = NemotronFlashMLP(config, layer_idx=layer_idx)
            self.pre_ffn_layernorm = NemotronFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.ffn = None
            self.pre_ffn_layernorm = None

        self.input_layernorm = NemotronFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[AttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        use_swa=False,
        mamba_inference_params=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        if position_ids is not None and position_ids.shape[1] != hidden_states.shape[1]:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)

        residual = hidden_states

        if self.input_layernorm is not None:
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value = self.mamba(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            inference_params=mamba_inference_params,
        )

        attn_key_value = None

        hidden_states = residual + hidden_states

        if self.intermediate_size > 0:
            residual = hidden_states

            if self.pre_ffn_layernorm is not None:
                hidden_states = self.pre_ffn_layernorm(hidden_states)

            hidden_states = self.ffn(hidden_states)

            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (attn_key_value,)

        return outputs

    def _get_past_seqlen(self, past_key_value, seqlen):
        if past_key_value is None:
            return seqlen
        past_seqlen = past_key_value.get_seq_length(self.layer_idx)

        if past_seqlen == 0:
            return seqlen

        return past_seqlen


class NemotronFlashHybridDecoderLayer(nn.Module):
    def __init__(self, config: NemotronFlashConfig, layer_idx: int):
        super().__init__()

        self.config = config

        self.layer_idx = layer_idx

        if config.hybrid_decoder_layer == "mamba":
            self.mamba = Mamba2(config=config, layer_idx=layer_idx)
        if config.hybrid_decoder_layer == "deltanet":
            if config.layer_types is not None:
                deltanet_idx = sum(1 for i in range(layer_idx) if config.layer_types[i] == "deltanet")
            else:
                deltanet_idx = layer_idx

            self.gla = DeltaNet(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                layer_idx=deltanet_idx,
                config=self.config,
            )
        else:
            raise ValueError(f"Not supported: {config.hybrid_decoder_layer}")

        self.config = config

        if self.config.intermediate_size > 0:
            self.ffn = NemotronFlashMLP(config, layer_idx=layer_idx)
            self.pre_ffn_layernorm = NemotronFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.ffn = None
            self.pre_ffn_layernorm = None

        self.input_layernorm = NemotronFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[AttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        fla_past_key_values=None,
        mamba_inference_params=None,
        use_swa=False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        if self.config.hybrid_decoder_layer == "mamba":
            hybrid_op_hidden_states, mamba_present_key_value = self.mamba(
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                inference_params=mamba_inference_params,
            )

        else:
            hybrid_op_hidden_states, _, fla_past_key_values = self.gla(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=fla_past_key_values,
                use_cache=use_cache,
            )

        self_attn_weights = self_attn_present_key_value = current_kv = None

        hidden_states = residual + hybrid_op_hidden_states

        if self.ffn is not None:
            residual = hidden_states
            hidden_states = self.pre_ffn_layernorm(hidden_states)

            hidden_states = self.ffn(hidden_states)

            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (self_attn_present_key_value,)

        outputs += (current_kv,)

        return outputs


# Adapted from transformers.models.mistral.modeling_mistral.MistralPreTrainedModel
class NemotronFlashPreTrainedModel(PreTrainedModel):
    config_class = NemotronFlashConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NemotronFlashAttentionDecoderLayer", "NemotronFlashMambaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# Adapted from transformers.models.mistral.modeling_mistral.MistralModel
class NemotronFlashModel(NemotronFlashPreTrainedModel):
    def __init__(self, config: NemotronFlashConfig):
        super().__init__(config)

        config.attn_implementation = config.attn_implementation_new
        config._attn_implementation = config.attn_implementation_new

        self.config = config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        decoder_layers = []

        layer_type = []
        for i in range(config.num_hidden_layers):
            if config.layer_types[i] in ["deltanet"]:
                layer_type.append("m")
                config_new = copy.deepcopy(config)
                config_new.hybrid_decoder_layer = "deltanet"
                decoder_layer = NemotronFlashHybridDecoderLayer(config_new, layer_idx=i)
            elif config.layer_types[i] in ["m", "m2"]:
                layer_type.append("m")
                decoder_layer = NemotronFlashMambaDecoderLayer(config, layer_idx=i)
            elif config.layer_types[i] == "a":
                layer_type.append("a")
                decoder_layer = NemotronFlashAttentionDecoderLayer(config, layer_idx=i)
            elif config.layer_types[i] == "f":
                layer_type.append("a")
                decoder_layer = FFNDecoderLayer(config, layer_idx=i)
            else:
                raise ValueError(f"Unsupported layer type {config.layer_types[i]}")

            decoder_layers.append(decoder_layer)

        config.layer_type = layer_type

        if config.sliding_window is not None:
            self.sliding_window = config.sliding_window
            self.global_attn_idx = config.global_attn_idx
        else:
            self.sliding_window = None
            self.global_attn_idx = None

        self.layers = nn.ModuleList(decoder_layers)

        self._attn_implementation = config.attn_implementation

        self.final_layernorm = NemotronFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.config.num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(self.config.num_memory_tokens, self.config.hidden_size))

        self.gradient_checkpointing = False

        self.post_init()

        self.has_previous_state = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], AttentionDynamicCache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        fla_past_key_values=None,
        mamba_inference_params=None,
        **kwargs,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        attention_mask = _resolve_attention_mask(attention_mask, padding_mask)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            if self.config.num_memory_tokens > 0 and past_key_values is not None and not self.has_previous_state:
                position_ids = position_ids.view(-1, seq_length + self.config.num_memory_tokens).long()
            else:
                position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        ori_b, ori_n = inputs_embeds.shape[0], inputs_embeds.shape[1]  # noqa: F841

        if self.config.num_memory_tokens > 0 and (past_key_values is None or not self.has_previous_state):
            mem = repeat(
                self.memory_tokens, "n d -> b n d", b=inputs_embeds.shape[0]
            )  # prepend the memory to every segment of m by repeating the memory tokens
            inputs_embeds, mem_packed_shape = pack((mem, inputs_embeds), "b * d")

            if position_ids is not None and position_ids.shape[1] != inputs_embeds.shape[1]:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

            if attention_mask is not None and attention_mask.shape[1] < inputs_embeds.shape[1]:
                assert attention_mask.shape[1] + self.config.num_memory_tokens == inputs_embeds.shape[1]
                attention_mask = torch.cat(
                    [
                        torch.ones(inputs_embeds.shape[0], self.config.num_memory_tokens, device=attention_mask.device),
                        attention_mask,
                    ],
                    dim=1,
                )

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of NemotronFlash. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    use_swa=self.sliding_window is not None and i not in self.global_attn_idx,
                    fla_past_key_values=fla_past_key_values,
                    mamba_inference_params=mamba_inference_params,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.config.num_memory_tokens > 0 and (past_key_values is None or not self.has_previous_state):
            mem, hidden_states = unpack(hidden_states, mem_packed_shape, "b * d")
            hidden_states = hidden_states[:, :ori_n, :]

        if past_key_values is not None and not self.has_previous_state:
            self.has_previous_state = True

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values
            if (fla_past_key_values is None and mamba_inference_params is None)
            else (past_key_values, fla_past_key_values, mamba_inference_params),
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# Adapted from transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM with MIXTRAL->JAMBA, Mixtral->NemotronFlash
class NemotronFlashForCausalLM(NemotronFlashPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: NemotronFlashConfig):
        super().__init__(config)
        self.config = config
        self.model = NemotronFlashModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()
        if getattr(config, "tie_word_embeddings", False):
            self.tie_weights()
        if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
            cast_model_to_dtype(self, config.torch_dtype)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @replace_return_docstrings(output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        calc_logits_for_entire_prompt: Optional[bool] = True,
        logits_to_keep: Optional[Union[int, torch.Tensor]] = None,
        fla_past_key_values=None,
        mamba_inference_params=None,
        **kwargs,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            calc_logits_for_entire_prompt (`bool`, *optional*):
                Whether or not to calculate the logits for the entire prompt, or just the last token. Only last token
                logits are needed for generation, and calculating them only for that token can save memory,
                which becomes pretty significant for long sequences.

        Returns:
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        attention_mask = _resolve_attention_mask(attention_mask, padding_mask)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            fla_past_key_values=fla_past_key_values,
            mamba_inference_params=mamba_inference_params,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        if logits_to_keep is None and calc_logits_for_entire_prompt:
            logits = self.lm_head(hidden_states)
        elif logits_to_keep is None:
            logits = self.lm_head(hidden_states[..., -1:, :])
        elif isinstance(logits_to_keep, int) and logits_to_keep == 0:
            logits = self.lm_head(hidden_states)
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        logits = logits / self.lm_head.weight.norm(p=2, dim=1)

        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_init_cache(self, max_seqlen, batch_size=1):
        past_key_values = AttentionDynamicCache(
            self.config, batch_size, self.dtype, device=self.device, layer_type=self.config.layer_type
        )

        mamba_inference_params = InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size)

        fla_past_key_values = fla_cache.from_legacy_cache(None)

        return past_key_values, fla_past_key_values, mamba_inference_params

    def init_cuda_graph_generation(
        self,
        max_new_tokens=128,
        batch_size=1,
        device=None,
    ):
        """
        Initialize CUDA graph for generation with proper cache handling and warmup.
        This function should be called once before generation to set up the graph.

        Args:
            max_new_tokens: Maximum number of new tokens to generate
            batch_size: Batch size for generation
            device: Device to use (defaults to model device)

        Returns:
            generation_state: Dictionary containing all necessary state for generation
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()

        # Initialize caches
        max_seqlen = max_new_tokens + 2048 + self.config.num_memory_tokens  # Add buffer for input
        past_key_values, fla_past_key_values, mamba_inference_params = self.get_init_cache(
            max_seqlen=max_seqlen, batch_size=batch_size
        )

        # Initialize KV caches for all modules
        for module in self.modules():
            if hasattr(module, "init_kv_cache"):
                module.init_kv_cache(max_batch_size=batch_size, max_seq_len=max_seqlen)

        with torch.no_grad():
            # Warmup runs
            dummy_input = torch.ones((batch_size, 10), dtype=torch.long, device=device)
            for _ in range(10):
                self(dummy_input)

            # Prepare static tensors for CUDA graph
            static_current_input = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            static_position_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            static_logits = torch.zeros((batch_size, self.config.vocab_size), device=device)

            # Set up for graph capture
            self.model.has_previous_state = True
            if mamba_inference_params is not None:
                mamba_inference_params.seqlen_offset = 1

            # Warmup runs for graph capture
            for _ in range(10):
                model_kwargs_warmup = {
                    "input_ids": static_current_input,
                    "fla_past_key_values": fla_past_key_values,
                    "mamba_inference_params": mamba_inference_params,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "position_ids": static_position_ids,
                }
                warmup_outputs = self(**model_kwargs_warmup)  # noqa: F841

            # Capture CUDA graph
            generation_graph = CUDAGraph()
            with torch.cuda.graph(generation_graph):
                model_kwargs_graph = {
                    "input_ids": static_current_input,
                    "fla_past_key_values": fla_past_key_values,
                    "mamba_inference_params": mamba_inference_params,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "position_ids": static_position_ids,
                }
                graph_outputs = self(**model_kwargs_graph)
                static_logits.copy_(graph_outputs.logits[:, -1, :])

        if fla_past_key_values is not None:
            fla_past_key_values.reset()

        if mamba_inference_params is not None:
            mamba_inference_params.reset(mamba_inference_params.max_seqlen, mamba_inference_params.max_batch_size)
            for key in mamba_inference_params.key_value_memory_dict:
                conv_state, ssm_state = mamba_inference_params.key_value_memory_dict[key]
                conv_state.zero_()
                ssm_state.zero_()

        for module in self.modules():
            if hasattr(module, "reset_kv_cache"):
                module.reset_kv_cache()

        self.model.has_previous_state = False

        # Return generation state
        generation_state = {
            "generation_graph": generation_graph,
            "static_current_input": static_current_input,
            "static_position_ids": static_position_ids,
            "static_logits": static_logits,
            "past_key_values": past_key_values,
            "fla_past_key_values": fla_past_key_values,
            "mamba_inference_params": mamba_inference_params,
            "max_seqlen": max_seqlen,
            "batch_size": batch_size,
            "device": device,
        }

        return generation_state

    def generate_with_cuda_graph(
        self,
        input_ids,
        generation_state,
        max_new_tokens=128,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        eos_token_id=None,
        verbose=False,
        profiling=False,
    ):
        """
        Generate text using pre-initialized CUDA graph state.

        Args:
            input_ids: Input token IDs tensor of shape (batch_size, seq_len)
            generation_state: State dictionary returned by init_cuda_graph_generation
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0 for greedy)
            top_k: Top-k filtering (0 to disable)
            top_p: Top-p filtering (1.0 to disable)
            eos_token_id: End-of-sequence token ID
            pad_token_id: Padding token ID
            verbose: Whether to print generated tokens
            profiling: Whether to return timing information

        Returns:
            generated_ids: Tensor of shape (batch_size, input_len + generated_len)
            or decode_latency if profiling=True
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Extract state
        generation_graph = generation_state["generation_graph"]
        static_current_input = generation_state["static_current_input"]
        static_position_ids = generation_state["static_position_ids"]
        static_logits = generation_state["static_logits"]
        past_key_values = generation_state["past_key_values"]
        fla_past_key_values = generation_state["fla_past_key_values"]
        mamba_inference_params = generation_state["mamba_inference_params"]

        with torch.no_grad():
            if mamba_inference_params.seqlen_offset == 0:
                if fla_past_key_values is not None:
                    fla_past_key_values.reset()

                if mamba_inference_params is not None:
                    mamba_inference_params.reset(
                        mamba_inference_params.max_seqlen, mamba_inference_params.max_batch_size
                    )
                    for key in mamba_inference_params.key_value_memory_dict:
                        conv_state, ssm_state = mamba_inference_params.key_value_memory_dict[key]
                        conv_state.zero_()
                        ssm_state.zero_()

                for module in self.modules():
                    if hasattr(module, "reset_kv_cache"):
                        module.reset_kv_cache()

                self.model.has_previous_state = False

                # Prefill phase - process input sequence
                position_ids = (
                    torch.arange(self.config.num_memory_tokens + input_ids.shape[1], dtype=torch.long, device=device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )

            else:
                # Prefill phase - process input sequence
                position_ids = (
                    torch.arange(
                        mamba_inference_params.seqlen_offset,
                        mamba_inference_params.seqlen_offset + input_ids.shape[1],
                        dtype=torch.long,
                        device=device,
                    )
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )

            current_input = input_ids

            model_kwargs = {
                "input_ids": current_input,
                "past_key_values": past_key_values,
                "fla_past_key_values": fla_past_key_values,
                "mamba_inference_params": mamba_inference_params,
                "use_cache": True,
                "position_ids": position_ids,
            }

            if profiling:
                torch.cuda.synchronize()
                t1 = time.time()

            # Forward pass for prefill
            outputs = self(**model_kwargs)

            if mamba_inference_params is not None:
                if mamba_inference_params.seqlen_offset == 0:
                    mamba_inference_params.seqlen_offset = current_input.shape[1] + self.config.num_memory_tokens
                else:
                    mamba_inference_params.seqlen_offset += current_input.shape[1]

            static_position_ids.fill_(position_ids[0, -1])

            logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
            generated_tokens = []

            # Generation loop using CUDA graph replay
            for step in range(max_new_tokens):
                # Sample next token using current logits
                if temperature == 0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    next_token = sample_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)

                generated_tokens.append(next_token)

                # Check for EOS
                if not profiling and eos_token_id is not None and (next_token == eos_token_id).all():
                    if verbose:
                        print("\nEOS reached")
                    break

                # Update static tensors for graph replay
                static_current_input.copy_(next_token)
                static_position_ids.add_(1)

                # Replay the captured graph
                generation_graph.replay()

                if mamba_inference_params is not None:
                    mamba_inference_params.seqlen_offset += 1

                logits = static_logits.clone()

            generated_ids = torch.cat([input_ids] + generated_tokens, dim=1)

            if profiling:
                torch.cuda.synchronize()
                t2 = time.time()
                decode_latency = t2 - t1
                return generated_ids, decode_latency

            return generated_ids

    def generate_with_cache(
        self,
        input_ids,
        max_new_tokens=128,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        eos_token_id=None,
        verbose=False,
    ):
        """
        Generate text using the hybrid model with proper cache handling using pre-initialized CUDA graph state.

        Args:
            input_ids: Input token IDs tensor of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0 for greedy)
            top_k: Top-k filtering (0 to disable)
            top_p: Top-p filtering (1.0 to disable)
            eos_token_id: End-of-sequence token ID
            verbose: Whether to print generated tokens

        Returns:
            generated_ids: Tensor of shape (batch_size, input_len + generated_len)
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        with torch.no_grad():
            max_seqlen = input_ids.shape[1] + max_new_tokens + self.config.num_memory_tokens
            past_key_values, fla_past_key_values, mamba_inference_params = self.get_init_cache(
                max_seqlen=max_seqlen, batch_size=batch_size
            )

            for module in self.model.modules():
                if hasattr(module, "init_kv_cache"):
                    module.init_kv_cache(max_batch_size=batch_size, max_seq_len=max_seqlen)

            # Prefill phase - process input sequence
            current_input = input_ids
            position_ids = (
                torch.arange(
                    self.model.config.num_memory_tokens + current_input.shape[1], dtype=torch.long, device=device
                )
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

            model_kwargs = {
                "input_ids": current_input,
                "past_key_values": past_key_values,
                "fla_past_key_values": fla_past_key_values,
                "mamba_inference_params": mamba_inference_params,
                "use_cache": True,
                "position_ids": position_ids,
            }

            outputs = self(**model_kwargs)

            # past_key_values, fla_past_key_values, mamba_inference_params = outputs.past_key_values
            mamba_inference_params.seqlen_offset = current_input.shape[1] + self.model.config.num_memory_tokens

            logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)

            generated_tokens = []

            # Generation loop
            for step in range(max_new_tokens):
                # Sample next token
                if temperature == 0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    next_token = sample_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)

                generated_tokens.append(next_token)

                # Check for EOS
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    if verbose:
                        print("\nEOS reached")
                    break

                current_input = next_token  # Shape: (batch_size, 1)

                # Update position_ids for decoding
                if position_ids is not None:
                    position_ids = torch.full((batch_size, 1), position_ids[0, -1] + 1, dtype=torch.long, device=device)

                # Forward pass for next token
                model_kwargs = {
                    "input_ids": current_input,
                    "fla_past_key_values": fla_past_key_values,
                    "mamba_inference_params": mamba_inference_params,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "position_ids": position_ids,
                }

                outputs = self(**model_kwargs)

                mamba_inference_params.seqlen_offset += 1

                logits = outputs.logits[:, -1, :]

            generated_ids = torch.cat([input_ids] + generated_tokens, dim=1)

            return generated_ids

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if self.config.num_memory_tokens > 0:
            attention_mask = torch.cat(
                [
                    torch.ones(input_ids.shape[0], self.config.num_memory_tokens, device=attention_mask.device),
                    attention_mask,
                ],
                dim=1,
            )

        ### Note that KV cache is disable when using model.generate; Please use model.generate_with_cuda_graph or model.generate_with_cache instead.
        past_key_values = None

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None:
            if input_ids.shape[1] == 0:
                model_inputs = {"inputs_embeds": inputs_embeds}
            else:
                inputs_embeds_new = self.model.embed_tokens(input_ids)
                model_inputs = {"inputs_embeds": torch.cat([inputs_embeds, inputs_embeds_new], dim=1)}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


def sample_token(logits, temperature=1.0, top_k=0, top_p=0.9):
    """
    Sample a token from logits with temperature, top-k, and top-p filtering.

    Args:
        logits: Tensor of shape (batch_size, vocab_size)
        temperature: Sampling temperature
        top_k: Top-k filtering (0 to disable)
        top_p: Top-p filtering (1.0 to disable)

    Returns:
        next_token: Tensor of shape (batch_size, 1)
    """
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits.masked_fill_(indices_to_remove, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits.masked_fill_(indices_to_remove, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
