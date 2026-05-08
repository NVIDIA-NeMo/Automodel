# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import contextlib
from typing import List, Optional, Set

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.thd_utils import split_batch_into_thd_chunks


_NON_TEXT_ATTENTION_PATH_PARTS = {
    "audio_encoder",
    "audio_model",
    "audio_tower",
    "image_encoder",
    "image_model",
    "image_tower",
    "video_encoder",
    "video_model",
    "video_tower",
    "vision_encoder",
    "vision_model",
    "vision_tower",
    "visual",
    "visual_model",
}


def _is_cp_attention_module_name(name: str) -> bool:
    if not name.endswith("self_attn"):
        return False
    return not any(part in _NON_TEXT_ATTENTION_PATH_PARTS for part in name.split("."))


def _packed_sequence_allowed_mask(
    packed_seq_ids: torch.Tensor | None,
    q_indices: torch.Tensor,
    kv_indices: torch.Tensor,
) -> torch.Tensor | None:
    """Build a [B, Q, KV] mask that isolates packed documents."""
    if packed_seq_ids is None:
        return None
    q_pack_ids = packed_seq_ids[:, q_indices]
    kv_pack_ids = packed_seq_ids[:, kv_indices]
    return (q_pack_ids[:, :, None] == kv_pack_ids[:, None, :]) & (q_pack_ids[:, :, None] > 0)


def _build_position_ids(batch, device):
    """Add position_ids to the batch only if they are missing."""
    # TODO(@boxiangw): Refractor. Needed for SP support
    # If 'position_ids' does not exist in batch already then override it.
    # In case of Packed sequence contains 'position_ids' and we don't want to override it.
    if "position_ids" not in batch:
        seq_len = batch["input_ids"].shape[1]
        batch["position_ids"] = torch.arange(seq_len, device=device).unsqueeze(0)
    return batch


# based on https://github.com/pytorch/torchtitan/blob/0b44d4c437c424b6bf719661c0eb4283dc4068bc/torchtitan/distributed/utils.py#L180  # pylint: disable=C0301
def get_train_context(enable_loss_parallel: bool, enable_compiled_autograd: bool, cp_context=None):
    """
    Create a train context.

    Args:
        enable_loss_parallel (bool): Whether to enable loss parallelism.
        enable_compiled_autograd (bool): Whether to enable compiled autograd.
    """

    @contextlib.contextmanager
    def context():
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())

            if enable_compiled_autograd:
                stack.enter_context(torch._dynamo.utils.maybe_enable_compiled_autograd(True))

            if cp_context is not None:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                # currently we only support these two SDP backends.
                # SDPBackend.MATH is not currently compatible with DTensor
                stack.enter_context(sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]))
                stack.enter_context(cp_context)

            yield

    return context


# based on https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/utils.py#L113
def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: List[torch.Tensor],
    cp_seq_dims: List[int],
    cp_no_restore_buffers: Set[torch.Tensor],
    cp_rotate_method: Optional[str] = None,
):
    """
    Create a context parallel context.

    Args:
        cp_mesh (DeviceMesh): The device mesh for context parallel.
        cp_buffers (List[torch.Tensor]): The buffers for context parallel.
        cp_seq_dims (List[int]): The sequence dimensions for context parallel.
        cp_no_restore_buffers (Set[torch.Tensor]): The no restore buffers for context parallel.
        cp_rotate_method (str): The rotation method for context parallel,
            such as "allgather" or "addtoall".
    """
    from torch.distributed.tensor.experimental import context_parallel
    from torch.distributed.tensor.experimental._attention import set_rotate_method

    if cp_rotate_method is not None:
        set_rotate_method(cp_rotate_method)

    # TODO: uncomment this when torch.distributed.tensor.experimental._attention.set_rotate_method
    # is available
    # from torch.distributed.tensor.experimental._attention import set_rotate_method
    # set_rotate_method(cp_rotate_method)
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )


def attach_context_parallel_hooks(model: torch.nn.Module):
    """Attach forward pre-hooks to self_attn modules to fix attention masks for context parallelism.

    Context parallelism shards Q/K/V on the sequence dimension as DTensors,
    so explicit 4D attention masks would have mismatched shapes.  This function
    registers a hook on every ``self_attn`` sub-module that strips the
    ``attention_mask`` kwarg and sets ``is_causal=True`` instead, letting
    SDPA handle causal masking internally.

    Based on ``accelerate.big_modeling._attach_context_parallel_hooks``.
    """

    def _self_attn_pre_forward_hook(_module, module_args, module_kwargs):
        if getattr(_module, "_cp_uses_sdpa_hook", False):
            module_kwargs["attention_mask"] = None
            module_kwargs["is_causal"] = True
            return module_args, module_kwargs
        if "attention_mask" in module_kwargs:
            module_kwargs["attention_mask"] = None
            module_kwargs["is_causal"] = True
        return module_args, module_kwargs

    for name, module in model.named_modules():
        if _is_cp_attention_module_name(name):
            module.register_forward_pre_hook(_self_attn_pre_forward_hook, with_kwargs=True, prepend=True)


def attach_cp_sdpa_hooks(model: torch.nn.Module, cp_mesh) -> None:
    """Inject CP-aware SDPA into self-attention modules.

    Generic non-TE models keep PyTorch DTensor ``context_parallel``: this hook
    only re-wraps local Q/K/V tensors so DTensor SDPA dispatch can run the
    existing all-gather/ring path.

    Gemma4 batches marked by ``prepare_model_inputs_for_cp`` use a manual path
    instead. The batch is sliced contiguously on the sequence dimension; at the
    SDPA call site we all-gather K/V and token types, build the local-query /
    global-key multimodal mask, and return the local output shard.

    Seq dim at the SDPA call is 2: tensors are [B, nH, S/cp_size, D] after HF reshape.
    """
    import logging
    import math

    import torch.nn.functional as F_module
    from torch.distributed.tensor import DTensor, Shard

    _original_sdpa = F_module.scaled_dot_product_attention
    _cp_group = cp_mesh.get_group()
    _cp_size = cp_mesh.size()
    _active_module = {"module": None}
    _log = logging.getLogger(__name__)

    try:
        from torch.distributed.nn.functional import all_gather as _dist_all_gather

        _use_differentiable_ag = True
    except (ImportError, AttributeError):
        _dist_all_gather = None
        _use_differentiable_ag = False

    _flex_attn_compiled = {"fn": None}

    def _get_compiled_flex_attn():
        if _flex_attn_compiled["fn"] is None:
            from torch.nn.attention.flex_attention import flex_attention

            _flex_attn_compiled["fn"] = torch.compile(flex_attention, dynamic=False)
        return _flex_attn_compiled["fn"]

    def _all_gather_seq(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.contiguous()
        if _use_differentiable_ag:
            parts = _dist_all_gather(tensor, group=_cp_group)
        else:
            parts = [torch.empty_like(tensor) for _ in range(_cp_size)]
            torch.distributed.all_gather(parts, tensor, group=_cp_group)
        return torch.cat(tuple(parts), dim=2)

    def _all_gather_seq_metadata(metadata: torch.Tensor | None) -> torch.Tensor | None:
        if metadata is None:
            return None
        local = metadata.contiguous()
        parts = [torch.empty_like(local) for _ in range(_cp_size)]
        torch.distributed.all_gather(parts, local, group=_cp_group)
        return torch.cat(parts, dim=1)

    def _vision_group_ids(mm_token_type_ids: torch.Tensor | None) -> torch.Tensor | None:
        if mm_token_type_ids is None:
            return None
        is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
        prev_is_vision = torch.roll(is_vision, shifts=1, dims=-1)
        prev_is_vision[..., 0] = False
        new_vision_starts = is_vision & ~prev_is_vision
        group_ids = torch.cumsum(new_vision_starts.int(), dim=1) - 1
        return torch.where(is_vision, group_ids, torch.full_like(group_ids, -1))

    @torch._dynamo.disable
    def _cp_sdpa(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kwargs
    ):
        if _cp_size <= 1:
            return _original_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
                **kwargs,
            )

        module = _active_module["module"]
        if not bool(getattr(module, "_cp_manual_allgather_active", False)):
            if not isinstance(query, DTensor):
                query = DTensor.from_local(query, device_mesh=cp_mesh, placements=[Shard(2)])
                key = DTensor.from_local(key, device_mesh=cp_mesh, placements=[Shard(2)])
                value = DTensor.from_local(value, device_mesh=cp_mesh, placements=[Shard(2)])
            out = _original_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
                **kwargs,
            )
            return out.to_local() if isinstance(out, DTensor) else out

        cp_rank = torch.distributed.get_rank(group=_cp_group)
        seq_local = key.shape[2]
        seq_global_start = cp_rank * seq_local

        key_full = _all_gather_seq(key)
        value_full = _all_gather_seq(value)
        seq_full = key_full.shape[2]

        orig_head_dim = query.shape[-1]
        if query.shape[1] != key_full.shape[1]:
            enable_gqa = True

        mm_token_type_ids = getattr(module, "_cp_mm_token_type_ids", None)
        mm_token_type_ids_full = _all_gather_seq_metadata(mm_token_type_ids)
        packed_seq_ids = getattr(module, "_cp_packed_seq_ids", None)
        packed_seq_ids_full = _all_gather_seq_metadata(packed_seq_ids)
        vision_group_ids = _vision_group_ids(mm_token_type_ids_full)
        # HF Gemma4TextAttention marks sliding layers by setting sliding_window;
        # it does not expose an is_sliding flag.
        sliding_window = getattr(module, "sliding_window", None)
        is_sliding = sliding_window is not None
        use_vision_bidirectional = is_sliding and vision_group_ids is not None

        q_indices = torch.arange(seq_local, device=query.device) + seq_global_start
        kv_indices = torch.arange(seq_full, device=query.device)
        if is_causal:
            sdpa_allowed = kv_indices.unsqueeze(0) <= q_indices.unsqueeze(1)
        else:
            sdpa_allowed = torch.ones(seq_local, seq_full, dtype=torch.bool, device=query.device)
        if sliding_window is not None:
            sdpa_allowed = sdpa_allowed & ((q_indices.unsqueeze(1) - kv_indices.unsqueeze(0)) < sliding_window)

        if use_vision_bidirectional:
            q_groups = vision_group_ids[:, q_indices]
            kv_groups = vision_group_ids[:, kv_indices]
            same_vision_group = (q_groups[:, :, None] == kv_groups[:, None, :]) & (q_groups[:, :, None] >= 0)
            sdpa_mask = (sdpa_allowed.unsqueeze(0) | same_vision_group).unsqueeze(1)
        else:
            sdpa_mask = sdpa_allowed.unsqueeze(0).unsqueeze(0)
        packed_allowed = _packed_sequence_allowed_mask(packed_seq_ids_full, q_indices, kv_indices)
        if packed_allowed is not None:
            sdpa_mask = sdpa_mask & packed_allowed.unsqueeze(1)
        empty_query_rows = ~sdpa_mask.any(dim=-1, keepdim=True)
        sdpa_mask_for_kernel = sdpa_mask
        if empty_query_rows.any():
            # Boolean SDPA backward can produce NaNs when a query row is
            # completely masked. Padding rows are ignored by the loss, so point
            # them at a dummy key for the kernel and zero their outputs.
            sdpa_mask_for_kernel = sdpa_mask.clone()
            sdpa_mask_for_kernel[..., :1] = sdpa_mask_for_kernel[..., :1] | empty_query_rows

        try:
            key_for_sdpa = key_full
            value_for_sdpa = value_full
            if query.shape[1] != key_full.shape[1]:
                if query.shape[1] % key_full.shape[1] != 0:
                    raise RuntimeError(
                        f"Cannot expand KV heads for GQA: query heads={query.shape[1]}, kv heads={key_full.shape[1]}"
                    )
                repeat_factor = query.shape[1] // key_full.shape[1]
                key_for_sdpa = key_full.repeat_interleave(repeat_factor, dim=1)
                value_for_sdpa = value_full.repeat_interleave(repeat_factor, dim=1)
            out = _original_sdpa(
                query,
                key_for_sdpa,
                value_for_sdpa,
                attn_mask=sdpa_mask_for_kernel,
                dropout_p=dropout_p,
                is_causal=False,
                scale=scale,
                **kwargs,
            )
            if empty_query_rows.any():
                out = out.masked_fill(empty_query_rows, 0)
            if not getattr(_cp_sdpa, "_sdpa_ok_logged", False):
                _log.info(
                    "CP using %sSDPA all-gather. Q=%s K=%s cp_rank=%s",
                    "packed full-mask " if packed_seq_ids_full is not None else "",
                    tuple(query.shape),
                    tuple(key_full.shape),
                    cp_rank,
                )
                _cp_sdpa._sdpa_ok_logged = True
            return out
        except Exception as sdpa_err:
            if not getattr(_cp_sdpa, "_sdpa_err_logged", False):
                _log.warning(
                    "CP SDPA all-gather failed; trying compiled flex_attention. "
                    "Q=%s K=%s V=%s cp_rank=%s seq_local=%s seq_full=%s error=%s: %s",
                    tuple(query.shape),
                    tuple(key_full.shape),
                    tuple(value_full.shape),
                    cp_rank,
                    seq_local,
                    seq_full,
                    type(sdpa_err).__name__,
                    sdpa_err,
                )
                _cp_sdpa._sdpa_err_logged = True

        padded_head_dim = 1 << (orig_head_dim - 1).bit_length()
        if padded_head_dim != orig_head_dim:
            pad_len = padded_head_dim - orig_head_dim
            query = F_module.pad(query, (0, pad_len))
            key_full = F_module.pad(key_full, (0, pad_len))
            value_full = F_module.pad(value_full, (0, pad_len))
            if scale is None:
                scale = 1.0 / math.sqrt(orig_head_dim)

        def _base_mask(q_idx, kv_idx):
            q_global_idx = q_idx + seq_global_start
            if not is_causal:
                allowed = torch.ones_like(q_global_idx >= kv_idx)
            else:
                allowed = kv_idx <= q_global_idx
            if sliding_window is not None:
                allowed = allowed & ((q_global_idx - kv_idx) < sliding_window)
            return allowed

        out = None
        try:
            from torch.nn.attention.flex_attention import create_block_mask

            if use_vision_bidirectional or packed_seq_ids_full is not None:

                def _cp_mask(batch_idx, head_idx, q_idx, kv_idx):
                    q_global_idx = q_idx + seq_global_start
                    allowed = _base_mask(q_idx, kv_idx)
                    if use_vision_bidirectional:
                        q_group = vision_group_ids[batch_idx, q_global_idx]
                        kv_group = vision_group_ids[batch_idx, kv_idx]
                        same_vision_group = (q_group == kv_group) & (q_group >= 0)
                        allowed = allowed | same_vision_group
                    if packed_seq_ids_full is not None:
                        q_pack_id = packed_seq_ids_full[batch_idx, q_global_idx]
                        kv_pack_id = packed_seq_ids_full[batch_idx, kv_idx]
                        allowed = allowed & (q_pack_id == kv_pack_id) & (q_pack_id > 0)
                    return allowed

                block_mask_batch = query.shape[0]
            else:

                def _cp_mask(batch_idx, head_idx, q_idx, kv_idx):
                    return _base_mask(q_idx, kv_idx)

                block_mask_batch = None

            block_mask = create_block_mask(
                _cp_mask,
                B=block_mask_batch,
                H=None,
                Q_LEN=seq_local,
                KV_LEN=seq_full,
                device=query.device,
            )
            flex_kwargs = {"block_mask": block_mask, "scale": scale, "enable_gqa": enable_gqa}
            if query.shape[-1] >= 256:
                flex_kwargs["kernel_options"] = {
                    "BLOCK_M": 32,
                    "BLOCK_N": 32,
                    "BLOCK_M1": 32,
                    "BLOCK_N1": 32,
                    "BLOCK_M2": 32,
                    "BLOCK_N2": 32,
                    "num_stages": 1,
                    "num_warps": 4,
                }
            try:
                out = _get_compiled_flex_attn()(query.contiguous(), key_full, value_full, **flex_kwargs)
            except TypeError as exc:
                if "kernel_options" in str(exc) and "kernel_options" in flex_kwargs:
                    flex_kwargs.pop("kernel_options")
                    out = _get_compiled_flex_attn()(query.contiguous(), key_full, value_full, **flex_kwargs)
                else:
                    raise
            if not getattr(_cp_sdpa, "_flex_ok_logged", False):
                _log.info(
                    "CP using compiled flex_attention all-gather. Q=%s K=%s head_dim=%s->%s cp_rank=%s",
                    tuple(query.shape),
                    tuple(key_full.shape),
                    orig_head_dim,
                    padded_head_dim,
                    cp_rank,
                )
                _cp_sdpa._flex_ok_logged = True
        except Exception as flex_err:
            if not getattr(_cp_sdpa, "_flex_err_logged", False):
                _log.warning(
                    "CP flex_attention all-gather failed; using chunked attention fallback. "
                    "Q=%s K=%s V=%s cp_rank=%s seq_local=%s seq_full=%s error=%s: %s",
                    tuple(query.shape),
                    tuple(key_full.shape),
                    tuple(value_full.shape),
                    cp_rank,
                    seq_local,
                    seq_full,
                    type(flex_err).__name__,
                    flex_err,
                )
                _cp_sdpa._flex_err_logged = True
            out = None

        if out is None:
            attn_scale = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
            num_q_heads = query.shape[1]
            num_kv_heads = key_full.shape[1]
            if num_q_heads != num_kv_heads:
                if num_q_heads % num_kv_heads != 0:
                    raise RuntimeError(
                        f"Cannot expand KV heads for GQA: query heads={num_q_heads}, kv heads={num_kv_heads}"
                    )
                repeat_factor = num_q_heads // num_kv_heads
                key_for_attn = key_full.repeat_interleave(repeat_factor, dim=1)
                value_for_attn = value_full.repeat_interleave(repeat_factor, dim=1)
            else:
                key_for_attn = key_full
                value_for_attn = value_full

            q_float = query.float()
            k_float = key_for_attn.float()
            v_float = value_for_attn.float()
            batch_size, num_heads, _, head_dim = q_float.shape
            out_acc = torch.zeros(batch_size, num_heads, seq_local, head_dim, dtype=torch.float32, device=query.device)
            lse = torch.full((batch_size, num_heads, seq_local), float("-inf"), dtype=torch.float32, device=query.device)
            q_indices = torch.arange(seq_local, device=query.device) + seq_global_start
            kv_chunk = 256

            for kv_start in range(0, seq_full, kv_chunk):
                kv_end = min(kv_start + kv_chunk, seq_full)
                kv_indices = torch.arange(kv_start, kv_end, device=query.device)
                scores = torch.matmul(q_float, k_float[:, :, kv_start:kv_end, :].transpose(-2, -1)) * attn_scale

                if is_causal:
                    allowed = kv_indices.unsqueeze(0) <= q_indices.unsqueeze(1)
                else:
                    allowed = torch.ones(seq_local, kv_end - kv_start, dtype=torch.bool, device=query.device)
                if sliding_window is not None:
                    allowed = allowed & ((q_indices.unsqueeze(1) - kv_indices.unsqueeze(0)) < sliding_window)
                if use_vision_bidirectional:
                    q_groups = vision_group_ids[:, q_indices]
                    kv_groups = vision_group_ids[:, kv_start:kv_end]
                    same_vision_group = (q_groups[:, :, None] == kv_groups[:, None, :]) & (q_groups[:, :, None] >= 0)
                    allowed = allowed.unsqueeze(0) | same_vision_group
                else:
                    allowed = allowed.unsqueeze(0)
                packed_allowed = _packed_sequence_allowed_mask(packed_seq_ids_full, q_indices, kv_indices)
                if packed_allowed is not None:
                    allowed = allowed & packed_allowed
                scores = scores.masked_fill(~allowed[:, None, :, :], float("-inf"))

                chunk_valid = allowed.any(dim=-1)
                chunk_max = scores.amax(dim=-1)
                safe_chunk_max = torch.where(chunk_valid[:, None, :], chunk_max, torch.zeros_like(chunk_max))
                exp_scores = torch.where(
                    allowed[:, None, :, :],
                    torch.exp(scores - safe_chunk_max.unsqueeze(-1)),
                    torch.zeros_like(scores),
                )
                chunk_sum = exp_scores.sum(dim=-1)
                chunk_lse = torch.where(
                    chunk_valid[:, None, :],
                    safe_chunk_max + torch.log(chunk_sum.clamp(min=1e-9)),
                    torch.full_like(safe_chunk_max, float("-inf")),
                )
                new_lse = torch.logaddexp(lse, chunk_lse)
                new_lse_finite = torch.isfinite(new_lse)
                alpha = torch.where(
                    torch.isfinite(lse) & new_lse_finite,
                    torch.exp(lse - new_lse),
                    torch.zeros_like(lse),
                ).unsqueeze(-1)
                beta = torch.where(
                    torch.isfinite(chunk_lse) & new_lse_finite,
                    torch.exp(safe_chunk_max - new_lse),
                    torch.zeros_like(chunk_lse),
                ).unsqueeze(-1)
                out_acc = out_acc * alpha + beta * torch.matmul(exp_scores, v_float[:, :, kv_start:kv_end, :])
                lse = new_lse
            out = out_acc.to(query.dtype)

        if padded_head_dim != orig_head_dim:
            out = out[..., :orig_head_dim]
        return out

    def _pre_hook(module, args, kwargs):
        mm_token_type_ids = kwargs.pop("mm_token_type_ids", None)
        packed_seq_ids = kwargs.pop("_packed_seq_ids", None)
        module._cp_mm_token_type_ids = mm_token_type_ids
        module._cp_packed_seq_ids = packed_seq_ids
        module._cp_manual_allgather_active = mm_token_type_ids is not None or packed_seq_ids is not None
        _active_module["module"] = module
        F_module.scaled_dot_product_attention = _cp_sdpa
        return args, kwargs

    def _post_hook(module, inputs, output):
        module._cp_mm_token_type_ids = None
        module._cp_packed_seq_ids = None
        module._cp_manual_allgather_active = False
        _active_module["module"] = None
        F_module.scaled_dot_product_attention = _original_sdpa

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    for name, module in model.named_modules():
        if _is_cp_attention_module_name(name):
            # Hook on the inner attention module so the hook fires during both
            # the original forward AND gradient-checkpointing recompute.
            # CheckpointWrapper's recompute bypasses __call__ (and thus pre-hooks
            # on the wrapper itself), so we must hook on the wrapped module directly.
            target = module._checkpoint_wrapped_module if isinstance(module, CheckpointWrapper) else module
            target._cp_uses_sdpa_hook = True
            target.register_forward_pre_hook(_pre_hook, with_kwargs=True)
            # always_call=True ensures _original_sdpa is restored even if the forward raises.
            target.register_forward_hook(_post_hook, always_call=True)


def attach_linear_attn_position_hooks(model: torch.nn.Module):
    """Forward pre-hook on decoder layers to pass position_ids to linear_attn.

    HF Qwen3.5 decoder layers don't pass position_ids to linear_attn, but
    CPAwareGatedDeltaNet needs them under CP to undo load-balanced sharding.
    This hook captures position_ids from the decoder layer's kwargs and
    stores it on the linear_attn module so its forward can read it.
    """

    def _decoder_pre_hook(_module, _args, kwargs):
        _module.linear_attn._cached_position_ids = kwargs.get("position_ids", None)
        return None

    for _, mod in model.named_modules():
        if hasattr(mod, "linear_attn") and hasattr(mod, "layer_type"):
            if not getattr(mod, "_linear_attn_pos_hook_registered", False):
                mod.register_forward_pre_hook(_decoder_pre_hook, with_kwargs=True)
                mod._linear_attn_pos_hook_registered = True


def _pad_tensor_seq_dim_(tensor: torch.Tensor, seq_dim: int, pad_len: int, value: float | int = 0) -> torch.Tensor:
    if pad_len <= 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[seq_dim] = pad_len
    pad = torch.full(pad_shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, pad), dim=seq_dim)


def _pad_position_ids_seq_dim_(position_ids: torch.Tensor, seq_dim: int, pad_len: int) -> torch.Tensor:
    if pad_len <= 0:
        return position_ids
    last_position = position_ids.select(seq_dim, position_ids.shape[seq_dim] - 1).unsqueeze(seq_dim)
    increment_shape = [1] * position_ids.ndim
    increment_shape[seq_dim] = pad_len
    increments = torch.arange(1, pad_len + 1, device=position_ids.device, dtype=position_ids.dtype).view(
        increment_shape
    )
    return torch.cat((position_ids, last_position + increments), dim=seq_dim)


def make_cp_batch_and_ctx(
    device_mesh,
    batch,
    loss_mask=None,
    use_te: bool = False,
    padding_token_id: int = 0,
    num_chunks: int = 1,
    seq_lens_padding_value: int = -1000,
):
    """
    Build a CP context manager and shards a batch. If the input device_mesh is None or the size
    of the context_parallel submesh is 1, this function is effectively a no-op.

    Args:
        cp_mesh (DeviceMesh): The device mesh for context parallel.
        batch (Dict[str, torch.Tensor]): The input batch containing (string, torch.Tensor)

    Returns:
        tuple (contextmanager, dict[str, torch.Tensor]): Returns a tuple with a context manager
        and a new batch. The context manager is either nullcontext (no CP) or CP context manager as
        returned by `create_context_parallel_ctx`. The batch has also been passed to
        `create_context_parallel_ctx` and is accordingly sharded.
    """
    from contextlib import nullcontext

    def _get_submesh(device_mesh, name):
        if name in getattr(device_mesh, "mesh_dim_names", {}):
            return device_mesh[name]
        return None

    def _get_mesh_size(mesh):
        if mesh is None:
            return 0
        return mesh.size()

    cp_mesh = _get_submesh(device_mesh, "cp")
    tp_mesh = _get_submesh(device_mesh, "tp")

    if use_te:
        return nullcontext, make_cp_batch_for_te(
            cp_mesh,
            batch,
            padding_token_id=padding_token_id,
            qkv_format="thd",
            num_chunks=num_chunks,
            seq_lens_padding_value=seq_lens_padding_value,
        )

    if _get_mesh_size(cp_mesh) <= 1:
        return nullcontext, batch

    # Gemma4 needs a local-query/global-key attention mask that PyTorch's
    # ring-template CP path cannot represent. Its pre-embed step marks the
    # batch so we use explicit contiguous sequence sharding and let
    # attach_cp_sdpa_hooks all-gather K/V and token types inside attention.
    manual_allgather = (
        bool(batch.pop("_cp_manual_allgather", False)) or "mm_token_type_ids" in batch or "_packed_seq_ids" in batch
    )

    # Remove attention_mask from the batch so the model does not attempt to
    # build a local 4D mask with the wrong key length. Preserve padding
    # semantics for modules such as MoE.
    attention_mask = batch.pop("attention_mask", None)
    if attention_mask is not None and "padding_mask" not in batch:
        if attention_mask.ndim == 4:
            diagonal = torch.diagonal(attention_mask[:, 0], dim1=-2, dim2=-1)
            batch["padding_mask"] = diagonal.logical_not() if attention_mask.dtype == torch.bool else diagonal != 0
        else:
            batch["padding_mask"] = attention_mask.bool().logical_not()

    # Determine the primary sequence tensor: inputs_embeds (VLM with CP, where
    # multimodal token replacement happened pre-shard) or input_ids (standard LLM).
    has_inputs_embeds = "inputs_embeds" in batch
    has_input_ids = "input_ids" in batch
    assert has_inputs_embeds ^ has_input_ids, (
        "make_cp_batch_and_ctx requires exactly one of 'inputs_embeds' or 'input_ids' in batch"
    )
    primary_key = "inputs_embeds" if has_inputs_embeds else "input_ids"
    primary_seq_tensor = batch[primary_key]
    seq_len = primary_seq_tensor.shape[1]

    # Skip 1D injection if position_ids already in batch (e.g. mRoPE pre-computed).
    if "position_ids" not in batch and (_get_mesh_size(cp_mesh) > 1 or _get_mesh_size(tp_mesh) > 1):
        batch["position_ids"] = torch.arange(0, seq_len).unsqueeze(0).to(primary_seq_tensor.device)

    position_ids = batch["position_ids"]
    pos_seq_dim = 2 if position_ids.ndim == 3 else 1

    labels = batch.get("labels")
    if labels is None and loss_mask is not None:
        labels = loss_mask
        loss_mask = None
    if labels is None:
        raise KeyError("Context parallelism requires `labels` in the batch, or labels passed as `loss_mask`.")

    if manual_allgather:
        cp_size = cp_mesh.size()
        pad_len = (-seq_len) % (2 * cp_size)
        if pad_len:
            if "input_ids" in batch:
                batch["input_ids"] = _pad_tensor_seq_dim_(batch["input_ids"], 1, pad_len, padding_token_id)
            if "inputs_embeds" in batch:
                batch["inputs_embeds"] = _pad_tensor_seq_dim_(batch["inputs_embeds"], 1, pad_len, 0)
            labels = _pad_tensor_seq_dim_(labels, 1, pad_len, -100)
            position_ids = _pad_position_ids_seq_dim_(position_ids, pos_seq_dim, pad_len)
            batch["position_ids"] = position_ids
            if "mm_token_type_ids" in batch:
                batch["mm_token_type_ids"] = _pad_tensor_seq_dim_(batch["mm_token_type_ids"], 1, pad_len, 0)
            if "_packed_seq_ids" in batch:
                batch["_packed_seq_ids"] = _pad_tensor_seq_dim_(batch["_packed_seq_ids"], 1, pad_len, 0)
            if "per_layer_inputs" in batch:
                batch["per_layer_inputs"] = _pad_tensor_seq_dim_(batch["per_layer_inputs"], 1, pad_len, 0)
            if loss_mask is not None:
                loss_mask = _pad_tensor_seq_dim_(loss_mask, 1, pad_len, 0)
            if "padding_mask" in batch:
                batch["padding_mask"] = _pad_tensor_seq_dim_(batch["padding_mask"], 1, pad_len, True)

        # Manual sequence slicing. Every CP rank in the same CP group starts
        # from the same full batch, then keeps one contiguous sequence shard.
        batch["labels"] = labels
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            cp_rank = torch.distributed.get_rank(group=cp_mesh.get_group())
        else:
            cp_rank = getattr(cp_mesh, "get_local_rank", lambda: 0)()

        seq_len = batch[primary_key].shape[1]
        if seq_len % cp_size != 0:
            raise ValueError(f"CP sequence length must be divisible by cp_size after padding, got {seq_len=} {cp_size=}")
        local_seq_len = seq_len // cp_size
        seq_start = cp_rank * local_seq_len
        seq_end = seq_start + local_seq_len

        def _slice_seq(key: str, seq_dim: int = 1) -> None:
            if key not in batch:
                return
            slices = [slice(None)] * batch[key].ndim
            slices[seq_dim] = slice(seq_start, seq_end)
            batch[key] = batch[key][tuple(slices)].contiguous()

        _slice_seq("input_ids", 1)
        _slice_seq("inputs_embeds", 1)
        _slice_seq("labels", 1)
        _slice_seq("position_ids", pos_seq_dim)
        _slice_seq("mm_token_type_ids", 1)
        _slice_seq("_packed_seq_ids", 1)
        _slice_seq("per_layer_inputs", 1)
        _slice_seq("padding_mask", 1)
        if loss_mask is not None:
            batch["loss_mask"] = loss_mask[:, seq_start:seq_end].contiguous()

        return nullcontext, batch

    # Generic non-TE CP path: keep PyTorch context_parallel's load-balanced
    # all-gather implementation from main.
    cp_buffers = [primary_seq_tensor, labels, position_ids]
    cp_seq_dims = [1, 1, pos_seq_dim]
    cp_no_restore_buffers = {primary_seq_tensor, labels}
    batch_buffer_keys: dict[int, str] = {0: primary_key, 1: "labels", 2: "position_ids"}

    if loss_mask is not None:
        cp_buffers.append(loss_mask)
        cp_seq_dims.append(1)
        cp_no_restore_buffers.add(loss_mask)

    if "padding_mask" in batch:
        batch_buffer_keys[len(cp_buffers)] = "padding_mask"
        cp_buffers.append(batch["padding_mask"])
        cp_seq_dims.append(1)
        cp_no_restore_buffers.add(batch["padding_mask"])

    PAD_FILL = {
        "labels": -100,
        "padding_mask": True,
        "attention_mask": False,
    }
    cp_divisor = cp_mesh.size() * 2
    if seq_len % cp_divisor != 0:
        pad_len = cp_divisor - (seq_len % cp_divisor)
        new_no_restore = set()
        for i, (buf, dim) in enumerate(zip(cp_buffers, cp_seq_dims)):
            pad_shape = list(buf.shape)
            pad_shape[dim] = pad_len
            if buf.dtype.is_floating_point:
                pad_val = torch.zeros(pad_shape, dtype=buf.dtype, device=buf.device)
            else:
                fill_val = PAD_FILL.get(batch_buffer_keys.get(i), 0)
                pad_val = torch.full(pad_shape, fill_val, dtype=buf.dtype, device=buf.device)
            old_buf = buf
            cp_buffers[i] = torch.cat([buf, pad_val], dim=dim)
            if old_buf in cp_no_restore_buffers:
                new_no_restore.add(cp_buffers[i])
        cp_no_restore_buffers = new_no_restore
        for idx, key in batch_buffer_keys.items():
            batch[key] = cp_buffers[idx]

    cp_ctx = create_context_parallel_ctx(
        cp_mesh=cp_mesh,
        cp_buffers=cp_buffers,
        cp_seq_dims=cp_seq_dims,
        cp_no_restore_buffers=cp_no_restore_buffers,
        cp_rotate_method="allgather",  # TODO: expose through cfg
    )
    enable_loss_parallel: bool = False
    enable_compiled_autograd: bool = False
    return get_train_context(enable_loss_parallel, enable_compiled_autograd, cp_ctx), batch


def make_cp_batch_for_te(
    cp_mesh,
    batch,
    qkv_format="thd",
    padding_token_id: int = 0,
    num_chunks: int = 1,
    seq_lens_padding_value: int = -1000,
):
    """
    Build a CP batch for Transformer Engine using THD format.

    This function converts BSHD format batches to THD format and shards them across
    context parallel ranks for use with Transformer Engine. It processes the batch
    in chunks if num_chunks > 1, allowing for better memory efficiency with large
    sequences.

    The function performs three main steps:
    1. Converts BSHD format to THD format using split_batch_into_thd_chunks
    2. Optionally splits the batch into multiple chunks for memory efficiency
    3. Shards each chunk across CP ranks using Transformer Engine's partitioning

    Args:
        cp_mesh (DeviceMesh or None): The device mesh for context parallel. If None or
            size <= 1, returns the batch in THD format without sharding.
        batch (Dict[str, torch.Tensor]): The input batch in BSHD format containing:
            - input_ids: Input token IDs [batch_size, seq_len] or [batch_size, seq_len, hidden_dim]
            - labels: Label token IDs [batch_size, seq_len]
            - position_ids (optional): Position IDs [batch_size, seq_len]
            - seq_lens: Actual sequence lengths [batch_size, num_packs]
            - seq_lens_padded: Padded sequence lengths [batch_size, num_packs]
        qkv_format (str): Format for QKV tensors. Currently only "thd" is supported.
        padding_token_id (int): Token ID used for padding in input_ids (default: 0)
        num_chunks (int): Number of chunks to split the batch into. If > 1, the batch
            dimension is split and each chunk is processed separately (default: 1)
        seq_lens_padding_value (int): Sentinel value used to indicate padding in
            seq_lens/seq_lens_padded tensors (default: -1000)

    Returns:
        dict: Processed batch in THD format with the following keys:
            - input_ids: Sharded input token IDs [total_tokens] or [num_chunks, chunk_tokens]
            - labels: Sharded labels [total_tokens] or [num_chunks, chunk_tokens]
            - position_ids: Generated and sharded position IDs [total_tokens] or [num_chunks, chunk_tokens]
            - cu_seqlens: Cumulative sequence lengths [num_seqs+1] or [num_chunks, max_seqs+1]
            - cu_seqlens_padded: Cumulative padded sequence lengths [num_seqs+1] or [num_chunks, max_seqs+1]
            - max_seqlen: Maximum sequence length (int32 tensor)
            - qkv_format: Format string ("thd")
            - padding_mask: Boolean mask indicating padding tokens

    Raises:
        ValueError: If qkv_format is not "thd"
        KeyError: If required fields (seq_lens, seq_lens_padded) are missing from batch

    Example:
        >>> # Single chunk, no CP
        >>> batch = {
        ...     'input_ids': torch.tensor([[1, 2, 3, 4]]),
        ...     'labels': torch.tensor([[2, 3, 4, 5]]),
        ...     'seq_lens': torch.tensor([[4]]),
        ...     'seq_lens_padded': torch.tensor([[4]])
        ... }
        >>> result = make_cp_batch_for_te(None, batch)
        >>> result['input_ids'].shape  # [4] in THD format
        torch.Size([4])

        >>> # Multiple chunks with CP
        >>> batch = {
        ...     'input_ids': torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        ...     'labels': torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]]),
        ...     'seq_lens': torch.tensor([[4], [4]]),
        ...     'seq_lens_padded': torch.tensor([[4], [4]])
        ... }
        >>> result = make_cp_batch_for_te(cp_mesh, batch, num_chunks=2)
        >>> result['input_ids'].shape  # [2, chunk_tokens] - 2 chunks
        torch.Size([2, 2])  # Example: 2 chunks, 2 tokens each after sharding
    """
    if qkv_format != "thd":
        raise ValueError(f"Currently only 'thd' format is supported, got: {qkv_format}")

    batch = split_batch_into_thd_chunks(
        batch, num_chunks=num_chunks, seq_lens_padding_value=seq_lens_padding_value, padding_token_id=padding_token_id
    )

    if cp_mesh is None or cp_mesh.size() <= 1:
        return batch

    if num_chunks <= 1:
        return _shard_thd_chunk_for_te(batch, cp_mesh, qkv_format, seq_lens_padding_value, padding_token_id)

    # Extract each chunk from the batched result and shard it
    chunks = []
    for i in range(num_chunks):
        chunk_batch = {k: v[i] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        chunks.append(
            _shard_thd_chunk_for_te(chunk_batch, cp_mesh, qkv_format, seq_lens_padding_value, padding_token_id)
        )

    return_dict = {
        "input_ids": torch.stack([chunk["input_ids"] for chunk in chunks]),
        "labels": torch.stack([chunk["labels"] for chunk in chunks]),
        "position_ids": torch.stack([chunk["position_ids"] for chunk in chunks]),
        "cu_seqlens": torch.stack([chunk["cu_seqlens"] for chunk in chunks]),
        "max_seqlen": torch.stack([chunk["max_seqlen"] for chunk in chunks]),
        "qkv_format": qkv_format,
        "padding_mask": torch.stack([chunk["padding_mask"] for chunk in chunks]),
        "cp_size": cp_mesh.size() if cp_mesh is not None else 1,
        "cp_rank": torch.distributed.get_rank(group=cp_mesh.get_group()) if cp_mesh is not None else 0,
    }

    return return_dict


def _shard_thd_chunk_for_te(
    batch,
    cp_mesh,
    qkv_format,
    seq_lens_padding_value,
    padding_token_id,
):
    import transformer_engine_torch as tex

    cu_seqlens = batch.get("cu_seqlens", None)
    cu_seqlens_padded = batch.get("cu_seqlens_padded", batch["cu_seqlens"])
    filtered_cu_seqlens_padded = cu_seqlens_padded[cu_seqlens_padded != seq_lens_padding_value]

    # Check for required fields - BSHD format is not supported
    if cu_seqlens is None or cu_seqlens_padded is None:
        raise ValueError(
            "BSHD format is not supported. Both 'cu_seqlens' and 'cu_seqlens_padded' must be present in the batch. "
            "Please use packed sequence format with cu_seqlens and cu_seqlens_padded."
        )

    cp_size = cp_mesh.size()

    cp_rank = torch.distributed.get_rank(group=cp_mesh.get_group()) if cp_mesh is not None else 0

    # Handle all mask keys that may be present in the batch
    mask_keys = ["input_ids", "labels", "position_ids", "padding_mask"]

    for key in mask_keys:
        if key in batch:
            val = batch[key]
            index = tex.thd_get_partitioned_indices(filtered_cu_seqlens_padded, val.size(0), cp_size, cp_rank)
            val = val.index_select(0, index)
            batch[key] = val

    max_seqlen = (filtered_cu_seqlens_padded[1:] - filtered_cu_seqlens_padded[:-1]).max().item()
    output_batch = {
        "input_ids": batch["input_ids"].to(torch.int64).contiguous(),
        "labels": batch["labels"].to(torch.int64).contiguous(),
        "position_ids": batch["position_ids"].to(torch.int64).contiguous(),
        "cu_seqlens": cu_seqlens_padded.to(torch.int32).contiguous(),
        "max_seqlen": torch.tensor(max_seqlen).to(torch.int32).to(device=cu_seqlens_padded.device),
        "qkv_format": qkv_format,
        "padding_mask": (batch["input_ids"] == padding_token_id).bool().contiguous(),
        "cp_size": cp_size,
        "cp_rank": cp_rank,
    }

    return output_batch
