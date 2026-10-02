# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""
Transformer Engine compatibility patches.

Runtime monkey-patches applied directly to TE classes in memory so they
take effect immediately in the current process.

Call `apply_te_patches()` early in the process, before TE attention or optimizers
are used.
"""

from __future__ import annotations

import logging

import torch

_logger = logging.getLogger(__name__)

_TE_PATCHES_APPLIED = False


def _apply_fused_adam_quantized_tensor_patch() -> None:
    """Patch FusedAdam._initialize_state to handle QuantizedTensor params.

    TE's FusedAdam uses ``torch.zeros(param.shape, ...)`` /
    ``torch.empty(param.shape, ...)`` in ``_initialize_state``, which fails for
    QuantizedTensor parameters because their ``.shape`` does not carry the
    correct metadata for allocation.  The fix dequantizes the param first and
    uses ``torch.zeros_like`` / ``torch.empty_like`` instead.

    The fix was merged upstream in TE 2.12 via
    https://github.com/NVIDIA/TransformerEngine/pull/2535.
    """
    from nemo_automodel.shared.import_utils import is_te_min_version

    if is_te_min_version("2.12"):
        _logger.debug("TE >= 2.12 detected; FusedAdam QuantizedTensor patch not needed.")
        return

    try:
        import inspect

        from transformer_engine.pytorch.optimizers.fused_adam import FusedAdam
        from transformer_engine.pytorch.quantized_tensor import QuantizedTensor
    except ImportError as e:
        _logger.debug(f"Skipping FusedAdam QuantizedTensor patch (import failed): {e}")
        return

    # Skip if upstream already contains the full fix from
    # https://github.com/NVIDIA/TransformerEngine/pull/2535
    _src = inspect.getsource(FusedAdam._initialize_state)
    _upstream_fix_lines = (
        "param.dequantize() if isinstance(param, QuantizedTensor)",
        "torch.zeros_like(param_for_empty",
        "torch.empty_like(param_for_empty",
    )
    if all(line in _src for line in _upstream_fix_lines):
        _logger.debug("FusedAdam._initialize_state already contains upstream QuantizedTensor fix, skipping patch.")
        return

    def _patched_initialize_state(self, param, state_name, zero_buffer, store_param_remainders=False):
        """Initialize one of the optimizer states according to `state_name`.

        Arguments:
            param (torch.nn.Parameter): One of parameters in this optimizer.
            state_name (string): Name of optimizer states, can be one of 'exp_avg', 'exp_avg_sq',
                and 'master_param`.
            zero_buffer (bool): Whether to initialize the optimizer state with zeros.
            store_param_remainders (bool): Store only trailing remainder bits.
        """
        dtype = self.name_to_dtype_map[state_name]
        # Handle QuantizedTensor by dequantizing first
        param_for_empty = param.dequantize() if isinstance(param, QuantizedTensor) else param
        if store_param_remainders:
            data = torch.zeros_like(param_for_empty, dtype=torch.int16)
        else:
            data = torch.empty_like(param_for_empty, dtype=dtype)
        if zero_buffer:
            data.zero_()

        if dtype == torch.uint8:
            import transformer_engine_torch as tex
            from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

            quantizer = Float8Quantizer(
                scale=torch.ones([1], dtype=torch.float32, device=param.device),
                amax=torch.zeros([1], dtype=torch.float32, device=param.device),
                fp8_dtype=tex.DType.kFloat8E4M3,
            )
            self.state[param][state_name] = quantizer.make_empty(param.shape)
            self.state[param][state_name].quantize_(data.float())
        else:
            self.state[param][state_name] = data

        # Create scale if necessary.
        if dtype != torch.float32:
            if param not in self._scales:
                self._scales[param] = {}
            self._scales[param][state_name] = torch.ones([1], dtype=torch.float32, device=param.device)

    FusedAdam._initialize_state = _patched_initialize_state
    _logger.info("Applied FusedAdam QuantizedTensor monkey-patch.")


def _apply_fused_adam_empty_shard_patch() -> None:
    """Keep zero-element FSDP shards out of TE's native multi-tensor kernel.

    TE's multi_tensor_apply metadata writer can overrun its final slot when a
    zero-element tensor occupies that slot (TE issue #3202).  Filter complete
    parameter/state columns at the Python/native boundary so state_dict keys,
    parameter groups, and the update for every nonempty shard stay intact.
    """
    from nemo_automodel.shared.import_utils import safe_import, safe_import_te

    have_te, _ = safe_import_te()
    available, fused_adam = (
        safe_import("transformer_engine.pytorch.optimizers.fused_adam") if have_te else (False, None)
    )
    if not available:
        _logger.debug("Skipping FusedAdam empty-shard patch: Transformer Engine is unavailable")
        return

    original = fused_adam.multi_tensor_applier
    if getattr(original, "_nemo_empty_shard_patch", False) is True:
        return

    reported_empty_shards = False

    def _apply_without_empty_shards(op, noop_flag_buffer, tensor_lists, *args):
        """Forward aligned nonempty shard columns to TE's existing applier.

        Args:
            op: Native multi-tensor optimizer kernel.
            noop_flag_buffer: Device integer tensor of shape [1], passed unchanged.
            tensor_lists: Role-major lists [role][parameter]. Each entry is a tensor
                of arbitrary shape or a DTensor whose local shard has that shape.
                Roles start with gradients and parameters, followed by optimizer
                state; corresponding columns describe the same local parameter.
                Tensor storage is preserved and may be updated in place by op.
            *args: Native kernel arguments, forwarded unchanged.

        Returns:
            The original applier's result, or None when every local shard is empty.
        """
        nonlocal reported_empty_shards
        if not tensor_lists or not tensor_lists[0]:
            return original(op, noop_flag_buffer, tensor_lists, *args)

        width = len(tensor_lists[0])
        if any(len(tensors) != width for tensors in tensor_lists):
            raise RuntimeError("FusedAdam multi-tensor lists have different lengths")

        def local_numel(tensor):
            """Return the element count of a tensor's local shard.

            Args:
                tensor: Tensor of arbitrary shape, or DTensor with an arbitrary
                    global shape and possibly empty local shard.

            Returns:
                Number of local elements, independent of the global DTensor shape.
            """
            # TE's original applier unwraps DTensor immediately before the
            # native call.  Check the same local shard here.
            return tensor._local_tensor.numel() if hasattr(tensor, "_local_tensor") else tensor.numel()

        nonempty = [index for index, grad in enumerate(tensor_lists[0]) if local_numel(grad)]
        if len(nonempty) == width:
            return original(op, noop_flag_buffer, tensor_lists, *args)

        # Each column is one parameter, gradient, and optimizer state tuple.
        # A zero gradient with a nonempty parameter would indicate an unrelated
        # error; do not silently omit its update.
        for index in set(range(width)) - set(nonempty):
            if local_numel(tensor_lists[1][index]):
                raise RuntimeError("FusedAdam has an empty gradient for a nonempty parameter shard")

        if not reported_empty_shards:
            _logger.info("FusedAdam filtered %d empty local shards from native update", width - len(nonempty))
            reported_empty_shards = True
        if not nonempty:
            return None
        filtered = [[tensors[index] for index in nonempty] for tensors in tensor_lists]
        return original(op, noop_flag_buffer, filtered, *args)

    _apply_without_empty_shards._nemo_empty_shard_patch = True
    fused_adam.multi_tensor_applier = _apply_without_empty_shards
    _logger.info("Applied FusedAdam empty-shard monkey-patch.")


def _thd_rank_order_indices(
    tokens: int, cu_seqlens: torch.Tensor, cp_size: int, seq_chunk_ids: torch.Tensor | None = None
) -> torch.Tensor:
    """Map chronological tokens to TE's rank-major dual-chunk layout on device.

    Args:
        tokens: Global padded token count.
        cu_seqlens: Integer tensor of shape [documents + 1] on the input device,
            containing global padded document boundaries from zero to tokens.
            Each document length must be divisible by 2 * cp_size, as required
            by TE's THD all-to-all path. Empty documents are allowed.
        cp_size: Number of context-parallel ranks.
        seq_chunk_ids: Optional integer tensor of shape [2 * cp_size] on the
            same device, mapping chronological chunks to TE's rank-major chunks.
            None uses TE's standard front/back dual-chunk assignment.

    Returns:
        Long tensor of shape [tokens], indexing rank-major storage in
        chronological order. Indices form a permutation for TE's chunk mapping.
    """
    positions = torch.arange(tokens, device=cu_seqlens.device, dtype=torch.long)
    documents = torch.bucketize(positions, cu_seqlens[1:], right=True)
    starts = cu_seqlens[documents].long()
    half_lengths = (cu_seqlens[documents + 1].long() - starts) // (2 * cp_size)
    offsets = positions - starts
    chunks = offsets // half_lengths
    ranks = (
        torch.where(chunks < cp_size, chunks, 2 * cp_size - 1 - chunks)
        if seq_chunk_ids is None
        else seq_chunk_ids[chunks].long() // 2
    )
    return (
        ranks * (tokens // cp_size)
        + starts // cp_size
        + torch.where(chunks < cp_size, 0, half_lengths)
        + offsets % half_lengths
    )


def _reorder_thd_after_a2a(
    x: torch.Tensor, cu_seqlens: torch.Tensor, seq_chunk_ids: torch.Tensor, cp_size: int, seq_dim: int = 0
) -> torch.Tensor:
    """Restore chronological document order after TE's all-to-all.

    Args:
        x: Tensor with arbitrary dimensions and tokens on seq_dim, usually
            [tokens, local_heads, head_dim]. Contains all CP ranks' tokens for
            this rank's head shard, concatenated in rank-major dual-chunk order.
        cu_seqlens: Integer tensor of shape [documents + 1] on x.device, with
            padded global boundaries; document lengths are divisible by 2 * cp_size.
        seq_chunk_ids: Integer tensor of shape [2 * cp_size] on x.device,
            produced by TE's get_seq_chunk_ids_for_reordering_before_attn.
        cp_size: Number of context-parallel ranks.
        seq_dim: Token axis, preserved along with all other dimensions.

    Returns:
        Tensor with x's shape and dtype, in chronological packed document order.
        Does not mutate or alias x.
    """
    indices = _thd_rank_order_indices(x.shape[seq_dim], cu_seqlens, cp_size, seq_chunk_ids)
    return x.index_select(seq_dim, indices)


def _reorder_thd_before_a2a(x: torch.Tensor, cu_seqlens: torch.Tensor, cp_size: int, seq_dim: int = 0) -> torch.Tensor:
    """Restore TE's rank-major dual-chunk order before the inverse all-to-all.

    Args:
        x: Tensor with arbitrary dimensions and tokens on seq_dim, usually
            [tokens, local_heads, head_dim]. Contains global tokens for this
            rank's head shard in chronological packed document order.
        cu_seqlens: Integer tensor of shape [documents + 1] on x.device, with
            padded global boundaries; document lengths are divisible by 2 * cp_size.
        cp_size: Number of context-parallel ranks.
        seq_dim: Token axis, preserved along with all other dimensions.

    Returns:
        Tensor with x's shape and dtype, in rank-major dual-chunk order.
        Does not mutate or alias x.
    """
    indices = _thd_rank_order_indices(x.shape[seq_dim], cu_seqlens, cp_size)
    positions = torch.arange(x.shape[seq_dim], device=x.device, dtype=torch.long)
    # This is a permutation: every destination is written exactly once.
    inverse = torch.empty_like(indices).scatter_(0, indices, positions)
    return x.index_select(seq_dim, inverse)


def _apply_thd_a2a_reorder_patch() -> None:
    """Avoid per-document CUDA scalar readbacks in TE 2.16/2.17 THD CP.

    Only the two local permutations change; TE owns attention, communication
    and backward. Newer TE uses native CUDA reordering and is left untouched.
    Remove this compatibility patch once TE >= 2.18 is the runtime minimum
    across supported containers.
    """
    from nemo_automodel.shared.import_utils import is_te_min_version, safe_import, safe_import_te

    have_te, _ = safe_import_te()
    if not have_te or not is_te_min_version("2.16") or is_te_min_version("2.18"):
        return
    available, cp = safe_import("transformer_engine.pytorch.attention.dot_product_attention.context_parallel")
    if not available:
        return
    # Also preserve native kernels backported into a vendor build of older TE.
    if hasattr(cp, "thd_cp_rank_order_to_sequence_order"):
        return
    after = getattr(cp, "reorder_seq_chunks_after_a2a_before_attn_thd", None)
    before = getattr(cp, "reorder_seq_chunks_before_a2a_after_attn_thd", None)
    if after is _reorder_thd_after_a2a and before is _reorder_thd_before_a2a:
        return
    if not callable(after) or not callable(before):
        _logger.warning("Skipping TE THD all-to-all reorder patch: expected TE 2.16/2.17 interface is unavailable")
        return
    cp.reorder_seq_chunks_after_a2a_before_attn_thd = _reorder_thd_after_a2a
    cp.reorder_seq_chunks_before_a2a_after_attn_thd = _reorder_thd_before_a2a
    _logger.info("Applied GPU-vectorized TE THD all-to-all reorder patch.")


def apply_te_patches() -> None:
    """Apply all Transformer Engine runtime patches.

    This function is idempotent and safe to call multiple times.
    """
    global _TE_PATCHES_APPLIED
    if _TE_PATCHES_APPLIED:
        return

    _apply_fused_adam_quantized_tensor_patch()
    _apply_fused_adam_empty_shard_patch()
    _apply_thd_a2a_reorder_patch()

    _TE_PATCHES_APPLIED = True
