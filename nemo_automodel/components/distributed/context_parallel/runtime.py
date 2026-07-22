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

"""Setup-time context-parallel backend resolution and per-forward preparation."""

import contextlib
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.context_parallel.magi import MagiState, setup_magi
from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelismSharder,
    CPShardResult,
    CPTokenLayout,
    ShardLayout,
)
from nemo_automodel.components.distributed.context_parallel.utils import _prepare_thd_batch


def _get_submesh(device_mesh: DeviceMesh | None, dim: str) -> DeviceMesh | None:
    """Return a named submesh, or ``None`` when the mesh or axis is absent."""
    if device_mesh is None or dim not in (getattr(device_mesh, "mesh_dim_names", None) or ()):
        return None
    return device_mesh[dim]


def _is_multimodal_model(model: torch.nn.Module | None) -> bool:
    """Return the live model's multimodal-input capability."""
    if model is None:
        return False
    supports = getattr(model, "supports", None)
    if supports is not None and hasattr(supports, "is_multimodal"):
        return bool(supports.is_multimodal)
    config = getattr(model, "config", None)
    if config is not None and any(
        getattr(config, name, None) is not None for name in ("vision_config", "audio_config")
    ):
        return True
    has_language_model = any(getattr(model, name, None) is not None for name in ("language_model", "text_model"))
    has_media_model = any(
        getattr(model, name, None) is not None
        for name in ("visual", "vision_model", "vision_tower", "audio_model", "audio_tower")
    )
    return has_language_model and has_media_model


@dataclass(frozen=True)
class CPForward:
    """Prepared context-parallel state for one model forward.

    Attributes:
        context: Context manager entered around the model forward.
        batch: Prepared batch. Tensor values may have layouts such as
            ``[batch, local_sequence]`` or packed THD ``[local_tokens]``
            according to the selected backend. The input mapping is mutated and
            returned by reference.
        tokens: Bound token-coordinate operations for sharding additional global
            tensors and gathering local tensors produced by this forward.
    """

    context: AbstractContextManager[None]
    batch: dict[str, Any]
    tokens: CPTokenLayout


@dataclass(frozen=True)
class ContextParallelRuntime:
    """Resolved CP backend state shared by all forwards in one recipe runtime.

    Backend intent is resolved once from the model construction config. Runtime
    resources such as the CP mesh and Magi process group are bound here, while
    data-dependent token layouts remain in each returned :class:`CPForward`.
    """

    device_mesh: DeviceMesh | None = None
    _magi: MagiState = field(default_factory=MagiState, repr=False)

    @classmethod
    def build(
        cls,
        model_config: object,
        *,
        device_mesh: DeviceMesh | None,
    ) -> "ContextParallelRuntime":
        """Resolve backend intent from model config and bind CP resources.

        This runs before model construction because the Magi HF attention
        implementation must be registered before the model is instantiated.

        Args:
            model_config: Model construction config containing the resolved
                attention backend.
            device_mesh: Full runtime device mesh containing optional ``cp`` and
                ``tp`` axes.

        Returns:
            Runtime ready to prepare batches for model forwards.
        """
        return cls(
            device_mesh=device_mesh,
            _magi=setup_magi(model_config, device_mesh),
        )

    @property
    def requires_full_logits(self) -> bool:
        """Whether the selected backend is incompatible with fused linear cross entropy."""
        return self._magi.hf_dispatch

    def _resolve_sharder(
        self,
        model_sharder: ContextParallelismSharder | None,
        *,
        is_thd: bool,
        num_chunks: int,
        model: torch.nn.Module | None,
    ) -> ContextParallelismSharder:
        """Resolve model-owned, Magi, THD, generic, or identity strategy."""
        cp_mesh = _get_submesh(self.device_mesh, "cp")
        cp_active = cp_mesh is not None and cp_mesh.size() > 1

        if model_sharder is not None:
            return model_sharder

        if self._magi.enabled:

            def shard_batch_magi(
                cp_mesh: DeviceMesh | None,
                tp_mesh: DeviceMesh | None,
                batch: dict[str, Any],
                *,
                loss_mask: torch.Tensor | None = None,
                padding_token_id: int = 0,
            ) -> CPShardResult:
                """Prepare a full-sequence batch with the bound Magi runtime.

                Args:
                    cp_mesh: One-dimensional context-parallel mesh, or ``None``.
                    tp_mesh: Tensor-parallel mesh; unused because Magi owns its
                        transport.
                    batch: Mapping containing ``input_ids`` of shape ``[batch,
                        sequence]`` and optional token-aligned tensors.
                    loss_mask: Optional tensor of shape ``[batch, sequence]``;
                        unused because Magi prepares labels through ``batch``.
                    padding_token_id: Token ID used for Magi sequence padding.

                Returns:
                    Magi-prepared batch with a null context and captured token
                    layout.
                """
                del tp_mesh, loss_mask
                input_ids = batch.get("input_ids")
                row_shape = tuple(input_ids.shape[:2]) if input_ids is not None and input_ids.dim() >= 2 else None
                prepped, local_indices = self._magi.prepare_batch(
                    batch,
                    padding_token_id=padding_token_id,
                    num_chunks=num_chunks,
                    is_thd=is_thd,
                    model=model,
                )
                layout = None
                if local_indices is not None:
                    padded = local_indices.numel() * max(self._magi.cp_size or 1, 1)
                    original, input_rows = None, None
                    if row_shape is not None:
                        if padded == row_shape[0] * row_shape[1]:
                            input_rows = row_shape
                        elif row_shape[0] == 1 and padded >= row_shape[1]:
                            original = row_shape[1]
                    layout = ShardLayout(
                        local_token_global_indices=local_indices,
                        original_seq_len=original,
                        padded_seq_len=padded,
                        input_row_shape=input_rows,
                    )
                return CPShardResult(contextlib.nullcontext(), prepped, layout)

            return ContextParallelismSharder(shard_batch=shard_batch_magi, local_token_global_indices=None)

        if is_thd:

            def shard_batch_thd(
                cp_mesh: DeviceMesh | None,
                tp_mesh: DeviceMesh | None,
                batch: dict[str, Any],
                *,
                loss_mask: torch.Tensor | None = None,
                padding_token_id: int = 0,
            ) -> CPShardResult:
                """Convert a full-sequence batch to THD and shard its token stream.

                Args:
                    cp_mesh: One-dimensional context-parallel mesh, or ``None``.
                    tp_mesh: Tensor-parallel mesh; unused by THD preparation.
                    batch: Mapping containing ``input_ids``, ``labels``, and
                        ``position_ids`` of shape ``[batch, sequence]`` plus
                        ``seq_lens`` and ``seq_lens_padded`` of shape ``[batch,
                        packed_sequences]``.
                    loss_mask: Optional tensor of shape ``[batch, sequence]``;
                        unused by the THD conversion.
                    padding_token_id: Token ID used for input padding.

                Returns:
                    Packed local THD batch with a null context and captured
                    flat-stream token layout.
                """
                del tp_mesh, loss_mask
                input_ids = batch.get("input_ids")
                row_shape = tuple(input_ids.shape[:2]) if input_ids is not None and input_ids.dim() >= 2 else None
                prepped, local_indices = _prepare_thd_batch(
                    cp_mesh,
                    batch,
                    padding_token_id=padding_token_id,
                    num_chunks=num_chunks,
                )
                layout = None
                if local_indices is not None:
                    layout = ShardLayout(
                        local_token_global_indices=local_indices,
                        padded_seq_len=row_shape[0] * row_shape[1] if row_shape is not None else None,
                        input_row_shape=row_shape,
                    )
                return CPShardResult(contextlib.nullcontext(), prepped, layout)

            return ContextParallelismSharder(shard_batch=shard_batch_thd, local_token_global_indices=None)

        if cp_active:
            return ContextParallelismSharder.sdpa()

        return ContextParallelismSharder.identity()

    def prepare_forward(
        self,
        model: torch.nn.Module | None,
        batch: dict[str, Any],
        *,
        padding_token_id: int = 0,
        num_chunks: int = 1,
        loss_mask: torch.Tensor | None = None,
    ) -> CPForward:
        """Prepare one batch and return its bound CP context and token layout.

        THD input format is inferred from ``batch["qkv_format"]`` rather than
        from the model backend or dataloader configuration. Model-owned CP hooks
        are invoked automatically whenever CP is active or the model owns native
        THD preparation.

        Args:
            model: Model or pipeline part for this forward, or ``None`` when no
                model-owned CP hook applies.
            batch: Full-sequence mapping containing ``input_ids`` or
                ``inputs_embeds`` with shape ``[batch, sequence, ...]``, labels
                of shape ``[batch, sequence]`` when required, and optional
                token-aligned metadata. The mapping is mutated in place.
            padding_token_id: Token ID used when the selected strategy pads
                ``input_ids``.
            num_chunks: Number of THD chunks aligned with pipeline microbatches.
            loss_mask: Optional tensor of shape ``[batch, sequence]`` sharded
                alongside the batch on strategies that consume it.

        Returns:
            Prepared forward state with a context manager, mutated batch, and
            immutable token layout bound to this batch.

        Raises:
            ValueError: If a THD batch omits required sequence metadata.
            NotImplementedError: If multimodal THD context parallelism is requested.
            TypeError: If a model CP hook does not return a sharder.
        """
        is_thd = batch.get("qkv_format") == "thd"
        if is_thd:
            missing = [key for key in ("seq_lens", "seq_lens_padded") if key not in batch]
            if missing:
                raise ValueError(f"THD batch is missing required field(s): {', '.join(missing)}")

        cp_mesh = _get_submesh(self.device_mesh, "cp")
        cp_active = cp_mesh is not None and cp_mesh.size() > 1
        is_multimodal = _is_multimodal_model(model)
        if is_multimodal and is_thd and cp_active:
            raise NotImplementedError(
                "THD packing for multimodal models currently supports cp_size=1 only; "
                "multimodal THD context parallelism is not implemented."
            )

        model_sharder = None
        hook = getattr(model, "prepare_model_inputs_for_cp", None) if model is not None else None
        model_owns_thd = is_thd and bool(getattr(model, "supports_thd", False))
        magi_replaces_hook = self._magi.enabled and not is_multimodal
        if (cp_active or model_owns_thd) and callable(hook) and not magi_replaces_hook:
            model_sharder = hook(batch, num_chunks=num_chunks)
            if not isinstance(model_sharder, ContextParallelismSharder):
                raise TypeError("prepare_model_inputs_for_cp must return ContextParallelismSharder")

        sharder = self._resolve_sharder(
            model_sharder,
            is_thd=is_thd,
            num_chunks=num_chunks,
            model=model,
        )
        tp_mesh = _get_submesh(self.device_mesh, "tp")
        prepared = sharder.shard_batch(
            cp_mesh,
            tp_mesh,
            batch,
            loss_mask=loss_mask,
            padding_token_id=padding_token_id,
        )
        tokens = CPTokenLayout(
            cp_mesh=cp_mesh,
            local_token_global_indices=sharder.local_token_global_indices,
            shard_layout=prepared.layout or ShardLayout(),
        )
        return CPForward(context=prepared.context, batch=prepared.batch, tokens=tokens)
