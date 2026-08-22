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

"""State-dict conversion for Qwen4-Exp and its owner-sharded Engram table.

The dense, grouped-MoE, shared-expert, and GatedDeltaNet parameters use the
same checkpoint layouts as Qwen3.5-MoE.  Qwen4-Exp's PLE table is different:
the checkpoint stores 128 physical tensors while the native module registers
one rank-local owner shard.  This adapter exposes only the complete physical
checkpoint shards owned by the current rank as views of that local parameter.
The checkpoint reader therefore writes directly into final model storage and
the 51.2-billion-parameter global table is never concatenated or materialized.

The released checkpoint also contains an MTP predictor.  The current
Qwen4-Exp target is ordinary full SFT and intentionally does not construct
MTP, so those checkpoint keys are explicitly ignored in both directions.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5_moe.state_dict_adapter import Qwen3_5MoeStateDictAdapter
from nemo_automodel.components.models.qwen4_exp.config import Qwen4ExpTextConfig
from nemo_automodel.components.models.qwen4_exp.engram import Qwen4ExpOwnerShardedEmbedding
from nemo_automodel.components.moe import state_dict_utils
from nemo_automodel.components.moe.layers import MoEConfig


def _same_tensor_region(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two tensors describe the same writable storage region.

    Args:
        left: Expected checkpoint destination with shape
            ``[checkpoint_shard_rows, embedding_dim]``.
        right: Candidate tensor with shape
            ``[checkpoint_shard_rows, embedding_dim]``.

    Returns:
        ``True`` when shape, stride, storage offset, dtype, device, and backing
        storage all match.  Object identity is accepted as a fast path.
    """
    if left is right:
        return True
    if (
        left.shape != right.shape
        or left.stride() != right.stride()
        or left.storage_offset() != right.storage_offset()
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        return False
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


class Qwen4ExpStateDictAdapter(Qwen3_5MoeStateDictAdapter):
    """Convert Qwen4-Exp checkpoints without gathering the global PLE table.

    Args:
        config: Qwen4-Exp text configuration.  Exactly one one-based PLE layer
            ID is required by the released architecture.
        moe_config: Native grouped-MoE configuration.
        backend: Native backend configuration used by the model.
        engram_table: Rank-local owner-sharded PLE table.  Its global row range
            determines which physical checkpoint shards this rank exposes.
        dtype: Native parameter dtype used by inherited expert conversion.
        pretrained_model_name_or_path: Optional path used by inherited
            checkpoint-layout discovery.

    Tensor layout:
        The native PLE weight is
        ``[global_row_end - global_row_start, embedding_dim]``.  Each emitted
        checkpoint view is
        ``[num_embeddings / split_ngram_parts, embedding_dim]``.  Grouped
        expert tensors retain the inherited Qwen3.5-MoE transposes between HF
        ``[experts, output, input]`` and native ``[experts, input, output]``.
    """

    # For a valid Qwen4-Exp model state, dense tensors are identities, grouped
    # experts are transpose views, intrinsic GDN state is already fp32, and PLE
    # shards are narrow views.  Checkpointer separately disables its
    # write-through fast path for dequantizing base-checkpoint loads, whose FP8
    # expert conversion necessarily allocates.
    _supports_write_through_checkpoint_load = True

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        moe_config: MoEConfig,
        backend: BackendConfig,
        engram_table: Qwen4ExpOwnerShardedEmbedding,
        dtype: torch.dtype = torch.bfloat16,
        pretrained_model_name_or_path: str | None = None,
    ) -> None:
        super().__init__(
            config=config,
            moe_config=moe_config,
            backend=backend,
            dtype=dtype,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            mtp_expert_hf_layout="grouped",
            text_only=False,
        )
        if len(config.ple_layer_ids) != 1:
            raise ValueError(
                "Qwen4-Exp checkpoint conversion requires exactly one PLE layer; "
                f"got ple_layer_ids={config.ple_layer_ids}"
            )
        if config.split_ngram_parts <= 0:
            raise ValueError(f"split_ngram_parts must be positive, got {config.split_ngram_parts}")

        self.engram_table = engram_table
        self.split_ngram_parts = int(config.split_ngram_parts)
        self._ple_layer_idx = int(config.ple_layer_ids[0]) - 1
        self._table_native_key = (
            f"model.language_model.layers.{self._ple_layer_idx}.ple.ple_embedding.ngram_embedding.weight"
        )
        self._table_hf_prefix = self._table_native_key.removesuffix(".weight")
        self._table_hf_key_pattern = re.compile(rf"{re.escape(self._table_hf_prefix)}\.shard_(\d+)\.weight")

        if engram_table.num_embeddings % self.split_ngram_parts != 0:
            raise ValueError(
                "The global Engram row count must divide evenly over physical checkpoint shards: "
                f"{engram_table.num_embeddings} % {self.split_ngram_parts} != 0"
            )
        self._rows_per_checkpoint_shard = engram_table.num_embeddings // self.split_ngram_parts
        start = int(engram_table.global_row_start)
        end = int(engram_table.global_row_end)
        if not 0 <= start < end <= engram_table.num_embeddings:
            raise ValueError(
                f"Invalid rank-local Engram global row range: [{start}, {end}) for {engram_table.num_embeddings} rows"
            )
        if start % self._rows_per_checkpoint_shard or end % self._rows_per_checkpoint_shard:
            raise ValueError(
                "The owner-sharded Engram row range must align to complete physical checkpoint shards; "
                f"got [{start}, {end}) with {self._rows_per_checkpoint_shard} rows per shard. "
                "Use an owner group whose size divides split_ngram_parts."
            )

        self._first_local_checkpoint_shard = start // self._rows_per_checkpoint_shard
        self._end_local_checkpoint_shard = end // self._rows_per_checkpoint_shard
        self._pending_table_aliases: dict[str, torch.Tensor] = {}
        self._view_loaded_native_keys: set[str] = set()

    @property
    def view_loaded_native_keys(self) -> set[str]:
        """Return native parameters already populated through checkpoint views."""
        return set(self._view_loaded_native_keys)

    def get_hf_state_dict_keys(self, state_dict: dict[str, Any]) -> list[str]:
        """Return the rank-independent global HF key set without gathering PLE.

        The model constructs the Engram table in its final owner-sharded layout,
        before ``apply_model_infrastructure`` snapshots the pre-shard HF keys.
        Consequently, deriving keys through :meth:`to_hf` would expose only the
        physical PLE shards owned by the calling rank.  Consolidated checkpoint
        planning requires the same global key list on every rank, so synthesize
        all physical PLE names while retaining the inherited conversion for
        every ordinary tensor.  No table view, copy, or collective is created.

        Args:
            state_dict: Native model state. Its PLE weight is either the
                single-rank local tensor or the globally shaped ``Shard(0)``
                DTensor used by distributed training.

        Returns:
            HF-format keys in native state iteration order, with the one local
            PLE weight replaced by all ``split_ngram_parts`` global shard names.
            Unsupported MTP keys and non-persistent ``_extra_state`` entries are
            omitted just as they are during checkpoint export.
        """
        keys: list[str] = []
        for fqn, tensor in state_dict.items():
            if fqn.startswith("mtp.") or re.match(r".*_extra_state.*", fqn):
                continue
            if fqn == self._table_native_key:
                keys.extend(
                    f"{self._table_hf_prefix}.shard_{shard_idx}.weight" for shard_idx in range(self.split_ngram_parts)
                )
                continue
            keys.extend(
                key
                for key, _ in self.convert_single_tensor_to_hf(
                    fqn,
                    tensor,
                    exclude_key_regex=r".*_extra_state.*",
                    quantization=False,
                )
            )
        return keys

    def _table_checkpoint_views(
        self,
        tensor: Any,
        exclude_key_regex: str | None,
    ) -> list[tuple[str, torch.Tensor]]:
        """Split one local table shard into aliasing physical-checkpoint views.

        Args:
            tensor: Single-rank local table, or a globally shaped ``Shard(0)``
                DTensor whose local shard has shape ``[global_row_end -
                global_row_start, embedding_dim]``.
            exclude_key_regex: Optional regular expression applied to emitted
                checkpoint keys.

        Returns:
            Ordered ``(key, view)`` pairs.  Every view has shape
            ``[rows_per_checkpoint_shard, embedding_dim]`` and aliases
                the local table storage along its first axis.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("The Engram weight must be a Tensor")
        expected_local_shape = (
            int(self.engram_table.global_row_end) - int(self.engram_table.global_row_start),
            self.engram_table.embedding_dim,
        )
        if isinstance(tensor, DTensor):
            expected_global_shape = (self.engram_table.num_embeddings, self.engram_table.embedding_dim)
            if tuple(tensor.shape) != expected_global_shape:
                raise ValueError(
                    f"Native Engram DTensor shape {tuple(tensor.shape)} does not match global table shape "
                    f"{expected_global_shape}"
                )
            if tuple(tensor.placements) != (Shard(0),):
                raise ValueError(f"Native Engram DTensor must use placement Shard(0), got {tensor.placements}")
            if self.engram_table.process_group is None:
                raise RuntimeError("A single-rank reference Engram table must not be a DTensor")
            owner_ranks = tuple(torch.distributed.get_process_group_ranks(self.engram_table.process_group))
            mesh_ranks = tuple(torch.distributed.get_process_group_ranks(tensor.device_mesh.get_group()))
            if owner_ranks != mesh_ranks:
                raise ValueError(
                    "Native Engram DTensor mesh does not match the owner process group: "
                    f"owner={owner_ranks}, mesh={mesh_ranks}"
                )
            local_tensor = tensor.to_local()
        else:
            local_tensor = tensor
        if tuple(local_tensor.shape) != expected_local_shape:
            raise ValueError(
                f"Native Engram local weight shape {tuple(local_tensor.shape)} does not match local owner range "
                f"and embedding dimension {expected_local_shape}"
            )

        result: list[tuple[str, torch.Tensor]] = []
        for shard_idx in range(self._first_local_checkpoint_shard, self._end_local_checkpoint_shard):
            local_row_start = shard_idx * self._rows_per_checkpoint_shard - int(self.engram_table.global_row_start)
            key = f"{self._table_hf_prefix}.shard_{shard_idx}.weight"
            if exclude_key_regex and re.match(exclude_key_regex, key):
                continue
            view = local_tensor.narrow(0, local_row_start, self._rows_per_checkpoint_shard)
            result.append((key, view))
        return result

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert native tensors and expose the local PLE table as views.

        Args:
            state_dict: Native tensors. The PLE entry is either local or a
                globally shaped ``Shard(0)`` DTensor; inherited grouped expert
                entries have shape ``[experts, input, output]``.
            exclude_key_regex: Optional regular expression for checkpoint keys.
            quantization: Whether inherited expert export should use the
                checkpoint's split FP8 representation.
            **kwargs: Additional inherited conversion options.

        Returns:
            HF-layout tensors.  PLE physical shards have shape
            ``[rows_per_checkpoint_shard, embedding_dim]`` and alias the native
            table.  MTP entries are omitted.
        """
        self._pending_table_aliases.clear()
        self._view_loaded_native_keys.clear()
        return super().to_hf(
            state_dict,
            exclude_key_regex=exclude_key_regex,
            quantization=quantization,
            **kwargs,
        )

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Restore native keys after checkpoint data was written into views.

        Args:
            hf_state_dict: HF tensors.  Local PLE entries must be the exact
                ``[rows_per_checkpoint_shard, embedding_dim]`` alias views
                previously returned by :meth:`to_hf`.
            device_mesh: Optional expert-parallel mesh used by inherited MoE
                conversion.
            **kwargs: Additional inherited conversion options.

        Returns:
            Native-layout tensors excluding the table weight, which is already
            populated in place.  ``view_loaded_native_keys`` records that
            ``[local_table_rows, embedding_dim]`` native parameter as loaded.
            MTP checkpoint tensors are omitted.
        """
        self._view_loaded_native_keys.clear()
        filtered_state_dict: dict[str, Any] = {}
        table_entries: dict[str, torch.Tensor] = {}
        for key, value in hf_state_dict.items():
            if key.startswith("mtp."):
                continue
            if self._table_hf_key_pattern.fullmatch(key):
                if not isinstance(value, torch.Tensor) or state_dict_utils.is_dtensor(value):
                    raise TypeError(f"PLE checkpoint destination {key!r} must be a rank-local torch.Tensor view")
                table_entries[key] = value
                continue
            filtered_state_dict[key] = value

        if table_entries:
            expected_keys = set(self._pending_table_aliases)
            actual_keys = set(table_entries)
            if actual_keys != expected_keys:
                missing = sorted(expected_keys - actual_keys)
                unexpected = sorted(actual_keys - expected_keys)
                raise RuntimeError(
                    "PLE checkpoint shards do not match this rank's owner range: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            for key, value in table_entries.items():
                if not _same_tensor_region(self._pending_table_aliases[key], value):
                    raise RuntimeError(
                        f"PLE checkpoint tensor {key!r} is not the write-through view returned by to_hf; "
                        "loading a materialized global table is intentionally unsupported"
                    )
            self._view_loaded_native_keys.add(self._table_native_key)
        elif self._pending_table_aliases:
            raise RuntimeError(
                "No local PLE checkpoint shards were returned after to_hf exposed write-through destinations"
            )

        self._pending_table_aliases.clear()
        return super().from_hf(filtered_state_dict, device_mesh=device_mesh, **kwargs)

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs: Any) -> list[tuple[str, Any]]:
        """Convert one native tensor, specializing the PLE and MTP entries.

        Args:
            fqn: Native fully-qualified tensor name.
            tensor: Native tensor.  The PLE weight is
                ``[local_table_rows, embedding_dim]``; other layouts follow the
                inherited Qwen3.5-MoE contract.
            **kwargs: Inherited conversion options, including
                ``exclude_key_regex`` and ``quantization``.

        Returns:
            Zero entries for ordinary-SFT MTP tensors, multiple aliasing
            ``[rows_per_checkpoint_shard, embedding_dim]`` entries for the PLE
            table, or the inherited conversion result for all other tensors.
        """
        if fqn.startswith("mtp."):
            return []
        if fqn == self._table_native_key:
            views = self._table_checkpoint_views(tensor, kwargs.get("exclude_key_regex"))
            self._pending_table_aliases = dict(views)
            return views
        return super().convert_single_tensor_to_hf(fqn, tensor, **kwargs)
