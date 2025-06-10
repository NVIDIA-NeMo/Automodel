# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# taken and edited from https://github.com/pytorch/pytorch/blob/c8d39a10457ea5d65184c6e8f037f46c5525d869/torch/distributed/checkpoint/_hf_storage.py  # pylint: disable=line-too-long

import dataclasses
import json
import os
import queue
from typing import Optional

import fsspec  # type: ignore[import-untyped]

from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader, FsspecWriter
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    Metadata,
    STORAGE_TYPES,
    StorageMeta,
)
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
)
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future

from nemo_automodel.checkpoint.checkpointing import SerializationFormat
from nemo_automodel.checkpoint.hf_planner import HuggingFaceLoadPlanner, _FqnToFileMapping

__all__ = ["HuggingFaceStorageWriter", "HuggingFaceStorageReader"]

_metadata_fn: str = "model.safetensors.index.json"

FILE_NAME = "model-{cpt_idx}-of-{num_shards}"
SUFFIX = ".safetensors"


class HuggingFaceStorageWriter(FsspecWriter):
    """
    A writer that writes to a huggingface repository in the huggingface format.
    Uses in Fsspec back-end to communicate with the huggingface hub.
    """

    def __init__(
        self,
        path: str,
        fqn_to_index_mapping: dict[str, int],
        token: Optional[str] = None,
    ) -> None:
        """
        Initialize the huggingface writer pointing to path.

        Args:
            path: hf directory where the checkpoint will be written to. Should begin with hf://.
            token: The token to use to authenticate with huggingface hub.
            fqn_to_index_mapping: A mapping from tensor FQN to the index of the file that the
                tensor should be written to. Indices are from 1 to N, where N is the number
                of files.

        """
        from huggingface_hub import HfFileSystem  # type: ignore[import-not-found]

        if HfFileSystem.protocol not in fsspec.available_protocols():
            fsspec.register_implementation(HfFileSystem.protocol, HfFileSystem)

        super().__init__(
            path=path,
            token=token,
            serialization_format=SerializationFormat.SAFETENSORS,
        )
        self._fqn_to_index_mapping: dict[str, int] = fqn_to_index_mapping

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        """
        Enrich the provided ``SavePlan`` with Hugging-Face specific routing
        information.

        The base :class:`torch.distributed.checkpoint._fsspec_filesystem.FsspecWriter`
        converts the planner's generic *plan* into one that is compatible with
        the target filesystem. Hugging Face sharded checkpoints require one
        additional piece of information: a mapping that tells which fully-
        qualified tensor name (FQN) should be written into which shard. This
        method attaches that mapping so that :py:meth:`write_data` can later
        group ``WriteItem`` objects by their target shard.

        Args:
            plan: The local plan produced by the checkpoint *planner*.

        Returns:
            A shallow copy of *plan* whose ``storage_data`` field now contains
            an instance of :class:`_FqnToFileMapping` wrapping
            ``self._fqn_to_index_mapping``.
        """
        plan = super().prepare_local_plan(plan)
        return dataclasses.replace(
            plan, storage_data=_FqnToFileMapping(self._fqn_to_index_mapping)
        )

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        """
        Forward the list of device-local plans unchanged.

        The Hugging Face writer does not need to merge or otherwise transform
        the per-rank plans produced by the planner, therefore this method
        simply returns *plans* untouched.

        Args:
            plans: A list of :class:`SavePlan` objects (one per rank).

        Returns:
            The *plans* argument unmodified.
        """
        return plans

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> Future[list[WriteResult]]:
        """
        Materialise the tensors described in *plan* to ``.safetensors`` files.

        Workflow:
            1. Group the incoming :class:`WriteItem` objects by their target
               shard index as defined by ``plan.storage_data``.
            2. Derive the correct filename for each shard following Hugging
               Face conventions (single-shard -> ``model.safetensors``;
               multi-shard -> ``model-{00000}-of-{00000}.safetensors``).
            3. Hand off the per-shard work to
               :py:meth:`FsspecWriter._write_data` which performs the actual
               I/O in background threads.

        Args:
            plan: The per-rank :class:`SavePlan` that should be written to
                disk.
            planner: The ``SavePlanner`` instance that generated *plan*.

        Returns:
            A :class:`torch.futures.Future` that resolves to a list of
            :class:`WriteResult` objects once all shards have been written.
        """
        if len(plan.items) == 0:
            fut: Future = Future()
            fut.set_result([])
            return fut

        # storage_plan is a map from key to file index
        storage_plan: dict[str, int] = plan.storage_data.fqn_to_file_index_mapping

        buckets = self._split_by_storage_plan(storage_plan, plan.items)
        highest_index = max(storage_plan.values())

        file_queue: queue.Queue = queue.Queue()
        for file_index, write_items in buckets.items():
            # Use simple filename for single shard models
            if highest_index == 1:
                file_name = "model" + SUFFIX
            else:
                file_name = self._gen_file_name(file_index, highest_index)
            file_queue.put(
                (self.fs.concat_path(self.path, file_name), file_name, write_items)
            )

        return super()._write_data(planner, file_queue)

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        """
        Finalise the checkpoint by writing the Hugging Face *weight map*.

        After all shards have been persisted, this method aggregates the
        information contained in the returned :class:`WriteResult` objects,
        computes the total byte size of the checkpoint and, when necessary,
        creates ``model.safetensors.index.json`` which is the file expected by
        the Hugging Face Hub.

        The index file is generated when *either* of the following holds:

        1. The checkpoint is split across multiple shards.
        2. The (single) shard is not named ``model.safetensors``.

        Args:
            metadata: The global :class:`Metadata` object passed in from the
                distributed-checkpoint runtime.
            results: Nested list with the :class:`WriteResult` objects returned
                from :py:meth:`write_data`.
        """
        storage_md = {}
        total_size = 0
        for wr_list in results:
            storage_md.update(
                {wr.index.fqn: wr.storage_data.relative_path for wr in wr_list}
            )
            total_size += sum([wr.storage_data.length for wr in wr_list])
        
        # Only create metadata file for multi-shard models
        if len(
            set(storage_md.values())
        ) > 1 or not any(
            path.endswith("model.safetensors"
        ) for path in storage_md.values()):
            metadata_to_write = {}
            metadata_to_write["metadata"] = {"total_size": total_size}
            metadata_to_write["weight_map"] = storage_md

            metadata_path = self.fs.concat_path(self.path, f"{_metadata_fn}")
            with self.fs.create_stream(metadata_path, "w") as metadata_file:
                json.dump(metadata_to_write, metadata_file, indent=2)

    def _split_by_storage_plan(
        self, storage_plan: dict[str, int], items: list[WriteItem]
    ) -> dict[int, list[WriteItem]]:
        """
        Group ``WriteItem`` instances by their destination shard.

        Args:
            storage_plan: Mapping ``FQN -> shard_index`` (1-based) that was
                created from the reference Hugging Face checkpoint.
            items: List of :class:`WriteItem` objects to be grouped.

        Returns:
            A dictionary mapping the shard index to the list of items that
            belong to it.

        Raises:
            KeyError: If an item's FQN (with or without a possible "model." or
                "optim." prefix) cannot be found in *storage_plan*.
        """
        # storage_plan is a map from key to index
        buckets = {}
        for item in items:
            key = item.index.fqn

            # Allow the caller to wrap a Stateful object under a root key such
            # as "model" or "optim" (i.e. they passed {"model": ModelState(...)}
            # to dcp.save).  In that case every FQN gets the extra "model." or
            # "optim." prefix, while the reference weight-map coming from the
            # original HF checkpoint does *not* include it.  Detect this case
            # and fall back to the prefix-stripped key.
            if key not in storage_plan and "." in key:
                _, stripped_key = key.split(".", 1)
                key_lookup = stripped_key if stripped_key in storage_plan else key
            else:
                key_lookup = key

            if key_lookup not in storage_plan:
                raise KeyError(
                    f"Key '{key}' (or '{stripped_key}') was not found in the FQN->file mapping."
                )

            idx = storage_plan[key_lookup]
            buckets.setdefault(idx, []).append(item)

        return buckets

    def _gen_file_name(self, index: int, largest_index: int) -> str:
        """
        Return the canonical shard filename used by Hugging Face.

        Args:
            index: Current shard index (1-based).
            largest_index: Highest shard index in the checkpoint (and thus the
                total number of shards).

        Returns:
            str: ``"model-{index:05d}-of-{largest_index:05d}.safetensors"``
        """
        return (
            FILE_NAME.format(
                cpt_idx=f"{index}".zfill(5), num_shards=f"{largest_index}".zfill(5)
            )
            + SUFFIX
        )

    @property
    def metadata_path(self) -> str:
        """Returns the path to the metadata file."""
        return _metadata_fn


class HuggingFaceStorageReader(FsspecReader):
    """
    A reader that reads from a huggingface repository in the huggingface format.
    Uses in Fsspec back-end to communicate with the huggingface hub.

    **Note**: This is currently experimental. Functionality is not guaranteed.
    """

    def __init__(self, path: str, token: Optional[str] = None) -> None:
        """
        Initialize the huggingface reader pointing to path.

        Args:
            path: hf directory where the checkpoint will be read from. Should begin with hf://.
            token: The token to use to authenticate with huggingface hub.
        """
        from huggingface_hub import HfFileSystem  # type: ignore[import-not-found]

        if HfFileSystem.protocol not in fsspec.available_protocols():
            fsspec.register_implementation(HfFileSystem.protocol, HfFileSystem)
        super().__init__(path=path, token=token)
        self.storage_data: dict[str, str] = {}

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        """Set up the storage reader with metadata."""
        super().set_up_storage_reader(metadata, is_coordinator)
        # Populate storage_data from metadata for use in read_data
        self.storage_data = metadata.storage_data

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        """Reads the data from the storage."""
        from safetensors.torch import load  # type: ignore[import-not-found]

        per_file: dict[str, list[ReadItem]] = {}

        for read_item in plan.items:
            file_name = self.storage_data[read_item.storage_index.fqn]
            per_file.setdefault(file_name, []).append(read_item)

        for file_name, reqs in per_file.items():
            new_path = self.fs.concat_path(self.path, file_name)
            with self.fs.create_stream(new_path, "rb") as stream:
                loaded_tensors = load(stream.read())
                for req in reqs:
                    tensor = loaded_tensors[req.dest_index.fqn]

                    target_tensor = planner.resolve_tensor(req)
                    if (
                        isinstance(planner, HuggingFaceLoadPlanner)
                        and planner.allow_tensor_resize
                    ):
                        target_tensor.resize_(tensor.size())
                    else:
                        assert target_tensor.size() == tensor.size(), (
                            f"Tensor size mismatch for {req.dest_index.fqn}: {target_tensor.size()} != {tensor.size()}"
                        )
                    target_tensor = target_tensor.detach()
                    target_tensor.copy_(tensor)
                    planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def read_metadata(self) -> Metadata:
        """
        Retrieve (or synthesise) the checkpoint's :class:`Metadata`.

        Two cases are supported:

        1. **Standard HF layout** – ``model.safetensors.index.json`` is present.
           The method parses the JSON and converts the declared weight-map into
           DCP-compatible structures.
        2. **Legacy/Single-file layout** – the index file is absent and exactly
           one ``.safetensors`` file is found. The method inspects the file and
           builds an in-memory weight-map on the fly so that downstream code
           can treat the checkpoint as if it had an index.

        Returns:
            A fully populated :class:`Metadata` instance whose
            ``storage_data`` attribute maps FQNs to shard filenames and whose
            ``state_dict_metadata`` marks every entry as
            :class:`BytesStorageMetadata`.
        """
        metadata_path = self.fs.concat_path(self.path, _metadata_fn)

        state_dict_metadata: dict[str, STORAGE_TYPES] = {}
        storage_data: dict[str, str] = {}

        if not self.fs.exists(metadata_path):
            # if metadata file doesn't exist, create it from the safetensors file
            from safetensors.torch import safe_open  # type: ignore[import-not-found]

            safetensors_files = []
            for file in self.fs.ls(self.path):
                basename = os.path.basename(file)
                if basename.endswith(SUFFIX):
                    safetensors_files.append(basename)

            if len(safetensors_files) != 1:
                raise ValueError(
                    f"Need exactly one safetensors file to load without metadata, found {len(safetensors_files)} files"
                )
            
            # Use the single safetensors file
            safetensor_file = safetensors_files[0]
            full_path = self.fs.concat_path(self.path, safetensor_file)
            
            storage_data = {}
            with safe_open(full_path, framework="pt") as f:
                for k in f.keys():
                    state_dict_metadata[k] = BytesStorageMetadata()
                    storage_data[k] = safetensor_file
        else:
            with self.fs.create_stream(metadata_path, "r") as metadata_file:
                metadata = json.load(metadata_file)

            for key in metadata["weight_map"].keys():
                state_dict_metadata[key] = BytesStorageMetadata()
            storage_data = metadata["weight_map"]

        metadata = Metadata(
            state_dict_metadata=state_dict_metadata,
            storage_data=storage_data,
        )

        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id

        return metadata


def get_fqn_to_file_index_mapping(reference_model_path: str) -> dict[str, int]:
    """
    Get the FQN to file index mapping from the metadata.

    Args:
        reference_model_path: Path to reference model to copy file structure from.

    Returns:
        A mapping from tensor FQN to the index of the file that the tensor should be written to.
        Indices are from 1 to N, where N is the number of files.
    """
    hf_storage_reader = HuggingFaceStorageReader(reference_model_path)
    metadata = hf_storage_reader.read_metadata()
    weight_map = metadata.storage_data
    
    fqn_to_file_index_mapping = {}
    for fqn, filename in weight_map.items():
        if "-" in filename:
            index = int(filename.split("-")[1])
            fqn_to_file_index_mapping[fqn] = index
        else:
            # For single-file models, all tensors go to index 1
            fqn_to_file_index_mapping[fqn] = 1

    return fqn_to_file_index_mapping