# taken and edited from https://github.com/pytorch/pytorch/blob/6ebe9a4f47e9cd1c9ccd467bcdfdea9445fd98d6/torch/distributed/checkpoint/hf_storage.py  # pylint: disable=line-too-long
# pylint: disable=missing-function-docstring

import dataclasses
import json
import queue
from typing import Any, Optional

import torch
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
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

from nemo_automodel.checkpoint._backports._fsspec_filesystem import FsspecReader, FsspecWriter
from nemo_automodel.checkpoint._backports.consolidate_hf_safetensors import consolidate_safetensors_files
from nemo_automodel.checkpoint._backports.filesystem import SerializationFormat
from nemo_automodel.checkpoint._backports.hf_utils import (
    CUSTOM_METADATA_KEY,
    DATA_KEY,
    DATA_OFFSETS_KEY,
    DEFAULT_EXTRA_METADATA_KEY,
    DTYPE_KEY,
    SAVED_OFFSETS_KEY,
    SHAPE_KEY,
    SUFFIX,
    _gen_file_name,
    _get_dtype,
    _get_safetensors_file_metadata,
    _HFStorageInfo,
    _metadata_fn,
)


__all__ = ["_HuggingFaceStorageWriter", "_HuggingFaceStorageReader"]


class _HuggingFaceStorageWriter(FsspecWriter):
    """
    A writer that writes to a huggingface repository in the huggingface format.
    Uses Fsspec back-end to communicate with back-end storage.
    Fsspec registration of the storage solution is required.
    """

    def __init__(
        self,
        path: str,
        fqn_to_index_mapping: Optional[dict[str, int]] = None,
        thread_count: int = 1,
        token: Optional[str] = None,
        save_sharded: bool = False,
        consolidated_output_path: Optional[str] = None,
        num_threads_consolidation: Optional[int] = None,
    ) -> None:
        """
        Initialize the huggingface writer pointing to path.

        Args:
            path: hf directory where the checkpoint will be read from.
                  Needs to have .safetensors files, but can be from any fsspec supported storage,
                  including localFS and hf://.
                  This needs to be a remote path if you want to enable consolidation after saving.
            fqn_to_index_mapping: A mapping from tensor FQN to the index of the file that the tensor should be written to.
                              Indices are from 1 to N, where N is the number of files. If not provided,
                              the tensors will be written to a single file. If none, then all the tensors on the
                              same rank will be written to the same file.
            token: The token to use to authenticate with huggingface hub.
            save_sharded: If True, save the checkpoint as a sharded checkpoint where every rank saves its own shard.
                        Default is False which assumes full tensors are being saved.
            consolidated_output_path: If provided, the output path where the consolidated files will be written in the finish step. This needs to be a local fs path right now.
            num_threads_consolidation: Number of threads to use for parallel processing of saving data to output files. If not provided, the default value is the number of output files.
        """
        if token is not None:
            super().__init__(
                path=path,
                token=token,
                serialization_format=SerializationFormat.SAFETENSORS,
            )
        else:
            super().__init__(
                path=path,
                serialization_format=SerializationFormat.SAFETENSORS,
            )
        self._fqn_to_index_mapping: Optional[dict[str, int]] = fqn_to_index_mapping
        self._save_sharded = save_sharded
        self._consolidated_output_path = consolidated_output_path

        if num_threads_consolidation:
            self._num_threads_consolidation = num_threads_consolidation
        elif self._fqn_to_index_mapping:
            self._num_threads_consolidation = max(self._fqn_to_index_mapping.values())
        else:
            self._num_threads_consolidation = 1

        self.thread_count = thread_count

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        new_plans = []
        for i, plan in enumerate(plans, start=1):
            storage_data: dict[str, Any] = {}
            # save default shard mapping. We only use fqn_to_index_mapping for consolidation.
            # if self._fqn_to_index_mapping is not None:
            #     storage_data["fqn_to_index_mapping"] = self._fqn_to_index_mapping
            if self._save_sharded:
                storage_data["shard_index"] = i

            new_plans.append(dataclasses.replace(plan, storage_data=storage_data))

        return new_plans

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> Future[list[WriteResult]]:
        if len(plan.items) == 0:
            fut: Future = Future()
            fut.set_result([])
            return fut

        # storage_plan is a map from key to file index
        storage_data: dict[str, Any] = plan.storage_data
        storage_plan: Optional[dict[str, int]] = None
        shard_index: Optional[int] = None
        if "fqn_to_index_mapping" in storage_data:
            storage_plan = storage_data["fqn_to_index_mapping"]
        if "shard_index" in storage_data:
            shard_index = storage_data["shard_index"]

        buckets = self._split_by_storage_plan(storage_plan, plan.items)
        highest_index = max(storage_plan.values()) if storage_plan is not None else 1

        file_queue: queue.Queue = queue.Queue()
        for file_index, write_items in buckets.items():
            file_name = _gen_file_name(file_index, highest_index, shard_index)
            file_queue.put(
                (self.fs.concat_path(self.path, file_name), file_name, write_items)
            )

        return super()._write_data(planner, file_queue)

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        if self._save_sharded and not self._consolidated_output_path:
            return
        if self._save_sharded:
            return consolidate_safetensors_files(
                input_dir=self.path,
                output_dir=self._consolidated_output_path,
                num_threads=self._num_threads_consolidation,
                fqn_to_index_mapping=self._fqn_to_index_mapping
            )

        metadata_to_write = {}
        storage_md = {}
        total_size = 0
        for wr_list in results:
            storage_md.update(
                {wr.index.fqn: wr.storage_data.relative_path for wr in wr_list}
            )
            total_size += sum([wr.storage_data.length for wr in wr_list])
        metadata_to_write["metadata"] = {"total_size": total_size}
        metadata_to_write["weight_map"] = storage_md

        metadata_path = self.fs.concat_path(self.path, f"{_metadata_fn}")
        with self.fs.create_stream(metadata_path, "w") as metadata_file:
            json.dump(metadata_to_write, metadata_file, indent=2)

    def _split_by_storage_plan(
        self, storage_plan: Optional[dict[str, int]], items: list[WriteItem]
    ) -> dict[int, list[WriteItem]]:
        # storage_plan is a map from key to index
        if storage_plan is None:
            return {1: items}

        buckets = {}
        for item in items:
            key = item.index.fqn

            idx = storage_plan[key]
            if idx not in buckets:
                buckets[idx] = [item]
            else:
                buckets[idx].append(item)

        return buckets

    @property
    def metadata_path(self) -> str:
        return _metadata_fn


class _HuggingFaceStorageReader(FsspecReader):
    """
    A reader that reads from a huggingface repository in the huggingface format.
    Uses in Fsspec back-end to communicate with storage.
    Fsspec registration of the storage solution is required.
    """

    def __init__(self, path: str, token: Optional[str] = None) -> None:
        """
        Initialize the huggingface reader pointing to path.

        Args:
            path: hf directory where the checkpoint will be read from.
            Needs to have .safetensors file, but can be from any fsspec supported storage,
            including localFS and hf://.
            token: The token to use to authenticate with huggingface hub.
        """
        if token is not None:
            super().__init__(path=path, token=token)
        else:
            super().__init__(path=path)

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        from safetensors import deserialize  # type: ignore[import-not-found]

        per_file: dict[str, list[ReadItem]] = {}

        for read_item in plan.items:
            item_md: _HFStorageInfo = self.storage_data[read_item.storage_index]
            file_name = item_md.relative_path
            per_file.setdefault(file_name, []).append(read_item)

        for file_name, reqs in per_file.items():
            with self.fs.create_stream(file_name, "rb") as stream:
                # TODO: make this more efficient by doing offset reads instead of a
                # full deserialization of the file
                deserialized = deserialize(stream.read())
                deserialized_dict: dict[str, dict[str, Any]] = {
                    tensor_info[0]: tensor_info[1] for tensor_info in deserialized
                }

                for req in reqs:
                    item_md: _HFStorageInfo = self.storage_data[req.storage_index]

                    tensor_bytes = deserialized_dict[req.dest_index.fqn][DATA_KEY]

                    tensor = torch.frombuffer(
                        tensor_bytes,
                        dtype=item_md.dtype,
                    )
                    tensor = tensor.reshape(item_md.shape)
                    tensor = narrow_tensor_by_index(
                        tensor, req.storage_offsets, req.lengths
                    )
                    target_tensor = planner.resolve_tensor(req).detach()

                    assert target_tensor.size() == tensor.size(), (
                        f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                    )

                    target_tensor.copy_(tensor)
                    planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def read_metadata(self) -> Metadata:
        state_dict_metadata: dict[str, TensorStorageMetadata] = {}
        storage_data: dict[MetadataIndex, _HFStorageInfo] = {}

        safetensors_files = []
        for file in self.fs.ls(self.path):
            if file.endswith(SUFFIX):
                safetensors_files.append(file)

        for safetensor_file in safetensors_files:
            with self.fs.create_stream(safetensor_file, "rb") as f:
                safetensors_metadata, _ = _get_safetensors_file_metadata(f)
                custom_metadata = safetensors_metadata.get(DEFAULT_EXTRA_METADATA_KEY)

                dcp_sharding_info = None
                if custom_metadata and custom_metadata.get(CUSTOM_METADATA_KEY):
                    dcp_sharding_info = json.loads(
                        custom_metadata.get(CUSTOM_METADATA_KEY)
                    )

                for key, val in safetensors_metadata.items():
                    if key == DEFAULT_EXTRA_METADATA_KEY:
                        continue

                    # construct state_dict_metadata
                    if dcp_sharding_info is not None:
                        offset = dcp_sharding_info[key][SAVED_OFFSETS_KEY]
                    else:
                        offset = [0] * len(val[SHAPE_KEY])

                    if key not in state_dict_metadata:
                        state_dict_metadata[key] = TensorStorageMetadata(
                            properties=TensorProperties(
                                dtype=_get_dtype(val[DTYPE_KEY])
                            ),
                            size=torch.Size(
                                [
                                    saved + offset
                                    for saved, offset in zip(val[SHAPE_KEY], offset)
                                ]
                            ),
                            chunks=[
                                ChunkStorageMetadata(
                                    offsets=torch.Size(offset),
                                    sizes=torch.Size(val[SHAPE_KEY]),
                                )
                            ],
                        )
                    else:
                        state_dict_metadata[key].chunks.append(
                            ChunkStorageMetadata(
                                torch.Size(offset), sizes=torch.Size(val[SHAPE_KEY])
                            )
                        )
                        size = list(state_dict_metadata[key].size)
                        for i in range(len(size)):
                            size[i] = max(size[i], val[SHAPE_KEY][i] + offset[i])
                        state_dict_metadata[key].size = torch.Size(size)

                    # construct storage data
                    if dcp_sharding_info is not None:
                        metadata_index = MetadataIndex(
                            fqn=key, offset=dcp_sharding_info[key][SAVED_OFFSETS_KEY]
                        )
                    else:
                        metadata_index = MetadataIndex(
                            fqn=key, offset=[0] * len(val[SHAPE_KEY])
                        )
                    storage_data[metadata_index] = _HFStorageInfo(
                        relative_path=safetensor_file,
                        offset=val[DATA_OFFSETS_KEY][0],
                        length=val[DATA_OFFSETS_KEY][1] - val[DATA_OFFSETS_KEY][0],
                        shape=torch.Size(val[SHAPE_KEY]),
                        dtype=_get_dtype(val[DTYPE_KEY]),
                    )

        metadata = Metadata(
            state_dict_metadata=state_dict_metadata,  # type: ignore[arg-type]
            storage_data=storage_data,
        )

        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id  # type: ignore[union-attr]

        return metadata


def _extract_file_index(filename: str) -> int:
    """Return the 1-based shard index encoded in a safetensors filename.

    Supported patterns::

        model-00001-of-00008.safetensors
        shard-00000-model-00002-of-00008.safetensors
        model.safetensors  (single-file checkpoints)

    Args:
        filename: The (relative) safetensors filename.

    Returns:
        The numeric shard index, defaulting to ``1`` when no explicit index is
        present or when the filename cannot be parsed.
    """
    # Strip any leading directory components so we only deal with the basename.
    basename = filename.split("/")[-1]

    # Single-file checkpoints which usually carry the name ``model.safetensors``.
    if basename == "model.safetensors":
        return 1

    parts = basename.split("-")

    # Common HF pattern: *model-{idx}-of-{n}.safetensors*
    if "model" in parts:
        idx_pos = parts.index("model") + 1
        if idx_pos < len(parts):
            token = parts[idx_pos].split(".")[0]  # Remove extension if present.
            try:
                return int(token.lstrip("0") or "0")
            except ValueError:
                pass

    # default to the first shard.
    return 1


def get_fqn_to_file_index_mapping(reference_model_path: str) -> dict[str, int]:
    """
    Get the FQN to file index mapping from the metadata.

    Args:
        reference_model_path: Path to reference model to copy file structure from.

    Returns:
        A mapping from tensor FQN to the index of the file that the tensor should be written to.
        Indices are from 1 to N, where N is the number of files.
    """
    hf_reader = _HuggingFaceStorageReader(reference_model_path)

    fqn_to_file_index_mapping: dict[str, int] = {}

    # ``read_metadata`` transparently handles both the presence of an index
    # JSON **and** the single-file fallback, so we can simply rely on it.
    metadata = hf_reader.read_metadata()

    for md_index, storage_info in metadata.storage_data.items():
        fqn = getattr(md_index, "fqn", md_index)
        filename = storage_info.relative_path
        fqn_to_file_index_mapping[str(fqn)] = _extract_file_index(filename)

    return fqn_to_file_index_mapping