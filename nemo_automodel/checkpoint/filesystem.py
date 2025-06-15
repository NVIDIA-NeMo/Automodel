# mypy: allow-untyped-defs
import json
import os
import queue
from io import UnsupportedOperation
from typing import Callable

import torch
from torch.distributed.checkpoint.planner import SavePlanner, WriteItemType

from nemo_automodel.checkpoint.checkpointing import SerializationFormat
from torch.distributed.checkpoint.filesystem import (
    _StorageWriterTransforms,
    _TensorLoader,
    _OverlappingCpuLoader,
    _SerialCpuLoader,
    _item_size,
    _write_item,
)



def _write_files_from_queue(
    create_stream: Callable,
    file_queue: queue.Queue,
    result_queue: queue.Queue,
    planner: SavePlanner,
    transforms: _StorageWriterTransforms,
    inflight_threshhold: int,
    use_fsync: bool,
    thread_count: int,
    serialization_format: SerializationFormat,
) -> None:
    # Convert incoming enum (could be from torch.distributed.checkpoint) to our local
    # SerializationFormat so that identity checks inside torch\'s _write_item succeed.
    if not isinstance(serialization_format, SerializationFormat):
        try:
            serialization_format = SerializationFormat[serialization_format.name]  # type: ignore[arg-type]
        except Exception:  # pragma: no cover – fallback for enum value conversion
            serialization_format = SerializationFormat(serialization_format.value)  # type: ignore[arg-type]

    try:
        while True:
            file_name, storage_key, write_items = file_queue.get_nowait()
            loader: _TensorLoader

            custom_backend_name = torch._C._get_privateuse1_backend_name()
            custom_device_mod = getattr(torch, custom_backend_name, None)

            # TODO: Using the OverlappingCpuLoader with multiple threads creates significant
            # performance degredation, observed as being related to cuda stream syncs. We
            # should try to fix this and use _OverlappingCpuLoader for all threaded cases
            if (
                thread_count == 1
                and (
                    torch.cuda.is_available()
                    or (custom_device_mod and custom_device_mod.is_available())
                )
                and inflight_threshhold > 0
            ):
                loader = _OverlappingCpuLoader(
                    planner.resolve_data,
                    inflight_threshhold=inflight_threshhold,
                )
            else:
                loader = _SerialCpuLoader(
                    planner.resolve_data,
                )

            tensor_w = [wi for wi in write_items if wi.type != WriteItemType.BYTE_IO]
            for write_item in tensor_w:
                loader.add(_item_size(write_item), write_item)
            loader.start_loading()

            bytes_w = [wi for wi in write_items if wi.type == WriteItemType.BYTE_IO]
            write_results = []

            with create_stream(file_name, "wb") as stream:
                for write_item in bytes_w:
                    data = planner.resolve_data(write_item)
                    write_results.append(
                        _write_item(
                            transforms,
                            stream,
                            data,
                            write_item,
                            storage_key,
                            serialization_format,
                        )
                    )

                tensor_dict = {}
                metadata_dict = {}
                for tensor, write_item in loader.values():
                    assert tensor.is_cpu
                    write_results.append(
                        _write_item(
                            transforms,
                            stream,
                            tensor,
                            write_item,
                            storage_key,
                            serialization_format,
                        )
                    )
                    tensor_dict[write_item.index.fqn] = tensor
                    metadata_dict[write_item.index.fqn] = {
                        "saved_offsets": write_item.tensor_data.chunk.offsets
                    }

                if serialization_format == SerializationFormat.SAFETENSORS:
                    from safetensors.torch import save  # type: ignore[import-not-found]

                    stream.write(
                        save(
                            tensor_dict,
                            metadata={
                                "DCP_SHARDING_INFO": json.dumps(metadata_dict),
                                "DCP_VERSION": "1.0",
                            },
                        )
                    )

                if use_fsync:
                    try:
                        os.fsync(stream.fileno())
                    except (AttributeError, UnsupportedOperation):
                        os.sync()
                stream.close()
            result_queue.put(write_results)
    except queue.Empty:
        pass

__all__: list[str] = [
    "_write_files_from_queue",
    "SerializationFormat",
]
