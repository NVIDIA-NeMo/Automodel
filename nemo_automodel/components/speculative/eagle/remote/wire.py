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

"""Compact binary tensor serialization for the remote target data plane.

This is the fallback path used when NCCL GPU-to-GPU transfer is unavailable:
tensors are encoded as dtype + shape + raw contiguous bytes and shipped inside
the HTTP body. The format is little-endian and self-delimiting.

Format::

    [4B]  magic         0x4E4D4554  ("NMET")
    per entry:
      [4B]  key_len      (uint32)
      [key_len B] key     UTF-8
      [1B]  flags        bit0 = is_none
      if not none:
        [1B]  dtype_code (see _DTYPE_TABLE)
        [1B]  ndim
        [ndim x 8B] shape (int64)
        [8B]  nbytes     (uint64)
        [nbytes B] data   raw contiguous tensor bytes
"""

from __future__ import annotations

import ctypes
import struct
from typing import Optional

import torch

MAGIC = 0x4E4D4554  # "NMET" (NeMo target)

# dtype_code -> torch.dtype. The inverse map drives encoding.
_DTYPE_TABLE: dict[int, torch.dtype] = {
    0: torch.float32,
    1: torch.float64,
    2: torch.float16,
    3: torch.bfloat16,
    4: torch.int64,
    5: torch.int32,
    6: torch.int16,
    7: torch.int8,
    8: torch.uint8,
    9: torch.bool,
}
_DTYPE_TO_CODE: dict[torch.dtype, int] = {dt: c for c, dt in _DTYPE_TABLE.items()}

# struct layouts (little-endian)
_HEADER_FMT = struct.Struct("<I")  # magic
_KEYLEN_FMT = struct.Struct("<I")  # key_len
_FLAG_FMT = struct.Struct("<B")  # flags
_DTYPE_FMT = struct.Struct("<B")  # dtype_code
_NDIM_FMT = struct.Struct("<B")  # ndim
_SHAPE_FMT = struct.Struct("<q")  # single int64 dim
_NBYTES_FMT = struct.Struct("<Q")  # uint64

_FLAG_NONE = 0x01


def encode(tensor_dict: dict[str, Optional[torch.Tensor]]) -> bytearray:
    """Encode a dict of CPU tensors into the wire format.

    ``None`` values are preserved. The caller is responsible for moving tensors
    to CPU first; CUDA tensors are rejected to keep the data path explicit.
    """
    total_size = _HEADER_FMT.size
    entries: list[tuple[bytes, Optional[torch.Tensor], int]] = []
    for key, tensor in tensor_dict.items():
        key_bytes = key.encode("utf-8")
        entry_size = _KEYLEN_FMT.size + len(key_bytes) + _FLAG_FMT.size
        if tensor is None:
            entries.append((key_bytes, None, 0))
        else:
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            if tensor.is_cuda:
                raise ValueError("wire encoding expects CPU tensors, got a CUDA tensor")
            code = _DTYPE_TO_CODE.get(tensor.dtype)
            if code is None:
                raise TypeError(f"Unsupported dtype for wire encoding: {tensor.dtype}")
            nbytes = tensor.numel() * tensor.element_size()
            entry_size += _DTYPE_FMT.size + _NDIM_FMT.size + tensor.ndim * _SHAPE_FMT.size + _NBYTES_FMT.size + nbytes
            entries.append((key_bytes, tensor, nbytes))
        total_size += entry_size

    buf = bytearray(total_size)
    pos = 0
    _HEADER_FMT.pack_into(buf, pos, MAGIC)
    pos += _HEADER_FMT.size

    for key_bytes, tensor, nbytes in entries:
        _KEYLEN_FMT.pack_into(buf, pos, len(key_bytes))
        pos += _KEYLEN_FMT.size
        buf[pos : pos + len(key_bytes)] = key_bytes
        pos += len(key_bytes)

        if tensor is None:
            _FLAG_FMT.pack_into(buf, pos, _FLAG_NONE)
            pos += _FLAG_FMT.size
            continue

        _FLAG_FMT.pack_into(buf, pos, 0)
        pos += _FLAG_FMT.size
        _DTYPE_FMT.pack_into(buf, pos, _DTYPE_TO_CODE[tensor.dtype])
        pos += _DTYPE_FMT.size
        _NDIM_FMT.pack_into(buf, pos, tensor.ndim)
        pos += _NDIM_FMT.size
        for dim in tensor.shape:
            _SHAPE_FMT.pack_into(buf, pos, dim)
            pos += _SHAPE_FMT.size
        _NBYTES_FMT.pack_into(buf, pos, nbytes)
        pos += _NBYTES_FMT.size
        # Bulk copy the raw bytes (avoids a Python-level per-element loop).
        ctypes.memmove((ctypes.c_char * nbytes).from_buffer(buf, pos), tensor.data_ptr(), nbytes)
        pos += nbytes

    return buf


def encode_to_bytes(tensor_dict: dict[str, Optional[torch.Tensor]]) -> bytes:
    """Encode and return immutable ``bytes`` (HTTP body)."""
    return bytes(encode(tensor_dict))


def decode(raw: bytes, map_location: str = "cpu") -> dict[str, Optional[torch.Tensor]]:
    """Decode a wire-format blob back into a dict of tensors on ``map_location``."""
    target_device = torch.device(map_location)

    mv = memoryview(raw)
    pos = 0
    magic = _HEADER_FMT.unpack_from(mv, pos)[0]
    pos += _HEADER_FMT.size
    if magic != MAGIC:
        raise ValueError(f"Bad wire-format magic: 0x{magic:08x} (expected 0x{MAGIC:08x})")

    result: dict[str, Optional[torch.Tensor]] = {}
    while pos < len(mv):
        key_len = _KEYLEN_FMT.unpack_from(mv, pos)[0]
        pos += _KEYLEN_FMT.size
        key = bytes(mv[pos : pos + key_len]).decode("utf-8")
        pos += key_len

        flags = _FLAG_FMT.unpack_from(mv, pos)[0]
        pos += _FLAG_FMT.size
        if flags & _FLAG_NONE:
            result[key] = None
            continue

        code = _DTYPE_FMT.unpack_from(mv, pos)[0]
        pos += _DTYPE_FMT.size
        dtype = _DTYPE_TABLE.get(code)
        if dtype is None:
            raise ValueError(f"Unknown dtype code: {code}")

        ndim = _NDIM_FMT.unpack_from(mv, pos)[0]
        pos += _NDIM_FMT.size
        shape = []
        for _ in range(ndim):
            shape.append(_SHAPE_FMT.unpack_from(mv, pos)[0])
            pos += _SHAPE_FMT.size

        nbytes = _NBYTES_FMT.unpack_from(mv, pos)[0]
        pos += _NBYTES_FMT.size
        # Copy the slice into a fresh bytearray so the tensor owns writable
        # storage independent of the request buffer.
        storage = torch.frombuffer(bytearray(mv[pos : pos + nbytes]), dtype=torch.uint8)
        tensor = storage.view(dtype).reshape(tuple(shape))
        if target_device.type != "cpu":
            tensor = tensor.to(target_device)
        result[key] = tensor
        pos += nbytes

    return result
