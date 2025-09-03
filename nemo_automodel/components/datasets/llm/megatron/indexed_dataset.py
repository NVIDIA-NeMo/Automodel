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

# taken and modified from https://github.com/NVIDIA/Megatron-LM/blob/5e798111e60f45e82c336ef7b89d8d793c93208f/megatron/core/datasets/indexed_dataset.py

"""A self-contained port of Megatron-Core's indexed dataset loader.

Supports the original mmap and file-pointer readers for local *.bin / *.idx
pairs. The file pair is expected to live on a local filesystem.

All three calls below are equivalent:

    from nemo_automodel.datasets.llm.indexed_dataset import IndexedDataset

    ds = IndexedDataset("/path/to/shard_00_text_document")
    print(len(ds), ds[0][:20])

    ds = IndexedDataset("/path/to/shard_00_text_document.bin")
    print(len(ds), ds[0][:20])

    ds = IndexedDataset("/path/to/shard_00_text_document.idx")
    print(len(ds), ds[0][:20])
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import struct
from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache
from itertools import accumulate
from types import TracebackType
from typing import Any, List, Optional, Tuple, Type, Union

import numpy
import torch

logger = logging.getLogger(__name__)

_INDEX_HEADER = b"MMIDIDX\x00\x00"


class DType(Enum):
    """The NumPy data type Enum for reading the IndexedDataset indices"""

    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, value: Type[numpy.number]) -> int:
        """Get the code from the dtype

        Args:
            value (Type[numpy.number]): The dtype

        Returns:
            int: The code
        """
        return cls[value.__name__].value

    @classmethod
    def dtype_from_code(cls, value: int) -> Type[numpy.number]:
        """Get the dtype from the code

        Args:
            value (int): The code

        Returns:
            Type[numpy.number]: The dtype
        """
        return getattr(numpy, cls(value).name)

    @classmethod
    def size(cls, key: Union[int, Type[numpy.number]]) -> int:
        """Get the size of the dtype/code in bytes

        Args:
            key (Union[int, Type[numpy.number]]): The dtype or code

        Raises:
            ValueError: If the key is neither dtype nor integer code

        Returns:
            int: The size of the dtype/code in bytes
        """
        if isinstance(key, int):
            return cls.dtype_from_code(key)().itemsize
        elif numpy.number in key.__mro__:
            return key().itemsize
        else:
            raise ValueError("Invalid key passed to DType.size()")

    @classmethod
    def optimal_dtype(cls, cardinality: Optional[int]) -> Type[numpy.number]:
        """Get the dtype to use for an index of a certain cardinality

        Args:
            cardinality (Optional[int]): The number of elements to be indexed

        Returns:
            Type[numpy.number]: The dtype to use for the index
        """
        if cardinality is not None and cardinality < 65500:
            return numpy.uint16
        return numpy.int32


class _IndexWriter(object):
    """Object class to write the index (.idx) file

    Args:
        idx_path (str): The path to the index file

        dtype (Type[numpy.number]): The dtype of the index file
    """

    def __init__(self, idx_path: str, dtype: Type[numpy.number]) -> None:
        self.idx_path = idx_path
        self.dtype = dtype

    def __enter__(self) -> "_IndexWriter":
        """Enter the context introduced by the 'with' keyword

        Returns:
            _IndexWriter: The instance
        """
        self.idx_writer = open(self.idx_path, "wb")
        # fixed, vestigial practice
        self.idx_writer.write(_INDEX_HEADER)
        # fixed, vestigial practice
        self.idx_writer.write(struct.pack("<Q", 1))
        # the numeric code for the dtype
        self.idx_writer.write(struct.pack("<B", DType.code_from_dtype(self.dtype)))
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        """Exit the context introduced by the 'with' keyword

        Args:
            exc_type (Optional[Type[BaseException]]): Exception type

            exc_val (Optional[BaseException]): Exception value

            exc_tb (Optional[TracebackType]): Exception traceback object

        Returns:
            Optional[bool]: Whether to silence the exception
        """
        self.idx_writer.close()
        return None

    def write(
        self,
        sequence_lengths: List[int],
        sequence_modes: Optional[List[int]],
        document_indices: List[int],
    ) -> None:
        """Write the index (.idx) file

        Args:
            sequence_lengths (List[int]): The length of each sequence

            sequence_modes (Optional[List[int]]): The mode of each sequences

            document_indices (List[int]): The seqyebce indices demarcating the end of each document
        """
        sequence_pointers = self._sequence_pointers(sequence_lengths)

        # the number of sequences in the dataset
        sequence_count = len(sequence_lengths)
        self.idx_writer.write(struct.pack("<Q", sequence_count))

        # the number of documents in the dataset
        document_count = len(document_indices)
        self.idx_writer.write(struct.pack("<Q", document_count))

        # the number of tokens per sequence
        self.idx_writer.write(numpy.array(sequence_lengths, dtype=numpy.int32).tobytes(order="C"))

        # the byte offsets for all sequences
        self.idx_writer.write(numpy.array(sequence_pointers, dtype=numpy.int64).tobytes(order="C"))

        # the sequence indices marking the end of each document
        self.idx_writer.write(numpy.array(document_indices, dtype=numpy.int64).tobytes(order="C"))

        # the mode per sequence
        if sequence_modes is not None:
            self.idx_writer.write(numpy.array(sequence_modes, dtype=numpy.int8).tobytes(order="C"))

    def _sequence_pointers(self, sequence_lengths: List[int]) -> List[int]:
        """Build the sequence pointers per the sequence lengths and dtype size

        Args:
            sequence_lengths (List[int]): The length of each sequence

        Returns:
            List[int]: The pointer to the beginning of each sequence
        """
        itemsize = DType.size(self.dtype)
        curr_ptr = 0
        list_ptr = []
        for length in sequence_lengths:
            list_ptr.append(curr_ptr)
            curr_ptr += length * itemsize
        return list_ptr


class _IndexReader:
    """Object class to read the index (.idx) file

    Args:
        idx_path (str): The path to the index file

        multimodal (bool): Whether the dataset is multimodal
    """

    def __init__(self, idx_path: str, multimodal: bool) -> None:
        logger.info("Loading index file %s", idx_path)

        with open(idx_path, "rb") as f:
            header = f.read(9)
            assert header == _INDEX_HEADER, f"Bad header in {idx_path}"

            version = struct.unpack("<Q", f.read(8))[0]
            assert version == 1, f"Unsupported index version {version} in {idx_path}"

            code = struct.unpack("<B", f.read(1))[0]
            self.dtype = DType.dtype_from_code(code)
            self.dtype_size = DType.size(self.dtype)

            self.sequence_count = struct.unpack("<Q", f.read(8))[0]
            self.document_count = struct.unpack("<Q", f.read(8))[0]
            payload_offset = f.tell()

        # memory-map the whole file for fast zero-copy slicing
        self._mmap = numpy.memmap(idx_path, mode="r", order="C")
        self._buffer = memoryview(self._mmap)

        # extract views
        logger.info("Extracting sequence lengths")
        self.sequence_lengths = numpy.frombuffer(
            self._buffer, dtype=numpy.int32, count=self.sequence_count, offset=payload_offset
        )
        logger.info("Extracting sequence pointers")
        self.sequence_pointers = numpy.frombuffer(
            self._buffer,
            dtype=numpy.int64,
            count=self.sequence_count,
            offset=payload_offset + self.sequence_lengths.nbytes,
        )
        logger.info("Extracting document indices")
        self.document_indices = numpy.frombuffer(
            self._buffer,
            dtype=numpy.int64,
            count=self.document_count,
            offset=payload_offset + self.sequence_lengths.nbytes + self.sequence_pointers.nbytes,
        )

        self.sequence_modes: Optional[numpy.ndarray] = None
        if multimodal:
            logger.info("Extracting sequence modes")
            self.sequence_modes = numpy.frombuffer(
                self._buffer,
                dtype=numpy.int8,
                count=self.sequence_count,
                offset=payload_offset
                + self.sequence_lengths.nbytes
                + self.sequence_pointers.nbytes
                + self.document_indices.nbytes,
            )

        assert self.sequence_lengths.shape[0] == len(self)
        assert self.sequence_lengths.shape[0] == self.sequence_count
        assert self.sequence_lengths.shape[0] == self.document_indices[-1]

        logger.info("Sequences: %d | Documents: %d", len(self), self.document_indices.shape[0] - 1)

    def __del__(self) -> None:
        """Clean up the object"""
        self._mmap._mmap.close()
        del self._mmap

    def __len__(self) -> int:
        """Get the number of sequences in the dataset

        Returns:
            int: The number of sequences in the dataset
        """
        return self.sequence_count

    @lru_cache(maxsize=8)
    def __getitem__(self, idx: int) -> Tuple[numpy.int32, numpy.int64, Optional[numpy.int8]]:
        """Return the pointer, length, and mode at the index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.int32, numpy.int64, Optional[numpy.int8]]: The pointer, length and mode
                at the index
        """
        return (
            self.sequence_pointers[idx],
            self.sequence_lengths[idx],
            self.sequence_modes[idx] if self.sequence_modes is not None else None,
        )


class _BinReader(ABC):
    """Abstract class to read the data (.bin) file"""

    @abstractmethod
    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (Type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype` constructed from
                reading bytes from the data file starting at `offset`.
        """
        pass


class _MMapBinReader(_BinReader):
    """A _BinReader that memory maps the data (.bin) file"""

    def __init__(self, bin_path: str) -> None:
        """Initialize the _MMapBinReader

        Args:
            bin_path (str): The path to the data (.bin) file.
        """
        self._file = open(bin_path, "rb")
        self._mmap = numpy.memmap(self._file, mode="r", order="C")
        self._buffer = memoryview(self._mmap.data)

    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (Type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype` constructed from
                reading bytes from the data file starting at `offset`.
        """
        return numpy.frombuffer(self._buffer, dtype=dtype, count=count, offset=offset)

    def __del__(self) -> None:
        """Clean up the object"""
        self._mmap._mmap.close()
        self._file.close()
        del self._mmap
        del self._file


class _FileBinReader(_BinReader):
    """A _BinReader that reads from the data (.bin) file using a file pointer"""

    def __init__(self, bin_path: str) -> None:
        """Initialize the _FileBinReader

        Args:
            bin_path (str): The path to the data (.bin) file.
        """
        self._bin_path = bin_path

    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (Type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype` constructed from
                reading bytes from the data file starting at `offset`.
        """
        out = numpy.empty(count, dtype=dtype)
        with open(self._bin_path, "rb", buffering=0) as f:
            f.seek(offset)
            f.readinto(out)
        return out


class IndexedDataset(torch.utils.data.Dataset):
    """A fast, on-disk dataset backed by Megatron-style index + binary files."""

    def __init__(self, path_prefix: str, multimodal: bool = False, mmap: bool = True) -> None:
        """Initialize the IndexedDataset

        Args:
        path_prefix (str): The index (.idx) and data (.bin) prefix

        multimodal (bool): Whether the dataset is multimodal. Defaults to False.

        mmap (bool): Whether to mmap the .bin files. Defaults to True.
        """
        super().__init__()
        normalized_prefix = _normalize_prefix(path_prefix)
        self.initialize(normalized_prefix, multimodal, mmap)

    def initialize(self, path_prefix: str, multimodal: bool, mmap: bool) -> None:
        idx_path, bin_path = get_idx_path(path_prefix), get_bin_path(path_prefix)
        assert os.path.exists(idx_path) and os.path.exists(bin_path), f"Missing .idx or .bin at prefix {path_prefix}"

        self.path_prefix = path_prefix
        self.multimodal = multimodal
        self.mmap = mmap

        self.bin_reader = _MMapBinReader(bin_path) if mmap else _FileBinReader(bin_path)
        self.index = _IndexReader(idx_path, multimodal)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(
        self, idx: Union[int, numpy.integer, slice]
    ) -> Union[
        numpy.ndarray,
        Tuple[numpy.ndarray, Any],  # mode attached
        List[numpy.ndarray],
        Tuple[List[numpy.ndarray], numpy.ndarray],
    ]:
        if isinstance(idx, (int, numpy.integer)):
            ptr, length, mode = self.index[idx]
            seq = self.bin_reader.read(self.index.dtype, length, ptr)
            return (seq, mode) if mode is not None else seq

        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into IndexedDataset must be contiguous (step=1)")
            lengths = self.index.sequence_lengths[idx]
            modes = self.index.sequence_modes[idx] if self.multimodal else None
            offsets = list(accumulate(lengths))
            buffer = self.bin_reader.read(
                self.index.dtype,
                int(sum(lengths)),
                int(self.index.sequence_pointers[start]),
            )
            sequences = numpy.split(buffer, offsets[:-1])
            return (sequences, modes) if modes is not None else sequences

        else:
            raise TypeError(f"Unexpected index type {type(idx)}")

    def get(
        self, idx: int, offset: int = 0, length: Optional[int] = None
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, Any]]:
        ptr, seq_len, mode = self.index[idx]
        length = seq_len - offset if length is None else length
        ptr += offset * DType.size(self.index.dtype)
        seq = self.bin_reader.read(self.index.dtype, length, ptr)
        return (seq, mode) if mode is not None else seq

    @property
    def sequence_lengths(self):  # numpy.ndarray[int32]
        return self.index.sequence_lengths

    @property
    def document_indices(self):  # numpy.ndarray[int64]
        return self.index.document_indices

    @staticmethod
    def exists(path_prefix: str) -> bool:
        return os.path.exists(get_idx_path(path_prefix)) and os.path.exists(get_bin_path(path_prefix))


class IndexedDatasetBuilder(object):
    """Builder class for the IndexedDataset class

    Args:
        bin_path (str): The path to the data (.bin) file

        dtype (Type[numpy.number], optional): The dtype of the index file. Defaults to numpy.int32.

        multimodal (bool, optional): Whether the dataset is multimodal. Defaults to False.
    """

    def __init__(self, bin_path: str, dtype: Type[numpy.number] = numpy.int32, multimodal: bool = False) -> None:
        self.data_file = open(bin_path, "wb")
        self.dtype = dtype
        self.multimodal = multimodal

        self.sequence_lengths = []
        self.document_indices = [0]
        self.sequence_modes = [] if self.multimodal else None

    def add_item(self, tensor: torch.Tensor, mode: int = 0) -> None:
        """Add a single item to the dataset

        Args:
            tensor (torch.Tensor): The item to add to the data file

            mode (int, optional): The mode for the item. Defaults to 0.
        """
        np_array = numpy.array(tensor.numpy(), dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.append(np_array.size)
        if self.multimodal:
            self.sequence_modes.append(mode)

    def add_document(self, tensor: torch.Tensor, lengths: List[int], modes: Optional[List[int]] = None) -> None:
        """Add an entire document to the dataset

        Args:
            tensor (torch.Tensor): The document to add

            lengths (List[int]): The lengths of each item in the document

            modes (Optional[List[int]], optional): The modes for each item in the document.
                Defaults to None.
        """
        np_array = numpy.array(tensor, dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.extend(lengths)
        self.document_indices.append(len(self.sequence_lengths))
        if self.multimodal:
            self.sequence_modes.extend(modes if modes is not None else [0] * lengths)

    def end_document(self) -> None:
        """Finalize the document, for use with IndexedDatasetBuilder.add_item"""
        self.document_indices.append(len(self.sequence_lengths))

    def add_index(self, path_prefix: str) -> None:
        """Add an entire IndexedDataset to the dataset

        Args:
            path_prefix (str): The index (.idx) and data (.bin) prefix
        """
        # Concatenate index
        index = _IndexReader(get_idx_path(path_prefix), multimodal=self.multimodal)
        assert index.dtype == self.dtype

        offset = len(self.sequence_lengths)
        self.sequence_lengths.extend(index.sequence_lengths)
        self.document_indices.extend((offset + index.document_indices)[1:])

        if self.multimodal:
            assert index.sequence_modes is not None, "sequence_modes cannot not be None"
            self.sequence_modes.extend(index.sequence_modes)

        # Free up memory to make space for new indices
        del index
        gc.collect()

        # Concatenate data
        with open(get_bin_path(path_prefix), "rb") as f:
            shutil.copyfileobj(f, self.data_file)

    def finalize(self, idx_path: str) -> None:
        """Clean up and write the index (.idx) file

        Args:
            idx_path (str): The path to the index file
        """
        self.data_file.close()
        with _IndexWriter(idx_path, self.dtype) as writer:
            writer.write(self.sequence_lengths, self.sequence_modes, self.document_indices)


def get_idx_path(path_prefix: str) -> str:
    return path_prefix + ".idx"


def get_bin_path(path_prefix: str) -> str:
    return path_prefix + ".bin"


def _normalize_prefix(path_prefix: str) -> str:
    if path_prefix.endswith(".bin") or path_prefix.endswith(".idx"):
        return path_prefix[:-4]
    return path_prefix
