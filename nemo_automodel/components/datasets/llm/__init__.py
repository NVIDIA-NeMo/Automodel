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

"""Load public dataset factories without initializing unrelated backends."""

import importlib
from types import FunctionType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_chat import make_agent_chat_dataset as make_agent_chat_dataset
    from .chat_dataset import ChatDataset as ChatDataset
    from .column_mapped_text_instruction_dataset import (
        ColumnMappedTextInstructionDataset as ColumnMappedTextInstructionDataset,
    )
    from .column_mapped_text_instruction_iterable_dataset import (
        ColumnMappedTextInstructionIterableDataset as ColumnMappedTextInstructionIterableDataset,
    )
    from .delta_lake_dataset import DeltaLakeDataset as DeltaLakeDataset
    from .delta_lake_dataset import is_delta_lake_path as is_delta_lake_path
    from .nanogpt_dataset import NanogptDataset as NanogptDataset
    from .neat_packing import neat_pack_dataset as neat_pack_dataset
    from .retrieval_collator import BiEncoderCollator as BiEncoderCollator
    from .retrieval_collator import CrossEncoderCollator as CrossEncoderCollator
    from .retrieval_collator import ProcessorMethodCollator as ProcessorMethodCollator
    from .retrieval_collator import (
        make_vision_collator_from_processor_method as make_vision_collator_from_processor_method,
    )
    from .retrieval_dataset import make_retrieval_dataset as make_retrieval_dataset
    from .retrieval_dataset_normalized import make_normalized_retrieval_dataset as make_normalized_retrieval_dataset
    from .retrieval_distill_collator import BiEncoderDistillCollator as BiEncoderDistillCollator
    from .squad import make_squad_dataset as make_squad_dataset
    from .xlam import make_xlam_dataset as make_xlam_dataset


# Keep submodule imports cheap in preprocessing workers. Resolving an export
# still returns the original factory/class, with its native module identity.
_LAZY_ATTRS = {
    "make_agent_chat_dataset": (".agent_chat", "make_agent_chat_dataset"),
    "ChatDataset": (".chat_dataset", "ChatDataset"),
    "ColumnMappedTextInstructionDataset": (
        ".column_mapped_text_instruction_dataset",
        "ColumnMappedTextInstructionDataset",
    ),
    "ColumnMappedTextInstructionIterableDataset": (
        ".column_mapped_text_instruction_iterable_dataset",
        "ColumnMappedTextInstructionIterableDataset",
    ),
    "DeltaLakeDataset": (".delta_lake_dataset", "DeltaLakeDataset"),
    "is_delta_lake_path": (".delta_lake_dataset", "is_delta_lake_path"),
    "NanogptDataset": (".nanogpt_dataset", "NanogptDataset"),
    "neat_pack_dataset": (".neat_packing", "neat_pack_dataset"),
    "BiEncoderCollator": (".retrieval_collator", "BiEncoderCollator"),
    "CrossEncoderCollator": (".retrieval_collator", "CrossEncoderCollator"),
    "ProcessorMethodCollator": (".retrieval_collator", "ProcessorMethodCollator"),
    "make_vision_collator_from_processor_method": (".retrieval_collator", "make_vision_collator_from_processor_method"),
    "make_retrieval_dataset": (".retrieval_dataset", "make_retrieval_dataset"),
    "make_normalized_retrieval_dataset": (".retrieval_dataset_normalized", "make_normalized_retrieval_dataset"),
    "BiEncoderDistillCollator": (".retrieval_distill_collator", "BiEncoderDistillCollator"),
    "make_squad_dataset": (".squad", "make_squad_dataset"),
    "make_xlam_dataset": (".xlam", "make_xlam_dataset"),
}


__all__ = [
    "NanogptDataset",
    "make_squad_dataset",
    "make_retrieval_dataset",
    "make_normalized_retrieval_dataset",
    "make_xlam_dataset",
    "make_agent_chat_dataset",
    "BiEncoderCollator",
    "BiEncoderDistillCollator",
    "CrossEncoderCollator",
    "ProcessorMethodCollator",
    "make_vision_collator_from_processor_method",
    "ColumnMappedTextInstructionDataset",
    "ColumnMappedTextInstructionIterableDataset",
    "ChatDataset",
    "DeltaLakeDataset",
    "is_delta_lake_path",
]


def __getattr__(name: str) -> type | FunctionType:
    """Resolve and cache an existing public export from its owning module."""
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _LAZY_ATTRS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Keep public factories discoverable without importing them."""
    return sorted(_LAZY_ATTRS)
