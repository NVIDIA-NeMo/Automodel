# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import logging
import os
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from diffusers import DiffusionPipeline

from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from nemo_automodel.shared.utils import dtype_from_str

logger = logging.getLogger(__name__)


def _choose_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _iter_pipeline_modules(pipe: DiffusionPipeline) -> Iterable[Tuple[str, nn.Module]]:
    # Prefer Diffusers' components registry when available
    if hasattr(pipe, "components") and isinstance(pipe.components, dict):
        for name, value in pipe.components.items():
            if isinstance(value, nn.Module):
                yield name, value
        return

    # Fallback: inspect attributes
    for name in dir(pipe):
        if name.startswith("_"):
            continue
        try:
            value = getattr(pipe, name)
        except Exception:
            continue
        if isinstance(value, nn.Module):
            yield name, value


def _move_module_to_device(module: nn.Module, device: torch.device, torch_dtype: Any) -> None:
    # torch_dtype can be "auto", torch.dtype, or string
    dtype: Optional[torch.dtype]
    if torch_dtype == "auto":
        dtype = None
    else:
        dtype = dtype_from_str(torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
    if dtype is not None:
        module.to(device=device, dtype=dtype)
    else:
        module.to(device=device)


class NeMoAutoDiffusionPipeline(DiffusionPipeline):
    """
    Drop-in Diffusers pipeline that adds optional FSDP2/TP parallelization during from_pretrained.

    Features:
    - Accepts a per-component mapping from component name to FSDP2Manager
    - Moves all nn.Module components to the chosen device/dtype
    - Parallelizes only components present in the mapping using their manager

    parallel_scheme:
    - Dict[str, FSDP2Manager]: component name -> manager used to parallelize that component
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        parallel_scheme: Optional[Dict[str, FSDP2Manager]] = None,
        device: Optional[torch.device] = None,
        torch_dtype: Any = "auto",
        move_to_device: bool = True,
        **kwargs,
    ) -> DiffusionPipeline:
        pipe: DiffusionPipeline = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        # Decide device
        dev = _choose_device(device)

        # Move modules to device/dtype first (helps avoid initial OOM during sharding)
        if move_to_device:
            for name, module in _iter_pipeline_modules(pipe):
                _move_module_to_device(module, dev, torch_dtype)

        # Use per-component FSDP2Manager mappings to parallelize components
        if parallel_scheme is not None:
            assert torch.distributed.is_initialized(), "Expect distributed environment to be initialized"
            for comp_name, comp_module in _iter_pipeline_modules(pipe):
                manager = parallel_scheme.get(comp_name)
                if manager is None:
                    continue
                try:
                    new_m = manager.parallelize(comp_module)
                    if new_m is not comp_module:
                        setattr(pipe, comp_name, new_m)
                except Exception as e:
                    logger.warning("FSDP2Manager.parallelize failed for %s: %s", comp_name, e)
        return pipe
