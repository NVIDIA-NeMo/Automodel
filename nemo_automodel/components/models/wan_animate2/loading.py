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

"""Load the released Wan-Animate-2 modular checkpoint."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from nemo_automodel.shared.import_utils import safe_import

if TYPE_CHECKING:
    from diffusers import ModularPipeline

_, diffusers = safe_import("diffusers")


def load_pipeline(
    model_dir: str,
    model_args: tuple[object, ...],
    *,
    torch_dtype: torch.dtype | str | dict[str, torch.dtype],
    components_to_load: Iterable[str] | None,
    **kwargs: object,
) -> ModularPipeline:
    """Load Wan-Animate-2 through its released modular pipeline.

    Components present in the resolved snapshot are loaded from that snapshot,
    preserving offline use and the selected checkpoint revision.

    Args:
        model_dir: Resolved local checkpoint directory.
        model_args: Positional loading arguments; must be empty for modular pipelines.
        torch_dtype: Requested component weight dtype, or a mapping by component.
        components_to_load: Component names, or all pretrained components.
        **kwargs: Loading arguments forwarded to the upstream Diffusers APIs.

    Returns:
        The loaded modular Wan-Animate-2 pipeline.
    """
    if model_args:
        raise TypeError("Modular pipelines accept keyword loading arguments only")

    pipe = diffusers.ModularPipeline.from_pretrained(model_dir, **kwargs)
    names = list(components_to_load) if components_to_load is not None else pipe.pretrained_component_names
    loaded = {}
    for name in names:
        spec = pipe.get_component_spec(name)
        load_kwargs = dict(kwargs, dtype=torch_dtype)
        if (Path(model_dir) / (spec.subfolder or name)).is_dir():
            load_kwargs.update(pretrained_model_name_or_path=model_dir, subfolder=spec.subfolder or name, revision=None)
        # ComponentSpec.load propagates errors; load_components only logs them,
        # which could leave a requested training component silently unloaded.
        loaded[name] = spec.load(**load_kwargs)
    pipe.update_components(**loaded)
    return pipe
