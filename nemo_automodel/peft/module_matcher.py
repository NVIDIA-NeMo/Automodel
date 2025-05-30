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

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set
import re
import torch.nn as nn

def wildcard_match(pattern, key):
    """
    Return whether the pattern (target module to add LoRA) matches the key (model weight name).

    Example:
    --------
        >>> wildcard_match("*.layers.0.*.linear_qkv", "decoder.layers.0.self_attention.linear_qkv")
        True
        >>> wildcard_match("*.layers.0.*.linear_qkv", "decoder.layers.1.self_attention.linear_qkv")
        False
    """
    if key is None:
        return None
    regex_pattern = re.compile("^" + pattern.replace("*", "(.*)") + "$")
    match = regex_pattern.match(key)
    return match is not None

@dataclass
class ModuleMatcher:
    """
    Matches Modules to apply PEFT adapters on.

    Args:
        target_modules (List[str], optional): A list of module names to apply LoRA to.
            Defaults to all linear layers ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'].
                - 'linear_qkv': Apply LoRA to the fused linear layer used for query, key, and value projections
                                in self-attention.
                - 'linear_proj': Apply LoRA to the linear layer used for projecting the output of self-attention.
                - 'linear_fc1': Apply LoRA to the first fully-connected layer in MLP.
                - 'linear_fc2': Apply LoRA to the second fully-connected layer in MLP.
            Target modules can also contain wildcards. For example, you can specify
                target_modules=['*.layers.0.*.linear_qkv', '*.layers.1.*.linear_qkv'] to add LoRA to only linear_qkv
                on the first two layers.
    """

    target_modules: List[str] = field(
        default_factory=lambda: ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2']
    )
    exclude_modules: List[str] = field(default_factory=list)
    canonical_mapping: Dict[str, Set] = field(default_factory=lambda: defaultdict(set))

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def match(self, m: nn.Module, name: str = None, prefix: str = None):
        """
        Return (pattern, full_name) if the module matches; otherwise None.
        """
        full_name = f"{prefix}.{name}" if prefix else name

        # 1. canonical_mapping takes absolute precedence
        if self.canonical_mapping:
            assert not self.exclude_modules, (
                "`exclude_modules` must be empty when `canonical_mapping` is used."
            )
            for pattern in self.canonical_mapping:
                if name == pattern or wildcard_match(pattern, full_name):
                    return (pattern, full_name)

        # 2. target_modules is the next most-specific rule set
        elif self.target_modules:
            assert not self.exclude_modules, (
                "`exclude_modules` must be empty when `target_modules` is used."
            )
            for pattern in self.target_modules:
                if name == pattern or wildcard_match(pattern, full_name):
                    return (pattern, full_name)

        # 3. Fallback: “all linear layers except those explicitly excluded”
        else:
            linear_types = (nn.Linear,)

            if (
                name not in self.exclude_modules
                and not any(wildcard_match(pattern, full_name) for pattern in self.exclude_modules)
                and isinstance(m, linear_types)
            ):
                return (name, full_name)

        return None
