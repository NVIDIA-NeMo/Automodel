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

import math
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn


# -----------------------------------------------------------------------------#
# 1.  Lightweight LoRA-enabled Linear layer                                    #
# -----------------------------------------------------------------------------#
class LinearLoRA(nn.Linear):
    """
    nn.Linear with an additive low-rank adapter (LoRA).
    """

    def __init__(
        self,
        orig_linear: nn.Linear,
        dim: int = 8,
        alpha: int = 32,
        dropout: float = 0.0,
        dropout_position: Literal["pre", "post"] = "post",
        lora_A_init: str = "xavier",
        lora_dtype: Optional[torch.dtype] = None,
    ):
        # Copy original Linear parameters
        super().__init__(
            in_features=orig_linear.in_features,
            out_features=orig_linear.out_features,
            bias=orig_linear.bias is not None,
            device=orig_linear.weight.device,
            dtype=orig_linear.weight.dtype,
        )
        self.weight.data.copy_(orig_linear.weight.data)
        if orig_linear.bias is not None:
            self.bias.data.copy_(orig_linear.bias.data)

        # ------------------------------------------------------------------ #
        # LoRA specific weights                                               #
        # ------------------------------------------------------------------ #
        self.rank = dim
        self.scale = alpha / dim
        dtype = lora_dtype or self.weight.dtype
        device = self.weight.device

        self.lora_A = nn.Linear(self.in_features, dim, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(dim, self.out_features, bias=False, device=device, dtype=dtype)

        if lora_A_init == "xavier":
            nn.init.xavier_uniform_(self.lora_A.weight)
        else:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout(dropout)
        assert dropout_position in ("pre", "post")
        self.dropout_position = dropout_position

        # Freeze base weights
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x):
        # Base linear (weight frozen)
        base = F.linear(x, self.weight, self.bias)

        # LoRA path
        if self.dropout_position == "pre":
            x = self.dropout(x)
        lora = self.lora_B(self.lora_A(x)) * self.scale
        if self.dropout_position == "post":
            lora = self.dropout(lora)

        return base + lora


# -----------------------------------------------------------------------------#
# 2.  Convenience: patch a model in-place                                      #
# -----------------------------------------------------------------------------#
def apply_lora_to_linear_modules(
    model: nn.Module,
    target_modules: List[str],
    dim: int = 8,
    alpha: int = 32,
    dropout: float = 0.0,
    dropout_position: Literal["pre", "post"] = "post",
    lora_A_init: str = "xavier",
    lora_dtype: Optional[torch.dtype] = None,
):
    """
    Replace selected nn.Linear layers with LinearLoRA layers (in-place).

    target_modules accepts wildcard fragments, e.g. ["q_proj", "k_proj", ".*fc.*"].
    """
    import re

    patterns = [re.compile(t) for t in target_modules]

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(p.fullmatch(name) or p.search(name) for p in patterns):
            parent, attr = _parent_and_attr(model, name)
            setattr(
                parent,
                attr,
                LinearLoRA(
                    module,
                    dim=dim,
                    alpha=alpha,
                    dropout=dropout,
                    dropout_position=dropout_position,
                    lora_A_init=lora_A_init,
                    lora_dtype=lora_dtype,
                ),
            )


def _parent_and_attr(root: nn.Module, fqname: str):
    """Return (parent_module, attribute_name) for fully-qualified module name."""
    parts = fqname.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


# -----------------------------------------------------------------------------#
# 3.  Merge utility                                                            #
# -----------------------------------------------------------------------------#
@torch.no_grad()
def merge_lora_weights(model: nn.Module):
    """
    Permanently adds the LoRA contribution into the base weights and
    deletes the LoRA parameters. Call after training if you want to
    discard adapters for inference.
    """
    for m in model.modules():
        if isinstance(m, LinearLoRA):
            # W ← W + (α/r) * B @ A
            delta_w = (m.scale * (m.lora_B.weight @ m.lora_A.weight)).to(m.weight.dtype)
            m.weight.data += delta_w

            # Remove LoRA parameters (turn layer into frozen plain Linear)
            del m.lora_A, m.lora_B, m.dropout
            m.__class__ = nn.Linear  # type: ignore
    return model


if __name__ == "__main__":
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(16, 16)
            self.linear2 = nn.Linear(16, 16)

        def forward(self, x):
            x = self.linear1(x).relu()
            x = self.linear2(x)
            return x

    model = ToyModel()
    apply_lora_to_linear_modules(model, target_modules=["linear1"], dim=4, alpha=16)
    dummy = torch.randn(2, 16)
    out = model(dummy)
    out.sum().backward()
    print("Forward/backward worked with LoRA.")

    # Merge & test
    merge_lora_weights(model)
    out2 = model(dummy)
    print("Forward after merge worked:", out2.shape)
