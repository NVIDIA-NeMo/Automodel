from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import math
import torch
from torch import nn


# -----------------------------------------------------------------------------#
# Helper: very small LoRA-style linear adapter (A ∘ B) with a weight-magnitude
# -----------------------------------------------------------------------------#
class LinearDoRAAdapter(nn.Module):
    """
    Implements a low-rank adaptation (LoRA-style) linear adapter with an additional
    per-output weight magnitude vector for adaptive scaling.

    This adapter computes:
        Output = (B @ A) * scaling

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        rank (int): Rank for low-rank decomposition. Default is 32.
        alpha (int): Scaling factor. Default is 64.
        dropout (float): Optional dropout probability. Default is 0.0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        alpha: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.linear_in = nn.Linear(in_features, rank, bias=False)
        self.linear_out = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.linear_in.weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear_out.weight)

        self.weight_magnitude = nn.Parameter(
            torch.ones(out_features, dtype=torch.get_default_dtype()),
            requires_grad=True,
        )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the adapter. Returns the low-rank adaptation.

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Adapted tensor output.
        """
        if self.dropout is not None and self.training:
            x = self.dropout(x)
        return self.linear_out(self.linear_in(x)) * self.scaling


class DoRALinear(nn.Module):
    """
    Wraps a standard nn.Linear module with a DoRA adapter, combining the base weight
    with a learnable low-rank adaptation and a per-output scaling vector.

    Implements Equation (5) from the DoRA paper, including dropout correction.

    Args:
        base_linear (nn.Linear): The original linear layer to wrap.
        adapter (LinearDoRAAdapter): The DoRA adapter to apply.
    """

    def __init__(self, base_linear: nn.Linear, adapter: LinearDoRAAdapter):
        super().__init__()
        self.base = base_linear
        self.adapter = adapter
        self._register_buffer("_cached_base_norm", self._get_weight_norm(), persistent=False)

    def _get_weight_norm(self) -> torch.Tensor:
        """
        Computes row-wise norm of the combined weight: ||W₀ + B·A||

        Returns:
            torch.Tensor: Row-wise L2 norm.
        """
        with torch.no_grad():
            rank_update = self.adapter.scaling * (
                self.adapter.linear_out.weight @ self.adapter.linear_in.weight
            )
            merged = self.base.weight.data + rank_update
            return torch.linalg.norm(merged, dim=1).to(merged.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining base and adapter output with magnitude scaling
        and optional dropout correction.

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after DoRA transformation.
        """
        base_out = self.base(x)
        adapter_out = self.adapter(x)

        current_norm = self._get_weight_norm()
        magnitude_scale = (self.adapter.weight_magnitude / current_norm).view(1, -1)

        if self.adapter.dropout is None or not self.training:
            dropout_correction = 0.0
        else:
            dropped_x = self.adapter.dropout(x)
            correction_term = magnitude_scale - 1.0
            dropout_correction = correction_term * self.base(dropped_x - x)

        return magnitude_scale * (base_out + adapter_out) + dropout_correction


@dataclass
class DoRA:
    """
    Utility class to apply DoRA transformations across specified submodules in a model.
    Finds `nn.Linear` layers with names matching `target_modules` and wraps them
    with `DoRALinear` using `LinearDoRAAdapter`.

    Args:
        target_modules (List[str]): List of substrings to match module names.
        rank (int): Rank of the low-rank adapter.
        alpha (int): Scaling factor for adapter.
        dropout (float): Dropout rate to use in adapter.
        dropout_position (Literal["pre", "post"]): Retained for API parity.
    """

    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "proj", "fc1", "fc2"]
    )
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "pre"

    def apply(self, model: nn.Module) -> nn.Module:
        """
        Applies DoRA transformation to specified modules within a model.

        Args:
            model (nn.Module): PyTorch model to modify.
        Returns:
            nn.Module: Model with specified linear layers wrapped in DoRA.
        """
        for full_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(tok in full_name for tok in self.target_modules):
                continue

            parent, child_name = self._parent_and_name(model, full_name)
            wrapped = DoRALinear(
                module,
                LinearDoRAAdapter(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                ),
            )
            setattr(parent, child_name, wrapped)
            print(f"[DoRA] wrapped → {full_name}")

        return model

    @staticmethod
    def _parent_and_name(root: nn.Module, dotted_name: str):
        """
        Resolves and returns the parent module and attribute name given a dotted path.

        Args:
            root (nn.Module): Root module.
            dotted_name (str): Dotted name path (e.g., 'layer1.linear').
        Returns:
            Tuple[nn.Module, str]: (parent_module, attribute_name)
        """
        parts = dotted_name.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent, parts[-1]
