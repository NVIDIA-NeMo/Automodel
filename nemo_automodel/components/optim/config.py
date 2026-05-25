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

"""Public config surface for the optim component.

Look here for the typed parameters that drive optimizer and LR scheduling.
Look at ``api.py`` for the builder functions that consume these configs.

Optimizer hierarchy::

    OptimizerConfig          (generic fallback — any optimizer via name + extra_kwargs)
    ├── AdamConfig           (torch.optim.Adam)
    ├── AdamWConfig          (torch.optim.AdamW)
    ├── FusedAdamConfig      (transformer_engine FusedAdam)
    ├── FlashAdamWConfig     (flashoptim.FlashAdamW)
    └── MuonConfig           (dion.Muon)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Optimizer configs
# ---------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    """Generic optimizer configuration (fallback for third-party optimizers).

    For known optimizers, prefer the typed subclasses below — they expose
    every parameter so you can read the config and know the full surface.

    For unknown / new optimizers, use this class directly with
    ``extra_kwargs`` for optimizer-specific parameters.

    Attributes:
        name: Dotted import path to the optimizer class
            (e.g. ``"torch.optim.AdamW"``).  Resolved by the builder
            via ``importlib``.
        lr: Learning rate.
        weight_decay: Weight decay coefficient.
        extra_kwargs: Pass-through keyword arguments forwarded to the
            optimizer constructor.
    """

    name: str = "torch.optim.AdamW"
    lr: float = 1e-4
    weight_decay: float = 0.01
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_kwargs(self) -> dict[str, Any]:
        """Return the full kwargs dict for the optimizer constructor."""
        return {"lr": self.lr, "weight_decay": self.weight_decay, **self.extra_kwargs}

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> OptimizerConfig:
        """Create the appropriate typed config from an optimizer name string.

        Looks up ``name`` in the optimizer registry.  Known optimizers
        return a typed subclass (e.g. ``AdamWConfig``) with full field
        validation.  Unknown optimizers return a base ``OptimizerConfig``
        with extra fields in ``extra_kwargs``.

        Args:
            name: Dotted import path (e.g. ``"torch.optim.AdamW"``).
            **kwargs: Optimizer hyper-parameters (``lr``, ``betas``, etc.).

        Returns:
            Typed ``OptimizerConfig`` subclass, or the generic base.
        """
        config_cls = _OPTIMIZER_REGISTRY.get(name, None)
        if config_cls is None:
            # Unknown optimizer — split known base fields from extra
            base_fields = {"lr", "weight_decay"}
            base_kwargs = {k: v for k, v in kwargs.items() if k in base_fields}
            extra = {k: v for k, v in kwargs.items() if k not in base_fields}
            return OptimizerConfig(name=name, extra_kwargs=extra, **base_kwargs)
        return config_cls(name=name, **kwargs)


@dataclass
class AdamConfig(OptimizerConfig):
    """Configuration for ``torch.optim.Adam``.

    Attributes:
        betas: Coefficients for computing running averages of gradient
            and its square.
        eps: Term added to the denominator for numerical stability.
        amsgrad: Whether to use the AMSGrad variant.
    """

    name: str = "torch.optim.Adam"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "betas": self.betas,
            "eps": self.eps,
            "amsgrad": self.amsgrad,
            **self.extra_kwargs,
        }


@dataclass
class AdamWConfig(OptimizerConfig):
    """Configuration for ``torch.optim.AdamW``.

    Attributes:
        betas: Coefficients for computing running averages of gradient
            and its square.
        eps: Term added to the denominator for numerical stability.
        amsgrad: Whether to use the AMSGrad variant.
        fused: Use the fused CUDA implementation (requires CUDA).
    """

    name: str = "torch.optim.AdamW"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    fused: bool = False

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "betas": self.betas,
            "eps": self.eps,
            "amsgrad": self.amsgrad,
            "fused": self.fused,
            **self.extra_kwargs,
        }


@dataclass
class FusedAdamConfig(OptimizerConfig):
    """Configuration for ``transformer_engine.pytorch.optimizers.FusedAdam``.

    Attributes:
        betas: Coefficients for computing running averages.
        eps: Numerical stability term.
        adam_w_mode: Use AdamW-style weight decay (decoupled).
        bias_correction: Apply bias correction to moments.
        master_weights: Keep FP32 master weights.
        master_weight_dtype: Dtype string for master weights
            (e.g. ``"torch.bfloat16"``).  Resolved by the builder.
    """

    name: str = "transformer_engine.pytorch.optimizers.FusedAdam"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    adam_w_mode: bool = True
    bias_correction: bool = True
    master_weights: bool = True
    master_weight_dtype: str | None = None

    def to_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "betas": self.betas,
            "eps": self.eps,
            "adam_w_mode": self.adam_w_mode,
            "bias_correction": self.bias_correction,
            "master_weights": self.master_weights,
            **self.extra_kwargs,
        }
        if self.master_weight_dtype is not None:
            kwargs["master_weight_dtype"] = self.master_weight_dtype
        return kwargs


@dataclass
class FlashAdamWConfig(OptimizerConfig):
    """Configuration for ``flashoptim.FlashAdamW``.

    Attributes:
        betas: Coefficients for computing running averages.
        eps: Numerical stability term.
        master_weight_bits: Bit-width for stochastic-rounding master weights
            (e.g. 24 for BF16-like precision with less memory).
    """

    name: str = "flashoptim.FlashAdamW"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    master_weight_bits: int = 24

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "betas": self.betas,
            "eps": self.eps,
            "master_weight_bits": self.master_weight_bits,
            **self.extra_kwargs,
        }


@dataclass
class MuonConfig(OptimizerConfig):
    """Configuration for ``dion.Muon``.

    Muon uses a matrix-aware update rule for 2D+ parameters and falls
    back to a scalar optimizer (AdamW by default) for 1D params
    (biases, normalization weights, embeddings).

    Attributes:
        mu: Muon momentum coefficient.
        betas: Adam betas for the scalar optimizer fallback.
        epsilon: Numerical stability term.
        adjust_lr: LR adjustment mode — ``"spectral_norm"`` or ``None``.
        scalar_opt: Optimizer name for scalar (non-matrix) parameters.
        scalar_betas: Adam betas for the scalar optimizer.
        scalar_eps: Epsilon for the scalar optimizer.
    """

    name: str = "dion.Muon"
    lr: float = 5e-4
    weight_decay: float = 0.0
    mu: float = 0.95
    betas: tuple[float, float] = (0.9, 0.95)
    epsilon: float = 1e-8
    adjust_lr: str = "spectral_norm"
    scalar_opt: str = "adamw"
    scalar_betas: tuple[float, float] = (0.9, 0.999)
    scalar_eps: float = 1e-8

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "mu": self.mu,
            "betas": self.betas,
            "epsilon": self.epsilon,
            "adjust_lr": self.adjust_lr,
            "scalar_opt": self.scalar_opt,
            "scalar_betas": self.scalar_betas,
            "scalar_eps": self.scalar_eps,
            **self.extra_kwargs,
        }


# ---------------------------------------------------------------------------
# LR scheduler config
# ---------------------------------------------------------------------------


@dataclass
class LRSchedulerConfig:
    """User-facing LR scheduler configuration.

    All fields are optional — the builder in ``api.py`` computes sensible
    defaults from the training schedule (total steps, optimizer base LR, etc.)
    for any field left as ``None``.

    Attributes:
        lr_warmup_steps: Number of linear warmup steps.  Default: min(1000, 10% of total steps).
        lr_decay_steps: Total steps over which the LR decays.  Default: total training steps.
        lr_decay_style: Decay curve — ``"cosine"``, ``"linear"``, ``"constant"``, ``"WSD"``,
            or ``"inverse-square-root"``.
        init_lr: LR at the start of warmup.  Default: 10% of base LR.
        max_lr: Peak LR after warmup.  Default: optimizer base LR.
        min_lr: Floor LR at end of decay.  Default: 1% of base LR.
        start_wd: Initial weight decay.  Default: optimizer weight_decay.
        end_wd: Final weight decay.  Default: same as ``start_wd``.
        wd_incr_steps: Steps over which WD ramps.  Default: ``lr_decay_steps``.
        wd_incr_style: WD ramp curve — ``"constant"``, ``"linear"``, or ``"cosine"``.
        use_checkpoint_opt_param_scheduler: Use checkpoint values when resuming.
        override_opt_param_scheduler: Force class values over checkpoint values.
        wsd_decay_steps: Decay steps for the WSD schedule tail.  Required when
            ``lr_decay_style="WSD"``.
        lr_wsd_decay_style: Sub-curve for the WSD tail — ``"linear"``, ``"cosine"``,
            ``"exponential"``, or ``"minus_sqrt"``.
    """

    lr_warmup_steps: int | None = None
    lr_decay_steps: int | None = None
    lr_decay_style: str = "cosine"
    init_lr: float | None = None
    max_lr: float | None = None
    min_lr: float | None = None
    start_wd: float | None = None
    end_wd: float | None = None
    wd_incr_steps: int | None = None
    wd_incr_style: str = "constant"
    use_checkpoint_opt_param_scheduler: bool = True
    override_opt_param_scheduler: bool = False
    wsd_decay_steps: int | None = None
    lr_wsd_decay_style: str | None = None


# ---------------------------------------------------------------------------
# Registry — maps dotted name → typed config subclass
# ---------------------------------------------------------------------------

_OPTIMIZER_REGISTRY: dict[str, type[OptimizerConfig]] = {
    "torch.optim.Adam": AdamConfig,
    "torch.optim.AdamW": AdamWConfig,
    "transformer_engine.pytorch.optimizers.FusedAdam": FusedAdamConfig,
    "transformer_engine.pytorch.optimizers.fused_adam.FusedAdam": FusedAdamConfig,
    "flashoptim.FlashAdamW": FlashAdamWConfig,
    "dion.Muon": MuonConfig,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _resolve_optimizer(name: str) -> Any:
    """Resolve a dotted path to an optimizer class.

    ``"torch.optim.AdamW"`` → ``torch.optim.AdamW``
    """
    import importlib

    parts = name.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Expected a dotted path like 'torch.optim.AdamW', got '{name}'")
    module_path, cls_name = parts
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name, None)
    if cls is None:
        raise ImportError(f"Cannot find '{cls_name}' in module '{module_path}'")
    return cls


__all__ = [
    "OptimizerConfig",
    "AdamConfig",
    "AdamWConfig",
    "FusedAdamConfig",
    "FlashAdamWConfig",
    "MuonConfig",
    "LRSchedulerConfig",
]
