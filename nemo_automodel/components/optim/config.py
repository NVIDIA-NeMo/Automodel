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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OptimizerConfig:
    """User-facing optimizer configuration.

    Follows the veRL/VeOmni pattern: a ``name`` string that the builder
    resolves to a callable, plus common hyper-parameters as explicit fields.
    Optimizer-specific kwargs go in ``extra_kwargs``.

    Works from CLI, YAML, or Python::

        # Python (e.g. veRL integration)
        OptimizerConfig(name="torch.optim.AdamW", lr=1e-4)

        # YAML
        optimizer:
          name: torch.optim.AdamW
          lr: 1e-4
          weight_decay: 0.01
          extra_kwargs:
            betas: [0.9, 0.95]

    Attributes:
        name: Dotted import path to the optimizer class (e.g.
            ``"torch.optim.AdamW"``, ``"flashoptim.FlashAdamW"``).
            Resolved by the builder via ``importlib``.
        lr: Learning rate.
        weight_decay: Weight decay coefficient.
        extra_kwargs: Arbitrary additional keyword arguments forwarded to
            the optimizer constructor.  Use this for optimizer-specific
            params like ``betas``, ``momentum``, ``eps``, etc.
    """

    name: str = "torch.optim.AdamW"
    lr: float = 1e-4
    weight_decay: float = 0.01
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


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


__all__ = ["OptimizerConfig", "LRSchedulerConfig"]
