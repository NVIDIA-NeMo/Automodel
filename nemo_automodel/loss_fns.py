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

"""Built-in tinker-style loss functions and the ``LossFn`` contract.

A ``LossFn`` follows the design doc's signature::

    LossFn = Callable[[ModelOutput, Sequence[Datum]], Sequence[torch.Tensor]]

It returns **one tensor per datum**, in input order. Each tensor is either:

* **token-level** — shape ``[T_i]`` (same as ``datum.loss_inputs["weights"]``):
  the Engine multiplies by ``weights`` and sums, normalizing by the global token
  count across data ranks.
* **sample-level** — a scalar: the Engine sums and normalizes by the global
  sample count.

The loss returns the *un-reduced, un-weighted* per-token signal; reduction,
masking, and the global denominator are the Engine's responsibility (see
``Engine._reduce_datum_losses`` / ``Engine._global_denominator``). This keeps
the loss free of any distributed/scaling knowledge — the doc's separation of
"users own the algorithm, infra owns the heavy lifting".

The built-ins read selected-token logprobs from ``model_output.logprobs`` (per
datum, ``[T_i]``, the logprob the model assigns to that datum's
``target_tokens``) and pull old logprobs / advantages from the datum's
``loss_inputs``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # excluded from the import-linter contract
    from nemo_automodel.components.datasets.datum import Datum
    from nemo_automodel.components.training.model_output import ModelOutput

LossFn = Callable[["ModelOutput", Sequence["Datum"]], Sequence[torch.Tensor]]

__all__ = ["LossFn", "BUILTIN_LOSSES", "cross_entropy", "importance_sampling", "ppo"]


def _require_logprobs(model_output: "ModelOutput") -> list[torch.Tensor]:
    if model_output.logprobs is None:
        raise ValueError("loss requires per-datum logprobs in ModelOutput (forward produced none).")
    return model_output.logprobs


def cross_entropy(model_output: "ModelOutput", datums: Sequence["Datum"], **kwargs) -> list[torch.Tensor]:
    """Token-level negative log-likelihood of each datum's target tokens."""
    return [-lp for lp in _require_logprobs(model_output)]


def importance_sampling(model_output: "ModelOutput", datums: Sequence["Datum"], **kwargs) -> list[torch.Tensor]:
    """Token-level importance-sampling policy-gradient loss.

    ``-(exp(logprob - old_logprob) * advantage)`` per token. Reads
    ``loss_inputs["logprobs"]`` (behavior-policy logprobs) and
    ``loss_inputs["advantages"]`` from each datum.
    """
    out = []
    for lp, d in zip(_require_logprobs(model_output), datums):
        old = d.loss_inputs["logprobs"].to(lp)
        adv = d.loss_inputs["advantages"].to(lp)
        out.append(-(torch.exp(lp - old) * adv))
    return out


def ppo(
    model_output: "ModelOutput",
    datums: Sequence["Datum"],
    *,
    clip_eps: float = 0.2,
    **kwargs,
) -> list[torch.Tensor]:
    """Token-level PPO clipped surrogate loss.

    ``-min(ratio * adv, clip(ratio, 1±clip_eps) * adv)`` per token.
    """
    out = []
    for lp, d in zip(_require_logprobs(model_output), datums):
        old = d.loss_inputs["logprobs"].to(lp)
        adv = d.loss_inputs["advantages"].to(lp)
        ratio = torch.exp(lp - old)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        out.append(-torch.minimum(unclipped, clipped))
    return out


BUILTIN_LOSSES: dict[str, LossFn] = {
    "cross_entropy": cross_entropy,
    "importance_sampling": importance_sampling,
    "ppo": ppo,
}
