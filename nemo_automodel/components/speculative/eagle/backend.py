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

"""Backend abstraction for the EAGLE-3 target model.

The frozen target model is a *supervision provider*: for every training batch
it produces the auxiliary hidden states and the per-token target distribution
the draft model is trained against. EAGLE-3 training never updates the target,
so it does not have to share the training GPU. This interface lets the recipe
consume the target uniformly whether it runs co-located in-process
(``HFEagle3TargetModel``) or, in a later change, as a remote inference service
on separate GPUs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from nemo_automodel.components.speculative.eagle.target import Eagle3TargetBatch


class Eagle3TargetBackend(ABC):
    """Abstract contract every EAGLE-3 target-model backend implements.

    Two supervision encodings are allowed, both consumed directly by
    :meth:`Eagle3TrainerModule.forward`:

    - **full logits** -- :attr:`Eagle3TargetBatch.logits` carries the target's
      full-vocab logits and the draft-vocab projection happens trainer-side.
      Cheap when co-located (the tensor never leaves the GPU); impractical to
      ship over a wire because it is full-vocab sized.
    - **precomputed** -- :attr:`Eagle3TargetBatch.target_probs` and
      :attr:`Eagle3TargetBatch.position_mask` carry the already-projected
      draft-vocab distribution, so a backend that computes them itself (e.g. a
      remote server) only has to transfer draft-vocab-sized tensors.

    A backend returns exactly one of the two encodings from
    :meth:`generate_batch`; the recipe forwards whichever is present.
    """

    @abstractmethod
    def generate_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> "Eagle3TargetBatch":
        """Run the target and return the supervision for one training batch."""

    @abstractmethod
    def get_input_embeddings(self) -> nn.Module:
        """Return the target input-embedding module (used to seed the draft)."""

    def set_vocab_mapping(
        self,
        selected_token_ids: torch.Tensor,
        selected_token_mask: torch.Tensor,
    ) -> None:
        """Provide the draft-vocab mapping needed to precompute supervision.

        Co-located backends keep the mapping on the trainer module and derive
        the distribution there, so the default is a no-op. A backend that
        computes ``target_probs`` itself overrides this to receive the mapping.
        """

    @property
    def supports_async(self) -> bool:
        """Whether :meth:`generate_batch_async` is implemented (prefetch-capable)."""
        return False

    def generate_batch_async(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ):
        """Submit an asynchronous :meth:`generate_batch` for prefetch pipelining.

        Only backends that overlap target inference with draft training
        implement this; the default signals a synchronous backend so callers
        fall back to :meth:`generate_batch`.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support async prefetch")

    def close(self) -> None:
        """Release backend resources (remote connections, server handles)."""
