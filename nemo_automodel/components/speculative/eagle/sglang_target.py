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

"""SGLang-backed EAGLE-3 target model.

A third :class:`Eagle3TargetBackend` implementation: it runs the frozen target
through SGLang instead of HuggingFace, capturing the same EAGLE-3 supervision
(three auxiliary hidden states plus full-vocab logits) the co-located backend
produces. SGLang is the fastest serving path for mainstream architectures, so a
remote target server (:mod:`nemo_automodel.components.speculative.serve_target`)
can hold the target on dedicated GPUs while the draft trains elsewhere.

The class is split into two layers so the contract is unit-testable without a
GPU or SGLang installed:

- :class:`SGLangEagle3TargetModel` (this file) owns the *contract*: it assembles
  an :class:`Eagle3TargetBatch` whose shift / aux-concatenation semantics are
  identical to :class:`HFEagle3TargetModel`, so a SGLang run is numerically
  equivalent to the co-located one. It depends only on a small runner surface
  (``forward_eagle3`` / ``input_embedding_weight`` / ``set_aux_layers`` and a
  ``model`` exposing ``.config`` + ``.parameters()``), which tests fake.
- :class:`~nemo_automodel.components.speculative.eagle.sglang_runner.SGLangTargetRunner`
  owns the SGLang-internal forward and is lazily imported only by
  :meth:`SGLangEagle3TargetModel.from_pretrained`, so importing this module
  never pulls in SGLang.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional, Protocol, Sequence

import torch

from nemo_automodel.components.speculative.eagle.backend import Eagle3TargetBackend
from nemo_automodel.components.speculative.eagle.target import (
    Eagle3TargetBatch,
    _shift_left_with_zero,
    default_eagle3_aux_layer_ids,
    validate_eagle3_aux_layer_ids,
)

if TYPE_CHECKING:
    import torch.nn as nn


class SGLangRunnerProtocol(Protocol):
    """Minimal surface :class:`SGLangEagle3TargetModel` needs from a runner.

    Implemented for real by ``SGLangTargetRunner`` (GPU/SGLang) and faked in
    unit tests, which is why the backend depends on this protocol rather than on
    SGLang directly.
    """

    #: Loaded model handle exposing ``.config`` (with ``num_hidden_layers`` /
    #: ``hidden_size`` / ``vocab_size``) and ``.parameters()`` for device
    #: inference, mirroring what the server reads off ``HFEagle3TargetModel``.
    model: "nn.Module"

    def set_aux_layers(self, aux_layer_ids: Sequence[int]) -> None:
        """Tell the underlying model which 3 decoder layers to capture."""

    def forward_eagle3(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the target once and return ``(logits, aux_hidden_states)``.

        ``logits`` is ``[batch, seq, vocab]`` (full vocab, unshifted) and
        ``aux_hidden_states`` is ``[batch, seq, 3 * hidden]`` (the three capture
        layers concatenated on the last dim, unshifted).
        """

    def input_embedding_weight(self) -> torch.Tensor:
        """Return the target input-embedding weight ``[vocab, hidden]``."""


class SGLangEagle3TargetModel(Eagle3TargetBackend):
    """EAGLE-3 target backend that runs the frozen target through SGLang.

    Parameters
    ----------
    runner:
        A loaded runner implementing :class:`SGLangRunnerProtocol`.
    aux_layer_ids:
        The three decoder layers to capture (low / mid / high). When ``None``
        the shared EAGLE-3 default recipe is used, matching every other backend.
    """

    def __init__(self, runner: SGLangRunnerProtocol, aux_layer_ids: Optional[Sequence[int]] = None):
        self._runner = runner
        # Expose ``.model`` so the remote server can read the target's config
        # and infer its device the same way it does for the co-located backend.
        self.model = runner.model
        num_layers = self.model.config.num_hidden_layers
        if aux_layer_ids is None:
            aux_layer_ids = default_eagle3_aux_layer_ids(num_layers)
        self.aux_layer_ids = validate_eagle3_aux_layer_ids(aux_layer_ids, num_layers)
        runner.set_aux_layers(self.aux_layer_ids)

    def get_input_embeddings(self) -> SimpleNamespace:
        """Return an object exposing ``.weight`` (the target input embeddings).

        Matches the offline-cache / remote path: the draft's
        ``copy_embeddings_from_target`` only reads ``.weight``.
        """
        return SimpleNamespace(weight=self._runner.input_embedding_weight())

    @torch.no_grad()
    def generate_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Eagle3TargetBatch:
        """Run the SGLang target and capture aux hidden states plus logits.

        Produces an :class:`Eagle3TargetBatch` byte-for-byte compatible with
        :meth:`HFEagle3TargetModel.generate_batch`: the logits, ``input_ids``
        and ``loss_mask`` are shifted left by one (next-token alignment) while
        the aux hidden states are kept position-aligned. The draft-vocab
        projection happens trainer-side / server-side from ``logits``, exactly
        as in the co-located path.

        Note: sequences are assumed right-padded (loss_mask zeros the pad). With
        causal attention, trailing pad tokens do not affect earlier positions,
        so the captured supervision matches a masked HuggingFace forward.
        """
        logits, aux_hidden_states = self._runner.forward_eagle3(input_ids, attention_mask)
        return Eagle3TargetBatch(
            aux_hidden_states=aux_hidden_states,
            logits=_shift_left_with_zero(logits),
            input_ids=_shift_left_with_zero(input_ids),
            attention_mask=attention_mask,
            loss_mask=_shift_left_with_zero(loss_mask),
        )

    def close(self) -> None:
        """Release the SGLang runner (frees GPU memory / engine handles)."""
        close = getattr(self._runner, "close", None)
        if close is not None:
            close()

    @classmethod
    def from_pretrained(  # pragma: no cover - requires GPU + SGLang
        cls,
        model_path: str,
        *,
        aux_layer_ids: Optional[Sequence[int]] = None,
        dtype: Optional[torch.dtype] = None,
        tp_size: int = 1,
        trust_remote_code: bool = False,
        **sglang_kwargs,
    ) -> "SGLangEagle3TargetModel":
        """Build a SGLang runner for ``model_path`` and wrap it as a target backend.

        SGLang is imported here (not at module load) so this module stays
        importable in environments without SGLang; ``sglang_kwargs`` are passed
        through to SGLang's ``ServerArgs`` for endpoint / parallelism tuning.
        """
        from nemo_automodel.components.speculative.eagle.sglang_runner import SGLangTargetRunner

        runner = SGLangTargetRunner.build(
            model_path,
            dtype=dtype,
            tp_size=tp_size,
            trust_remote_code=trust_remote_code,
            **sglang_kwargs,
        )
        return cls(runner, aux_layer_ids=aux_layer_ids)
