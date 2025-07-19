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

from abc import abstractmethod

import torch


class LossFunction:
    """Signature for loss functions used in training.

    Loss functions compute a scalar loss value from model logits and other data.
    """

    @abstractmethod
    def __call__(
        self,
        next_token_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss and metrics from logprobs and other data.

        Args:
            next_token_logits: Logits from the model, typically with shape [batch_size, seq_len, vocab_size].
                               For each position (b, i), contains the logit distribution over the entire vocabulary
                               for predicting the next token (at position i+1). For example, if processing "The cat sat on",
                               then next_token_logits[b, 3] would contain the logits for predicting the token
                               that follows "on".
            labels: torch.Tensor
                This tensor should contain the labels for the next token with shape [batch_size, seq_len].
                It's used for the loss computation.

        Returns:
            torch.Tensor: The computed loss as a scalar tensor.
        """
        raise NotImplementedError("Subclasses must implement this method")