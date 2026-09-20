# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Cross-document masking of MTP targets for `neat`-packed batches.

``calculate_mtp_loss`` rolls labels left by one position per depth. In a packed batch
that roll crosses document boundaries, so the depth-k target at a document's last k
tokens would come from the next document. The guard accepts either ``cu_seqlens`` (THD
packing) or ``seq_idx`` (the indexed document map `neat` packing emits as
``_packed_seq_ids``); the recipes pass both so whichever the batch carries applies.
"""

import pytest
import torch

from nemo_automodel.components.loss.mtp import calculate_mtp_loss


class _RecordingLoss(torch.nn.Module):
    """Stand-in per-token loss that records the labels each depth receives."""

    def __init__(self):
        super().__init__()
        self.seen: list[torch.Tensor] = []

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        """Record ``labels`` of shape [batch, sequence] and return a scalar.

        Args:
            logits: Per-token logits of shape [batch, sequence, vocab].
            labels: Target ids of shape [batch, sequence].
            **kwargs: Ignored; accepted so the shared loss dispatch can call this.

        Returns:
            Scalar tensor derived from ``logits`` so autograd stays valid.
        """
        self.seen.append(labels.clone())
        return logits.sum() * 0.0


def _run(seq_idx: torch.Tensor | None, labels: torch.Tensor) -> torch.Tensor:
    """Run one depth of MTP loss and return the labels the loss actually saw."""
    loss_fn = _RecordingLoss()
    logits = torch.zeros(labels.shape[0], labels.shape[1], 7, requires_grad=True)
    calculate_mtp_loss(
        loss_fn,
        mtp_per_depth_logits=[logits],
        labels=labels,
        model=torch.nn.Identity(),
        scaling_factor=0.1,
        ignore_index=-100,
        seq_idx=seq_idx,
    )
    assert len(loss_fn.seen) == 1
    return loss_fn.seen[0]


class TestPackedMTPTargets:
    """The depth-1 target must never come from a different document."""

    def test_without_seq_idx_targets_cross_the_boundary(self):
        """Baseline: the bug this guard prevents."""
        # Documents [1,1,1] and [2,2]; labels are distinct per position.
        labels = torch.tensor([[10, 11, 12, 20, 21]])

        seen = _run(None, labels)

        # Position 2 is the last token of document 1. Its depth-1 target is rolled in
        # from position 3 -- the first token of document 2.
        assert seen[0, 2].item() == 20

    def test_seq_idx_masks_the_boundary_target(self):
        """With the document map, the boundary target is ignored instead."""
        labels = torch.tensor([[10, 11, 12, 20, 21]])
        seq_idx = torch.tensor([[1, 1, 1, 2, 2]], dtype=torch.int32)

        seen = _run(seq_idx, labels)

        assert seen[0, 2].item() == -100
        # Positions inside a document keep their ordinary next-token target.
        assert seen[0, 0].item() == 11
        assert seen[0, 1].item() == 12
        # The final position of the row has no future token at all.
        assert seen[0, 4].item() == -100

    def test_padding_is_also_excluded(self):
        """Padding carries document id 0, so rolls into or out of it are masked."""
        labels = torch.tensor([[10, 11, 20, 21, -100]])
        seq_idx = torch.tensor([[1, 1, 2, 2, 0]], dtype=torch.int32)

        seen = _run(seq_idx, labels)

        # Last token of document 1 -> would read document 2.
        assert seen[0, 1].item() == -100
        # Last token of document 2 -> would read padding.
        assert seen[0, 3].item() == -100
        assert seen[0, 0].item() == 11

    def test_indexed_map_from_the_collater_is_accepted_directly(self):
        """`_packed_seq_ids` is int64 1-based with 0 padding; it needs no conversion."""
        from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater

        batch = neat_packed_vlm_collater(
            [
                {
                    "input_ids": torch.tensor([10, 11, 12, 20, 21]),
                    "labels": torch.tensor([10, 11, 12, 20, 21]),
                    "attention_mask": torch.tensor([1, 1, 1, 2, 2]),
                    "position_ids": torch.tensor([0, 1, 2, 0, 1]),
                    "n_images": 0,
                    "n_videos": 0,
                }
            ],
            attn_implementation="te",
        )
        packed_seq_ids = batch["_packed_seq_ids"]

        seen = _run(packed_seq_ids, batch["labels"])

        assert seen[0, 2].item() == -100

    def test_mismatched_seq_idx_shape_raises(self):
        """A wiring error must fail loudly rather than mask the wrong positions."""
        labels = torch.tensor([[10, 11, 12, 20, 21]])
        seq_idx = torch.tensor([[1, 1, 2]], dtype=torch.int32)

        with pytest.raises(ValueError, match="does not"):
            _run(seq_idx, labels)
