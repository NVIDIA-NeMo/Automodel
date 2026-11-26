# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Fixed-anchor CPU parity with MMFlash's released objective.

The expected losses and logits gradients were produced by ``OnlineMMFlashModel``
at https://github.com/fscdc/MMFlash/tree/5b3fa3fe28f5bc6d7dfe75934a156c52b865dc8c
(MIT, ``specforge/algorithms/common/mmflash_model.py``) with the same logits,
anchors, target ids, masks, dequantized scores, and identity LM head. The author
computed CE in the input dtype and accumulated prefix weights in float32.
"""

import pytest
import torch
from torch import nn

from nemo_automodel.components.speculative.dflash.core import DFlashTrainerModule

pytestmark = pytest.mark.cpu_only


class _FixedDraft(nn.Module):
    input_embedding_scale = 1.0

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.logits = nn.Parameter(
            torch.tensor(
                [
                    [
                        [[2, -1, 0], [-0.5, 0.25, 1.5], [1.5, -0.5, 0.25], [0.25, 1, -1.25]],
                        [[0, 2, -1], [-1.25, -0.5, 0.5], [1.5, 0.25, -0.25], [0, 0.5, -1]],
                    ],
                    [
                        [[0, 0.5, -1], [-0.75, 1, 0.25], [0.75, 0, -0.75], [1, 0.25, -0.5]],
                        [[0.5, -0.75, 1], [0.25, -1.25, 0.75], [1, -0.75, 0], [-1, 0.75, 0.25]],
                    ],
                ],
                dtype=dtype,
            )
        )

    def forward(
        self,
        *,
        position_ids: torch.Tensor,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the fixture's draft states to an identity LM head.

        Args:
            position_ids: Absolute positions of shape [batch, sequence + draft].
            noise_embedding: Draft embeddings of shape [batch, draft, hidden].
            target_hidden: Target features of shape [batch, sequence, hidden].
            attention_mask: Dense attention of shape [batch, 1, draft, sequence + draft].

        Returns:
            Tensor of shape [batch, draft, vocab], with blocks flattened in draft order.
        """
        return self.logits.flatten(1, 2)

    def compute_logits(self, hidden: torch.Tensor, output_head: nn.Module) -> torch.Tensor:
        """Apply the head to the fixture's draft states.

        Args:
            hidden: Tensor of shape [batch, draft, vocab].
            output_head: Identity projection used to expose the fixed logits.

        Returns:
            Tensor of shape [batch, draft, vocab].
        """
        return output_head(hidden)


class _FixedAnchors(DFlashTrainerModule):
    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
        doc_remaining: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Override sampling so production target gathering sees fixed anchors.

        Args:
            seq_len: Context length.
            loss_mask: Supervision mask of shape [batch, sequence].
            device: Device for the returned positions and validity mask.
            doc_remaining: Optional document remainder of shape [batch, sequence].

        Returns:
            Anchor positions and their keep mask, both of shape [batch, blocks].
        """
        return torch.tensor([[1, 5], [0, 2]], device=device), torch.tensor([[True, True], [True, False]], device=device)


def _trainer(draft: _FixedDraft, **objective_overrides: float) -> DFlashTrainerModule:
    return _FixedAnchors(
        draft,
        nn.Identity(),
        nn.Embedding(3, 1),
        mask_token_id=0,
        loss_type="mmflash",
        block_size=4,
        num_anchors=2,
        attention_backend="sdpa",
        **objective_overrides,
    )


@pytest.fixture
def batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[0, 1, 2, 0, 2, 1, 0], [2, 1, 0, 2, 1, 0, 2]]),
        "loss_mask": torch.tensor([[1, 1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0]]),
        "hidden_states": torch.zeros(2, 7, 1),
        "visual_score": torch.tensor([[10000, 3100, 8200, 10000, 2300, 9100, 5600], [-1] * 7]),
    }


@pytest.mark.parametrize(
    "dtype,expected_loss,expected_gradient,rtol,atol",
    [
        pytest.param(
            torch.float32,
            2.57734251,
            [
                [0.05039238, 0.10668066, -0.15707305],
                [0.08259836, 0.17486073, -0.25745910],
                [-0.36946461, 0.09936433, 0.27010027],
                [0.04242323, -0.15774153, 0.11531829],
                [-0.13103762, 0.08899794, 0.04203966],
                [0.10660177, 0.05035511, -0.15695691],
            ],
            1e-6,
            1e-7,
            id="fp32",
        ),
        pytest.param(
            torch.bfloat16,
            2.57881856,
            [
                [0.05102539, 0.10693359, -0.15722656],
                [0.08251953, 0.17480469, -0.25781250],
                [-0.36914062, 0.09960938, 0.26953125],
                [0.04248047, -0.15820312, 0.11523438],
                [-0.13085938, 0.08837891, 0.04174805],
                [0.10644531, 0.05004883, -0.15722656],
            ],
            4e-3,
            1e-4,
            id="bf16",
        ),
    ],
)
@pytest.mark.parametrize("score_encoding", ["quantized", "float"])
def test_mmflash_matches_author_loss_and_logits_gradient(
    batch: dict[str, torch.Tensor],
    dtype: torch.dtype,
    expected_loss: float,
    expected_gradient: list[list[float]],
    rtol: float,
    atol: float,
    score_encoding: str,
) -> None:
    """Check loss and gradient of fixed blocks with text, scores, and masking.

    Args:
        batch: Inputs with [batch, sequence] IDs, loss mask and scores, and
            features of shape [batch, sequence, hidden].
        dtype: Dtype used for logits and the per-position cross-entropy.
        expected_loss: Author's scalar loss.
        expected_gradient: Author's gradients of the six supervised logit rows.
        rtol: Relative tolerance accounting for CPU kernel rounding.
        atol: Absolute tolerance accounting for CPU kernel rounding.
        score_encoding: Author int64 wire format or normalized floating point.
    """
    draft = _FixedDraft(dtype)
    scores = batch["visual_score"]
    if score_encoding == "float":
        scores = torch.where(scores < 0, -1.0, scores.float() / 10_000).requires_grad_()
        batch["visual_score"] = scores
        # The objective's accumulation precision follows the CE dtype.
        batch["loss_mask"] = batch["loss_mask"].double()

    out = _trainer(draft)(**batch)
    # The weighted sum is float32 even when cross-entropy is in bf16.
    torch.testing.assert_close(out.loss, torch.tensor(expected_loss), rtol=1e-6, atol=1e-6)
    assert out.loss_weight.item() == 2  # sequence count
    assert out.valid_tokens.item() == 6
    assert out.correct_tokens.item() == 3
    assert out.accept_len_sum.item() == 6
    assert out.valid_blocks.item() == 3
    out.loss.backward()

    # Anchor slots, the unsupervised gap, out-of-bounds labels and the padded
    # block all have zero gradient. The supervised positions after a gap still
    # receive the prefix product of the preceding supervised positions.
    expected = torch.zeros_like(draft.logits)
    expected[[0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 0, 0], [1, 3, 1, 1, 2, 3]] = torch.tensor(expected_gradient, dtype=dtype)
    torch.testing.assert_close(draft.logits.grad, expected, rtol=rtol, atol=atol)
    assert scores.grad is None


def test_mmflash_zero_alpha_keeps_the_author_prefix_weights(batch: dict[str, torch.Tensor]) -> None:
    """Check the author's alpha=0 loss for the same token-aligned batch as the parity test."""
    draft = _FixedDraft(torch.float32)
    no_boost = _trainer(draft, visual_alpha=0.0)(**batch)
    batch["visual_score"] = torch.full_like(batch["visual_score"], -1)
    text_only = _trainer(draft)(**batch)
    batch["visual_score"] = torch.zeros_like(batch["visual_score"])
    zero_score = _trainer(draft)(**batch)

    torch.testing.assert_close(no_boost.loss, torch.tensor(2.10724163))
    torch.testing.assert_close(text_only.loss, no_boost.loss, rtol=0, atol=0)
    torch.testing.assert_close(zero_score.loss, no_boost.loss, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype,aligned_loss,shifted_loss",
    [(torch.float32, 2.57734251, 3.18489265), (torch.bfloat16, 2.57881856, 3.18642354)],
)
def test_mmflash_gathers_scores_at_the_predicted_token(
    batch: dict[str, torch.Tensor], dtype: torch.dtype, aligned_loss: float, shifted_loss: float
) -> None:
    """Use different author losses to distinguish label scores from one-token-shifted scores.

    Args:
        batch: Fixed fixture with label-aligned input IDs, masks, features, and scores.
        dtype: Dtype used for fixture logits.
        aligned_loss: Author loss for the label-aligned scores.
        shifted_loss: Author loss with each position given its previous token's score.
    """
    trainer = _trainer(_FixedDraft(dtype))
    torch.testing.assert_close(trainer(**batch).loss, torch.tensor(aligned_loss), rtol=0, atol=5e-7)
    batch["visual_score"] = batch["visual_score"].roll(1, dims=1)
    torch.testing.assert_close(trainer(**batch).loss, torch.tensor(shifted_loss), rtol=0, atol=5e-7)


def test_mmflash_supports_configured_smoothing_and_gain(batch: dict[str, torch.Tensor]) -> None:
    """Changing smoothing and gain produces the author's loss on the parity batch."""
    out = _trainer(_FixedDraft(torch.float32), mmflash_smoothing=0.25, visual_alpha=1.5)(**batch)
    torch.testing.assert_close(out.loss, torch.tensor(1.71888304))


@pytest.mark.parametrize(
    "visual_alpha,smoothing,invalid_key", [(-1.0, 0.5, "visual_alpha"), (1.0, 1.5, "mmflash_smoothing")]
)
def test_mmflash_rejects_invalid_parameters(visual_alpha: float, smoothing: float, invalid_key: str) -> None:
    with pytest.raises(ValueError, match=invalid_key):
        _trainer(_FixedDraft(torch.float32), visual_alpha=visual_alpha, mmflash_smoothing=smoothing)


@pytest.mark.parametrize("score_error", ["missing", "shape", "dtype"])
def test_mmflash_requires_token_aligned_scores(batch: dict[str, torch.Tensor], score_error: str) -> None:
    """Reject missing, misaligned, or unsupported score tensors in the parity batch."""
    if score_error == "missing":
        del batch["visual_score"]
    elif score_error == "shape":
        batch["visual_score"] = batch["visual_score"][:, :-1]
    else:
        batch["visual_score"] = batch["visual_score"].int()

    with pytest.raises(ValueError, match="visual_score"):
        _trainer(_FixedDraft(torch.float32))(**batch)
