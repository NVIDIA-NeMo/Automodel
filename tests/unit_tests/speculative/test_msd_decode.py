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

"""Unit tests for MSD recursive drafting and posterior verification."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.speculative.eagle.msd_decode import (
    MSDGreedyDecoder,
    MSDTreeDraftGenerator,
    MSDTreeNode,
    MSDTreeProposal,
    accept_or_resample,
    build_msd_tree_layout,
    verify_greedy_tree,
    verify_hf_greedy_tree,
    verify_stochastic_chain,
)


def _proposal() -> MSDTreeProposal:
    nodes = (
        MSDTreeNode(index=1, parent_index=0, token_id=2, depth=1, log_probability=-0.1),
        MSDTreeNode(index=2, parent_index=1, token_id=3, depth=2, log_probability=-0.2),
        MSDTreeNode(index=3, parent_index=0, token_id=4, depth=1, log_probability=-0.3),
    )
    return MSDTreeProposal(
        root_token_id=1,
        nodes=nodes,
        leaf_indices=(2, 3),
        layout=build_msd_tree_layout(nodes, (2, 3), device=torch.device("cpu")),
    )


def _logits(*token_ids: int, vocab_size: int = 6) -> torch.Tensor:
    logits = torch.full((len(token_ids), vocab_size), -20.0)
    for row, token_id in enumerate(token_ids):
        logits[row, token_id] = 20.0
    return logits


def test_msd_tree_layout_preserves_only_ancestor_attention() -> None:
    """Tree metadata exposes root-to-node attention and retrieval paths."""
    proposal = _proposal()

    assert proposal.candidate_paths() == ((1, 2, 3), (1, 4))
    assert torch.equal(proposal.layout.position_ids, torch.tensor([0, 1, 2, 1]))
    assert torch.equal(proposal.layout.attention_mask[2], torch.tensor([True, True, True, False]))
    assert torch.equal(proposal.layout.attention_mask[3], torch.tensor([True, False, False, True]))
    assert torch.equal(proposal.layout.retrieve_indices, torch.tensor([[0, 1, 2], [0, 3, -1]]))


def test_msd_greedy_verifier_selects_longest_accepted_leaf() -> None:
    """Target verification chooses the tree path with the longest prefix hit."""
    result = verify_greedy_tree(
        _proposal(),
        (
            _logits(1, 2, 0, 5),
            _logits(1, 0, 4),
        ),
    )

    assert result.accepted_token_ids == (1, 2)
    assert result.bonus_token_id == 0
    assert result.accepted_draft_tokens == 1
    assert result.leaf_index == 2


def test_msd_greedy_verifier_rejects_invalid_target_shapes() -> None:
    """A missing bonus distribution fails before posterior selection."""
    with pytest.raises(ValueError, match="candidate_tokens"):
        verify_greedy_tree(_proposal(), (_logits(1), _logits(1, 4, 0)))


def test_msd_lossless_acceptance_and_correction_distribution() -> None:
    """Rejected draft tokens are replaced from the normalized positive residual."""
    target_logits = torch.tensor([4.0, 0.0])
    draft_logits = torch.tensor([0.0, 4.0])

    accepted = accept_or_resample(
        target_logits=target_logits,
        draft_logits=draft_logits,
        candidate_token_id=1,
        random_value=0.0,
    )
    rejected = accept_or_resample(
        target_logits=target_logits,
        draft_logits=draft_logits,
        candidate_token_id=1,
        random_value=1.0,
    )

    assert accepted.accepted and accepted.token_id == 1
    assert not rejected.accepted and rejected.token_id == 0


def test_msd_stochastic_verifier_stops_at_first_rejection_or_samples_bonus() -> None:
    """Linear stochastic verification preserves accepted prefixes and emits one target token."""
    target_logits = _logits(1, 0, 2)
    draft_logits = _logits(1, 1, 0)
    generator = torch.Generator().manual_seed(0)
    result = verify_stochastic_chain(
        candidate_token_ids=(1, 1),
        target_logits=target_logits,
        draft_logits=draft_logits,
        generator=generator,
    )

    assert result.accepted_draft_tokens == 1
    assert result.emitted_token_ids == (1, 0)


class _IdentityFeatureDraft(nn.Module):
    """Draft whose hidden prediction is its final input embedding."""

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return input embeddings while keeping every argument in the graph."""
        return inputs_embeds + 0.0 * (
            target_hidden_states + attention_mask.unsqueeze(-1) + image_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        )


def test_msd_tree_generator_recursively_expands_and_prunes_feature_paths() -> None:
    """Feature drafting recursively grows a bounded top-k candidate tree."""
    embeddings = nn.Embedding.from_pretrained(torch.eye(5), freeze=True)
    lm_head = nn.Linear(5, 5, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.eye(5))
    generator = MSDTreeDraftGenerator(_IdentityFeatureDraft(), lm_head, embeddings)
    proposal = generator.propose(
        shifted_inputs_embeds=torch.zeros(1, 3, 5),
        input_hidden_states=torch.zeros(1, 3, 5),
        attention_mask=torch.ones(1, 3),
        shifted_image_mask=torch.zeros(1, 3, dtype=torch.bool),
        root_token_id=2,
        draft_steps=2,
        top_k=2,
        beam_width=2,
    )

    assert len(proposal.nodes) == 4
    assert len(proposal.leaf_indices) == 2
    assert all(len(path) == 3 for path in proposal.candidate_paths())
    assert all(path[0] == 2 for path in proposal.candidate_paths())


def test_msd_tree_generator_keeps_indices_dense_after_beam_pruning() -> None:
    """Pruned candidates cannot leave holes in tree-attention node indices."""
    embeddings = nn.Embedding.from_pretrained(torch.eye(5), freeze=True)
    lm_head = nn.Linear(5, 5, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.eye(5))
    proposal = MSDTreeDraftGenerator(_IdentityFeatureDraft(), lm_head, embeddings).propose(
        shifted_inputs_embeds=torch.zeros(1, 3, 5),
        input_hidden_states=torch.zeros(1, 3, 5),
        attention_mask=torch.ones(1, 3),
        shifted_image_mask=torch.zeros(1, 3, dtype=torch.bool),
        root_token_id=2,
        draft_steps=2,
        top_k=2,
        beam_width=1,
    )

    assert [node.index for node in proposal.nodes] == [1, 2]
    assert proposal.layout.attention_mask.shape == (3, 3)


class _TinyVerifierModel(nn.Module):
    """Target that predicts a deterministic successor for every input token."""

    def __init__(self) -> None:
        super().__init__()
        self.successor = torch.tensor([1, 2, 3, 0, 0, 0])

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool,
        use_cache: bool,
        **kwargs,
    ):
        """Return logits whose greedy token is the current token's successor."""
        del attention_mask, return_dict, use_cache, kwargs
        logits = torch.full((*input_ids.shape, 6), -20.0)
        next_ids = self.successor.to(input_ids.device)[input_ids]
        logits.scatter_(-1, next_ids.unsqueeze(-1), 20.0)
        return SimpleNamespace(logits=logits)


def test_hf_msd_verifier_replays_multimodal_prompt_paths() -> None:
    """The reference verifier extends text fields while preserving prompt media tensors."""
    result = verify_hf_greedy_tree(
        model=_TinyVerifierModel(),
        model_inputs={
            "input_ids": torch.tensor([[0]]),
            "attention_mask": torch.ones(1, 1),
            "mm_token_type_ids": torch.ones(1, 1, dtype=torch.long),
            "pixel_values": torch.ones(1, 3, 2, 2),
        },
        proposal=_proposal(),
    )

    assert result.accepted_token_ids == (1, 2, 3)
    assert result.bonus_token_id == 0
    assert result.accepted_draft_tokens == 2


class _TinyDecodeModel(_TinyVerifierModel):
    """Verifier model with the token embedding and LM head required by the decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(torch.eye(6), freeze=True)
        self.lm_head = nn.Linear(6, 6, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.eye(6))

    def get_input_embeddings(self) -> nn.Module:
        """Expose target token embeddings through the Hugging Face convention."""
        return self.embeddings


class _TinyDecodeTarget:
    """Target wrapper that returns deterministic feature supervision for a prompt."""

    def __init__(self) -> None:
        self.model = _TinyDecodeModel()

    def get_lm_head(self) -> nn.Module:
        """Return the frozen language head."""
        return self.model.lm_head

    def generate_batch(self, **kwargs):
        """Return a minimal MSD target batch aligned to input ids."""
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]
        embeddings = self.model.embeddings(input_ids)
        logits = self.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, use_cache=False
        ).logits
        shifted_logits = torch.cat((logits[:, 1:], torch.zeros_like(logits[:, :1])), dim=1)
        shifted_embeds = torch.cat((embeddings[:, 1:], torch.zeros_like(embeddings[:, :1])), dim=1)
        return SimpleNamespace(
            inputs_embeds=shifted_embeds,
            input_hidden_states=embeddings,
            attention_mask=attention_mask,
            image_mask=torch.zeros_like(input_ids, dtype=torch.bool),
            target_logits=shifted_logits,
        )


def test_msd_greedy_decoder_runs_recursive_draft_and_tree_verification() -> None:
    """The public decoder composes target prefill, drafting, and verification."""
    target = _TinyDecodeTarget()
    decoder = MSDGreedyDecoder(target, _IdentityFeatureDraft())

    proposal, result = decoder.decode_round(
        model_inputs={"input_ids": torch.tensor([[0, 1]]), "attention_mask": torch.ones(1, 2)},
        draft_steps=1,
        top_k=1,
        beam_width=1,
    )

    assert proposal.root_token_id == 2
    assert proposal.candidate_paths() == ((2, 2),)
    assert result.emitted_token_ids == (2, 3)
