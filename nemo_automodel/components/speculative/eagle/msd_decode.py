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

"""Reference multimodal speculative decoding primitives for MSD.

The module implements the feature-level EAGLE/MSD decoding contract in plain
PyTorch. It intentionally keeps the proposal tree and posterior verification
separate from serving-engine kernels: engines can use :class:`MSDTreeLayout` to
execute the tree in one attention pass, while the Hugging Face reference path
verifies each leaf independently for correctness and integration testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from nemo_automodel.components.speculative.eagle.msd_target import HFMSDTargetModel


@dataclass(frozen=True)
class MSDTreeNode:
    """One drafted text-token node in an MSD candidate tree."""

    index: int
    parent_index: int
    token_id: int
    depth: int
    log_probability: float


@dataclass(frozen=True)
class MSDTreeLayout:
    """Tree attention and retrieval metadata for an MSD proposal.

    Node zero represents the target-sampled root token. ``attention_mask`` is
    boolean and contains the root plus every ancestor for each node. Serving
    backends turn its false entries into additive ``-inf`` attention bias.
    """

    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    retrieve_indices: torch.Tensor


@dataclass(frozen=True)
class MSDTreeProposal:
    """A target root token plus recursively drafted candidate-tree nodes."""

    root_token_id: int
    nodes: tuple[MSDTreeNode, ...]
    leaf_indices: tuple[int, ...]
    layout: MSDTreeLayout

    def candidate_path(self, leaf_index: int) -> tuple[int, ...]:
        """Return the root-to-leaf candidate-token path."""
        node_by_index = {node.index: node for node in self.nodes}
        if leaf_index not in node_by_index:
            raise ValueError(f"leaf_index {leaf_index} is not part of this proposal.")

        path: list[int] = []
        current = leaf_index
        while current:
            node = node_by_index[current]
            path.append(node.token_id)
            current = node.parent_index
        return (self.root_token_id, *reversed(path))

    def candidate_paths(self) -> tuple[tuple[int, ...], ...]:
        """Return all leaf paths in deterministic tree order."""
        return tuple(self.candidate_path(leaf_index) for leaf_index in self.leaf_indices)


@dataclass(frozen=True)
class MSDVerificationResult:
    """Greedy posterior verification result for one candidate tree."""

    accepted_token_ids: tuple[int, ...]
    bonus_token_id: int
    accepted_draft_tokens: int
    leaf_index: int | None

    @property
    def emitted_token_ids(self) -> tuple[int, ...]:
        """Return accepted candidate tokens followed by the target bonus token."""
        return (*self.accepted_token_ids, self.bonus_token_id)


@dataclass(frozen=True)
class MSDStochasticStep:
    """Result of one lossless speculative acceptance decision."""

    token_id: int
    accepted: bool


@dataclass(frozen=True)
class MSDStochasticVerificationResult:
    """Lossless stochastic verification result for a linear candidate chain."""

    emitted_token_ids: tuple[int, ...]
    accepted_draft_tokens: int


@dataclass
class _DraftState:
    """Internal recursive feature-draft state for one tree node."""

    node_index: int
    inputs_embeds: torch.Tensor
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    image_mask: torch.Tensor
    log_probability: float


def build_msd_tree_layout(
    nodes: Iterable[MSDTreeNode], leaf_indices: Iterable[int], *, device: torch.device
) -> MSDTreeLayout:
    """Build engine-ready tree attention and root-to-leaf retrieval metadata."""
    ordered_nodes = tuple(sorted(nodes, key=lambda node: node.index))
    node_by_index = {node.index: node for node in ordered_nodes}
    expected_indices = list(range(1, len(ordered_nodes) + 1))
    if [node.index for node in ordered_nodes] != expected_indices:
        raise ValueError("MSD tree node indices must be consecutive and start at one.")
    if any(node.parent_index < 0 or node.parent_index >= node.index for node in ordered_nodes):
        raise ValueError("MSD tree parents must refer to earlier nodes or root index zero.")

    tree_size = len(ordered_nodes) + 1
    attention_mask = torch.zeros(tree_size, tree_size, dtype=torch.bool, device=device)
    position_ids = torch.zeros(tree_size, dtype=torch.long, device=device)
    attention_mask[0, 0] = True
    for node in ordered_nodes:
        position_ids[node.index] = node.depth
        attention_mask[node.index, node.index] = True
        attention_mask[node.index, attention_mask[node.parent_index]] = True

    paths: list[list[int]] = []
    for leaf_index in leaf_indices:
        if leaf_index not in node_by_index:
            raise ValueError(f"Leaf index {leaf_index} is not an MSD tree node.")
        path: list[int] = []
        current = leaf_index
        while current:
            path.append(current)
            current = node_by_index[current].parent_index
        paths.append([0, *reversed(path)])

    if not paths:
        raise ValueError("MSD tree proposals require at least one leaf.")
    max_depth = max(len(path) for path in paths)
    retrieve_indices = torch.full((len(paths), max_depth), -1, dtype=torch.long, device=device)
    for row, path in enumerate(paths):
        retrieve_indices[row, : len(path)] = torch.tensor(path, dtype=torch.long, device=device)
    return MSDTreeLayout(attention_mask=attention_mask, position_ids=position_ids, retrieve_indices=retrieve_indices)


class MSDTreeDraftGenerator:
    """Recursively draft a top-k MSD feature tree from target prefix features."""

    def __init__(self, draft_model: nn.Module, target_lm_head: nn.Module, target_embeddings: nn.Module) -> None:
        self.draft_model = draft_model
        self.target_lm_head = target_lm_head
        self.target_embeddings = target_embeddings

    @torch.inference_mode()
    def propose(
        self,
        *,
        shifted_inputs_embeds: torch.Tensor,
        input_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        shifted_image_mask: torch.Tensor,
        root_token_id: int,
        draft_steps: int,
        top_k: int,
        beam_width: int,
    ) -> MSDTreeProposal:
        """Create a feature-level candidate tree after a target-sampled root.

        ``shifted_inputs_embeds`` and ``shifted_image_mask`` use the same
        next-token alignment as :class:`HFMSDTargetModel`. The final shifted
        embedding is replaced with the target root token embedding, then each
        recursive draft prediction is appended as the feature context for its
        child nodes.
        """
        if draft_steps < 1 or top_k < 1 or beam_width < 1:
            raise ValueError("draft_steps, top_k, and beam_width must all be positive.")
        if shifted_inputs_embeds.shape != input_hidden_states.shape:
            raise ValueError("shifted_inputs_embeds and input_hidden_states must have matching shapes.")
        if shifted_inputs_embeds.shape[:2] != attention_mask.shape or attention_mask.shape != shifted_image_mask.shape:
            raise ValueError("MSD draft tensors must share matching [batch, sequence] dimensions.")
        if shifted_inputs_embeds.shape[0] != 1:
            raise ValueError("The reference MSD tree generator currently supports batch size one.")

        active_length = int(attention_mask[0].sum().item())
        if active_length < 2:
            raise ValueError("MSD drafting requires at least two non-padding prefix tokens.")
        root_ids = torch.tensor([[root_token_id]], dtype=torch.long, device=shifted_inputs_embeds.device)
        root_embedding = self.target_embeddings(root_ids).to(shifted_inputs_embeds.dtype)
        initial_inputs = shifted_inputs_embeds[:, :active_length].clone()
        initial_inputs[:, -1:] = root_embedding
        initial_state = _DraftState(
            node_index=0,
            inputs_embeds=initial_inputs,
            hidden_states=input_hidden_states[:, :active_length],
            attention_mask=attention_mask[:, :active_length],
            image_mask=shifted_image_mask[:, :active_length],
            log_probability=0.0,
        )

        nodes: list[MSDTreeNode] = []
        states = [initial_state]
        next_index = 1
        for depth in range(1, draft_steps + 1):
            candidates: list[tuple[_DraftState, torch.Tensor, float, torch.Tensor]] = []
            for state in states:
                predicted_hidden = self.draft_model(
                    inputs_embeds=state.inputs_embeds,
                    target_hidden_states=state.hidden_states,
                    attention_mask=state.attention_mask,
                    image_mask=state.image_mask,
                )[:, -1:]
                logits = self.target_lm_head(predicted_hidden)[:, 0]
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                values, token_ids = torch.topk(log_probs, k=top_k, dim=-1)
                for value, token_id in zip(values[0], token_ids[0]):
                    candidates.append((state, token_id, state.log_probability + float(value.item()), predicted_hidden))

            candidates.sort(key=lambda item: item[2], reverse=True)
            selected = candidates[:beam_width]
            states = []
            for parent_state, token_id, log_probability, predicted_hidden in selected:
                index = next_index
                next_index += 1
                token = int(token_id.item())
                nodes.append(
                    MSDTreeNode(
                        index=index,
                        parent_index=parent_state.node_index,
                        token_id=token,
                        depth=depth,
                        log_probability=log_probability,
                    )
                )
                token_embedding = self.target_embeddings(token_id.reshape(1, 1)).to(parent_state.inputs_embeds.dtype)
                states.append(
                    _DraftState(
                        node_index=index,
                        inputs_embeds=torch.cat((parent_state.inputs_embeds, token_embedding), dim=1),
                        hidden_states=torch.cat((parent_state.hidden_states, predicted_hidden), dim=1),
                        attention_mask=torch.cat(
                            (parent_state.attention_mask, torch.ones_like(parent_state.attention_mask[:, :1])), dim=1
                        ),
                        image_mask=torch.cat(
                            (
                                parent_state.image_mask,
                                torch.zeros_like(parent_state.image_mask[:, :1], dtype=torch.bool),
                            ),
                            dim=1,
                        ),
                        log_probability=log_probability,
                    )
                )

        leaf_indices = tuple(state.node_index for state in states)
        layout = build_msd_tree_layout(nodes, leaf_indices, device=shifted_inputs_embeds.device)
        return MSDTreeProposal(
            root_token_id=root_token_id,
            nodes=tuple(nodes),
            leaf_indices=leaf_indices,
            layout=layout,
        )


def verify_greedy_tree(
    proposal: MSDTreeProposal,
    target_logits_by_leaf: Iterable[torch.Tensor],
) -> MSDVerificationResult:
    """Select the longest target-greedy accepted path and its bonus token.

    Each logits tensor has one row for every candidate token in the matching
    leaf path plus one final row for the target bonus token. This is the greedy
    form of EAGLE/MSD posterior verification.
    """
    paths = proposal.candidate_paths()
    logits_per_leaf = tuple(target_logits_by_leaf)
    if len(paths) != len(logits_per_leaf):
        raise ValueError("Target verification must provide one logits tensor per MSD leaf path.")

    best_accept_length = -1
    best_leaf_index: int | None = None
    best_path: tuple[int, ...] = ()
    best_logits: torch.Tensor | None = None
    for leaf_index, path, logits in zip(proposal.leaf_indices, paths, logits_per_leaf):
        if logits.ndim != 2 or logits.shape[0] < len(path) + 1:
            raise ValueError("Each target logits tensor must have [candidate_tokens + bonus, vocab] shape.")
        candidate_ids = torch.tensor(path, device=logits.device)
        greedy_ids = logits[: len(path)].argmax(dim=-1)
        accept_length = int((candidate_ids == greedy_ids).cumprod(dim=0).sum().item())
        if accept_length > best_accept_length:
            best_accept_length = accept_length
            best_leaf_index = leaf_index
            best_path = path
            best_logits = logits

    assert best_logits is not None
    bonus_token = int(best_logits[best_accept_length].argmax().item())
    accepted = best_path[:best_accept_length]
    return MSDVerificationResult(
        accepted_token_ids=accepted,
        bonus_token_id=bonus_token,
        accepted_draft_tokens=max(0, best_accept_length - 1),
        leaf_index=best_leaf_index,
    )


def accept_or_resample(
    *,
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    candidate_token_id: int,
    generator: torch.Generator | None = None,
    random_value: float | None = None,
) -> MSDStochasticStep:
    """Apply the lossless speculative acceptance rule for one candidate token."""
    if target_logits.ndim != 1 or draft_logits.ndim != 1 or target_logits.shape != draft_logits.shape:
        raise ValueError("target_logits and draft_logits must be matching one-dimensional tensors.")
    if not 0 <= candidate_token_id < target_logits.numel():
        raise ValueError("candidate_token_id must be inside the shared vocabulary.")
    if random_value is not None and not 0.0 <= random_value <= 1.0:
        raise ValueError("random_value must be in [0, 1].")

    target_probs = torch.softmax(target_logits.float(), dim=-1)
    draft_probs = torch.softmax(draft_logits.float(), dim=-1)
    acceptance = min(1.0, float((target_probs[candidate_token_id] / draft_probs[candidate_token_id]).item()))
    draw = random_value if random_value is not None else float(torch.rand((), generator=generator).item())
    if draw <= acceptance:
        return MSDStochasticStep(token_id=candidate_token_id, accepted=True)

    residual = (target_probs - draft_probs).clamp_min(0)
    residual = residual / residual.sum().clamp_min(torch.finfo(residual.dtype).eps)
    token_id = int(torch.multinomial(residual, num_samples=1, generator=generator).item())
    return MSDStochasticStep(token_id=token_id, accepted=False)


def verify_stochastic_chain(
    *,
    candidate_token_ids: Iterable[int],
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    generator: torch.Generator | None = None,
) -> MSDStochasticVerificationResult:
    """Losslessly verify a linear draft chain and emit the correction token."""
    candidates = tuple(candidate_token_ids)
    if target_logits.ndim != 2 or draft_logits.ndim != 2 or target_logits.shape != draft_logits.shape:
        raise ValueError("target_logits and draft_logits must be matching [steps, vocab] tensors.")
    if target_logits.shape[0] != len(candidates) + 1:
        raise ValueError("Verification requires one target-logits row per candidate plus one bonus row.")
    if draft_logits.shape[0] != len(candidates) + 1:
        raise ValueError("draft_logits must include an unused final row for shape alignment.")

    accepted: list[int] = []
    for index, candidate in enumerate(candidates):
        result = accept_or_resample(
            target_logits=target_logits[index],
            draft_logits=draft_logits[index],
            candidate_token_id=candidate,
            generator=generator,
        )
        if not result.accepted:
            return MSDStochasticVerificationResult(
                emitted_token_ids=(*accepted, result.token_id),
                accepted_draft_tokens=len(accepted),
            )
        accepted.append(candidate)

    bonus = int(
        torch.multinomial(torch.softmax(target_logits[-1].float(), dim=-1), num_samples=1, generator=generator).item()
    )
    return MSDStochasticVerificationResult(
        emitted_token_ids=(*accepted, bonus),
        accepted_draft_tokens=len(accepted),
    )


@torch.inference_mode()
def verify_hf_greedy_tree(
    *,
    model: nn.Module,
    model_inputs: dict[str, torch.Tensor],
    proposal: MSDTreeProposal,
) -> MSDVerificationResult:
    """Reference target verification for a batch-one Hugging Face VLM.

    The path-wise implementation is intentionally engine neutral and works for
    VLMs whose visual tensors are prompt-only. A serving backend can instead
    consume ``proposal.layout`` to verify the same tree in one tree-attention
    target forward.
    """
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    if input_ids.shape[0] != 1 or attention_mask.shape[0] != 1:
        raise ValueError("The Hugging Face MSD reference verifier currently supports batch size one.")
    active_length = int(attention_mask[0].sum().item())
    if active_length < 1:
        raise ValueError("MSD verification requires a non-empty prompt.")

    logits_by_leaf = []
    for path in proposal.candidate_paths():
        candidate_ids = torch.tensor([path], dtype=input_ids.dtype, device=input_ids.device)
        extended = dict(model_inputs)
        extended["input_ids"] = torch.cat((input_ids[:, :active_length], candidate_ids), dim=1)
        extended["attention_mask"] = torch.ones_like(extended["input_ids"], dtype=attention_mask.dtype)
        if "mm_token_type_ids" in extended:
            suffix = torch.zeros_like(candidate_ids, dtype=extended["mm_token_type_ids"].dtype)
            extended["mm_token_type_ids"] = torch.cat((extended["mm_token_type_ids"][:, :active_length], suffix), dim=1)
        extended.pop("position_ids", None)
        extended.pop("labels", None)
        outputs = model(return_dict=True, use_cache=False, **extended)
        logits_by_leaf.append(outputs.logits[0, active_length - 1 : active_length + len(path)])
    return verify_greedy_tree(proposal, logits_by_leaf)


class MSDGreedyDecoder:
    """Batch-one reference MSD round with recursive drafting and tree verification."""

    def __init__(self, target: HFMSDTargetModel, draft_model: nn.Module) -> None:
        self.target = target
        self.draft_model = draft_model
        self.target_embeddings = self._resolve_target_embeddings(target.model)
        self.generator = MSDTreeDraftGenerator(draft_model, target.get_lm_head(), self.target_embeddings)

    @staticmethod
    def _resolve_target_embeddings(model: nn.Module) -> nn.Module:
        """Find the language token embedding module across common HF VLM layouts."""
        candidates = [
            model,
            getattr(model, "model", None),
            getattr(getattr(model, "model", None), "language_model", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            get_embeddings = getattr(candidate, "get_input_embeddings", None)
            if callable(get_embeddings):
                embeddings = get_embeddings()
                if isinstance(embeddings, nn.Module):
                    return embeddings
            embeddings = getattr(candidate, "embed_tokens", None)
            if isinstance(embeddings, nn.Module):
                return embeddings
        raise ValueError("MSD decoding requires a target VLM exposing language token embeddings.")

    @torch.inference_mode()
    def decode_round(
        self,
        *,
        model_inputs: dict[str, torch.Tensor],
        draft_steps: int,
        top_k: int,
        beam_width: int,
    ) -> tuple[MSDTreeProposal, MSDVerificationResult]:
        """Draft and greedily verify one MSD candidate tree for a VLM prompt."""
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        active_length = int(attention_mask[0].sum().item())
        if input_ids.shape[0] != 1 or active_length < 2:
            raise ValueError("The MSD reference decoder requires a batch-one prompt with at least two tokens.")
        target_batch = self.target.generate_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=torch.ones_like(input_ids, dtype=torch.bool),
            model_inputs=model_inputs,
        )
        root_token_id = int(target_batch.target_logits[0, active_length - 2].argmax().item())
        proposal = self.generator.propose(
            shifted_inputs_embeds=target_batch.inputs_embeds,
            input_hidden_states=target_batch.input_hidden_states,
            attention_mask=target_batch.attention_mask,
            shifted_image_mask=target_batch.image_mask,
            root_token_id=root_token_id,
            draft_steps=draft_steps,
            top_k=top_k,
            beam_width=beam_width,
        )
        result = verify_hf_greedy_tree(model=self.target.model, model_inputs=model_inputs, proposal=proposal)
        return proposal, result
