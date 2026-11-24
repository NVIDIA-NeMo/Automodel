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

"""Native EAGLE-3.1 Parallel Reconsideration training (arXiv:2611.05724v1)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributed.nn.functional import all_reduce
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig

from nemo_automodel.components.speculative.eagle.core import Eagle3StepMetrics, _shift_left_with_zero
from nemo_automodel.components.speculative.eagle.draft_llama import LlamaEagle3DraftModel


@dataclass(frozen=True)
class ReconsiderationKV:
    """Rotated keys and values of shape [batch, heads, origin, head_dim]."""

    key: Tensor
    value: Tensor


@dataclass(frozen=True)
class ReconsiderationStep:
    """One pass's features [batch, origin, hidden] and uncommitted KV."""

    hidden: Tensor
    kv: ReconsiderationKV


class LlamaParallelReconsiderationDraftModel(LlamaEagle3DraftModel):
    """The EAGLE-3.1 layer and LM head with a feature query and a shared halt gate."""

    def __init__(self, config: PretrainedConfig) -> None:
        if (getattr(config, "attn_implementation", None) or "eager") != "eager":
            raise ValueError("Parallel Reconsideration requires eager draft attention")
        if not (getattr(config, "fc_norm", False) and getattr(config, "norm_output", False)):
            raise ValueError("Parallel Reconsideration requires EAGLE-3.1 fc_norm and norm_output")
        super().__init__(config)
        self.self_query = nn.Linear(config.hidden_size, config.hidden_size)
        self.halt_gate = nn.Linear(config.hidden_size, 1)
        self.post_init()
        # The inherited RMSNorms otherwise start in BF16 even when loading
        # FP32 optimizer weights through from_pretrained.
        self.to(dtype=self.lm_head.weight.dtype)

    def draft_step(
        self,
        input_embeds: Tensor,
        hidden: Tensor,
        mask: Tensor,
        positions: Tensor,
        *,
        prefix: tuple[ReconsiderationKV, ...] = (),
    ) -> ReconsiderationStep:
        """Run the shared M with explicit positions and a branch-owned cache.

        Args:
            input_embeds: Token embeddings or self-queries of shape [batch, origin, hidden].
            hidden: Features of shape [batch, origin, hidden].
            mask: Additive base attention mask of shape [batch, 1, origin, origin].
            positions: Absolute RoPE indices of shape [batch, origin].
            prefix: Base keys/values followed by one diagonal block per prior
                speculative position. Blocks have shape [batch, heads, origin, head_dim].

        Returns:
            Updated features [batch, origin, hidden] and this pass's key/value
            tensors [batch, heads, origin, head_dim].
        """

        def run(embeddings: Tensor, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            """Decode a pass without mutating its input prefix.

            Args:
                embeddings: Tensor of shape [batch, origin, hidden].
                features: Tensor of shape [batch, origin, hidden].

            Returns:
                Features of shape [batch, origin, hidden] and projected keys and
                values of shape [batch, heads, origin, head_dim].
            """
            keys, values = [block.key for block in prefix], [block.value for block in prefix]
            output = self.model.layers[0](
                input_embeds=embeddings,
                hidden_states=features,
                attention_mask=mask,
                position_ids=positions,
                cache_hidden=[keys, values],
                position_offset=0,
            )
            return self.model.norm(output), keys[-1], values[-1]

        input_embeds = input_embeds.to(hidden.dtype)
        if self.gradient_checkpointing and torch.is_grad_enabled():
            output, key, value = checkpoint(run, input_embeds, hidden, use_reentrant=False)
        else:
            output, key, value = run(input_embeds, hidden)
        return ReconsiderationStep(output, ReconsiderationKV(key, value))


def halt_probabilities(logits: Tensor) -> Tensor:
    """Apply the shared gate's stick-breaking distribution.

    Args:
        logits: Tensor of shape [..., depth_cap] with one logit per optional halt.

    Returns:
        FP32 halt masses of shape [..., depth_cap + 1], including forced halt.
    """
    halt = logits.float().sigmoid()
    terminal = torch.ones_like(halt[..., :1])
    remaining = torch.cat((terminal, (1 - halt).cumprod(-1)), dim=-1)
    return torch.cat((halt, terminal), dim=-1) * remaining


def reconsideration_token_loss(ce: Tensor, mass: Tensor) -> Tensor:
    """Apply Eq. (6)'s token terms with separate gate and CE gradients.

    Args:
        ce: Cross entropy of shape [..., depth], with depth zero first.
        mass: Halt probabilities of shape [..., depth].

    Returns:
        Combined loss of shape [...] before the marginal entropy term.
    """
    return ce[..., 0] + (mass[..., 1:].detach() * ce[..., 1:]).sum(-1) + (mass * ce.detach()).sum(-1)


def _select_halt(levels: list[ReconsiderationStep], depth: Tensor) -> ReconsiderationStep:
    """Select the same depth for the feature and its cache.

    Args:
        levels: Candidate features [batch, origin, hidden] with projected keys
            and values [batch, heads, origin, head_dim], ordered by depth.
        depth: Sampled depth indices of shape [batch, origin].

    Returns:
        A gathered feature and KV block with the same layouts as a single level.
    """
    features = torch.stack([level.hidden for level in levels], dim=2)
    hidden = features.gather(2, depth[..., None, None].expand(-1, -1, 1, features.shape[-1])).squeeze(2)

    def gather_block(blocks: list[Tensor]) -> Tensor:
        """Gather the sampled level of each origin.

        Args:
            blocks: One tensor of shape [batch, heads, origin, head_dim] per depth.

        Returns:
            One block of shape [batch, heads, origin, head_dim].
        """
        cache = torch.stack(blocks, dim=3)
        index = depth[:, None, :, None, None].expand(-1, cache.shape[1], -1, 1, cache.shape[-1])
        return cache.gather(3, index).squeeze(3)

    return ReconsiderationStep(
        hidden,
        ReconsiderationKV(
            gather_block([level.kv.key for level in levels]),
            gather_block([level.kv.value for level in levels]),
        ),
    )


def _ce(logits: Tensor, labels: Tensor) -> Tensor:
    """Compute the hard target loss for each origin.

    Args:
        logits: Tensor of shape [batch, origin, draft_vocab].
        labels: Integer draft IDs of shape [batch, origin].

    Returns:
        FP32 cross entropy of shape [batch, origin].
    """
    return F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="none").reshape_as(labels)


@dataclass(frozen=True)
class ParallelReconsiderationConfig:
    """Training options for the paper's Eq. (6).

    Attributes:
        max_reconsiderations: Positive cap on additional passes at a position.
        balance_beta: Coefficient for entropy of the minibatch halt marginal.
    """

    max_reconsiderations: int = 2
    balance_beta: float = 0.01

    def __post_init__(self) -> None:
        if type(self.max_reconsiderations) is not int or self.max_reconsiderations < 1:
            raise ValueError("max_reconsiderations must be a positive integer")
        if type(self.balance_beta) not in (int, float) or not math.isfinite(self.balance_beta) or self.balance_beta < 0:
            raise ValueError("balance_beta must be finite and nonnegative")

    def build(
        self,
        draft_model: LlamaParallelReconsiderationDraftModel,
        *,
        selected_token_ids: Tensor,
        selected_token_mask: Tensor,
        ttt_steps: int,
    ) -> ParallelReconsiderationTrainerModule:
        """Build the trainer with the configured reconsideration loss.

        Args:
            draft_model: Draft that owns the shared decoder, query, and gate.
            selected_token_ids: Target token IDs of shape [draft_vocab] in draft order.
            selected_token_mask: Eligibility mask of shape [target_vocab].
            ttt_steps: Maximum rollout length in positions.

        Returns:
            The trainer with its own target-vocabulary mapping buffers.
        """
        return ParallelReconsiderationTrainerModule(
            draft_model,
            self,
            selected_token_ids=selected_token_ids,
            selected_token_mask=selected_token_mask,
            ttt_steps=ttt_steps,
        )


class ParallelReconsiderationTrainerModule(nn.Module):
    """Train two feature rollouts over the same recorded token sequence."""

    def __init__(
        self,
        draft_model: LlamaParallelReconsiderationDraftModel,
        settings: ParallelReconsiderationConfig,
        *,
        selected_token_ids: Tensor,
        selected_token_mask: Tensor,
        ttt_steps: int,
    ) -> None:
        """Bind the draft and its vocabulary.

        Args:
            draft_model: Trainable reconsideration draft.
            settings: Validated loss and halt-depth configuration.
            selected_token_ids: Target IDs of shape [draft_vocab] in draft order.
            selected_token_mask: Eligibility mask of shape [target_vocab].
            ttt_steps: Maximum rollout length in positions.
        """
        super().__init__()
        if ttt_steps < 1:
            raise ValueError("ttt_steps must be positive")
        self.draft_model = draft_model
        self.settings = settings
        self.ttt_steps = ttt_steps
        self.register_buffer("selected_token_ids", selected_token_ids.long(), persistent=True)
        self.register_buffer("selected_token_mask", selected_token_mask.bool(), persistent=True)

    def _teacher_labels(
        self,
        loss_mask: Tensor,
        *,
        target_logits: Tensor | None,
        target_probs: Tensor | None,
        position_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Resolve hard teacher labels in the aligned EAGLE frame.

        Args:
            loss_mask: Eligible teacher rows of shape [batch, origin].
            target_logits: Live logits of shape [batch, origin, target_vocab].
            target_probs: Cached probabilities of shape [batch, origin, draft_vocab].
            position_mask: Cache eligibility of shape [batch, origin, 1].

        Returns:
            Draft IDs and a boolean eligibility mask, each of shape [batch, origin].
        """
        if target_logits is not None and target_probs is None and position_mask is None:
            top = target_logits.argmax(-1)
            # Cached columns use selected_token_ids in frequency order. The
            # eligibility mask alone cannot recover that order.
            reverse = self.selected_token_ids.new_zeros(self.selected_token_mask.shape)
            reverse[self.selected_token_ids] = torch.arange(self.selected_token_ids.numel(), device=top.device)
            return reverse[top], loss_mask.bool() & self.selected_token_mask[top]
        if target_logits is None and target_probs is not None and position_mask is not None:
            return target_probs.argmax(-1), position_mask.squeeze(-1).bool() & loss_mask.bool()
        raise ValueError("Provide live target_logits or cached target_probs and position_mask")

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        loss_mask: Tensor,
        aux_hidden_states: Tensor,
        target_logits: Tensor | None = None,
        *,
        target_probs: Tensor | None = None,
        position_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        seq_lens: Tensor | None = None,
        doc_remaining: Tensor | None = None,
    ) -> Eagle3StepMetrics:
        """Compute the native trainer's loss and one-pass metrics.

        Args:
            input_ids: EAGLE-shifted recorded token IDs [batch, origin], shared by both rollouts.
            attention_mask: Original real-token mask [batch, origin].
            loss_mask: Aligned teacher supervision mask [batch, origin].
            aux_hidden_states: Target taps [batch, origin, concatenated_hidden].
            target_logits: Aligned live teacher logits [batch, origin, target_vocab].
            target_probs: Aligned cached probabilities [batch, origin, draft_vocab].
            position_mask: Cached supervision eligibility [batch, origin, 1].
            position_ids: Optional original RoPE indices [batch, origin].
            seq_lens: Packed document lengths [num_documents]. Must be absent.
            doc_remaining: Packed positions remaining [batch, origin]. Must be absent.

        Returns:
            Scalar loss, accuracy and supervised-token count in the recipe's metric record.
        """
        if seq_lens is not None or doc_remaining is not None:
            raise ValueError("Parallel Reconsideration requires unpacked sequences")
        if input_ids.shape[1] < 2:
            raise ValueError("Parallel Reconsideration requires at least two positions")
        labels, eligible = self._teacher_labels(
            loss_mask, target_logits=target_logits, target_probs=target_probs, position_mask=position_mask
        )
        length = input_ids.shape[1]
        anchor = torch.arange(length, device=input_ids.device)
        if position_ids is None:
            position_ids = anchor.expand_as(input_ids)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=input_ids.device.type == "cuda"):
            draft = self.draft_model
            features = draft.project_hidden_states(aux_hidden_states)
            padding = attention_mask.bool()
            seed_valid = padding & _shift_left_with_zero(padding)
            base_visible = (anchor[:, None] >= anchor[None, :]) & seed_valid[:, None, None, :]
            strict_visible = base_visible & (anchor[:, None] != anchor[None, :])
            minimum = torch.finfo(features.dtype).min
            base_mask = torch.zeros_like(base_visible, dtype=features.dtype).masked_fill(~base_visible, minimum)
            strict_mask = torch.zeros_like(base_mask).masked_fill(~strict_visible, minimum)
            tokens = input_ids
            one = mixed = draft.draft_step(draft.embed_input_ids(tokens), features, base_mask, position_ids)
            one_prefix = mixed_prefix = (one.kv,)
            loss_sum, count, correct, halt_count = (features.new_zeros((), dtype=torch.float32) for _ in range(4))
            mass_sum = loss_sum.new_zeros(self.settings.max_reconsiderations + 1)
            steps = min(self.ttt_steps, length - 1)
            for step in range(steps):
                mask = (eligible & seed_valid).float()
                n = mask.sum()
                position = position_ids + step
                if step:
                    logits = draft.compute_logits(one.hidden)
                    loss_sum = loss_sum + (_ce(logits, labels) * mask).sum()
                    correct = correct + ((logits.detach().argmax(-1) == labels) * mask).sum()
                    count = count + n

                levels = [mixed]
                for _ in range(self.settings.max_reconsiderations):
                    previous = levels[-1].hidden
                    levels.append(
                        draft.draft_step(
                            draft.self_query(previous), previous, strict_mask, position, prefix=mixed_prefix
                        )
                    )
                mass = halt_probabilities(torch.cat([draft.halt_gate(level.hidden) for level in levels[:-1]], dim=-1))
                logits = draft.compute_logits(mixed.hidden)
                ce = torch.stack(
                    [_ce(logits, labels)] + [_ce(draft.compute_logits(x.hidden), labels) for x in levels[1:]], -1
                )
                loss_sum = loss_sum + (reconsideration_token_loss(ce, mass) * mask).sum()
                correct = correct + ((logits.detach().argmax(-1) == labels) * mask).sum()
                count = count + n
                halt_count = halt_count + n
                mass_sum = mass_sum + (mass * mask[..., None]).sum(dim=(0, 1))
                if step == steps - 1:
                    break

                depths = torch.multinomial(mass.detach().flatten(0, 1), 1).reshape_as(input_ids)
                chosen = _select_halt(levels, depths)
                mixed_prefix = (*mixed_prefix, chosen.kv)
                # Follow EAGLE-3 TTT: carry each branch's own feature and cache,
                # and advance on the recorded tokens that condition the teacher labels.
                tokens = _shift_left_with_zero(tokens)
                next_embeds = draft.embed_input_ids(tokens)
                one = draft.draft_step(
                    next_embeds,
                    one.hidden,
                    base_mask,
                    position + 1,
                    prefix=one_prefix,
                )
                one_prefix = (*one_prefix, one.kv)
                mixed = draft.draft_step(
                    next_embeds,
                    chosen.hidden,
                    strict_mask,
                    position + 1,
                    prefix=mixed_prefix,
                )
                labels = _shift_left_with_zero(labels)
                eligible = _shift_left_with_zero(eligible)

        counts = torch.stack((count, halt_count))
        replicas = dist.get_world_size() if self.training and dist.is_initialized() else 1
        if replicas > 1:
            mass_sum = all_reduce(mass_sum)
            dist.all_reduce(counts)
        marginal = mass_sum / counts[1].clamp_min(1)
        entropy = -(marginal * marginal.clamp_min(torch.finfo(marginal.dtype).tiny).log()).sum()
        loss = replicas * loss_sum / counts[0].clamp_min(1) - self.settings.balance_beta * entropy
        return Eagle3StepMetrics(loss=loss, accuracy=correct / count.clamp_min(1), valid_tokens=count)
