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

"""DSpark training adapter for the native DeepSeek V4.1 draft backbone."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41TextConfig
from nemo_automodel.components.models.deepseek_v41.dspark import DeepseekV41DSparkBackbone
from nemo_automodel.components.speculative.dspark.common import (
    DSparkForwardOutput,
    build_eval_mask,
    create_noise_embed,
    sample_anchor_positions,
)
from nemo_automodel.shared.utils import dtype_from_str


class DeepseekV41DSparkModel(DeepseekV41DSparkBackbone):
    """Native V4.1 drafter adapted to AutoModel's shared DSpark loss contract."""

    _no_split_modules = ["_DeepseekV41DSparkBlock"]
    _keep_in_fp32_modules_strict = [
        "attn_hc",
        "ffn_hc",
        "attn.sinks_param",
        "bias_vl",
        "confidence_head",
    ]

    def __init__(self, config: DeepseekV41TextConfig, backend: BackendConfig | None = None) -> None:
        backend = backend or BackendConfig(
            attn="sdpa",
            linear="torch",
            rms_norm="torch_fp32",
            experts="torch",
            dispatcher="torch",
        )
        super().__init__(config, backend)
        if config.dspark_markov_rank <= 0:
            raise ValueError("DeepSeek V4.1 DSpark requires dspark_markov_rank > 0")
        if not hasattr(config, "dspark_num_anchors"):
            raise ValueError("DeepSeek V4.1 DSpark config requires dspark_num_anchors")
        dtype = dtype_from_str(config.dtype, torch.bfloat16)
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            dtype=dtype,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)
        self.num_anchors = int(config.dspark_num_anchors)
        self.enable_confidence_head = bool(getattr(config, "dspark_enable_confidence_head", True))
        if self.num_anchors <= 0:
            raise ValueError("dspark_num_anchors must be positive")
        if not self.enable_confidence_head:
            self.mtp[-1].confidence_head.requires_grad_(False)
        self.initialize_weights(self.embed_tokens.weight.device)

    @property
    def layers(self) -> nn.ModuleList:
        """Expose native MTP stages to the shared AC/FSDP helpers."""
        return self.mtp

    def initialize_embeddings_and_head(
        self,
        *,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        freeze: bool = True,
    ) -> None:
        """Copy the target's embedding and vocabulary projection.

        Args:
            embed_tokens: Target embedding with weight [vocab, hidden].
            lm_head: Target output projection with weight [vocab, hidden].
            freeze: Disable gradients for both copied modules when true.
        """
        if not freeze:
            raise ValueError("DeepSeek V4.1 DSpark requires frozen copied embedding and LM-head weights")
        if self.embed_tokens.weight.shape != embed_tokens.weight.shape:
            raise ValueError("DSpark and target embedding shapes must match")
        if self.lm_head.weight.shape != lm_head.weight.shape:
            raise ValueError("DSpark and target LM-head shapes must match")
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_tokens.weight.detach())
            self.lm_head.weight.copy_(lm_head.weight.detach())
        self.set_embedding_head_trainable(False)

    def set_embedding_head_trainable(self, trainable: bool) -> None:
        """Set whether the copied embedding and LM head receive gradients."""
        self.embed_tokens.requires_grad_(trainable)
        self.lm_head.requires_grad_(trainable)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project [batch, sequence, hidden] states to FP32 vocabulary logits."""
        return F.linear(hidden_states.float(), self.lm_head.weight.float())

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: torch.Tensor | None = None,
    ) -> DSparkForwardOutput:
        """Run native V4.1 DSpark training for sampled anchors.

        Args:
            input_ids: Target tokens [batch, sequence].
            target_hidden_states: Concatenated captured target features [batch,
                sequence, target_layers * hidden].
            loss_mask: Supervision mask [batch, sequence].
            target_last_hidden_states: Optional frozen final target states
                [batch, sequence, hidden] used by the probability-distance loss.

        Returns:
            Shared DSpark loss inputs. Draft/token/confidence tensors have shape
            [batch, num_anchors, block_size, ...].
        """
        batch, sequence = input_ids.shape
        if target_hidden_states.shape[:2] != (batch, sequence):
            raise ValueError("target_hidden_states must match input_ids [batch, sequence]")
        anchor_positions, block_keep_mask = sample_anchor_positions(
            seq_len=sequence,
            loss_mask=loss_mask,
            num_anchors=self.num_anchors,
            device=input_ids.device,
        )
        noise_embeddings = create_noise_embed(
            self.embed_tokens,
            input_ids,
            anchor_positions,
            block_keep_mask,
            mask_token_id=self.config.dspark_noise_token_id,
            block_size=self.config.dspark_block_size,
        )
        position_ids = self.build_position_ids(anchor_positions, sequence)
        attention_mask = self.build_attention_mask(
            anchor_positions,
            block_keep_mask,
            sequence,
            noise_embeddings.dtype,
        )
        backbone_output = super().forward(
            noise_embeddings,
            target_hidden_states.detach(),
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        num_blocks = anchor_positions.shape[1]
        block_size = self.config.dspark_block_size
        hidden = backbone_output.hidden_states.reshape(batch, num_blocks, block_size, -1)
        normalized = backbone_output.normalized_hidden_states.reshape(batch, num_blocks, block_size, -1)
        offsets = torch.arange(1, block_size + 1, device=input_ids.device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + offsets
        safe_label_indices = label_indices.clamp(max=sequence - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1), safe_label_indices, torch.zeros_like(safe_label_indices)
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, num_blocks, -1),
            2,
            safe_label_indices,
        )
        eval_mask = build_eval_mask(
            seq_len=sequence,
            loss_mask=loss_mask,
            label_indices=label_indices,
            safe_label_indices=safe_label_indices,
            block_keep_mask=block_keep_mask,
        )

        aligned_target_logits = None
        if target_last_hidden_states is not None:
            if target_last_hidden_states.shape[:2] != (batch, sequence):
                raise ValueError("target_last_hidden_states must match input_ids [batch, sequence]")
            target_prediction_indices = (safe_label_indices - 1).clamp(min=0)
            aligned_hidden = torch.gather(
                target_last_hidden_states.unsqueeze(1).expand(-1, num_blocks, -1, -1),
                2,
                target_prediction_indices.unsqueeze(-1).expand(-1, -1, -1, target_last_hidden_states.shape[-1]),
            )
            aligned_target_logits = self.compute_logits(aligned_hidden.detach())

        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)
        previous_token_ids = torch.cat((anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]), dim=-1)
        transition_logits, markov_embeddings = self.mtp[-1].markov_head(previous_token_ids)
        draft_logits = self.compute_logits(normalized) + transition_logits.float()
        confidence_pred = (
            self.mtp[-1].confidence_head(hidden, markov_embeddings) if self.enable_confidence_head else None
        )
        return DSparkForwardOutput(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            confidence_pred=confidence_pred,
            aligned_target_logits=aligned_target_logits,
        )


__all__ = ["DeepseekV41DSparkModel"]
