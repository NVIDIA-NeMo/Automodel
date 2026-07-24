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

"""ViSpec (vision-aware) draft model for speculative decoding on VLM targets.

ViSpec (Kang et al., NeurIPS 2025, arXiv:2509.15235) extends the EAGLE-1/2
draft with two vision-specific modules, keeping the rest of the draft
(``embed_tokens`` / ``fc`` / decoder layers / ``norm``) byte-identical to
:class:`~nemo_automodel.components.speculative.eagle.draft_llama_v12.LlamaEagleDraftModel`
so a text-only EAGLE-1/2 checkpoint can initialize stage-2 ViSpec training:

* :class:`VispecImageAdaptor` -- ``num_query_tokens`` learnable queries
  cross-attend over the target's image-token features and compress a whole
  image span (hundreds to thousands of tokens) into ``num_query_tokens``
  vectors. ``num_query_tokens - 1`` of them are spliced back into the draft
  sequence at the *original* trailing positions of the image span, so the
  positional layout of the surrounding text is untouched.
* ``img_fc`` -- the remaining ("global") image vector is broadcast onto every
  subsequent text position and mixed in with a ``[2*hidden -> hidden]``
  projection, giving each text token vision context without paying for image
  tokens in the draft's KV cache.

Reference implementation: ``vispec/model/cnets_ours.py`` in
https://github.com/KangJialiang/ViSpec.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from nemo_automodel.components.speculative.eagle.draft_llama_v12 import (
    LlamaEagleDraftModel,
    _build_causal_mask,
)


class VispecImageAdaptor(nn.Module):
    """Compress an image-token span into a small set of learnable-query vectors.

    A single non-causal cross-attention: ``num_query_tokens`` learnable queries
    attend over the image-token features, so the output length is independent
    of how many image tokens the target emitted.

    Args:
        config: Draft config supplying ``hidden_size`` and ``num_attention_heads``.
        num_query_tokens: Number of learnable queries (ViSpec's ``num_q``).
    """

    def __init__(self, config: PretrainedConfig, num_query_tokens: int):
        super().__init__()
        if num_query_tokens < 2:
            # One query would leave nothing to splice back into the sequence
            # after the global vector is taken (see ``VispecDraftModel``).
            raise ValueError(f"vispec_num_query_tokens must be >= 2, got {num_query_tokens}")
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # ViSpec derives the adaptor head dim from hidden_size (not config.head_dim):
        # the adaptor is its own module, not a copy of the target's attention.
        self.head_dim = self.hidden_size // self.num_heads
        self.num_query_tokens = num_query_tokens

        self.query = nn.Parameter(torch.empty(num_query_tokens, self.num_heads, self.head_dim))
        bias = bool(getattr(config, "attention_bias", False))
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def reset_query(self) -> None:
        """Re-draw the learnable queries from ``N(0, head_dim ** -0.5)`` (ViSpec's init)."""
        nn.init.normal_(self.query, mean=0.0, std=self.head_dim**-0.5)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Compress image features into ``num_query_tokens`` vectors.

        Args:
            image_features: Tensor of shape [batch, image_tokens, hidden] holding
                the target's embedding-layer output at the image positions of one
                image span.

        Returns:
            Tensor of shape [batch, num_query_tokens, hidden].
        """
        batch_size, seq_len, _ = image_features.shape
        query = self.query.view(1, self.num_query_tokens, self.num_heads, self.head_dim)
        query = query.to(image_features.dtype).transpose(1, 2).expand(batch_size, -1, -1, -1)
        key = self.k_proj(image_features).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(image_features).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query.contiguous(), key.contiguous(), value.contiguous(), is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, self.num_query_tokens, self.hidden_size)
        return self.o_proj(attn_output)


class VispecDraftModel(LlamaEagleDraftModel):
    """EAGLE-1/2 draft extended with ViSpec's image compression and global image feature.

    Adds two config fields on top of the EAGLE-1/2 draft config:

    * ``vispec_num_query_tokens`` (default 2) -- queries per image span.
    * ``draft_num_hidden_layers`` -- inherited, unchanged.

    The base draft's parameters (``embed_tokens``, ``fc``, ``layers``, ``norm``)
    keep their names, so ``load_state_dict(..., strict=False)`` restores a
    stage-1 EAGLE-1/2 checkpoint and leaves only ``img_adaptor`` / ``img_fc``
    freshly initialized.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.num_query_tokens = int(getattr(config, "vispec_num_query_tokens", 2))
        self.img_adaptor = VispecImageAdaptor(config, self.num_query_tokens)
        # ViSpec keeps a bias on ``img_fc``; AutoModel's EAGLE-1/2 ``fc`` is
        # bias-free, so ``img_fc`` follows the surrounding draft convention.
        self.img_fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.reset_vispec_parameters()

    def reset_vispec_parameters(self) -> None:
        """Initialize the ViSpec-only modules (identity-start for ``img_fc``).

        ``img_fc`` starts as ``[I | 0]``: it copies the target hidden state
        through and ignores the global image vector, so a stage-2 run
        initialized from a stage-1 EAGLE-1/2 checkpoint starts numerically
        equal to that checkpoint and only then learns to use vision context.
        """
        self.img_adaptor.reset_query()
        hidden_size = self.img_fc.weight.shape[0]
        with torch.no_grad():
            nn.init.eye_(self.img_fc.weight[:, :hidden_size])
            nn.init.zeros_(self.img_fc.weight[:, hidden_size:])

    def _fuse(
        self,
        inputs_embeds: torch.Tensor,
        target_hidden_states: torch.Tensor,
        global_image_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Mix the global image vector into the target hidden state, then run EAGLE's ``fc``.

        Args:
            inputs_embeds: Tensor of shape [tokens, hidden].
            target_hidden_states: Tensor of shape [tokens, hidden].
            global_image_feature: Tensor of shape [1, hidden], broadcast over ``tokens``.

        Returns:
            Tensor of shape [tokens, hidden].
        """
        vision_context = global_image_feature.expand_as(target_hidden_states)
        hidden_states = self.img_fc(torch.cat((target_hidden_states, vision_context), dim=-1))
        return self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))

    def _compress_sequence(
        self,
        inputs_embeds: torch.Tensor,
        target_hidden_states: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the compressed draft sequence for one (batch-size-1) sample.

        Walks the sample image span by image span. Each span contributes its
        leading text positions (fused with the *previous* span's global image
        vector) followed by ``num_query_tokens - 1`` compressed image tokens;
        the trailing text after the last span is fused with the last span's
        global vector. Because every span ends with its image run, this
        reordering preserves the original left-to-right token order.

        Args:
            inputs_embeds: Tensor of shape [sequence, hidden].
            target_hidden_states: Tensor of shape [sequence, hidden].
            image_mask: Bool tensor of shape [sequence]; True at image positions.

        Returns:
            Tuple of ``(hidden_states, source_index)``:
            ``hidden_states`` is a Tensor of shape [compressed_sequence, hidden];
            ``source_index`` is a long Tensor of shape [compressed_sequence]
            giving, for each compressed position, its position in the original
            sequence. It supplies both the draft's position ids and the scatter
            index that restores the original layout after the decoder stack.
        """
        seq_len = inputs_embeds.shape[0]
        positions = torch.arange(seq_len, device=inputs_embeds.device)
        # One entry per contiguous image run: the index just past its last token.
        span_ends = torch.cat((image_mask[:-1] & ~image_mask[1:], image_mask[-1:]))
        span_end_ids = torch.nonzero(span_ends, as_tuple=False).flatten() + 1

        num_spliced = self.num_query_tokens - 1
        global_image_feature = torch.zeros_like(inputs_embeds[:1])
        segments: list[torch.Tensor] = []
        source_index: list[torch.Tensor] = []
        span_start = 0
        for span_end in span_end_ids.tolist():
            span_image_mask = image_mask[span_start:span_end]
            text_index = positions[span_start:span_end][~span_image_mask]
            segments.append(
                self._fuse(
                    inputs_embeds[text_index],
                    target_hidden_states[text_index],
                    global_image_feature,
                )
            )
            source_index.append(text_index)

            image_index = positions[span_start:span_end][span_image_mask]
            if image_index.numel() < num_spliced:
                raise ValueError(
                    f"ViSpec splices {num_spliced} compressed tokens back into the positions of each image "
                    f"span, but this span holds only {int(image_index.numel())} image token(s). Lower "
                    "vispec_num_query_tokens or raise the processor's image resolution."
                )
            compressed = self.img_adaptor(inputs_embeds[image_index].unsqueeze(0)).squeeze(0)
            segments.append(compressed[:num_spliced])
            # The spliced tokens inherit the span's trailing original positions,
            # so the surrounding text keeps its own position ids untouched.
            source_index.append(positions[span_end - num_spliced : span_end])
            global_image_feature = compressed[-1:]
            span_start = span_end

        tail_index = positions[span_start:]
        segments.append(
            self._fuse(
                inputs_embeds[tail_index],
                target_hidden_states[tail_index],
                global_image_feature,
            )
        )
        source_index.append(tail_index)
        return torch.cat(segments, dim=0), torch.cat(source_index, dim=0)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the next-position target hidden states from vision-aware features.

        Batch size must be 1: image spans have per-sample lengths, so the
        compressed sequences of two samples would not share a length. This
        matches the reference implementation, which raises on batch size > 1.

        Args:
            inputs_embeds: Tensor of shape [1, sequence, hidden] -- the target's
                embedding-layer output (vision features already spliced in),
                shifted left by one position so index ``i`` holds the embedding
                of token ``i + 1``.
            target_hidden_states: Tensor of shape [1, sequence, hidden] -- the
                target's last hidden state, *not* shifted.
            attention_mask: Tensor of shape [1, sequence]; 1 for real tokens,
                0 for padding.
            image_mask: Bool tensor of shape [1, sequence], aligned with
                ``inputs_embeds`` (True where token ``i + 1`` is an image token).

        Returns:
            Tensor of shape [1, sequence, hidden]. Positions consumed by the
            image compression carry zeros: they are never supervised, because
            the loss mask covers assistant-response positions only.
        """
        batch_size = inputs_embeds.shape[0]
        if batch_size != 1:
            raise NotImplementedError(
                f"ViSpec draft training runs at micro_batch_size=1 (got batch size {batch_size}); "
                "image spans differ per sample, so their compressed sequences have different lengths."
            )
        inputs_embeds = inputs_embeds.to(target_hidden_states.dtype)
        image_mask = image_mask.bool()

        if bool(image_mask.any()):
            hidden_states, source_index = self._compress_sequence(
                inputs_embeds[0], target_hidden_states[0], image_mask[0]
            )
            hidden_states = hidden_states.unsqueeze(0)
            position_ids = source_index.unsqueeze(0)
            compressed_attention_mask = attention_mask[:, source_index]
        else:
            # Text-only sample: no span to compress, so ``img_adaptor`` would
            # receive no gradient and DDP (find_unused_parameters=False) would
            # error out on the rank that saw it. Keep it in the graph with a
            # zero-weighted term, exactly as the reference does.
            hidden_states = self._fuse(
                inputs_embeds[0],
                target_hidden_states[0],
                torch.zeros_like(inputs_embeds[0, :1]),
            ).unsqueeze(0)
            hidden_states = hidden_states + 0.0 * self.img_adaptor(inputs_embeds[:, :1]).sum(dim=1, keepdim=True)
            source_index = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            position_ids = source_index.unsqueeze(0)
            compressed_attention_mask = attention_mask

        causal_mask = _build_causal_mask(compressed_attention_mask, hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids)
        hidden_states = self.norm(hidden_states)

        # Scatter back to the original sequence layout so the caller can compare
        # against the (uncompressed) target supervision. Equivalent to the
        # reference's one-hot ``trans_mat`` einsum, without materializing a
        # [compressed_sequence, sequence] matrix.
        output = torch.zeros(
            (1, inputs_embeds.shape[1], hidden_states.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return output.index_copy(1, source_index, hidden_states)
