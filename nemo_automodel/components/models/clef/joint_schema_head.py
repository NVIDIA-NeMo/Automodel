# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Adapted from Cloudflare's Apache-2.0 CLEF joint_schema_model.py.
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

"""Clef joint schema head and record encoding.

Adapted from Cloudflare, Inc.'s Apache-2.0 ``joint_schema_model.py``:
https://huggingface.co/Cloudflare/clef/blob/ed3eed331870db2eff4b0db01237128ede8a00ce/joint_schema_model.py
The publisher's license is https://huggingface.co/Cloudflare/clef/blob/ed3eed331870db2eff4b0db01237128ede8a00ce/LICENSE.
Parameter names remain unchanged. NeMo adaptations add typed configuration, tensor validation,
a callable lexical-row table for sharded backbones, and fp32 head training.

Clef turns a ``state`` and a schema of typed questions into one logit per allowed option per question, in a single
forward pass of a Qwen3.5-architecture backbone followed by this small transformer head.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as functional
from torch import nn

# Publisher-derived prompt layout and head architecture, adapted as described above.
SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"
MEDIA_BATCH_KEYS = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
MEDIA_TOKEN_KEYS = ("mm_token_type_ids",)
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}


def render(value: Any) -> str:
    """Render a state / instruction value as text (strings verbatim, everything else as compact sorted JSON)."""
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def question_options(question: dict[str, Any]) -> list[tuple[str, Any]]:
    """``(option id, description)`` pairs in Clef order: noul ``[true, false]``, choice sorted, score by level."""
    question_type = str(question["type"])
    if question_type == "noul":
        criteria = {
            "true": "The proposition is true or the answer is yes.",
            "false": "The proposition is false or the answer is no.",
        }
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if question_type == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


@dataclass(frozen=True)
class EncodedQuestion:
    """Token spans of one question and its options inside an encoded record (end-exclusive)."""

    question_id: str
    question_type: int
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRecord:
    """A tokenised record: input ids, per-question spans, and optional media tensors."""

    input_ids: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]
    record_id: str
    media: dict[str, Any] | None = dataclass_field(default=None, compare=False, repr=False)


def _tokens(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _encode_media(processor: Any, record: dict[str, Any]) -> tuple[list[int], dict[str, Any] | None]:
    images = list(record.get("images") or [])
    videos = list(record.get("videos") or [])
    if not images and not videos:
        return [], None
    if processor is None:
        raise ValueError("records with images or videos require a processor")
    text = IMAGE_PLACEHOLDER * len(images) + VIDEO_PLACEHOLDER * len(videos) + "\n"
    encoded = processor(
        text=[text],
        images=images or None,
        videos=videos or None,
        return_tensors="pt",
        **(record.get("media_kwargs") or {}),
    )
    media = {key: encoded[key] for key in MEDIA_BATCH_KEYS if key in encoded}
    for key in MEDIA_TOKEN_KEYS:
        if key in encoded:
            media[key] = encoded[key][0].tolist()
    return encoded["input_ids"][0].tolist(), media


def encode_record(
    tokenizer: Any,
    record: dict[str, Any],
    max_length: int = 16384,
    max_state_tokens: int | None = None,
    processor: Any | None = None,
) -> EncodedRecord:
    """Tokenise a SystemOne record into Clef's prompt layout and record question / option token spans.

    The state is truncated (never the schema) so the prompt fits in ``max_length`` tokens.
    """
    schema_ids = _tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")
    questions: list[EncodedQuestion] = []
    for question_index, (question_id, question) in enumerate(record["questions"].items()):
        schema_ids.extend(
            _tokens(
                tokenizer,
                f"\nFIELD {question_index + 1}\nID: {question_id}\nTYPE: {question['type']}\nINSTRUCTION: ",
            )
        )
        question_start = len(schema_ids)
        instructions = question.get("instructions")
        if instructions is None or instructions == "":
            instructions = str(question_id)
        schema_ids.extend(_tokens(tokenizer, render(instructions)))
        question_end = len(schema_ids)
        schema_ids.extend(_tokens(tokenizer, "\nALLOWED OPTIONS:\n"))

        option_spans: list[tuple[int, int]] = []
        option_ids: list[str] = []
        for option_index, (option_id, description) in enumerate(question_options(question)):
            schema_ids.extend(_tokens(tokenizer, f"OPTION {option_index + 1}: "))
            option_start = len(schema_ids)
            semantics = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema_ids.extend(_tokens(tokenizer, render(semantics)))
            option_spans.append((option_start, len(schema_ids)))
            option_ids.append(option_id)
            schema_ids.extend(_tokens(tokenizer, "\n"))
        schema_ids.extend(_tokens(tokenizer, "END FIELD\n"))
        questions.append(
            EncodedQuestion(
                question_id=str(question_id),
                question_type=QUESTION_TYPES[str(question["type"])],
                question_span=(question_start, question_end),
                option_spans=tuple(option_spans),
                option_ids=tuple(option_ids),
            )
        )

    prefix_ids = _tokens(
        tokenizer,
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n",
    )
    suffix_ids = _tokens(
        tokenizer,
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )
    media_ids, media = _encode_media(processor, record)
    if media is not None:
        media["token_offset"] = len(prefix_ids)
        prefix_ids = prefix_ids + media_ids
    state_ids = _tokens(tokenizer, render(record["state"]))
    if max_state_tokens is not None:
        state_ids = state_ids[:max_state_tokens]
    fixed_length = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
    if fixed_length > max_length:
        raise ValueError(f"schema requires {fixed_length} tokens before state; maximum is {max_length}")
    state_ids = state_ids[: max_length - fixed_length]
    schema_offset = len(prefix_ids) + len(state_ids)
    shifted_questions = tuple(
        EncodedQuestion(
            question_id=question.question_id,
            question_type=question.question_type,
            question_span=(
                question.question_span[0] + schema_offset,
                question.question_span[1] + schema_offset,
            ),
            option_spans=tuple((start + schema_offset, end + schema_offset) for start, end in question.option_spans),
            option_ids=question.option_ids,
        )
        for question in questions
    )
    input_ids = tuple(prefix_ids + state_ids + schema_ids + suffix_ids)
    if not input_ids or not shifted_questions:
        raise ValueError("record produced no model input or questions")
    return EncodedRecord(
        input_ids=input_ids,
        questions=shifted_questions,
        record_id=str(record.get("id", "unknown")),
        media=media,
    )


def collate_records(
    records: list[EncodedRecord],
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    """Right-pad encoded records into a batch.

    Args:
        records: Batch-aligned token spans. Optional media contains pixel_values of shape
            [patches, patch_features], where patch_features flattens each pixel patch;
            image_grid_thw of shape [images, 3] (temporal, height, width patch counts);
            pixel_values_videos and video_grid_thw with corresponding video layouts;
            and mm_token_type_ids, a list of media-token types starting at token_offset.
        pad_token_id: Token used to fill right-padded positions.
        device: Device for input IDs, attention masks, and media token-type tensors.

    Returns:
        Mapping with input_ids and attention_mask, long Tensors of shape [batch, sequence]
        (mask one means a real token); records, the original encoded records; and media.
        Media pixel/patch and grid tensors concatenate along axis zero, preserving their
        source dtype/device. Media mm_token_type_ids is a long Tensor of shape [batch, sequence],
        zero outside media spans. Other input tensors are created on ``device``.
    """
    maximum_length = max(len(record.input_ids) for record in records)
    input_ids = torch.full(
        (len(records), maximum_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(records), maximum_length),
        dtype=torch.long,
        device=device,
    )
    for index, record in enumerate(records):
        length = len(record.input_ids)
        input_ids[index, :length] = torch.tensor(record.input_ids, device=device)
        attention_mask[index, :length] = 1
    media: dict[str, torch.Tensor] = {}
    for key in MEDIA_BATCH_KEYS:
        values = [record.media[key] for record in records if record.media and key in record.media]
        if values:
            media[key] = torch.cat(values, dim=0).to(device)
    for key in MEDIA_TOKEN_KEYS:
        if any(record.media and key in record.media for record in records):
            token_values = torch.zeros((len(records), maximum_length), dtype=torch.long, device=device)
            for index, record in enumerate(records):
                if record.media and key in record.media:
                    offset = record.media["token_offset"]
                    values = torch.tensor(record.media[key], dtype=torch.long, device=device)
                    token_values[index, offset : offset + len(values)] = values
            media[key] = token_values
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "records": records,
        "media": media,
    }


@dataclass(frozen=True)
class ClefHeadConfig:
    """Declarative configuration matching joint_head_config.json.

    Args:
        hidden_size: Backbone hidden-state and output-embedding width.
        width: Internal decision-head width.
        routing_layers: Number of option-to-evidence cross-attention blocks.
        layers: Number of joint field decoder blocks.
        heads: Attention head count.
        feedforward: Feed-forward width in attention blocks.
        dropout: Dropout probability used while post-training the head.
    """

    hidden_size: int
    width: int
    routing_layers: int
    layers: int
    heads: int
    feedforward: int
    dropout: float = 0.0

    def __post_init__(self) -> None:
        dimensions = (
            self.hidden_size,
            self.width,
            self.routing_layers,
            self.layers,
            self.heads,
            self.feedforward,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("CLEF head dimensions and layer counts must be positive")
        if self.width % self.heads:
            raise ValueError("CLEF head width must be divisible by heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def build(self) -> JointSchemaHead:
        """Build a checkpoint-compatible joint schema head.

        Returns:
            Newly initialized CLEF decision head.
        """
        return JointSchemaHead(self)


class EvidenceRoutingLayer(nn.Module):
    """Route state evidence into per-option queries."""

    def __init__(self, config: ClefHeadConfig) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(config.width)
        self.memory_norm = nn.LayerNorm(config.width)
        self.attention = nn.MultiheadAttention(
            config.width,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.feedforward_norm = nn.LayerNorm(config.width)
        self.feedforward = nn.Sequential(
            nn.Linear(config.width, config.feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward, config.width),
            nn.Dropout(config.dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Cross-attend option queries to the encoded record.

        Args:
            queries: Tensor of shape [batch, options, width].
            memory: Tensor of shape [batch, sequence, width].

        Returns:
            Tensor of shape [batch, options, width].
        """
        normalized_queries = self.query_norm(queries)
        normalized_memory = self.memory_norm(memory)
        routed, _ = self.attention(
            normalized_queries,
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )
        queries = queries + self.attention_dropout(routed)
        return queries + self.feedforward(self.feedforward_norm(queries))


class JointSchemaHead(nn.Module):
    """Cloudflare CLEF joint head over one or more typed questions."""

    def __init__(self, config: ClefHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_norm = nn.LayerNorm(config.hidden_size)
        self.memory_projection = nn.Linear(config.hidden_size, config.width, bias=False)
        self.question_projection = nn.Linear(config.hidden_size, config.width, bias=False)
        self.option_question_projection = nn.Linear(config.hidden_size, config.width, bias=False)
        self.global_projection = nn.Linear(config.hidden_size, config.width, bias=False)
        self.option_context_projection = nn.Linear(config.hidden_size, config.width, bias=False)
        self.option_lexical_projection = nn.Linear(config.hidden_size, config.width, bias=False)
        self.type_embedding = nn.Embedding(3, config.width)
        self.evidence_layers = nn.ModuleList(EvidenceRoutingLayer(config) for _ in range(config.routing_layers))
        self.option_summary_norm = nn.LayerNorm(config.width)
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                d_model=config.width,
                nhead=config.heads,
                dim_feedforward=config.feedforward,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(config.layers)
        )
        self.field_norm = nn.LayerNorm(config.width)
        self.option_norm = nn.LayerNorm(config.width)
        self.residual_scorer = nn.Sequential(
            nn.Linear(config.width * 4, config.width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.width, 1),
        )
        self.prior_logit_scale = nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = nn.Parameter(torch.zeros(()))
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def score_records(
        self,
        backbone: nn.Module,
        records: list[EncodedRecord],
        lexical: LexicalTable,
        *,
        pad_token_id: int,
        device: torch.device,
    ) -> list[list[torch.Tensor]]:
        """Run CLEF's backbone-to-head composition through the public hidden-state interface.

        Args:
            backbone: Language model returning hidden states of shape [batch, sequence, hidden].
            records: Publisher-ordered token spans. Optional media contains pixel_values
                of shape [patches, features], pixel_values_videos of shape [video_patches, features],
                and image/video_grid_thw of shape [items, 3] (temporal, height, width).
            lexical: Frozen token-row lookup returning tensors of shape [tokens, hidden].
            pad_token_id: Token used for right padding.
            device: Execution device for collated input tensors.

        Returns:
            Batch/question lists of fp32 logit tensors of shape [options].
        """
        collated = collate_records(records, pad_token_id, device)
        output = backbone(
            input_ids=collated["input_ids"],
            attention_mask=collated["attention_mask"],
            use_cache=False,
            logits_to_keep=1,
            output_hidden_states=True,
            **collated["media"],
        )
        if output.hidden_states is None:
            raise RuntimeError("the backbone must return final hidden states for CLEF training")
        return self(output.hidden_states[-1], collated["input_ids"], collated["attention_mask"], records, lexical)

    @staticmethod
    def _mean_span(values: torch.Tensor, span: tuple[int, int]) -> torch.Tensor:
        """Mean-pool a non-empty token span.

        Args:
            values: Tensor of shape [sequence, hidden].
            span: Half-open (start, end) token interval.

        Returns:
            Tensor of shape [hidden].
        """
        start, end = span
        if start < 0 or end <= start or end > values.shape[0]:
            raise ValueError(f"invalid token span {span} for sequence length {values.shape[0]}")
        return values[start:end].mean(dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        records: list[EncodedRecord],
        output_embedding_weight: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    ) -> list[list[torch.Tensor]]:
        """Produce one variable-length option-logit tensor per question.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            input_ids: Tensor of shape [batch, sequence].
            attention_mask: Tensor of shape [batch, sequence], with one for real tokens.
            records: Batch-aligned token-span metadata.
            output_embedding_weight: Tensor of shape [vocab, hidden], or a callable mapping
                long token IDs [tokens] to rows [tokens, hidden].

        Returns:
            Nested batch/question list. Each Tensor has shape [options].
        """
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        if input_ids.shape != attention_mask.shape or input_ids.shape != hidden_states.shape[:2]:
            raise ValueError("input_ids, attention_mask, and hidden_states must share batch/sequence dimensions")
        if len(records) != hidden_states.shape[0]:
            raise ValueError("records must contain one item per batch row")
        if not callable(output_embedding_weight) and (
            output_embedding_weight.ndim != 2 or output_embedding_weight.shape[1] != self.config.hidden_size
        ):
            raise ValueError("output_embedding_weight must have shape [vocab, hidden]")

        lexical_rows = (
            output_embedding_weight if callable(output_embedding_weight) else output_embedding_weight.__getitem__
        )
        hidden_states = hidden_states.to(self.hidden_norm.weight.dtype)
        results: list[list[torch.Tensor]] = []
        normalized_hidden = self.hidden_norm(hidden_states)
        for batch_index, record in enumerate(records):
            sequence_length = int(attention_mask[batch_index].sum().item())
            sequence_hidden = normalized_hidden[batch_index, :sequence_length]
            memory = self.memory_projection(sequence_hidden).unsqueeze(0)
            global_vector = sequence_hidden[-1]
            question_vectors = torch.stack(
                [self._mean_span(sequence_hidden, question.question_span) for question in record.questions]
            )
            type_ids = torch.tensor(
                [question.question_type for question in record.questions],
                dtype=torch.long,
                device=hidden_states.device,
            )

            option_contexts: list[torch.Tensor] = []
            lexical_options: list[torch.Tensor] = []
            option_counts: list[int] = []
            for question in record.questions:
                contexts = torch.stack([self._mean_span(sequence_hidden, span) for span in question.option_spans])
                lexical = torch.stack(
                    [
                        lexical_rows(input_ids[batch_index, start:end]).to(hidden_states.dtype).mean(dim=0)
                        for start, end in question.option_spans
                    ]
                )
                option_contexts.append(contexts)
                lexical_options.append(lexical)
                option_counts.append(len(question.option_spans))

            option_queries = [
                self.option_context_projection(contexts)
                + self.option_lexical_projection(lexical)
                + self.option_question_projection(question_vectors[index]).unsqueeze(0)
                for index, (contexts, lexical) in enumerate(zip(option_contexts, lexical_options))
            ]
            routed_options = torch.cat(option_queries, dim=0).unsqueeze(0)
            for layer in self.evidence_layers:
                routed_options = layer(routed_options, memory)
            split_options = list(torch.split(routed_options[0], option_counts, dim=0))

            base_fields = self.question_projection(question_vectors)
            option_summaries = []
            for field, options in zip(base_fields, split_options):
                weights = torch.softmax(options @ field / math.sqrt(options.shape[-1]), dim=0)
                option_summaries.append(torch.sum(weights.unsqueeze(-1) * options, dim=0))
            fields = (
                base_fields
                + self.option_summary_norm(torch.stack(option_summaries))
                + self.global_projection(global_vector).unsqueeze(0)
                + self.type_embedding(type_ids)
            ).unsqueeze(0)
            for layer in self.layers:
                fields = layer(fields, memory)
            fields = self.field_norm(fields[0])

            record_logits: list[torch.Tensor] = []
            for index, (field, lexical, routed) in enumerate(zip(fields, lexical_options, split_options)):
                anchor = functional.normalize(question_vectors[index] + global_vector, dim=-1)
                lexical_anchor = functional.normalize(lexical, dim=-1)
                prior_scale = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
                prior = prior_scale * (lexical_anchor @ anchor)
                options = self.option_norm(routed)
                repeated_field = field.unsqueeze(0).expand_as(options)
                cosine = functional.cosine_similarity(repeated_field, options, dim=-1)
                features = torch.cat(
                    [
                        repeated_field,
                        options,
                        repeated_field * options,
                        torch.abs(repeated_field - options),
                    ],
                    dim=-1,
                )
                residual = self.residual_scorer(features).squeeze(-1)
                joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
                joint = joint_scale * cosine + residual
                record_logits.append(prior + torch.sigmoid(self.residual_gate) * joint)
            results.append(record_logits)
        return results


def load_joint_head(
    path: str | Path, dtype: torch.dtype = torch.float32, dropout: float = 0.0, *, revision: str | None = None
) -> JointSchemaHead:
    """Build a :class:`JointSchemaHead` from a directory with ``joint_head_config.json`` / ``joint_head.safetensors``.

    ``path`` may also be a Hugging Face repo id (e.g. ``"Cloudflare/clef"``); only the head files are downloaded.
    """
    from safetensors.torch import load_file

    path = Path(path)
    if not (path / "joint_head_config.json").exists():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(str(path), revision=revision, allow_patterns=["joint_head*"]))
    config = json.loads((path / "joint_head_config.json").read_text())
    head = ClefHeadConfig(**config, dropout=dropout).build()
    head.load_state_dict(load_file(str(path / "joint_head.safetensors")), strict=True)
    return head.to(dtype=dtype)


def save_joint_head(head: JointSchemaHead, path: str | Path, dtype: torch.dtype = torch.bfloat16) -> None:
    """Save a head in the release layout (``joint_head_config.json`` + ``joint_head.safetensors``)."""
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "hidden_size": head.hidden_norm.normalized_shape[0],
        "width": head.memory_projection.out_features,
        "routing_layers": len(head.evidence_layers),
        "layers": len(head.layers),
        "heads": head.layers[0].self_attn.num_heads,
        "feedforward": head.layers[0].linear1.out_features,
    }
    (path / "joint_head_config.json").write_text(json.dumps(config, indent=2))
    state = {k: v.detach().to(dtype).cpu().contiguous() for k, v in head.state_dict().items()}
    save_file(state, str(path / "joint_head.safetensors"))


def clef_option_orders(encoded: EncodedRecord) -> dict[str, tuple[str, ...]]:
    """``{question_id: option ids}`` in the order the Clef head returns logits."""
    return {q.question_id: q.option_ids for q in encoded.questions}


def option_token_ids(encoded_records: Sequence[EncodedRecord]) -> list[int]:
    """Sorted unique token ids inside option spans (the only output-embedding rows the head reads)."""
    ids: set[int] = set()
    for record in encoded_records:
        for question in record.questions:
            for start, end in question.option_spans:
                ids.update(record.input_ids[start:end])
    return sorted(ids)


@torch.no_grad()
def build_lexical_table(model: torch.nn.Module, records: Sequence[EncodedRecord]) -> "LexicalTable":
    """Cache the frozen output-embedding rows consumed by the CLEF readout.

    All model-parallel ranks must participate when the output weights are sharded.

    Args:
        model: Backbone exposing output embeddings of shape [vocabulary, hidden].
        records: Encoded records whose option spans determine the cached token rows.

    Returns:
        Frozen table of shape [unique_option_tokens, hidden] on the embedding device.

    Raises:
        ValueError: If the output embeddings are trainable or no option tokens exist.
    """
    weight = model.get_output_embeddings().weight
    if weight.requires_grad:
        raise ValueError("CLEF lexical caching requires frozen output embeddings")
    ids = option_token_ids(records)
    if not ids:
        raise ValueError("CLEF lexical caching requires at least one option token")
    full = weight.full_tensor() if hasattr(weight, "full_tensor") else weight
    rows = full[torch.tensor(ids, device=full.device)].detach().clone()
    return LexicalTable(ids, rows).to(rows.device)


class LexicalTable(torch.nn.Module):
    """Frozen cache of the output-embedding rows the head needs, indexed by token id.

    The released head reads ``lm_head.weight[token_ids]`` for option tokens.  When the LM head is sharded we gather
    the full weight once to select these rows (they never change, the LM head is frozen),
    then retain only the selected rows in this table.
    """

    def __init__(self, token_ids: Sequence[int], rows: torch.Tensor):
        """Create the table.

        Args:
            token_ids: Unique token ids, one per row of ``rows``.
            rows: Tensor of shape [len(token_ids), hidden] with the corresponding LM-head rows (unsharded).
        """
        super().__init__()
        lookup = torch.full((max(token_ids) + 1,), -1, dtype=torch.long)
        lookup[torch.tensor(token_ids, dtype=torch.long)] = torch.arange(len(token_ids))
        self.register_buffer("lookup", lookup, persistent=False)
        self.register_buffer("rows", rows.detach(), persistent=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up LM-head rows.

        Args:
            token_ids: Long tensor of shape [tokens].

        Returns:
            Tensor of shape [tokens, hidden] on ``token_ids.device``.
        """
        index = self.lookup.to(token_ids.device)[token_ids]
        if bool((index < 0).any()):
            raise KeyError("token id missing from LexicalTable; rebuild it with option_token_ids() over all records")
        return self.rows.to(token_ids.device)[index]
