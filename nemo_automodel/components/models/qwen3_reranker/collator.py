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

"""Context-aware collator for the Qwen3 reranker.

Owned by this model package rather than the generic dataset collators because every
constant it carries is part of the Qwen3-Reranker prompt contract: the ChatML markers,
the empty ``<think>`` block that makes the final tokens a yes/no next-token prediction,
the model card's system message, and the yes/no label semantics themselves.

It stands alone on :class:`~transformers.DataCollatorWithPadding` and imports neither
``model.py`` nor the generic retrieval collator module, so training data assembly and
model definition stay independently importable.
"""

import hashlib
from typing import TYPE_CHECKING, Any, Dict, List

import torch
from transformers import DataCollatorWithPadding
from transformers.file_utils import PaddingStrategy

if TYPE_CHECKING:
    from transformers import BatchEncoding


class Qwen3ContextAwareRerankerCollator(DataCollatorWithPadding):
    """Collate query-document pairs using the Qwen3 reranker chat template.

    Wraps each (query, document) pair in the instruction-aware chat format used by
    ``Qwen/Qwen3-Reranker-*`` so that the final tokens are the assistant think-prefix
    and the next-token prediction is "yes"/"no". The middle (instruct/query/document)
    is truncated to fit, then the fixed prefix/suffix token ids are concatenated so the
    template markers are never truncated away.

    Optional context fields ``reasoning`` and ``global_query`` are supported. When
    present they are embedded inside the ``<Query>:`` block as ``#``-prefixed markers.
    The instruction is selected from ``instructions`` by which fields SURVIVE the drop,
    so it always describes the context the prompt actually carries.

    Templates per mode::

        # base -- byte-identical to the out-of-the-box Qwen3-Reranker prompt
        <Instruct>: {instruction}
        <Query>: {query}
        <Document>: {document}

        # reasoning
        <Instruct>: {instruction}
        <Query>: #Reasoning Trace: {reasoning}
        #Query: {query}
        <Document>: {document}

        # globalq
        <Instruct>: {instruction}
        <Query>: #Original Question (global query): {global_query}
        #Query: {query}
        <Document>: {document}

        # full
        <Instruct>: {instruction}
        <Query>: #Original Question (global query): {global_query}
        #Reasoning Trace: {reasoning}
        #Query: {query}
        <Document>: {document}

    Context dropout (multimode training)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``reasoning_drop_prob`` and ``global_query_drop_prob`` (both 0.5 by default) drop each
    field independently per query, so a row carrying both yields all four modes above. A
    drop REMOVES the field -- no placeholder is substituted, so the prompt shape changes
    with it and the instruction follows. The draw is a hash of
    (drop_seed, epoch, field, query): identical across runs, workers and ranks, and
    redrawn per epoch once ``set_epoch`` is called. Set both to 0.0 at
    eval time to force full mode.
    """

    DEFAULT_PREFIX_TEMPLATE = "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n"
    DEFAULT_SUFFIX_TEMPLATE = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    DEFAULT_SYSTEM = (
        "Judge whether the Document meets the requirements based on the Query and the "
        'Instruct provided. Note that the answer can only be "yes" or "no".'
    )

    # Instructions keyed by frozenset of present context field names.
    # Supported fields: "reasoning", "global_query".
    # frozenset()                             → base (no context)
    # frozenset({"reasoning"})                → reasoning only
    # frozenset({"global_query"})             → global_query only
    # frozenset({"reasoning","global_query"}) → both present
    #
    # The base instruction is Qwen3-Reranker's own, verbatim, so a row with no context is
    # byte-identical to the out-of-the-box prompt.
    DEFAULT_INSTRUCTIONS: Dict[frozenset, str] = {
        frozenset(): ("Given a web search query, retrieve relevant passages that answer the query"),
        frozenset({"reasoning"}): (
            "Given a user generated web search query and the user's reasoning trace that "
            "motivated the query (when available), retrieve relevant passages that answer "
            "the query in light of the reasoning context"
        ),
        frozenset({"global_query"}): (
            "Given a user generated query issued during an iterative research session and "
            "the original question (global query) that motivated the session, retrieve "
            "relevant passages that answer the query in light of the original question"
        ),
        frozenset({"reasoning", "global_query"}): (
            "Given a user generated query, the original question (global query), and the "
            "user's reasoning trace (when available) that motivated the query, retrieve "
            "relevant passages that answer the query in light of the original question and "
            "reasoning context"
        ),
    }

    @staticmethod
    def _normalize_instructions(raw: dict) -> Dict[frozenset, str]:
        """Normalize instruction keys to frozensets of field name strings.

        Accepts three key formats so callers can use whichever is most natural:
        - ``frozenset`` — used directly (Python API).
        - ``tuple`` of strings — converted to frozenset (Python API).
        - ``str`` — comma-separated field names (YAML-friendly); empty string maps to
          the no-context frozenset. Examples::

              ""                    → frozenset()
              "reasoning"           → frozenset({"reasoning"})
              "global_query,reasoning" → frozenset({"global_query","reasoning"})
        """
        result: Dict[frozenset, str] = {}
        for key, val in raw.items():
            if isinstance(key, frozenset):
                result[key] = val
            elif isinstance(key, tuple):
                result[frozenset(key)] = val
            elif isinstance(key, str):
                fields = frozenset(f.strip() for f in key.split(",") if f.strip())
                result[fields] = val
            else:
                raise TypeError(f"instructions keys must be str, tuple, or frozenset; got {type(key)}")
        return result

    def __init__(
        self,
        rerank_max_length: int,
        *args,
        instructions: dict = None,
        system_message: str = None,
        reasoning_drop_prob: float = 0.5,
        global_query_drop_prob: float = 0.5,
        drop_seed: int = 42,
        global_query_max_length: int = None,
        sub_query_max_length: int = None,
        reasoning_max_length: int = None,
        passage_max_length: int = None,
        prefix_template: str = None,
        suffix_template: str = None,
        **kwargs,
    ):
        """
        Args:
            rerank_max_length: Maximum total token length (prefix + middle + suffix).
            global_query_max_length: Token cap on the original/global question.
            sub_query_max_length: Token cap on the user-generated query.
            reasoning_max_length: Token cap on the reasoning trace.
            passage_max_length: Token cap on the document.
            prefix_template: Chat prefix; ``{system_message}`` is substituted. Defaults to
                ChatML (``DEFAULT_PREFIX_TEMPLATE``).
            suffix_template: Chat suffix that ends the prompt on the token whose next-token
                prediction is the yes/no decision. Defaults to ``DEFAULT_SUFFIX_TEMPLATE``.
            instructions: Dict mapping present context field names to instruction strings.
                Keys may be ``frozenset``, ``tuple``, or comma-separated ``str``
                (YAML-friendly). Defaults to ``DEFAULT_INSTRUCTIONS``. Ignored when
                ``instruction`` (singular) is set.
            system_message: Override the system prompt. Defaults to ``DEFAULT_SYSTEM``.
            reasoning_drop_prob: Probability of REMOVING a non-empty reasoning trace at
                collation time (no placeholder is substituted). Drawn per query, not
                per passage. Default 0.5. Set to 0.0 at eval/inference.
            global_query_drop_prob: Same, for the original question. Default 0.5.
            drop_seed: Seed for both draws; with the epoch it fixes the whole schedule.
        """
        self.rerank_max_length = rerank_max_length
        # DataCollatorWithPadding does not accept an ``args`` kwarg; the retrieval
        # collators accept and stash it for callers that pass a config object through.
        self.args = kwargs.pop("args", None)
        super().__init__(*args, **kwargs)
        # PER-ITEM caps, each applied to its own field BEFORE the prompt is assembled, so
        # every item is guaranteed its own share and no item can be starved by another.
        # Capping the assembled block instead would truncate it from the end, and the
        # query sits last -- a long trace would delete the very thing being asked.
        #
        # Cap the QUERY-side fields and leave passage_max_length None: rerank_max_length
        # then applies to the assembled result, and <Document> is last, so the document
        # takes whatever the query items leave and absorbs all overflow by itself. Setting
        # passage_max_length as well only cuts the document EARLIER than the total
        # requires, discarding text that would have fit.
        #
        # All default to None (uncapped) so existing configs are unaffected; set them
        # explicitly. Defaults per field cannot be chosen here because they only make
        # sense relative to rerank_max_length and to the document's share of it.
        self.global_query_max_length = global_query_max_length
        self.sub_query_max_length = sub_query_max_length
        self.reasoning_max_length = reasoning_max_length
        self.passage_max_length = passage_max_length
        raw = instructions if instructions is not None else self.DEFAULT_INSTRUCTIONS
        self.instructions = self._normalize_instructions(raw)
        # Validate up front rather than at collation time. _format_one selects the
        # instruction by which context fields SURVIVE the drop, so a misspelled key silently
        # yields no entry for a real mode; falling back would train a context-bearing prompt
        # under the no-context instruction, which is invisible in the loss and only shows up
        # as a model that ignores its context at eval. Both directions are errors: an
        # unrecognised key is a typo, and a missing one leaves a reachable mode unspecified.
        supported = set(self.DEFAULT_INSTRUCTIONS)
        unknown = set(self.instructions) - supported
        if unknown:
            raise ValueError(
                "instructions contains unsupported context modes: "
                f"{sorted(sorted(k) for k in unknown)}; supported modes are "
                f"{sorted(sorted(k) for k in supported)}"
            )
        missing = supported - set(self.instructions)
        if missing:
            raise ValueError(
                "instructions is missing an entry for context modes: "
                f"{sorted(sorted(k) for k in missing)}; every mode reachable under the drop "
                "probabilities needs its own instruction"
            )
        self.reasoning_drop_prob = reasoning_drop_prob
        self.global_query_drop_prob = global_query_drop_prob
        self.drop_seed = drop_seed
        # Shared memory rather than a plain int because collate_fn runs inside the
        # DataLoader worker processes. Under persistent_workers=True those workers outlive
        # the epoch boundary, so an attribute assigned in the parent would never reach the
        # already-forked children and every epoch would replay epoch 0's drops. A shared
        # tensor is visible to the workers as the parent mutates it in place.
        #
        # Left untouched the epoch stays 0, which is what the validation collator wants:
        # the same fixed mix every time.
        self._epoch = torch.zeros(1, dtype=torch.int64).share_memory_()
        system_message = system_message if system_message is not None else self.DEFAULT_SYSTEM
        # Defaults are ChatML plus Qwen3's empty think block, which is what makes the final
        # tokens a yes/no next-token prediction. Override both for a backbone with different
        # chat markers. They are encoded once and concatenated around the truncated middle,
        # so the markers themselves can never be truncated away.
        prefix = (prefix_template or self.DEFAULT_PREFIX_TEMPLATE).format(system_message=system_message)
        suffix = suffix_template or self.DEFAULT_SUFFIX_TEMPLATE
        self.prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used for deterministic context dropout.

        Called by ``StepScheduler.set_epoch`` on the training dataloader's collate
        function, so drops are redrawn each epoch. A validation collator is never given an
        epoch and stays at 0, keeping its prompt mix fixed -- otherwise a val_loss that
        moved because the sampled modes changed would be indistinguishable from one that
        moved because the model did.

        Args:
            epoch: Zero-based index of the epoch about to run.
        """
        self._epoch[0] = epoch

    def _keep_field(self, kind: str, query: str, prob: float) -> bool:
        """Whether to KEEP a context field for this query, deterministically.

        Keyed on the QUERY, not the (query, document) pair: the dataset repeats the
        context fields across every passage of a group, and a listwise group is scored as
        one softmax over 1 positive + n negatives. If passages within a group disagreed
        about which context was present, the comparison would be between different
        prompts rather than between documents.

        Hashed rather than sampled from a shared RNG so the draw depends only on
        (drop_seed, epoch, field, query) -- identical across runs, workers and ranks, and
        independent of batch order. hashlib rather than hash(): PYTHONHASHSEED randomises
        str hashing per process, which would make runs unreproducible.
        """
        if prob <= 0.0:
            return True
        if prob >= 1.0:
            return False
        epoch = int(self._epoch[0].item())
        key = f"{self.drop_seed}|{epoch}|{kind}|{query}".encode()
        u = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") / 2.0**64
        return u >= prob

    def _format_one(self, query: str, doc: str, reasoning: str = None, global_query: str = None) -> str:
        """Build the user-turn text for a single (query, doc) pair.

        Order: decide which context fields survive the drop draws, cap each surviving
        item on its own token budget, pick the instruction matching what survived, then
        assemble with the context sub-fields embedded inside ``<Query>:``.

        Capping precedes assembly so each item is guaranteed its share. The document is
        normally left uncapped and takes the remainder of ``rerank_max_length``, absorbing
        all overflow on its own because it is assembled last.
        """
        # A field is available only if the data actually carries it. A row with no trace
        # is not a "drop" -- there was nothing to drop -- so no draw is made for it.
        has_gq = bool(global_query and global_query.strip())
        has_r = bool(reasoning and reasoning.strip())

        # DROP FIRST. Dropping REMOVES the field, so the prompt shape changes; it does not
        # substitute a placeholder. The two draws are independent, so a dataset carrying
        # both fields yields all four modes at rates
        #   full (1-pr)(1-pg) | globalq pr(1-pg) | reasoning (1-pr)pg | base pr*pg
        if has_r and not self._keep_field("reasoning", query, self.reasoning_drop_prob):
            has_r = False
        if has_gq and not self._keep_field("global_query", query, self.global_query_drop_prob):
            has_gq = False

        # Cap each surviving item on its own budget, before assembly. A field that is
        # over its cap loses only its own tail; the others are untouched.
        query = self._truncate_tokens(query, self.sub_query_max_length)
        global_query_text = global_query.strip() if has_gq else None
        if global_query_text is not None:
            global_query_text = self._truncate_tokens(global_query_text, self.global_query_max_length)
        reasoning_text = reasoning.strip() if has_r else None
        if reasoning_text is not None:
            reasoning_text = self._truncate_tokens(reasoning_text, self.reasoning_max_length)

        # Instruction is chosen from what SURVIVED, so it always matches the content.
        # Choosing it beforehand made a dropped row keep the richer instruction while its
        # prompt no longer carried that context.
        present = frozenset(f for f, flag in (("reasoning", has_r), ("global_query", has_gq)) if flag)
        instruction = self.instructions[present]

        # Build the <Query> block. The user-generated query is tagged #Query in every mode
        # that has a marker, matching DEFAULT_SYSTEM's "based on the Query"; the wider
        # question carries its parenthetical so the two can never be confused.
        if not has_gq and reasoning_text is None:
            # Base mode: bare query, no markers at all -- byte-identical to the
            # out-of-the-box Qwen3-Reranker prompt. Do not add markers here.
            query_block = query
        elif not has_gq:
            # Reasoning mode.
            query_block = f"#Reasoning Trace: {reasoning_text}\n#Query: {query}"
        elif reasoning_text is None:
            # GlobalQ mode.
            query_block = f"#Original Question (global query): {global_query_text}\n#Query: {query}"
        else:
            # Full mode.
            query_block = (
                f"#Original Question (global query): {global_query_text}\n"
                f"#Reasoning Trace: {reasoning_text}\n"
                f"#Query: {query}"
            )

        # The block itself is NOT re-truncated: its items are already capped, and cutting
        # it here would take the tail, which is the query itself.
        doc = self._truncate_tokens(doc, self.passage_max_length)
        return f"<Instruct>: {instruction}\n<Query>: {query_block}\n<Document>: {doc}"

    def _truncate_tokens(self, text: str, limit: int = None) -> str:
        """Cut ``text`` to at most ``limit`` tokens. No-op when limit is None."""
        if not limit or not text:
            return text
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= limit:
            return text
        return self.tokenizer.decode(ids[:limit])

    def __call__(self, features: List[Dict[str, Any]]) -> "BatchEncoding":
        query_examples = [x["question"] for x in features]
        doc_examples = [x["doc_text"] for x in features]
        reasoning_examples = [x.get("reasoning") for x in features]
        global_query_examples = [x.get("global_query") for x in features]
        num_labels = features[0].get("num_labels") if features else None

        examples = [
            self._format_one(q, d, r, gq)
            for q, d, r, gq in zip(query_examples, doc_examples, reasoning_examples, global_query_examples)
        ]

        middle_max = max(self.rerank_max_length - len(self.prefix_ids) - len(self.suffix_ids), 1)
        encodings = self.tokenizer(
            examples,
            max_length=middle_max,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=True,
            add_special_tokens=False,
        )

        tok_features = []
        for i in range(len(examples)):
            ids = self.prefix_ids + encodings["input_ids"][i] + self.suffix_ids
            tok_features.append({"input_ids": ids, "attention_mask": [1] * len(ids)})

        batch_dict = self.tokenizer.pad(
            tok_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if num_labels is not None:
            batch_dict["labels"] = torch.zeros(num_labels, dtype=torch.long)

        return batch_dict


__all__ = ["Qwen3ContextAwareRerankerCollator"]
