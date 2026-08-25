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

import hashlib
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union, cast

import torch
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase, ProcessorMixin
from transformers.file_utils import PaddingStrategy


def _doc_id_str_to_int64(doc_id: str) -> int:
    """Stable 63-bit int for corpus doc id strings (for in-batch duplicate masking)."""
    h = hashlib.md5(doc_id.encode("utf-8")).digest()[:8]
    return int.from_bytes(h, "little", signed=False) & ((1 << 63) - 1)


if TYPE_CHECKING:
    from transformers import BatchEncoding


def _unpack_doc_values(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Unpack document lists into individual examples.

    Example:
        Input: [{'input_ids': [[1,2], [3,4]], 'attention_mask': [[1,1], [1,1]]}]
        Output: [{'input_ids': [1,2], 'attention_mask': [1,1]},
                 {'input_ids': [3,4], 'attention_mask': [1,1]}]
    """
    doc_examples = []
    for f in features:
        keys = list(f.keys())
        lists_per_key = len(f[keys[0]])
        for idx in range(lists_per_key):
            doc_examples.append({k: f[k][idx] for k in keys})
    return doc_examples


class BiEncoderCollator:
    """
    Collator for encoder retrieval training.

    This collator handles tokenization of queries and documents at batch time,
    which is more memory-efficient than pre-tokenization and allows for
    dynamic padding based on batch max length.

    Based on EncoderCollator from nemo-retriever-research but adapted for Automodel.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        q_max_len: int = 512,
        p_max_len: int = 512,
        query_prefix: str = "",
        passage_prefix: str = "",
        padding: Union[bool, str, PaddingStrategy] = True,
        pad_to_multiple_of: int = None,
        use_dataset_instruction: bool = False,
    ):
        """
        Initialize the collator.

        Args:
            tokenizer: Tokenizer to use for encoding
            q_max_len: Maximum length for queries
            p_max_len: Maximum length for passages
            query_prefix: Prefix to add to queries (e.g., "query: ")
            passage_prefix: Prefix to add to passages (e.g., "passage: ")
            padding: Padding strategy ("longest", "max_length", or "do_not_pad")
            pad_to_multiple_of: Pad to multiple of this value (e.g., 8 for FP16)
            use_dataset_instruction: Whether to use instruction from dataset's metadata
        """
        self.tokenizer = tokenizer
        self.q_max_len = q_max_len
        self.p_max_len = p_max_len
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.use_dataset_instruction = use_dataset_instruction

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.

        Args:
            batch: List of examples, each with 'question', 'doc_text', 'doc_image' keys

        Returns:
            Dictionary with:
            - q_input_ids: Query input IDs [batch_size, q_seq_len]
            - q_attention_mask: Query attention mask [batch_size, q_seq_len]
            - d_input_ids: Document input IDs [batch_size * num_docs, d_seq_len]
            - d_attention_mask: Document attention mask [batch_size * num_docs, d_seq_len]
            - labels: Dummy labels for compatibility [batch_size]
        """
        # Extract queries and documents
        query_examples = [x["question"] for x in batch]
        doc_examples = [x["doc_text"] for x in batch]

        # Flatten documents (each example has multiple docs)
        doc_examples_flat = []
        doc_size = len(doc_examples[0])

        if self.use_dataset_instruction:
            query_instruction_examples = [x["query_instruction"] for x in batch]
            passage_instruction_examples = [x["passage_instruction"] for x in batch]
            passage_instruction_examples_flat = []

            # Flatten documents with instructions
            for doc, passage_instruction in zip(doc_examples, passage_instruction_examples):
                doc_examples_flat += doc
                passage_instruction_examples_flat += [passage_instruction] * len(doc)
        else:
            # Flatten documents without instructions
            for doc in doc_examples:
                doc_examples_flat += doc

        # Add prefixes
        if self.use_dataset_instruction:
            query_examples = [
                f"{query_instruction} {question}" if query_instruction else question
                for query_instruction, question in zip(query_instruction_examples, query_examples)
            ]
            doc_examples_flat = [
                f"{passage_instruction} {passage}" if passage_instruction else passage
                for passage_instruction, passage in zip(passage_instruction_examples_flat, doc_examples_flat)
            ]
        else:
            if self.query_prefix:
                query_examples = [self.query_prefix + " " + question for question in query_examples]
            if self.passage_prefix:
                doc_examples_flat = [self.passage_prefix + " " + passage for passage in doc_examples_flat]

        # Some tokenizer backends, including MistralCommonBackend, do not support
        # the return_token_type_ids kwarg and do not advertise token type IDs.
        token_type_kwargs = {}
        if "token_type_ids" in getattr(self.tokenizer, "model_input_names", []):
            token_type_kwargs["return_token_type_ids"] = False

        # Tokenize queries (no padding yet)
        query_encodings = self.tokenizer(
            query_examples,
            max_length=self.q_max_len,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=True,
            **token_type_kwargs,
        )

        # Tokenize documents (no padding yet)
        doc_encodings = self.tokenizer(
            doc_examples_flat,
            max_length=self.p_max_len,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=True,
            **token_type_kwargs,
        )

        # Merge into features format for unpacking
        features = self._merge_batch_dict(
            query_batch_dict=query_encodings, doc_batch_dict=doc_encodings, train_n_passages=doc_size
        )
        features = self._convert_dict_to_list(features)

        # Separate query and document features with prefixes
        q_prefix, d_prefix = "q_", "d_"
        query_features = [{k[len(q_prefix) :]: v for k, v in f.items() if k.startswith(q_prefix)} for f in features]
        doc_features = _unpack_doc_values(
            [{k[len(d_prefix) :]: v for k, v in f.items() if k.startswith(d_prefix)} for f in features]
        )

        assert len(doc_features) % len(query_features) == 0, (
            f"{len(doc_features)} doc and {len(query_features)} queries"
        )

        # Pad queries based on batch max length
        q_collated = self.tokenizer.pad(
            query_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt"
        )

        # Pad documents based on batch max length
        d_collated = self.tokenizer.pad(
            doc_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt"
        )

        # Add prefixes to keys
        merged_batch_dict = {}
        for k in q_collated.keys():
            merged_batch_dict[q_prefix + k] = q_collated[k]
        for k in d_collated.keys():
            merged_batch_dict[d_prefix + k] = d_collated[k]

        # Add dummy labels (required by some training frameworks)
        labels = torch.zeros(len(query_features), dtype=torch.long)
        merged_batch_dict["labels"] = labels

        # Per-passage corpus doc ids (positive + negatives, flattened in d_input_ids
        # order) for distributed in-batch same-doc negative masking. Top-level key
        # so it bypasses the q_/d_ unpacking in the trainer.
        doc_id_groups = [x.get("doc_id") for x in batch]
        # Inline records may not provide IDs; incomplete IDs are unsafe for same-doc masking.
        if doc_id_groups and all(doc_ids and all(doc_ids) for doc_ids in doc_id_groups):
            doc_id_flat = [doc_id for doc_ids in doc_id_groups for doc_id in doc_ids]
            merged_batch_dict["passage_doc_ids"] = torch.tensor(
                [_doc_id_str_to_int64(s) for s in doc_id_flat],
                dtype=torch.long,
            )

        return merged_batch_dict

    def _merge_batch_dict(
        self, query_batch_dict: Dict[str, List], doc_batch_dict: Dict[str, List], train_n_passages: int
    ) -> Dict[str, List]:
        """
        Merge query and document batches into a single dictionary.

        Adapted from nemo-retriever-research/src/loaders/loader_utils.py
        """
        batch_size = len(query_batch_dict["input_ids"])

        merged_batch_dict = {}
        for key in query_batch_dict:
            merged_batch_dict["q_" + key] = query_batch_dict[key]

        for key in doc_batch_dict:
            # Reshape doc features: [batch_size * train_n_passages, seq_len]
            # -> [batch_size, train_n_passages, seq_len]
            doc_values = doc_batch_dict[key]
            doc_values_reshaped = []
            for i in range(batch_size):
                doc_values_reshaped.append(doc_values[i * train_n_passages : (i + 1) * train_n_passages])
            merged_batch_dict["d_" + key] = doc_values_reshaped

        return merged_batch_dict

    def _convert_dict_to_list(self, input_dict: Dict[str, List]) -> List[Dict[str, Any]]:
        """
        Convert dictionary of lists to list of dictionaries.

        Example:
            Input: {'a': [1, 2], 'b': [3, 4]}
            Output: [{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]
        """
        out_list = []
        length = len(input_dict[list(input_dict.keys())[0]])
        for i in range(length):
            tmp = {}
            for key in input_dict.keys():
                tmp[key] = input_dict[key][i]
            out_list.append(tmp)
        return out_list


class CrossEncoderCollator(DataCollatorWithPadding):
    """Collate query-document pairs for cross-encoder reranking."""

    def __init__(
        self, rerank_max_length: int, *args, prompt_template: str = "question:{query} \n \n passage:{passage}", **kwargs
    ):
        self.rerank_max_length = rerank_max_length
        self.prompt_template = prompt_template
        # Call Base with all args and kwargs
        self.args = None
        if "args" in kwargs:
            self.args = kwargs.pop("args")
        super().__init__(*args, **kwargs)

    def __call__(self, features: List[Dict[str, Any]]) -> "BatchEncoding":
        query_examples = [x["question"] for x in features]
        doc_examples = [x["doc_text"] for x in features]
        num_labels = features[0].get("num_labels") if features else None

        def format_text(q, p):
            return self.prompt_template.format(query=q, passage=p)

        examples = [format_text(q, d) for q, d in zip(query_examples, doc_examples)]

        # Tokenize without tensors first (so NeMoAutoTokenizer BOS/EOS insertion works on lists),
        # then pad and convert to tensors in a separate step.
        encodings = self.tokenizer(
            examples,
            max_length=self.rerank_max_length,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=True,
        )
        tok_features = [{k: encodings[k][i] for k in encodings} for i in range(len(examples))]
        batch_dict = self.tokenizer.pad(
            tok_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if num_labels is not None:
            batch_dict["labels"] = torch.zeros(num_labels, dtype=torch.long)

        return batch_dict


class ProcessorMethodCollator:
    """Expose one method of a multimodal processor as a dataloader collator."""

    def __init__(self, tokenizer: ProcessorMixin, collator_fn_name: str) -> None:
        """Resolve the processor method once during dataloader construction.

        Args:
            tokenizer: Runtime multimodal processor.
            collator_fn_name: Processor method used to collate each batch.
        """
        self.collate_fn = cast(
            Callable[[list[dict[str, object]]], dict[str, object]],
            getattr(tokenizer, collator_fn_name),
        )

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, object]:
        """Collate retrieval examples with the resolved processor method.

        Args:
            batch: Retrieval examples for one local batch.

        Returns:
            Processor-produced tensor batch.
        """
        return self.collate_fn(batch)


def make_vision_collator_from_processor_method(
    tokenizer: ProcessorMixin,
    collator_fn_name: str,
) -> Callable[[list[dict[str, object]]], dict[str, object]]:
    """
    Turns a method of a processor into a collator function.

    Args:
        tokenizer: The processor instance.
        collator_fn_name: The name of the processor method to turn into a collator function.

    Returns:
        A collator for vision/multimodal retrieval datasets.
    """
    warnings.warn(
        "make_vision_collator_from_processor_method is deprecated; use ProcessorMethodCollator instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return cast(Callable[[list[dict[str, object]]], dict[str, object]], getattr(tokenizer, collator_fn_name))


class ContextAwareRerankerCollator(CrossEncoderCollator):
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
    redrawn per epoch once the recipe calls ``set_epoch_source``. Set both to 0.0 at
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
        frozenset(): (
            "Given a web search query, retrieve relevant passages that answer the query"
        ),
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
                raise TypeError(
                    f"instructions keys must be str, tuple, or frozenset; got {type(key)}"
                )
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
        super().__init__(rerank_max_length, *args, **kwargs)
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
        # Supplied by the recipe so drops differ per epoch. Left unset the epoch reads 0,
        # which is what the validation collator wants: the same fixed mix every time.
        self._epoch_fn = None
        system_message = system_message if system_message is not None else self.DEFAULT_SYSTEM
        # Defaults are ChatML plus Qwen3's empty think block, which is what makes the final
        # tokens a yes/no next-token prediction. Override both for a backbone with different
        # chat markers. They are encoded once and concatenated around the truncated middle,
        # so the markers themselves can never be truncated away.
        prefix = (prefix_template or self.DEFAULT_PREFIX_TEMPLATE).format(system_message=system_message)
        suffix = suffix_template or self.DEFAULT_SUFFIX_TEMPLATE
        self.prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)

    def set_epoch_source(self, fn) -> None:
        """Register a zero-arg callable returning the current epoch.

        Drops are then redrawn each epoch. Without it the epoch is 0 forever, so the mix
        is fixed -- correct for validation, where a val_loss that moves because the
        prompt mix changed would be indistinguishable from one that moved because the
        model did.
        """
        self._epoch_fn = fn

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
        epoch = self._epoch_fn() if self._epoch_fn is not None else 0
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
