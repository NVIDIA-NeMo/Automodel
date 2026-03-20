# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import TYPE_CHECKING, Any, Dict, List, Union

import torch
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

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
        self.tokenizer = tokenizer
        self.q_max_len = q_max_len
        self.p_max_len = p_max_len
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.use_dataset_instruction = use_dataset_instruction

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        query_examples = [x["question"] for x in batch]
        doc_examples = [x["doc_text"] for x in batch]

        doc_examples_flat = []
        doc_size = len(doc_examples[0])

        if self.use_dataset_instruction:
            query_instruction_examples = [x["query_instruction"] for x in batch]
            passage_instruction_examples = [x["passage_instruction"] for x in batch]
            passage_instruction_examples_flat = []

            for doc, passage_instruction in zip(doc_examples, passage_instruction_examples):
                doc_examples_flat += doc
                passage_instruction_examples_flat += [passage_instruction] * len(doc)
        else:
            for doc in doc_examples:
                doc_examples_flat += doc

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

        query_encodings = self.tokenizer(
            query_examples,
            max_length=self.q_max_len,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=True,
            return_token_type_ids=False,
        )

        doc_encodings = self.tokenizer(
            doc_examples_flat,
            max_length=self.p_max_len,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=True,
            return_token_type_ids=False,
        )

        features = self._merge_batch_dict(
            query_batch_dict=query_encodings, doc_batch_dict=doc_encodings, train_n_passages=doc_size
        )
        features = self._convert_dict_to_list(features)

        q_prefix, d_prefix = "q_", "d_"
        query_features = [{k[len(q_prefix) :]: v for k, v in f.items() if k.startswith(q_prefix)} for f in features]
        doc_features = _unpack_doc_values(
            [{k[len(d_prefix) :]: v for k, v in f.items() if k.startswith(d_prefix)} for f in features]
        )

        assert len(doc_features) % len(query_features) == 0, (
            f"{len(doc_features)} doc and {len(query_features)} queries"
        )

        q_collated = self.tokenizer.pad(
            query_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt"
        )

        d_collated = self.tokenizer.pad(
            doc_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt"
        )

        merged_batch_dict = {q_prefix + k: v for k, v in q_collated.items()}
        merged_batch_dict.update({d_prefix + k: v for k, v in d_collated.items()})
        merged_batch_dict["labels"] = torch.zeros(len(query_features), dtype=torch.long)

        return merged_batch_dict

    def _merge_batch_dict(
        self, query_batch_dict: Dict[str, List], doc_batch_dict: Dict[str, List], train_n_passages: int
    ) -> Dict[str, List]:
        batch_size = len(query_batch_dict["input_ids"])

        merged_batch_dict = {"q_" + k: v for k, v in query_batch_dict.items()}
        # Reshape doc features from [batch_size * n_passages, seq_len] to [batch_size, n_passages, seq_len]
        for key, values in doc_batch_dict.items():
            merged_batch_dict["d_" + key] = [
                values[i * train_n_passages : (i + 1) * train_n_passages] for i in range(batch_size)
            ]
        return merged_batch_dict

    def _convert_dict_to_list(self, input_dict: Dict[str, List]) -> List[Dict[str, Any]]:
        """Convert ``{'a': [1, 2], 'b': [3, 4]}`` to ``[{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]``."""
        keys = list(input_dict.keys())
        return [{k: input_dict[k][i] for k in keys} for i in range(len(input_dict[keys[0]]))]


class CrossEncoderCollator(DataCollatorWithPadding):
    def __init__(self, rerank_max_length: int, *args, prompt_template: str = "question:{query} \n \n passage:{passage}", **kwargs):
        self.rerank_max_length = rerank_max_length
        self.prompt_template = prompt_template
        self.args = kwargs.pop("args", None)
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
