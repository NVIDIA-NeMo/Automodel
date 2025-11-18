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

from typing import Any, Dict, List, Union

import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


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


class RetrievalBiencoderCollator:
    """
    Collator for biencoder retrieval training.

    This collator handles tokenization of queries and documents at batch time,
    which is more memory-efficient than pre-tokenization and allows for
    dynamic padding based on batch max length.

    Based on BiencoderCollator from nemo-retriever-research but adapted for Automodel.
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
        """
        self.tokenizer = tokenizer
        self.q_max_len = q_max_len
        self.p_max_len = p_max_len
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of

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
        for doc in doc_examples:
            doc_examples_flat += doc

        # Add prefixes if provided
        if self.query_prefix:
            query_examples = [self.query_prefix + " " + question for question in query_examples]
        if self.passage_prefix:
            doc_examples_flat = [self.passage_prefix + " " + passage for passage in doc_examples_flat]

        # Tokenize queries (no padding yet)
        query_encodings = self.tokenizer(
            query_examples,
            max_length=self.q_max_len,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=True,
            return_token_type_ids=False,
        )

        # Tokenize documents (no padding yet)
        doc_encodings = self.tokenizer(
            doc_examples_flat,
            max_length=self.p_max_len,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=True,
            return_token_type_ids=False,
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
