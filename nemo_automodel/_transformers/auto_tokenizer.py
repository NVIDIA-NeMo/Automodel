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

import logging
from typing import Any, Callable, Optional, Union

import transformers
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

logger = logging.getLogger(__name__)


def _check_supports_system_role(tokenizer):
    if getattr(tokenizer, "chat_template", None) is not None:
        if "System role not supported" in tokenizer.chat_template:
            logging.warning("Tokenizer: assistant role will be dropped from the first message.")
            logging.warning("Tokenizer: chat template: %s", tokenizer.chat_template)
            return False
    return True


class NeMoAutoTokenizer(AutoTokenizer):
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *args, force_hf=False, add_bos_token=True, add_eos_token=True, **kwargs
    ):
        """
        Load the HF tokenizer class via AutoTokenizer and (optionally) wrap it to add BOS/EOS.

        There are pre-existing issues with some tokenizers (e.g. GPT2Tokenizer) where the BOS/EOS tokens are not added
        """
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if force_hf:
            return tokenizer

        # Create a new class that inherits from NeMoAutoTokenizer AND the specific HF tokenizer
        tokenizer.__class__ = type("NeMoAutoTokenizer", (cls, type(tokenizer)), {})
        tokenizer.supports_system_role = _check_supports_system_role(tokenizer)
        assert isinstance(add_bos_token, bool), f"Expected add_bos_token to be a boolean, got {type(add_bos_token)}"
        assert isinstance(add_eos_token, bool), f"Expected add_eos_token to be a boolean, got {type(add_eos_token)}"
        tokenizer.add_bos_token = add_bos_token
        tokenizer.add_eos_token = add_eos_token
        return tokenizer

    def apply_chat_template(
        self,
        conversation: Union[list[dict[str, str]], list[list[dict[str, str]]]],
        tools: Optional[list[Union[dict, Callable]]] = None,
        documents: Optional[list[dict[str, str]]] = None,
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: bool = True,
        padding: Union[bool, str, transformers.utils.generic.PaddingStrategy] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Union[str, transformers.utils.generic.TensorType, None] = None,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Union[str, list[int], list[str], list[list[int]], transformers.tokenization_utils_base.BatchEncoding]:
        if not self.supports_system_role and conversation[0].get("role", None) == "system":
            if not any(map(lambda x: x.get("role", None) == "system", conversation[1:])):
                # in this case we will drop the first message
                conversation = conversation[1:]
            else:
                raise ValueError("System role appeared in multiple messages.")

        return super().apply_chat_template(
            conversation,
            tools=tools,
            documents=documents,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=tokenize,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_dict=return_dict,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
            tokenizer_kwargs=tokenizer_kwargs,
            **kwargs,
        )

    def __call__(self, *args, **kwargs):
        tokenized = super().__call__(*args, **kwargs)
        if not kwargs.get("add_special_tokens", True):
            return tokenized
        if isinstance(tokenized, BatchEncoding):
            _tokenized_keys = {"input_ids", "attention_mask", "assistant_masks"}
            add_bos_ids = self.add_bos_token and (getattr(self, "bos_token_id", None) is not None)
            add_eos_ids = self.add_eos_token and (getattr(self, "eos_token_id", None) is not None)
            if not "input_ids" in tokenized:
                return tokenized
            if add_bos_ids:
                add_bos_ids = _add_token(tokenized, self.bos_token_id, 0, "input_ids")
            if add_eos_ids:
                add_eos_ids = _add_token(tokenized, self.eos_token_id, -1, "input_ids")

            for key in {"attention_mask", "assistant_masks"}:
                if key not in tokenized:
                    continue
                if add_bos_ids:
                    _add_token(tokenized, 1, 0, key)
                if add_eos_ids:
                    _add_token(tokenized, 1, -1, key)
        return tokenized

    def encode(self, *args, **kwargs):
        encoded = super().encode(*args, **kwargs)
        if not kwargs.get("add_special_tokens", True):
            return encoded
        if self.add_bos_token:
            if encoded and (getattr(self, "bos_token_id", None) is not None) and encoded[0] != self.bos_token_id:
                encoded = [self.bos_token_id] + encoded
        if self.add_eos_token:
            if encoded and (getattr(self, "eos_token_id", None) is not None) and encoded[-1] != self.eos_token_id:
                encoded = encoded + [self.eos_token_id]
        return encoded


def _add_token(tokenized, value, position, key):
    def _extend_single(sequence, val, pos, always_add):
        if pos == 0:
            if always_add or not sequence or sequence[0] != val:
                return [val] + sequence, True
            return sequence, False
        if pos == -1:
            if always_add or not sequence or sequence[-1] != val:
                return sequence + [val], True
            return sequence, False
        raise ValueError(f"Invalid position: {pos}")

    sequences = tokenized[key]
    always_add = key != "input_ids"
    if isinstance(sequences, list) and sequences and isinstance(sequences[0], list):
        ans = [_extend_single(seq, value, position, always_add) for seq in sequences]
        tokenized[key] = list(map(lambda x: x[0], ans))
        return any(map(lambda x: x[1], ans))
    elif isinstance(sequences, list):
        ans = _extend_single(sequences, value, position, always_add)
        tokenized[key] = ans[0]
        return ans[1]
    else:
        raise ValueError(f"Invalid sequence type: {type(sequences)}")
    return False
