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

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class NeMoAutoTokenizerWithBosEosEnforced(AutoTokenizer):
    """
    A wrapper around HuggingFace's AutoTokenizer that ensures consistent BOS/EOS token handling.

    There are pre-existing issues with some tokenizers (e.g. GPT2Tokenizer) where the BOS/EOS tokens
    are not added automatically. This wrapper ensures they are always added when requested.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, add_bos_token=True, add_eos_token=True, **kwargs):
        """
        Load the HF tokenizer class via AutoTokenizer and (optionally) wrap it to add BOS/EOS.

        Args:
            pretrained_model_name_or_path: Model identifier or path
            add_bos_token: Whether to add BOS token (default: True)
            add_eos_token: Whether to add EOS token (default: True)
        """
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        # Transformers >=5.0.0 defaults special_tokens_pattern to "cls_sep", which inserts
        # cls_token_id / sep_token_id into input_ids via build_inputs_with_special_tokens.
        # Moonlight's TikTokenTokenizer doesn't define CLS/SEP, so those IDs are None,
        # resulting in None values in input_ids and a downstream ValueError in pad().
        # Fix: when the pattern is "cls_sep" but the required tokens are missing, fall
        # back to "none" so build_inputs_with_special_tokens passes through unchanged.
        if getattr(tokenizer, "special_tokens_pattern", None) == "cls_sep" and (
            getattr(tokenizer, "cls_token_id", None) is None or getattr(tokenizer, "sep_token_id", None) is None
        ):
            tokenizer.special_tokens_pattern = "none"

        if add_bos_token and getattr(tokenizer, "bos_token", None) is not None:
            try:
                tokenizer.add_bos_token = add_bos_token
            except ValueError:
                tokenizer._add_bos_token = add_bos_token
        if add_eos_token and getattr(tokenizer, "eos_token", None) is not None:
            try:
                tokenizer.add_eos_token = add_eos_token
            except ValueError:
                tokenizer._add_eos_token = add_eos_token
        # Keep the wrapper class name at runtime, but remember the original HF tokenizer class
        # so we can save an HF-compatible `tokenizer_class` in `save_pretrained()`.
        base_tokenizer_cls = type(tokenizer)
        tokenizer._base_class = base_tokenizer_cls
        tokenizer.__class__ = type(cls.__name__, (cls, base_tokenizer_cls), {})
        return tokenizer

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

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        # HF writes `tokenizer_class` using `self.__class__.__name__`. Our runtime class name is
        # the NeMo wrapper, but for portability we want to save the original HF tokenizer class.
        # base_name = getattr(self, "_nemo_base_tokenizer_class_name", None)
        base_class = getattr(self, "_base_class", None)
        if not base_class:
            return super().save_pretrained(save_directory, push_to_hub=push_to_hub, **kwargs)

        original_cls = self.__class__
        try:
            self.__class__ = base_class
            return base_class.save_pretrained(self, save_directory, push_to_hub=push_to_hub, **kwargs)
        finally:
            self.__class__ = original_cls


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
