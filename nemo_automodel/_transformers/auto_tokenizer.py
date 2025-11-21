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

from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


class NeMoAutoTokenizer(PreTrainedTokenizerBase):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, force_hf=False, add_bos_token=True, add_eos_token=True, **kwargs):
        """
        Load the HF tokenizer class via AutoTokenizer and (optionally) wrap it to add BOS/EOS.

        There are pre-existing issues with some tokenizers (e.g. GPT2Tokenizer) where the BOS/EOS tokens are not added
        """
        import types
        from transformers import AutoTokenizer

        hf_tok = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if force_hf:
            return hf_tok

        # Capture flags in closure (avoid mutating the tokenizer instance)
        nemo_add_bos = bool(add_bos_token)
        nemo_add_eos = bool(add_eos_token)

        original_call = hf_tok.__call__
        original_encode = hf_tok.encode

        def _patched_call(self, *c_args, **c_kwargs):
            tokenized = original_call(*c_args, **c_kwargs)
            if isinstance(tokenized, BatchEncoding):
                _tokenized_keys = {"input_ids", "attention_mask", "assistant_masks"}
                add_bos_ids = nemo_add_bos and (getattr(self, "bos_token_id", None) is not None)
                add_eos_ids = nemo_add_eos and (getattr(self, "eos_token_id", None) is not None)
                for key in _tokenized_keys:
                    if key not in tokenized:
                        continue
                    if key == "input_ids":
                        if add_bos_ids:
                            _add_token(tokenized, self.bos_token_id, 0, key)
                        if add_eos_ids:
                            _add_token(tokenized, self.eos_token_id, -1, key)
                    else:
                        if add_bos_ids:
                            _add_token(tokenized, 1, 0, key)
                        if add_eos_ids:
                            _add_token(tokenized, 1, -1, key)
            return tokenized

        def _patched_encode(self, *e_args, **e_kwargs):
            encoded = original_encode(*e_args, **e_kwargs)
            if nemo_add_bos:
                if encoded and (self.bos_token_id is not None) and encoded[0] != self.bos_token_id:
                    encoded = [self.bos_token_id] + encoded
            if nemo_add_eos:
                if encoded and (self.eos_token_id is not None) and encoded[-1] != self.eos_token_id:
                    encoded = encoded + [self.eos_token_id]
            return encoded

        hf_tok.__call__ = types.MethodType(_patched_call, hf_tok)
        hf_tok.encode = types.MethodType(_patched_encode, hf_tok)
        return hf_tok


def _add_token(tokenized, value, position, key):
    def _extend_single(sequence, val, pos, always_add):
        if pos == 0:
            if always_add or not sequence or sequence[0] != val:
                return [val] + sequence
            return sequence
        if pos == -1:
            if always_add or not sequence or sequence[-1] != val:
                return sequence + [val]
            return sequence
        raise ValueError(f"Invalid position: {pos}")

    sequences = tokenized[key]
    always_add = key != "input_ids"
    if isinstance(sequences, list) and sequences and isinstance(sequences[0], list):
        tokenized[key] = [_extend_single(seq, value, position, always_add) for seq in sequences]
    elif isinstance(sequences, list):
        tokenized[key] = _extend_single(sequences, value, position, always_add)
    else:
        raise ValueError(f"Invalid sequence type: {type(sequences)}")
    return tokenized
