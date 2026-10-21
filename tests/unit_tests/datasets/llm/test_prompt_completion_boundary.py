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

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

from nemo_automodel.components.datasets.llm.formatting_utils import format_prompt_completion


@pytest.fixture(params=[False, True], ids=["plain", "bos-eos"])
def boundary_tokenizer(request: pytest.FixtureRequest) -> PreTrainedTokenizerFast:
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        [
            "Context: The observatory is located in Kyoto. Question: Where is the observatory located? Answer: "
            + answer
            for answer in ["Kyoto", "Cobalt"]
        ],
        trainers.BpeTrainer(
            vocab_size=400,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<pad>", "<bos>", "<eos>"],
        ),
    )
    if request.param:
        backend.post_processor = processors.TemplateProcessing(
            single="<bos> $A <eos>",
            special_tokens=[("<bos>", backend.token_to_id("<bos>")), ("<eos>", backend.token_to_id("<eos>"))],
        )
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>")


@pytest.mark.parametrize("answer", ["Kyoto", "Cobalt"])
@pytest.mark.parametrize("unshifted", [False, True])
@pytest.mark.parametrize("padding,seq_length", [(False, 512), ("max_length", 512), (False, 8)])
def test_prompt_completion_supervises_boundary_token(
    boundary_tokenizer: PreTrainedTokenizerFast,
    answer: str,
    unshifted: bool,
    padding: bool | str,
    seq_length: int,
) -> None:
    tok = boundary_tokenizer
    prompt = "Context: The observatory is located in Kyoto. Question: Where is the observatory located? Answer: "
    encoded = tok(prompt + answer, return_offsets_mapping=True, truncation=True, max_length=seq_length)
    answer_positions = [i for i, (start, end) in enumerate(encoded["offset_mapping"]) if end > len(prompt)]
    if seq_length == 512:
        # A real BPE merges the prompt's trailing space with the answer.
        assert encoded["offset_mapping"][answer_positions[0]][0] < len(prompt)
    else:
        assert not answer_positions

    result = format_prompt_completion(
        tok,
        prompt,
        answer,
        tok.eos_token_id,
        tok.pad_token_id,
        padding=padding,
        truncation=True,
        seq_length=seq_length,
        unshifted=unshifted,
    )
    expected = [
        int(end > len(prompt) or (bool(answer_positions) and token == tok.eos_token_id))
        for token, (start, end) in zip(encoded["input_ids"], encoded["offset_mapping"])
    ]
    if unshifted:
        actual = result["loss_mask"]
    else:
        expected = expected[1:]
        actual = [int(label != -100) for label in result["labels"]]
    expected += [0] * (len(actual) - len(expected))
    assert actual == expected
    for i in answer_positions:
        pos = i if unshifted else i - 1
        assert actual[pos] == 1
        if not unshifted:
            assert result["labels"][pos] == encoded["input_ids"][i]
