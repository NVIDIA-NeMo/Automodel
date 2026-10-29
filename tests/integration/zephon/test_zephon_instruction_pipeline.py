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

"""Built-in SFT tokenization, supervision and replay through real CPU pipelines."""

import json
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

from nemo_automodel.components.datasets.zephon import ZephonDataloaderConfig
from nemo_automodel.components.datasets.zephon.config import (
    ZephonPretokenizedTextSourceConfig,
    ZephonInstructionSourceConfig,
)
from tests.utils.zephon_text_fixtures import config as config
from tests.utils.zephon_text_fixtures import tokenizer as tokenizer

TAGGED = "{% for message in messages %}{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] }} [EOS]{% endgeneration %}{% else %}{{ message['content'] }} {% endif %}{% endfor %}"
PREFIX = "{% for message in messages %}{% if message['role'] == 'assistant' %}w4 {% endif %}{{ message['content'] }} {% endfor %}"


def _instruction_config(config, payloads, **settings):
    path = Path(config.zephon.sources[0].path) / "text.jsonl"
    path.write_text("".join(json.dumps(payload) + "\n" for payload in payloads))
    return replace(
        config,
        dataset_config=ZephonInstructionSourceConfig(column_mapping={"question": "q", "answer": "a"}, **settings),
        batch_size=1,
        zephon=replace(
            config.zephon,
            packing=replace(
                config.zephon.packing,
                seq_len=15,
                pack_algorithm="best_fit",
                pack_num_bins=2,
                max_num_documents=1,
                max_num_documents_scope="bin",
            ),
            source=replace(config.zephon.source, chunk_size=1),
        ),
    )


@pytest.mark.parametrize(
    "template,answer_only,tokens,labels",
    [
        (TAGGED, True, [2, 3, 4, 5, 1], [-100, 4, 5, 1]),
        (PREFIX, True, [2, 3, 6, 4, 5], [-100, 6, 4, 5]),
        ({"default": PREFIX}, False, [2, 3, 6, 4, 5], [3, 6, 4, 5]),
        ({"named_only": TAGGED}, True, [2, 3, 4, 5, 1], [-100, 4, 5, 1]),
    ],
)
def test_chat_supervision_is_selected_inside_zephon(config, tokenizer, template, answer_only, tokens, labels):
    tokenizer.chat_template = template
    payload = {"q": "w0 w1", "a": "w2 w3"}
    config = _instruction_config(config, [payload], use_hf_chat_template=True, answer_only_loss_mask=answer_only)
    with closing(config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)) as loader:
        batch = next(loader)
        length = len(tokens) - 1
        assert batch["input_ids"][0, :length].tolist() == tokens[:-1]
        assert batch["labels"][0, :length].tolist() == labels
        assert batch["labels"][0, length:].eq(-100).all()
        assert list(loader) == []


def test_chat_template_keeps_its_turn_marker_without_canonical_eos(config, tokenizer):
    tokenizer.add_special_tokens({"additional_special_tokens": ["[EOT]"]})
    turn_end = tokenizer.convert_tokens_to_ids("[EOT]")
    assert turn_end != tokenizer.eos_token_id
    tokenizer.chat_template = TAGGED.replace("[EOS]", "[EOT]")
    config = _instruction_config(config, [{"q": "w0 w1", "a": "w2 w3"}], use_hf_chat_template=True)
    with closing(config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)) as loader:
        batch = next(loader)
        assert batch["input_ids"][0, :4].tolist() == [2, 3, 4, 5]
        assert batch["labels"][0, :4].tolist() == [-100, 4, 5, turn_end]
        assert batch["labels"][0, 4:].eq(-100).all()
        assert list(loader) == []


def test_sft_explicit_ids_override_bracketing_without_changing_template_or_tokenizer(
    config: ZephonDataloaderConfig, tokenizer: PreTrainedTokenizerFast
) -> None:
    tokenizer.add_special_tokens({"bos_token": "[BOS]", "additional_special_tokens": ["[INSERT_BOS]", "[INSERT_EOS]"]})
    original_bos, original_eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    inserted_bos = tokenizer.convert_tokens_to_ids("[INSERT_BOS]")
    inserted_eos = tokenizer.convert_tokens_to_ids("[INSERT_EOS]")
    tokenizer.chat_template = TAGGED
    config = _instruction_config(
        config,
        [{"q": "w0 w1", "a": "w2 w3"}],
        use_hf_chat_template=True,
        special_tokens="bos_eos",
        bos_token_id=inserted_bos,
        eos_token_id=inserted_eos,
    )
    with closing(config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)) as loader:
        batch = next(loader)
        assert batch["input_ids"][0, :6].tolist() == [inserted_bos, 2, 3, 4, 5, original_eos]
        assert batch["labels"][0, :6].tolist() == [-100, -100, 4, 5, original_eos, inserted_eos]
        assert batch["labels"][0, 6:].eq(-100).all()
        assert list(loader) == []
    assert tokenizer.bos_token_id == original_bos
    assert tokenizer.eos_token_id == original_eos
    assert tokenizer.chat_template == TAGGED


def test_prompt_worker_and_pretokenized_tokens_have_identical_delivery_and_replay(config, tokenizer):
    payloads = [{"q": f"w{i} w{i + 1}", "a": f"w{i + 2} w{i + 3}"} for i in range(10)]
    config = _instruction_config(config, payloads)
    config = replace(
        config,
        zephon=replace(
            config.zephon,
            runtime=replace(config.zephon.runtime, runner="process", mtp_mode=True),
            workers=replace(config.zephon.workers, map_parallelism=2, tokenize_parallelism=2, pack_parallelism=2),
        ),
    )
    pretokenized_path = Path(config.zephon.sources[0].path).parent / "pretokenized"
    pretokenized_path.mkdir()
    (pretokenized_path / "tokens.jsonl").write_text(
        "".join(
            json.dumps({"tokens": [i + 2, i + 3, i + 4, i + 5], "loss_mask": [0, 0, 1, 1]}) + "\n" for i in range(10)
        )
    )
    pretokenized_config = replace(
        config,
        dataset_config=ZephonPretokenizedTextSourceConfig(tokens_field="tokens"),
        zephon=replace(config.zephon, sources=(replace(config.zephon.sources[0], path=str(pretokenized_path)),)),
    )
    raw = config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)
    tokens = pretokenized_config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)
    resumed = config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)
    try:
        torch.testing.assert_close(next(tokens), next(raw))
        checkpoint = raw.state_dict()
        resumed.load_state_dict(checkpoint)
        for expected in raw:
            torch.testing.assert_close(next(tokens), expected)
            torch.testing.assert_close(next(resumed), expected)
    finally:
        raw.close()
        tokens.close()
        resumed.close()


@pytest.mark.parametrize(
    "tokenizer_eos,special_tokens", [(True, None), (False, "bos_eos")], ids=["tokenizer-specials", "explicit-bos-eos"]
)
def test_plain_sft_preserves_special_tokens_without_duplicates(
    config: ZephonDataloaderConfig,
    tokenizer: PreTrainedTokenizerFast,
    tokenizer_eos: bool,
    special_tokens: str | None,
) -> None:
    tokenizer.add_special_tokens({"bos_token": "[BOS]"})
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]" if tokenizer_eos else "[BOS] $A",
        special_tokens=[("[BOS]", bos), ("[EOS]", eos)],
    )
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = tokenizer_eos
    assert tokenizer("w0 w1 w2 w3")["input_ids"] == [bos, 2, 3, 4, 5] + ([eos] if tokenizer_eos else [])
    config = _instruction_config(config, [{"q": "w0 w1", "a": "w2 w3"}], special_tokens=special_tokens)
    with closing(config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)) as loader:
        batch = next(loader)
        assert batch["input_ids"][0, :5].tolist() == [bos, 2, 3, 4, 5]
        assert batch["labels"][0, :5].tolist() == [-100, -100, 4, 5, eos]
        assert batch["labels"][0, 5:].eq(-100).all()
        assert list(loader) == []


def test_plain_sft_supervises_answer_token_merged_with_prompt_whitespace(config):
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        ["hello world", "hello ", "lead hello world"] * 10,
        trainers.BpeTrainer(vocab_size=100, special_tokens=["[UNK]", "[EOS]"]),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[EOS]"
    )
    prompt = tokenizer("hello ", add_special_tokens=False)["input_ids"]
    full = tokenizer("hello world", add_special_tokens=False)["input_ids"]
    assert len(prompt) == len(full) == 2 and prompt[-1] != full[-1]
    config = _instruction_config(config, [{"q": "hello", "a": "world"}])
    with closing(config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)) as loader:
        batch = next(loader)
        assert batch["input_ids"][0, 0].item() == full[0]
        assert batch["labels"][0, 0].item() == full[1]
        assert batch["labels"][0, 1:].eq(-100).all()


def test_instruction_mixture_uses_builtin_token_calibration(config, tokenizer, tmp_path):
    from nemo_automodel.components.datasets.zephon.config import ZephonSource

    sources = []
    for name, payload in (("short", {"q": "w0", "a": "w1"}), ("long", {"q": "w2 w2 w2", "a": "w3 " * 5})):
        path = tmp_path / name
        path.mkdir()
        (path / "data.jsonl").write_text((json.dumps(payload) + "\n") * 64)
        sources.append(ZephonSource(name=name, path=str(path), fmt="jsonl"))
    config = replace(
        config,
        batch_size=1,
        dataset_config=ZephonInstructionSourceConfig(column_mapping={"question": "q", "answer": "a"}),
        zephon=replace(
            config.zephon,
            sources=tuple(sources),
            packing=replace(config.zephon.packing, seq_len=31),
            source=replace(
                config.zephon.source,
                chunk_size=8,
                exhausted_policy="repeat",
                stop_after_passes=4,
                token_estimation={"calibration_samples": 8, "calibration_shards_min": 1, "calibration_shards_max": 1},
            ),
        ),
    )
    with closing(config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)) as loader:
        short = total = 0
        for batch in loader:
            tokens = batch["input_ids"]
            short += int((tokens.eq(2) | tokens.eq(3)).sum())
            total += tokens.numel()
        assert 0.42 < short / total < 0.58
        assert 0 < total < 5000


def test_instruction_tokenization_without_packing_keeps_answer_supervision(
    config: ZephonDataloaderConfig, tokenizer: PreTrainedTokenizerFast
) -> None:
    Path(config.zephon.sources[0].path, "text.jsonl").write_text('{"q":"w0 w1","a":"w2 w3"}\n' * 2)
    selected = replace(
        config,
        dataset_config=ZephonInstructionSourceConfig(column_mapping={"question": "q", "answer": "a"}),
        zephon=replace(
            config.zephon,
            packing=replace(config.zephon.packing, pack_algorithm=None),
            source=replace(config.zephon.source, chunk_size=1),
        ),
    )
    with closing(selected.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)) as loader:
        batch = next(loader)
        assert batch["input_ids"].tolist() == [[2, 3, 4, 5, 1, 1, 1]] * 2
        assert batch["labels"].tolist() == [[-100, 4, 5, -100, -100, -100, -100]] * 2
        assert batch["_packed_seq_ids"].tolist() == [[1, 1, 1, 1, 0, 0, 0]] * 2
