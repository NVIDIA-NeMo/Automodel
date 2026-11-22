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

"""A saved tokenizer does not carry the truncation and padding of the last training call."""

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from transformers import PreTrainedTokenizerFast

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.addons import ConsolidatedHFAddon, PeftAddon, _save_generated_hf_assets
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState

pytestmark = pytest.mark.cpu_only

_WORDS = [f"w{index}" for index in range(20)]
_TEXT = " ".join(_WORDS)
_SHORT_TEXT = "w0 w1 w2"
_MAX_LENGTH = 8

SaveFn = Callable[[PreTrainedTokenizerFast, Path, str | None], None]


def _tokenizer() -> PreTrainedTokenizerFast:
    """Build a word-level fast tokenizer without any file or download."""
    vocab = {"[PAD]": 0, "[UNK]": 1, **{word: index + 2 for index, word in enumerate(_WORDS)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")


class _Model(nn.Module):
    """A model that neither has a metadata exporter hook nor contains a module with one."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.query = nn.Linear(4, 4)


class _GeneratedAssetsExporter:
    """Writes the metadata with ``_save_generated_hf_assets``, the call the retrieval metadata exporters make."""

    def __init__(self, model_part: nn.Module) -> None:
        self.model_part = model_part

    def validate(self, *, tokenizer: PreTrainedTokenizerFast, original_model_path: str | None) -> None:
        del tokenizer, original_model_path

    def save(
        self,
        *,
        hf_metadata_dir: str,
        tokenizer: PreTrainedTokenizerFast,
        original_model_path: str | None,
        v4_compatible: bool,
    ) -> None:
        _save_generated_hf_assets(
            self.model_part,
            original_model_path,
            hf_metadata_dir,
            tokenizer,
            v4_compatible=v4_compatible,
            save_custom_model_code=False,
        )


class _ModelWithExporterHook(_Model):
    """Stands in for a model with a metadata exporter hook, such as the retrieval encoders."""

    def __init__(self, hook_returns_exporter: bool) -> None:
        super().__init__()
        self.hook_returns_exporter = hook_returns_exporter

    def _get_consolidated_hf_metadata_exporter(
        self, *, tokenizer: PreTrainedTokenizerFast, original_model_path: str | None
    ) -> _GeneratedAssetsExporter | None:
        del tokenizer, original_model_path
        return _GeneratedAssetsExporter(self) if self.hook_returns_exporter else None


class _WrapperOfModelWithExporterHook(nn.Module):
    """Stands in for a model built around one with the hook, as the retrieval distillation student is."""

    def __init__(self) -> None:
        super().__init__()
        self.student = _ModelWithExporterHook(False)


def _under_ddp(model: nn.Module) -> DistributedDataParallel:
    """Put ``model`` inside a DDP wrapper without starting a process group."""
    wrapped = object.__new__(DistributedDataParallel)
    nn.Module.__init__(wrapped)
    wrapped.module = model
    return wrapped


def _pre_save_full_model(
    model: nn.Module, tokenizer: PreTrainedTokenizerFast, directory: Path, original_model_path: str | None
) -> None:
    """Write the metadata of a full-model checkpoint, as ``Checkpointer.save_model`` does before the weights."""
    ConsolidatedHFAddon().pre_save(
        model_state=SimpleNamespace(model=[model]),
        hf_metadata_dir=str(directory),
        tokenizer=tokenizer,
        fqn_to_file_index_mapping={"encoder.query.weight": 1, "encoder.query.bias": 1},
        fqn_to_dtype_mapping=None,
        original_model_path=original_model_path,
        v4_compatible=False,
    )


def _pre_save_peft_adapter(
    model: nn.Module, tokenizer: PreTrainedTokenizerFast, directory: Path, original_model_path: str | None
) -> None:
    """Write the metadata of a PEFT checkpoint, as ``Checkpointer.save_model`` does before the adapter."""
    peft_config = PeftConfig(target_modules=["*.query"], dim=2, alpha=4, use_triton=False)
    assert apply_lora_to_linear_modules(model, peft_config) == 1
    PeftAddon().pre_save(
        model_path=str(directory),
        tokenizer=tokenizer,
        model_state=ModelState(model, is_peft=True),
        peft_config=peft_config,
        original_model_path=original_model_path,
    )


def _save_with_full_model(tokenizer: PreTrainedTokenizerFast, directory: Path, original_model_path: str | None) -> None:
    _pre_save_full_model(_Model(), tokenizer, directory, original_model_path)


def _save_with_peft_adapter(
    tokenizer: PreTrainedTokenizerFast, directory: Path, original_model_path: str | None
) -> None:
    _pre_save_peft_adapter(_Model(), tokenizer, directory, original_model_path)


def _save_through_exporter(
    tokenizer: PreTrainedTokenizerFast, directory: Path, original_model_path: str | None
) -> None:
    _pre_save_full_model(_ModelWithExporterHook(True), tokenizer, directory, original_model_path)


def _save_with_declined_hook(
    tokenizer: PreTrainedTokenizerFast, directory: Path, original_model_path: str | None
) -> None:
    _pre_save_full_model(_ModelWithExporterHook(False), tokenizer, directory, original_model_path)


def _save_with_peft_adapter_of_model_with_hook(
    tokenizer: PreTrainedTokenizerFast, directory: Path, original_model_path: str | None
) -> None:
    _pre_save_peft_adapter(_ModelWithExporterHook(False), tokenizer, directory, original_model_path)


def _save_with_full_model_under_ddp(
    tokenizer: PreTrainedTokenizerFast, directory: Path, original_model_path: str | None
) -> None:
    _pre_save_full_model(_under_ddp(_Model()), tokenizer, directory, original_model_path)


def _save_wrapper_of_model_with_hook(
    tokenizer: PreTrainedTokenizerFast, directory: Path, original_model_path: str | None
) -> None:
    _pre_save_full_model(_WrapperOfModelWithExporterHook(), tokenizer, directory, original_model_path)


def _save_with_peft_adapter_of_wrapper_of_model_with_hook(
    tokenizer: PreTrainedTokenizerFast, directory: Path, original_model_path: str | None
) -> None:
    _pre_save_peft_adapter(_WrapperOfModelWithExporterHook(), tokenizer, directory, original_model_path)


def _save_wrapper_of_model_with_hook_under_ddp(
    tokenizer: PreTrainedTokenizerFast, directory: Path, original_model_path: str | None
) -> None:
    _pre_save_full_model(_under_ddp(_WrapperOfModelWithExporterHook()), tokenizer, directory, original_model_path)


_SAVE_SITES = pytest.mark.parametrize(
    "save",
    [_save_with_full_model, _save_with_peft_adapter, _save_with_full_model_under_ddp],
    ids=["full_model", "peft_adapter", "full_model_under_ddp"],
)
_SAVES_LEFT_ALONE = pytest.mark.parametrize(
    "save",
    [
        _save_through_exporter,
        _save_with_declined_hook,
        _save_with_peft_adapter_of_model_with_hook,
        _save_wrapper_of_model_with_hook,
        _save_with_peft_adapter_of_wrapper_of_model_with_hook,
        _save_wrapper_of_model_with_hook_under_ddp,
    ],
    ids=[
        "exporter",
        "declined_hook",
        "peft_adapter_of_model_with_hook",
        "wrapper_of_model_with_hook",
        "peft_adapter_of_wrapper_of_model_with_hook",
        "wrapper_of_model_with_hook_under_ddp",
    ],
)


def _call_like_training(tokenizer: PreTrainedTokenizerFast) -> dict[str, list[list[int]]]:
    """Tokenize the way a dataset with ``truncation`` and ``padding: max_length`` does."""
    encoded = tokenizer([_TEXT, _SHORT_TEXT], truncation=True, max_length=_MAX_LENGTH, padding="max_length")
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


def _saved_state(directory: Path) -> tuple[dict | None, dict | None]:
    saved = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
    return saved["truncation"], saved["padding"]


@_SAVE_SITES
def test_saved_tokenizer_has_no_truncation_or_padding_of_the_last_call(tmp_path: Path, save: SaveFn) -> None:
    tokenizer = _tokenizer()
    _call_like_training(tokenizer)
    assert tokenizer.backend_tokenizer.truncation is not None
    assert tokenizer.backend_tokenizer.padding is not None
    directory = tmp_path / "saved"
    directory.mkdir()

    save(tokenizer, directory, None)

    assert _saved_state(directory) == (None, None)
    # A consumer that reads the file with the tokenizers library gets every token of its input.
    assert len(Tokenizer.from_file(str(directory / "tokenizer.json")).encode(_TEXT).ids) == len(_WORDS)


def _source_snapshot_with_state(tmp_path: Path) -> tuple[Path, tuple[dict | None, dict | None]]:
    """Write a source snapshot whose ``tokenizer.json`` ships a truncation and a padding entry."""
    shipped = _tokenizer()
    shipped.backend_tokenizer.enable_truncation(max_length=16)
    shipped.backend_tokenizer.enable_padding(length=16)
    source = tmp_path / "source"
    shipped.save_pretrained(source)
    source_state = _saved_state(source)
    assert None not in source_state
    return source, source_state


@_SAVE_SITES
def test_saved_tokenizer_keeps_the_state_of_the_source_snapshot(tmp_path: Path, save: SaveFn) -> None:
    source, source_state = _source_snapshot_with_state(tmp_path)
    tokenizer = _tokenizer()
    _call_like_training(tokenizer)
    directory = tmp_path / "saved"
    directory.mkdir()

    save(tokenizer, directory, str(source))

    assert _saved_state(directory) == source_state


@_SAVE_SITES
def test_saved_tokenizer_keeps_the_state_of_its_own_snapshot_not_the_model_snapshot(
    tmp_path: Path, save: SaveFn
) -> None:
    """The tokenizer comes from another snapshot than the weights, as with a separate ``dataset.tokenizer``."""
    tokenizer_snapshot, tokenizer_state = _source_snapshot_with_state(tmp_path)
    model_snapshot = tmp_path / "model_snapshot"
    other = _tokenizer()
    other.backend_tokenizer.enable_truncation(max_length=3)
    other.backend_tokenizer.enable_padding(length=3, pad_id=1)
    other.save_pretrained(model_snapshot)
    assert _saved_state(model_snapshot) != tokenizer_state
    tokenizer = _tokenizer()
    tokenizer.name_or_path = str(tokenizer_snapshot)
    _call_like_training(tokenizer)
    directory = tmp_path / "saved"
    directory.mkdir()

    save(tokenizer, directory, str(model_snapshot))

    assert _saved_state(directory) == tokenizer_state


@_SAVE_SITES
def test_saved_tokenizer_gets_the_state_of_the_source_when_the_last_call_left_none(
    tmp_path: Path, save: SaveFn
) -> None:
    source, source_state = _source_snapshot_with_state(tmp_path)
    tokenizer = _tokenizer()
    tokenizer(_TEXT)
    assert tokenizer.backend_tokenizer.truncation is None
    assert tokenizer.backend_tokenizer.padding is None
    directory = tmp_path / "saved"
    directory.mkdir()

    save(tokenizer, directory, str(source))

    assert _saved_state(directory) == source_state


@_SAVE_SITES
def test_live_tokenizer_truncates_and_pads_as_before_after_a_save(tmp_path: Path, save: SaveFn) -> None:
    tokenizer = _tokenizer()
    before = _call_like_training(tokenizer)
    assert [len(ids) for ids in before["input_ids"]] == [_MAX_LENGTH, _MAX_LENGTH]
    truncation, padding = tokenizer.backend_tokenizer.truncation, tokenizer.backend_tokenizer.padding
    backend_ids = tokenizer.backend_tokenizer.encode(_TEXT).ids
    directory = tmp_path / "saved"
    directory.mkdir()

    save(tokenizer, directory, None)

    assert tokenizer.backend_tokenizer.truncation == truncation
    assert tokenizer.backend_tokenizer.padding == padding
    assert tokenizer.backend_tokenizer.encode(_TEXT).ids == backend_ids
    assert _call_like_training(tokenizer) == before


@_SAVE_SITES
def test_tokenizer_without_such_state_is_saved_unchanged(tmp_path: Path, save: SaveFn) -> None:
    tokenizer = _tokenizer()
    tokenizer(_TEXT)
    assert tokenizer.backend_tokenizer.truncation is None
    assert tokenizer.backend_tokenizer.padding is None
    reference = tmp_path / "reference"
    tokenizer.save_pretrained(reference)
    directory = tmp_path / "saved"
    directory.mkdir()

    save(tokenizer, directory, None)

    for name in ("tokenizer.json", "tokenizer_config.json"):
        assert (directory / name).read_bytes() == (reference / name).read_bytes()


@_SAVES_LEFT_ALONE
def test_models_with_or_around_a_metadata_exporter_hook_are_saved_as_before(tmp_path: Path, save: SaveFn) -> None:
    """A model that has the hook, or holds a module that has it, keeps what ``save_pretrained`` writes."""
    source, _ = _source_snapshot_with_state(tmp_path)
    tokenizer = _tokenizer()
    _call_like_training(tokenizer)
    reference = tmp_path / "reference"
    tokenizer.save_pretrained(reference)
    directory = tmp_path / "saved"
    directory.mkdir()

    save(tokenizer, directory, str(source))

    for name in ("tokenizer.json", "tokenizer_config.json"):
        assert (directory / name).read_bytes() == (reference / name).read_bytes()
