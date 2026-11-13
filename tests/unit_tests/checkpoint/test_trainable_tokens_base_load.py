# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Gated-MTP's trainable mask-token embedding must be patched in *after* the base checkpoint
load, not during PEFT application like LoRA's in-place Linear patching: TrainableTokensWrapper
wraps the embedding (moving its weight to a nested `token_adapter.base_layer.weight` FQN), so
patching it before the base HF checkpoint loads would make the loader miss the checkpoint's
plain `embed_tokens.weight` key entirely."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch.distributed.checkpoint.api import CheckpointException

from nemo_automodel._transformers.infrastructure import _apply_trainability_policy
from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules, patch_embedding_module
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.shared.peft_config import PeftConfig

pytestmark = pytest.mark.cpu_only


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 4)
        self.proj = nn.Linear(4, 4, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


def _peft_checkpointer(tmp_path: Path) -> Checkpointer:
    return Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_cache_dir=str(tmp_path),
            model_repo_id="source",
            model_save_format="safetensors",
            save_consolidated=False,
            is_peft=True,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )


def test_base_checkpoint_loads_before_embedding_is_wrapped(tmp_path: Path) -> None:
    """Mirrors the real call order in apply_model_infrastructure: LoRA patch (in-place, same
    FQNs) -> base checkpoint load -> embedding patch (wraps, new FQNs) -> freeze."""
    torch.manual_seed(0)
    pretrained = {key: value.clone() for key, value in _Tiny().state_dict().items()}
    source = tmp_path / "source"
    source.mkdir()
    save_file(pretrained, source / "model.safetensors")

    model = _Tiny()
    peft_config = PeftConfig(target_modules=["proj"], dim=2, alpha=4, trainable_token_indices=[2, 5])
    apply_lora_to_linear_modules(model, peft_config, skip_freeze=True)

    # Base load must still find `embed_tokens.weight`: the embedding is not wrapped yet.
    _peft_checkpointer(tmp_path).load_base_model(model, torch.device("cpu"), str(tmp_path), str(source))
    torch.testing.assert_close(model.embed_tokens.weight, pretrained["embed_tokens.weight"])

    # Only after the base load does the embedding get wrapped, from the just-loaded values.
    wrapped = patch_embedding_module(model.get_input_embeddings(), list(peft_config.trainable_token_indices))
    model.set_input_embeddings(wrapped)

    torch.testing.assert_close(wrapped.token_adapter.base_layer.weight, pretrained["embed_tokens.weight"])
    torch.testing.assert_close(
        wrapped.token_adapter.trainable_tokens_delta["default"], pretrained["embed_tokens.weight"][[2, 5]]
    )

    # The post-load global freeze pass must keep the delta trainable alongside the LoRA weights,
    # not just params whose name contains "lora_".
    _apply_trainability_policy(model, peft_enabled=True, freeze_config=None, strict=False)
    assert wrapped.token_adapter.trainable_tokens_delta["default"].requires_grad
    assert wrapped.token_adapter.base_layer.weight.requires_grad is False
    assert model.proj.lora_A.weight.requires_grad


def test_wrapping_the_embedding_before_base_load_breaks_the_checkpoint_key(tmp_path: Path) -> None:
    """Regression guard: confirms *why* the ordering above matters, not just that it works."""
    torch.manual_seed(0)
    pretrained = {key: value.clone() for key, value in _Tiny().state_dict().items()}
    source = tmp_path / "source"
    source.mkdir()
    save_file(pretrained, source / "model.safetensors")

    model = _Tiny()
    peft_config = PeftConfig(target_modules=["proj"], dim=2, alpha=4, trainable_token_indices=[2, 5])
    apply_lora_to_linear_modules(model, peft_config, skip_freeze=True)
    wrapped = patch_embedding_module(model.get_input_embeddings(), list(peft_config.trainable_token_indices))
    model.set_input_embeddings(wrapped)

    with pytest.raises(CheckpointException, match="embed_tokens.token_adapter.base_layer.weight"):
        _peft_checkpointer(tmp_path).load_base_model(model, torch.device("cpu"), str(tmp_path), str(source))
