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

"""Recipe-side EAGLE-4 wiring: ``recipe_args.eagle4`` parsing, gates and the candidate-head warm start."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig

import nemo_automodel.recipes.llm.train_eagle3 as recipe_module
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.speculative.eagle.core import Eagle3TrainerModule
from nemo_automodel.components.speculative.eagle.draft_gpt_oss import GptOssEagle3DraftModel
from nemo_automodel.components.speculative.eagle.draft_llama import LlamaEagle3DraftModel
from nemo_automodel.components.speculative.eagle.eagle4 import (
    CANDIDATE_HEAD_KEYS,
    Eagle4CandidateHeadConfig,
    Eagle4CandidateHeadTrainerModule,
    Eagle4Config,
)
from nemo_automodel.recipes.llm.train_eagle3 import (
    _load_draft_weights,
    _parse_eagle4_config,
    _prepare_candidate_head_training,
    _validate_eagle4_gates,
)

STAGE1 = Eagle4Config(sliding_window=4096)
STAGE2 = Eagle4Config(sliding_window=4096, candidate_head=Eagle4CandidateHeadConfig(rank=4, num_candidates=8))


def _draft(eagle4: Eagle4Config) -> LlamaEagle3DraftModel:
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=64,
    )
    config.draft_vocab_size = 64
    config.update(eagle4.draft_config_updates(aux_layer_ids=[1, 2, 3]))
    torch.manual_seed(0)
    return LlamaEagle3DraftModel(config).to(torch.float32)


def test_parse_eagle4_config():
    assert _parse_eagle4_config(None) is None
    assert _parse_eagle4_config(ConfigNode({"sliding_window": 4096})) == STAGE1
    parsed = _parse_eagle4_config({"sliding_window": 4096, "candidate_head": {"rank": 4, "num_candidates": 8}})
    assert parsed == STAGE2
    with pytest.raises(TypeError):
        _parse_eagle4_config({"sliding_windw": 4096})


def _gate_kwargs(**overrides):
    kwargs = dict(
        draft_cls=LlamaEagle3DraftModel,
        parallel_drafting=False,
        cp_size=1,
        draft_vocab_size=None,
        lk_loss_type=None,
        has_peft=False,
        has_decode_eval=False,
        draft_weights_path=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_gates_accept_both_stages():
    _validate_eagle4_gates(STAGE1, **_gate_kwargs(lk_loss_type="alpha"))
    _validate_eagle4_gates(STAGE2, **_gate_kwargs(draft_weights_path="/stage1"))


@pytest.mark.parametrize(
    "eagle4, overrides, error, match",
    [
        (STAGE1, {"draft_cls": GptOssEagle3DraftModel}, ValueError, "GptOssEagle3DraftModel"),
        (STAGE1, {"parallel_drafting": True}, ValueError, "parallel_drafting"),
        (STAGE1, {"cp_size": 2}, NotImplementedError, "context parallelism"),
        (STAGE1, {"draft_vocab_size": 32000}, ValueError, "full target vocabulary"),
        (STAGE1, {"has_decode_eval": True}, ValueError, "decode_eval"),
        (STAGE2, {}, ValueError, "draft_weights_path"),
        (STAGE2, {"draft_weights_path": "/stage1", "has_peft": True}, ValueError, "peft"),
        (STAGE2, {"draft_weights_path": "/stage1", "lk_loss_type": "alpha"}, ValueError, "lk_loss_type"),
    ],
)
def test_gates_reject_unsupported_combinations(eagle4, overrides, error, match):
    with pytest.raises(error, match=match):
        _validate_eagle4_gates(eagle4, **_gate_kwargs(**overrides))


def _save(state_dict, tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({k: v.contiguous() for k, v in state_dict.items()}, str(path))
    return str(tmp_path)


def test_stage1_checkpoint_warm_starts_the_candidate_head_stage(tmp_path):
    """A stage-1 draft lacks the factors: they come from the SVD, and only they train."""
    stage1 = _draft(STAGE1)
    stage2 = _draft(STAGE2)
    missing = _load_draft_weights(stage2, _save(stage1.state_dict(), tmp_path), optional_keys=CANDIDATE_HEAD_KEYS)
    assert missing == CANDIDATE_HEAD_KEYS
    torch.testing.assert_close(stage2.lm_head.weight, stage1.lm_head.weight)

    _prepare_candidate_head_training(stage2, missing)
    u, s, vh = torch.linalg.svd(stage2.lm_head.weight.double(), full_matrices=False)
    product = stage2.efficient_lm_head.weight @ stage2.efficient_lm_head_down_proj.weight
    torch.testing.assert_close(product.double(), (u[:, :4] * s[:4]) @ vh[:4], atol=1e-4, rtol=1e-4)
    assert {n for n, p in stage2.named_parameters() if p.requires_grad} == CANDIDATE_HEAD_KEYS


def test_stage2_checkpoint_keeps_its_trained_factors(tmp_path):
    trained = _draft(STAGE2)
    with torch.no_grad():
        trained.efficient_lm_head.weight.fill_(0.5)
    fresh = _draft(STAGE2)
    missing = _load_draft_weights(fresh, _save(trained.state_dict(), tmp_path), optional_keys=CANDIDATE_HEAD_KEYS)
    assert missing == set()
    _prepare_candidate_head_training(fresh, missing)
    torch.testing.assert_close(fresh.efficient_lm_head.weight, trained.efficient_lm_head.weight)


def test_partial_candidate_head_is_rejected():
    with pytest.raises(ValueError, match="only part of the candidate head"):
        _prepare_candidate_head_training(_draft(STAGE2), {"efficient_lm_head.weight"})


def test_missing_non_optional_keys_still_fail(tmp_path):
    state = _draft(STAGE1).state_dict()
    state.pop("model.fc.weight")
    with pytest.raises(ValueError, match="missing 1 draft keys"):
        _load_draft_weights(_draft(STAGE2), _save(state, tmp_path), optional_keys=CANDIDATE_HEAD_KEYS)
    # Without optional keys, the absent candidate head is an error too.
    with pytest.raises(ValueError, match="missing 2 draft keys"):
        _load_draft_weights(_draft(STAGE2), _save(_draft(STAGE1).state_dict(), tmp_path))


# --- TrainEagle3Recipe.setup() with recipe_args.eagle4 ----------------------

TARGET_VOCAB, TARGET_HIDDEN = 128, 32


class _TrainerBuilt(Exception):
    """Raised once ``setup()`` has built the trainer module; carries it and skips the data setup."""

    def __init__(self, trainer):
        super().__init__("trainer built")
        self.trainer = trainer


def _run_setup_to_trainer(recipe_overrides: dict, *, selected_vocab: int = TARGET_VOCAB):
    """Drive ``TrainEagle3Recipe.setup()`` with stub target/tokenizer up to the trainer build."""
    target_config = LlamaConfig(
        hidden_size=TARGET_HIDDEN,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=TARGET_VOCAB,
        max_position_embeddings=64,
    )

    def fake_online_target(self, recipe_cfg, target_path, draft_base_config):
        self.target_wrapper = SimpleNamespace(
            get_input_embeddings=lambda: torch.nn.Embedding(TARGET_VOCAB, TARGET_HIDDEN)
        )
        return torch.arange(selected_vocab), torch.ones(TARGET_VOCAB, dtype=torch.bool)

    def stop(modules, mesh):
        raise _TrainerBuilt(modules[0])

    cfg = MagicMock()
    cfg.get = MagicMock(side_effect=lambda key, default=None: default)
    cfg.recipe_args = SimpleNamespace(target_model_name_or_path="ignored/path", ttt_steps=2)
    cfg.recipe_args.get = lambda key, default=None: recipe_overrides.get(key, default)
    dist_env = SimpleNamespace(device=torch.device("cpu"), world_size=1, is_main=True, rank=0, local_rank=0)
    recipe = recipe_module.TrainEagle3Recipe(cfg)
    with (
        patch.object(recipe_module, "initialize_distributed", return_value=dist_env),
        patch.object(recipe_module, "setup_logging"),
        patch.object(recipe_module, "AutoConfig") as auto_config,
        patch.object(recipe_module, "NeMoAutoTokenizer") as tokenizer,
        patch.object(
            recipe_module, "resolve_eagle3_draft_spec", return_value=SimpleNamespace(draft_cls=LlamaEagle3DraftModel)
        ),
        patch.object(
            recipe_module.TrainEagle3Recipe, "_setup_online_target", autospec=True, side_effect=fake_online_target
        ),
        patch.object(recipe_module, "broadcast_tp_replicas", side_effect=stop),
    ):
        auto_config.from_pretrained.return_value = target_config
        tokenizer.from_pretrained.return_value = MagicMock()
        with pytest.raises(_TrainerBuilt) as excinfo:
            recipe.setup()
    return excinfo.value.trainer


def test_setup_trains_both_eagle4_stages(tmp_path):
    stage1 = _run_setup_to_trainer({"eagle4": {"sliding_window": 16}, "fc_norm": True, "norm_output": True})
    assert isinstance(stage1, Eagle3TrainerModule)
    config = stage1.draft_model.config
    assert config.architectures == ["LlamaForCausalLMEagle4"]
    assert config.num_hidden_layers == 1 and config.layer_types == ["sliding_attention"]
    assert config.sliding_window == 16 and config.draft_vocab_size == TARGET_VOCAB
    # The default EAGLE-3 aux layers of an 8-layer target.
    assert config.eagle_config["eagle_aux_hidden_state_layer_ids"] == [1, 3, 4]
    assert stage1.draft_model.model.layers[0].self_attn.sliding_window == 16

    save_file(
        {k: v.contiguous() for k, v in stage1.draft_model.state_dict().items()}, str(tmp_path / "model.safetensors")
    )
    stage2 = _run_setup_to_trainer(
        {
            "eagle4": {"sliding_window": 16, "candidate_head": {"rank": 4, "num_candidates": 8}},
            "fc_norm": True,
            "norm_output": True,
            "draft_weights_path": str(tmp_path),
            "aux_layer_ids": [2, 4, 6],
        }
    )
    assert isinstance(stage2, Eagle4CandidateHeadTrainerModule)
    draft = stage2.draft_model
    assert draft.config.eagle_config["eagle_aux_hidden_state_layer_ids"] == [2, 4, 6]
    assert {n for n, p in draft.named_parameters() if p.requires_grad} == CANDIDATE_HEAD_KEYS
    torch.testing.assert_close(draft.lm_head.weight, stage1.draft_model.lm_head.weight)


def test_setup_rejects_a_compressed_draft_vocabulary():
    with pytest.raises(ValueError, match="full target vocabulary"):
        _run_setup_to_trainer({"eagle4": {"sliding_window": 16}}, selected_vocab=16)


def test_setup_gates_eagle4_before_loading_the_target():
    with pytest.raises(ValueError, match="draft_weights_path"):
        _run_setup_to_trainer({"eagle4": {"candidate_head": {"rank": 4}}})
