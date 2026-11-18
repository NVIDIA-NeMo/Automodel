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

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig

import nemo_automodel.recipes.speculative_decoding.train_eagle3 as recipe_module
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
from nemo_automodel.components.speculative.eagle.registry import DraftSpec
from nemo_automodel.components.speculative.eagle.remote import RemoteEagle3TargetModel
from nemo_automodel.recipes.speculative_decoding._spec_recipe_base import SpecDecodeRecipeBase
from nemo_automodel.recipes.speculative_decoding.train_eagle3 import (
    _check_candidate_head_warm_start,
    _load_draft_weights,
    _parse_eagle4_config,
    _prepare_candidate_head_training,
    _resolve_freeze_embeddings,
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


@pytest.mark.parametrize(
    "recipe_args, parallel_drafting, expected",
    [
        ({}, False, True),
        ({}, True, False),
        ({"freeze_embeddings": None}, False, True),
        ({"freeze_embeddings": None}, True, False),
        ({"freeze_embeddings": False}, False, False),
        ({"freeze_embeddings": True}, True, True),
    ],
)
def test_resolve_freeze_embeddings_treats_null_as_the_default(recipe_args, parallel_drafting, expected):
    assert _resolve_freeze_embeddings(ConfigNode(recipe_args), parallel_drafting) is expected


def _gate_kwargs(**overrides):
    kwargs = dict(
        draft_spec=DraftSpec(draft_cls=LlamaEagle3DraftModel, eagle4_draft_config=lambda config: config.to_dict()),
        parallel_drafting=False,
        cp_size=1,
        draft_vocab_size=None,
        lk_loss_type=None,
        has_peft=False,
        has_decode_eval=False,
        draft_weights_path=None,
        ttt_steps=7,
        freeze_embeddings=True,
    )
    kwargs.update(overrides)
    return kwargs


def test_gates_accept_both_stages():
    _validate_eagle4_gates(STAGE1, **_gate_kwargs(lk_loss_type="alpha"))
    _validate_eagle4_gates(STAGE2, **_gate_kwargs(draft_weights_path="/stage1"))
    # The smallest window that still leaves the deepest TTT step one context key.
    _validate_eagle4_gates(Eagle4Config(sliding_window=7), **_gate_kwargs())


@pytest.mark.parametrize(
    "eagle4, overrides, error, match",
    [
        (STAGE1, {"draft_spec": DraftSpec(draft_cls=GptOssEagle3DraftModel)}, ValueError, "GptOssEagle3DraftModel"),
        (STAGE1, {"parallel_drafting": True}, ValueError, "parallel_drafting"),
        (STAGE1, {"cp_size": 2}, NotImplementedError, "context parallelism"),
        (STAGE1, {"draft_vocab_size": 32000}, ValueError, "full target vocabulary"),
        (STAGE1, {"has_decode_eval": True}, ValueError, "decode_eval"),
        (Eagle4Config(sliding_window=6), {}, ValueError, "at least ttt_steps"),
        (STAGE1, {"freeze_embeddings": False}, ValueError, "freeze_embeddings"),
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


EMBED = "model.embed_tokens.weight"


def test_target_copied_keys_tolerate_a_lower_precision_checkpoint(tmp_path):
    """Untrained embeddings saved by a bf16 stage still match the fp32 copy this run seeded."""
    state = _draft(STAGE1).state_dict()
    state[EMBED] = state[EMBED].to(torch.bfloat16)
    draft = _draft(STAGE1)
    _load_draft_weights(draft, _save(state, tmp_path), target_copied_keys=frozenset({EMBED}))
    torch.testing.assert_close(draft.model.embed_tokens.weight, state[EMBED].float())


def test_target_copied_keys_reject_trained_values(tmp_path):
    state = _draft(STAGE1).state_dict()
    state[EMBED] = state[EMBED] + 1e-3
    with pytest.raises(ValueError, match="trained model.embed_tokens.weight"):
        _load_draft_weights(_draft(STAGE1), _save(state, tmp_path), target_copied_keys=frozenset({EMBED}))


def test_target_copied_keys_leave_a_vocabulary_mismatch_to_load_state_dict(tmp_path):
    state = _draft(STAGE1).state_dict()
    state[EMBED] = torch.zeros(32, state[EMBED].shape[1])
    with pytest.raises(RuntimeError, match="size mismatch"):
        _load_draft_weights(_draft(STAGE1), _save(state, tmp_path), target_copied_keys=frozenset({EMBED}))


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
# One target embedding for every setup() run, as a real target gives both stages.
TARGET_EMBEDDING = torch.randn(TARGET_VOCAB, TARGET_HIDDEN, generator=torch.Generator().manual_seed(0))


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
            get_input_embeddings=lambda: torch.nn.Embedding.from_pretrained(TARGET_EMBEDDING.clone())
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
            recipe_module,
            "resolve_eagle3_draft_spec",
            return_value=DraftSpec(
                draft_cls=LlamaEagle3DraftModel, eagle4_draft_config=lambda config: config.to_dict()
            ),
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


STAGE1_ARGS = {"eagle4": {"sliding_window": 16}, "fc_norm": True, "norm_output": True}
STAGE2_ARGS = {**STAGE1_ARGS, "eagle4": {"sliding_window": 16, "candidate_head": {"rank": 4, "num_candidates": 8}}}


def _save_stage1(draft, tmp_path):
    """Write ``draft`` as a stage-1 consolidated checkpoint (weights plus config.json)."""
    (tmp_path / "config.json").write_text(draft.config.to_json_string(use_diff=False))
    return _save(draft.state_dict(), tmp_path)


def test_setup_trains_both_eagle4_stages(tmp_path):
    stage1 = _run_setup_to_trainer(STAGE1_ARGS)
    assert isinstance(stage1, Eagle3TrainerModule)
    config = stage1.draft_model.config
    assert isinstance(config, LlamaConfig) and config.hidden_act == "silu"
    assert config.architectures == ["LlamaForCausalLMEagle4"]
    assert config.num_hidden_layers == 1 and config.layer_types == ["sliding_attention"]
    assert config.sliding_window == 16 and config.draft_vocab_size == TARGET_VOCAB
    # The default EAGLE-3 aux layers of an 8-layer target.
    assert config.eagle_config["eagle_aux_hidden_state_layer_ids"] == [1, 3, 4]
    assert stage1.draft_model.model.layers[0].self_attn.sliding_window == 16
    assert not stage1.draft_model.model.embed_tokens.weight.requires_grad

    stage2 = _run_setup_to_trainer({**STAGE2_ARGS, "draft_weights_path": _save_stage1(stage1.draft_model, tmp_path)})
    assert isinstance(stage2, Eagle4CandidateHeadTrainerModule)
    draft = stage2.draft_model
    assert draft.config.eagle_config["eagle_aux_hidden_state_layer_ids"] == [1, 3, 4]
    assert {n for n, p in draft.named_parameters() if p.requires_grad} == CANDIDATE_HEAD_KEYS
    torch.testing.assert_close(draft.lm_head.weight, stage1.draft_model.lm_head.weight)


@pytest.mark.parametrize("aux_layer_ids", [[4, 3, 1], [2, 4, 6]])
def test_setup_rejects_candidate_head_stage_fed_other_aux_layers(tmp_path, aux_layer_ids):
    """The frozen stage-1 fc expects its aux layers in their order; reordered or other layers change the teacher."""
    stage1_path = _save_stage1(_run_setup_to_trainer(STAGE1_ARGS).draft_model, tmp_path)
    with pytest.raises(ValueError, match="aux_layer_ids"):
        _run_setup_to_trainer({**STAGE2_ARGS, "draft_weights_path": stage1_path, "aux_layer_ids": aux_layer_ids})


@pytest.mark.parametrize("stage_args", [STAGE1_ARGS, STAGE2_ARGS])
def test_setup_rejects_eagle4_warm_start_with_trained_embeddings(tmp_path, stage_args):
    """The serving adapters replace the draft embeddings with the target's, so a warm start must not train them."""
    stage1 = _run_setup_to_trainer(STAGE1_ARGS).draft_model
    with torch.no_grad():
        stage1.model.embed_tokens.weight[0] += 1.0
    with pytest.raises(ValueError, match="trained model.embed_tokens.weight"):
        _run_setup_to_trainer({**stage_args, "draft_weights_path": _save_stage1(stage1, tmp_path)})


def test_setup_freezes_eagle4_embeddings_under_an_explicit_null():
    """An explicit null means the default like a missing key, so stage 1 cannot train what stage 2 rejects."""
    draft = _run_setup_to_trainer({**STAGE1_ARGS, "freeze_embeddings": None}).draft_model
    assert not draft.model.embed_tokens.weight.requires_grad


def test_setup_rejects_eagle4_trainable_embeddings():
    with pytest.raises(ValueError, match="freeze_embeddings"):
        _run_setup_to_trainer({**STAGE1_ARGS, "freeze_embeddings": False})


def test_setup_rejects_a_compressed_draft_vocabulary():
    with pytest.raises(ValueError, match="full target vocabulary"):
        _run_setup_to_trainer({"eagle4": {"sliding_window": 16}}, selected_vocab=16)


def test_setup_gates_eagle4_before_loading_the_target():
    with pytest.raises(ValueError, match="draft_weights_path"):
        _run_setup_to_trainer({"eagle4": {"candidate_head": {"rank": 4}}})


def test_candidate_head_stage_must_keep_the_stage1_structure(tmp_path):
    _save_stage1(_draft(STAGE1), tmp_path)
    _check_candidate_head_warm_start(_draft(STAGE2).config, str(tmp_path))
    _check_candidate_head_warm_start(_draft(STAGE2).config, str(tmp_path / "model.safetensors"))  # file path

    other_window = Eagle4Config(sliding_window=2048, candidate_head=STAGE2.candidate_head)
    with pytest.raises(ValueError, match="sliding_window"):
        _check_candidate_head_warm_start(_draft(other_window).config, str(tmp_path))
    no_norm_output = _draft(STAGE2).config
    no_norm_output.norm_output = True
    with pytest.raises(ValueError, match="norm_output"):
        _check_candidate_head_warm_start(no_norm_output, str(tmp_path))
    reordered_aux = _draft(STAGE2).config
    reordered_aux.eagle_config = {**reordered_aux.eagle_config, "eagle_aux_hidden_state_layer_ids": [3, 2, 1]}
    with pytest.raises(ValueError, match="aux_layer_ids"):
        _check_candidate_head_warm_start(reordered_aux, str(tmp_path))


def test_candidate_head_warm_start_requires_the_stage1_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        _check_candidate_head_warm_start(_draft(STAGE2).config, str(tmp_path))


def _aux_ids(target_wrapper, cached=None, recipe_ids=None):
    recipe = SimpleNamespace(target_wrapper=target_wrapper, _cached_aux_layer_ids=cached)
    recipe_cfg = SimpleNamespace(get=lambda key, default=None: recipe_ids if key == "aux_layer_ids" else default)
    base = LlamaConfig(num_hidden_layers=8)
    return recipe_module.TrainEagle3Recipe._supervision_aux_layer_ids(recipe, recipe_cfg, base)


def test_aux_layer_ids_come_from_the_supervision_source():
    """The recorded layers are the ones the backend or cache produced, not a recomputed default."""
    assert _aux_ids(SimpleNamespace(aux_layer_ids=[0, 2, 5]), recipe_ids=[1, 3, 4]) == [0, 2, 5]
    remote = MagicMock(spec=RemoteEagle3TargetModel)
    remote.model_info.return_value = {"aux_layer_ids": [2, 5, 6]}
    assert _aux_ids(remote) == [2, 5, 6]
    assert _aux_ids(None, cached=[2, 4, 6], recipe_ids=[1, 3, 4]) == [2, 4, 6]
    assert _aux_ids(None, recipe_ids=[1, 3, 4]) == [1, 3, 4]
    assert _aux_ids(None) == [1, 3, 4]  # default recipe for an 8-layer target


def test_candidate_head_trainer_round_trips_the_checkpoint_metadata(tmp_path):
    """The recipe saves and restores the trainer's vocab-mapping buffers; stage 2 must provide them."""
    draft = _draft(STAGE2)
    trainer = STAGE2.candidate_head.build(draft, ttt_steps=2)
    saver = SimpleNamespace(runtime=SimpleNamespace(global_step=7), _module=lambda: trainer)
    saver._save_meta = MethodType(SpecDecodeRecipeBase._save_meta, saver)
    recipe_module.TrainEagle3Recipe._save_extra_state(saver, str(tmp_path), epoch=1)

    resumed = STAGE2.candidate_head.build(_draft(STAGE2), ttt_steps=2)
    loader = SimpleNamespace(
        runtime=SimpleNamespace(global_step=0),
        _resume_epoch=0,
        checkpoint_config=SimpleNamespace(allow_legacy_pickle_restore=False),
        _module=lambda: resumed,
        draft_model=resumed.draft_model,
        target_wrapper=None,
    )
    loader._load_meta = MethodType(SpecDecodeRecipeBase._load_meta, loader)
    recipe_module.TrainEagle3Recipe._load_extra_state(loader, str(tmp_path))
    assert loader.runtime.global_step == 7 and loader._resume_epoch == 1
    torch.testing.assert_close(resumed.selected_token_ids, torch.arange(64))
    assert resumed.selected_token_mask.all()
