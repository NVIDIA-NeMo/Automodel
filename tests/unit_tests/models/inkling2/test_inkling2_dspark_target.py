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
"""Inkling 2 DSpark target: feature capture against the native model and the reference oracle, one recipe step."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

import nemo_automodel.recipes.speculative_decoding.train_dspark as train_dspark_module
from nemo_automodel.components.models.inkling2.config import Inkling2Config, Inkling2TextConfig, Inkling2VisionConfig
from nemo_automodel.components.models.inkling2.dspark import Inkling2DSparkModel, Inkling2DSparkReleasedConfig
from nemo_automodel.components.models.inkling2.dspark_target import Inkling2DSparkTargetConfig
from nemo_automodel.components.models.inkling2.layers import ResidualState
from nemo_automodel.components.models.inkling2.model import Inkling2ForCausalLM
from nemo_automodel.components.models.inkling2.multimodal import Inkling2ForConditionalGeneration
from nemo_automodel.components.speculative.dspark.registry import (
    ModelOwnedDSparkProvider,
    ModelOwnedDSparkTargetProvider,
)
from nemo_automodel.components.speculative.dspark.target import HFDSparkTargetModel
from nemo_automodel.recipes.speculative_decoding.train_dspark import TrainDSparkRecipe

from . import reference
from .test_loop import looped_config

pytestmark = pytest.mark.cpu_only

_HIDDEN, _LAYERS, _SEQ = 16, 7, 12
# Released-style labels: the streams entering layers 2, 5 (global) and 6 (a block boundary), then the
# post-norm sentinel 7. DSpark ids are one lower, after the normalized embedding (-1).
_MTP_CONFIG = {
    "hidden_size": _HIDDEN, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
    "head_dim": 4, "intermediate_size": 24, "max_position_embeddings": 4096, "q_bias": False, "o_bias": False,
    "rms_norm_eps": 1e-6, "rope_theta": 1e6, "local_layer_ids": [0], "sliding_window_size": 4, "block_size": 4,
    "markov_rank": 8, "markov_head_type": "rnn_hidden", "target_layer_ids": [2, 5, 6, 7],
}  # fmt: skip
_TARGET_LAYER_IDS = [-1, 1, 4, 5, 6]


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _text_config() -> Inkling2TextConfig:
    return Inkling2TextConfig(
        hidden_size=_HIDDEN,
        num_hidden_layers=_LAYERS,
        vocab_size=40,
        unpadded_vocab_size=37,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        d_rel=3,
        rel_extent=4,
        sliding_window_size=7,
        n_routed_experts=12,
        num_experts_per_tok=8,
        intermediate_size=8,
        log_scaling_n_floor=4,
        dtype="float32",
    )


def _config() -> Inkling2Config:
    vision = Inkling2VisionConfig(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=2, mlp_ratio=2.0, posemb_grid_size=4,
        projection_hidden_dim=24,
    )  # fmt: skip
    return Inkling2Config(
        text_config=_text_config(),
        vision_config=vision,
        mtp_config=dict(_MTP_CONFIG),
        architectures=["Inkling2ForConditionalGeneration"],
    )


def _target(multimodal: bool) -> Inkling2ForCausalLM:
    """Return a tiny target with non-trivial residual queries, norms and router biases."""
    torch.manual_seed(0)
    model = Inkling2ForConditionalGeneration(_config()) if multimodal else Inkling2ForCausalLM(_text_config())
    with torch.no_grad():
        model.output_query.weight.normal_(std=0.2)
        for layer in model.layers.values():
            if layer.attn_query is not None:
                layer.attn_query.weight.normal_(std=0.2)
            layer.mlp_query.weight.normal_(std=0.2)
            layer.mlp.gate.e_score_correction_bias.uniform_(-0.03, 0.03)
        for name, param in model.named_parameters():
            if name in ("embed_norm.weight", "norm.weight"):
                param.uniform_(0.5, 1.5)
    return model.eval()


def _input_ids() -> torch.Tensor:
    """Return random int64 token ids of shape [1, _SEQ] over the real vocabulary."""
    return torch.randint(0, 37, (1, _SEQ), generator=torch.Generator().manual_seed(3))


def test_released_labels_map_to_dspark_ids():
    released = Inkling2DSparkReleasedConfig.from_mtp_config(_MTP_CONFIG)
    assert released.dspark_target_layer_ids(_LAYERS) == tuple(_TARGET_LAYER_IDS)


@pytest.mark.parametrize("multimodal", [False, True])
def test_captured_features_are_the_embedding_the_partials_and_the_head_input(multimodal):
    """Id -1 is the normalized embedding entering layer 0, id i < last the partial block that layer i
    returns (what layer i + 1 receives), and the last id the post-final-norm hidden state."""
    model = _target(multimodal)
    input_ids = _input_ids()
    returned: dict[str, torch.Tensor] = {}

    def keep_partial(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple, output: ResidualState) -> None:
            """Record the [batch, sequence, hidden] partial block a layer returns."""
            returned[name] = output.partial.detach().clone()

        return hook

    handles = [layer.register_forward_hook(keep_partial(name)) for name, layer in model.layers.items()]
    try:
        batch = HFDSparkTargetModel(model, target_layer_ids=_TARGET_LAYER_IDS).generate_batch(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids), loss_mask=torch.ones_like(input_ids)
        )
    finally:
        for handle in handles:
            handle.remove()
    with torch.no_grad():
        final = model(input_ids, output_hidden_states=True).hidden_states
        embedded = model.embed_norm(model.embed_tokens(input_ids).float())
    features = batch.target_hidden_states.unflatten(-1, (len(_TARGET_LAYER_IDS), _HIDDEN))

    # Exactly what the native model computes.
    torch.testing.assert_close(features[..., 0, :], embedded, rtol=0, atol=0)
    for slot, layer_id in enumerate(_TARGET_LAYER_IDS[1:-1], start=1):
        torch.testing.assert_close(features[..., slot, :], returned[str(layer_id)], rtol=0, atol=0)
    torch.testing.assert_close(features[..., -1, :], final, rtol=0, atol=0)
    torch.testing.assert_close(batch.target_last_hidden_states, final, rtol=0, atol=0)

    # And what the independent functional oracle computes.
    weights = {name: value.detach() for name, value in model.state_dict().items()}
    expected = reference.forward(_text_config(), weights, input_ids)
    torch.testing.assert_close(
        features[..., 0, :],
        reference.rms(F.embedding(input_ids, weights["embed_tokens.weight"]), weights["embed_norm.weight"], 1e-6),
        rtol=1e-6,
        atol=1e-6,
    )
    for slot, layer_id in enumerate(_TARGET_LAYER_IDS[1:-1], start=1):
        torch.testing.assert_close(features[..., slot, :], expected.partials[layer_id], rtol=2e-4, atol=1e-5)
    logits = F.linear(features[..., -1, :], weights["lm_head.weight"])
    logits = 30.0 * torch.tanh(logits / 30.0)
    torch.testing.assert_close(logits[..., :37], expected.logits[..., :37], rtol=2e-4, atol=1e-5)


def test_feature_modules_reject_invalid_ids_and_pipeline_stages():
    model = _target(multimodal=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        model.get_dspark_target_feature_modules([4, 1])
    with pytest.raises(ValueError, match=r"integers in \[-1, 7\)"):
        model.get_dspark_target_feature_modules([1, 7])
    stage = copy.deepcopy(model)
    stage.lm_head = None
    with pytest.raises(ValueError, match="pipeline stage"):
        stage.get_dspark_target_feature_modules([1])


def test_feature_modules_reject_a_looped_decoder():
    looped = Inkling2ForCausalLM(looped_config())
    with pytest.raises(ValueError, match="unlooped"):
        looped.get_dspark_target_feature_modules([1])


def test_config_provides_the_draft_and_the_target():
    config = _config()
    assert isinstance(config, ModelOwnedDSparkProvider)
    assert isinstance(config, ModelOwnedDSparkTargetProvider)
    assert config.get_text_config() is config.text_config
    with pytest.raises(ValueError, match="no mtp_config"):
        Inkling2Config(text_config=_text_config()).build_dspark_draft(SimpleNamespace())


def test_target_requires_cuda():
    with pytest.raises(RuntimeError, match="requires CUDA"):
        Inkling2DSparkTargetConfig(target_path="unused").build(
            device=torch.device("cpu"), compute_dtype=torch.float32, distributed_setup=None
        )


class _CfgNode(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def to_dict(self):
        return dict(self)


def test_target_backend_defaults_and_overrides(monkeypatch):
    """The target loads with the fine-tuning recipes' backend unless the config overrides it."""
    from nemo_automodel._transformers import NeMoAutoModelForImageTextToText

    loaded = {}

    def from_pretrained(path, **kwargs):
        loaded[path] = kwargs["backend"]
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(NeMoAutoModelForImageTextToText, "from_pretrained", from_pretrained)
    cuda = torch.device("cuda")
    Inkling2DSparkTargetConfig(target_path="default").build(
        device=cuda, compute_dtype=torch.bfloat16, distributed_setup=None
    )
    Inkling2DSparkTargetConfig(target_path="override", attn_backend="eager", experts="torch", dispatcher="torch").build(
        device=cuda, compute_dtype=torch.bfloat16, distributed_setup=None
    )
    assert (loaded["default"].attn, loaded["default"].experts, loaded["default"].dispatcher) == (
        "flex",
        "torch_mm",
        "hybridep",
    )
    assert (loaded["override"].attn, loaded["override"].experts, loaded["override"].dispatcher) == (
        "eager",
        "torch",
        "torch",
    )


def _patch_recipe(monkeypatch, target, built, batch):
    """Run the recipe on CPU with ``target`` as the frozen Inkling 2 target.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        target: Tiny native Inkling 2 model returned by the patched target build.
        built: Dict that records the target build's path, mesh and backend.
        batch: The single training batch the patched dataloader yields; tensors of shape [1, sequence].
    """

    def build_target(self, *, device, compute_dtype, distributed_setup, load_metadata_with_gloo=False):
        built.update(
            path=self.target_path,
            distributed_setup=distributed_setup,
            backend=(self.attn_backend, self.experts, self.dispatcher),
        )
        return target

    monkeypatch.setattr(
        train_dspark_module,
        "initialize_distributed",
        lambda **_kwargs: SimpleNamespace(device=torch.device("cpu"), world_size=1, is_main=True),
    )
    monkeypatch.setattr(train_dspark_module, "setup_logging", lambda: None)
    monkeypatch.setattr(train_dspark_module, "_read_target_model_type", lambda *_args, **_kwargs: "inkling2")
    monkeypatch.setattr(train_dspark_module.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: _config())
    monkeypatch.setattr(train_dspark_module, "create_distributed_setup_from_config", lambda *_a, **_k: "mesh")
    monkeypatch.setattr(Inkling2DSparkTargetConfig, "build", build_target)
    monkeypatch.setattr(
        train_dspark_module.NeMoAutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(
            chat_template="{{ messages }}", apply_chat_template=lambda *a, **k: None
        ),
    )
    monkeypatch.setattr(train_dspark_module, "build_eagle3_dataloader", lambda **_kwargs: [batch])
    monkeypatch.setattr(TrainDSparkRecipe, "_build_checkpointer", lambda self, _target_path: None)
    monkeypatch.setattr(TrainDSparkRecipe, "load_checkpoint", lambda self, restore_from=None: None)


def _recipe_cfg(tmp_path, **recipe_overrides):
    recipe_args = _CfgNode(
        target_model_name_or_path="tiny-inkling2",
        train_data_path="unused.jsonl",
        seq_length=_SEQ,
        micro_batch_size=1,
        mask_token_id=0,
        num_epochs=1,
        output_dir=str(tmp_path / "out"),
        target_layer_ids=_TARGET_LAYER_IDS,
        draft_num_hidden_layers=2,
        num_anchors=3,
        block_size=3,
        markov_rank=8,
        markov_head_type="rnn_hidden",
        confidence_head_alpha=0.0,
        trust_remote_code=False,
        **recipe_overrides,
    )
    return _CfgNode(
        recipe_args=recipe_args,
        optimizer=_CfgNode(lr=1e-3, warmup_ratio=0.0, min_lr_ratio=0.1),
        checkpoint=_CfgNode(enabled=False),
        raw_config={},
    )


def _batch():
    input_ids = _input_ids()
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "loss_mask": torch.cat((torch.zeros(1, 3), torch.ones(1, _SEQ - 3)), dim=1).long(),
    }


def test_recipe_rejects_context_parallelism_for_the_native_target(monkeypatch, tmp_path):
    built = {}
    _patch_recipe(monkeypatch, _target(multimodal=True), built, _batch())
    monkeypatch.setattr(
        train_dspark_module,
        "create_distributed_setup_from_config",
        lambda *_a, **_k: SimpleNamespace(mesh_context=SimpleNamespace(device_mesh=None)),
    )
    monkeypatch.setattr(train_dspark_module, "get_flat_mesh", lambda *_args, **_kwargs: None)
    cfg = _recipe_cfg(tmp_path, target_force_hf=True, target_attn_implementation="sdpa")
    cfg["distributed.cp_size"] = 2
    recipe = TrainDSparkRecipe(cfg)
    with pytest.raises(NotImplementedError, match="Inkling2Config DSpark target"):
        recipe.setup()
    assert built == {}


def test_recipe_trains_the_released_draft_against_the_native_target(monkeypatch, tmp_path):
    target = _target(multimodal=True)
    built = {}
    batch = _batch()
    _patch_recipe(monkeypatch, target, built, batch)
    cfg = _recipe_cfg(tmp_path, target_dispatcher="torch")
    recipe = TrainDSparkRecipe(cfg)
    recipe.setup()

    assert built == {"path": "tiny-inkling2", "distributed_setup": "mesh", "backend": ("flex", "torch_mm", "torch")}
    draft = recipe.draft_model
    assert isinstance(draft, Inkling2DSparkModel)
    assert "embed_tokens.weight" not in draft.state_dict()
    torch.testing.assert_close(draft.lm_head.weight, target.lm_head.weight, rtol=0, atol=0)

    torch.manual_seed(4)
    metrics = recipe._forward_batch(batch)
    metrics.loss.backward()
    assert torch.isfinite(metrics.loss)
    assert metrics.tau_den > 0 and metrics.accept_rate_per_pos_num.shape == (3,)
    assert all(p.grad is None for p in target.parameters())
    trainable = [p for p in draft.parameters() if p.requires_grad]
    assert trainable and all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
