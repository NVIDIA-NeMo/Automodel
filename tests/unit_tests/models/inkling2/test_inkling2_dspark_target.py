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
"""Inkling 2 DSpark target: feature taps on the native target and one recipe training step on CPU."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import nemo_automodel.recipes.speculative_decoding.train_dspark as train_dspark_module
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.configuration import Inkling2Config
from nemo_automodel.components.models.inkling2.dspark import Inkling2DSparkModel, Inkling2DSparkReleasedConfig
from nemo_automodel.components.models.inkling2.dspark_target import Inkling2DSparkTargetConfig
from nemo_automodel.components.models.inkling2.model import Inkling2ForConditionalGeneration
from nemo_automodel.components.speculative.dspark.registry import (
    ModelOwnedDSparkProvider,
    ModelOwnedDSparkTargetProvider,
)
from nemo_automodel.components.speculative.dspark.target import HFDSparkTargetModel
from nemo_automodel.recipes.speculative_decoding.train_dspark import TrainDSparkRecipe

pytestmark = pytest.mark.cpu_only

# Seven target layers: one attention-residual block of six, then a boundary at layer 6.
_TEXT_CONFIG = dict(
    hidden_size=32,
    num_hidden_layers=7,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    swa_num_attention_heads=4,
    swa_num_key_value_heads=2,
    swa_head_dim=8,
    sliding_window_size=8,
    d_rel=4,
    rel_extent=12,
    vocab_size=64,
    unpadded_vocab_size=60,
    moe_intermediate_size=16,
    n_routed_experts=4,
    num_experts_per_tok=2,
    n_layers_per_attn_res_block=6,
    layer_types=["hybrid_sliding", "hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid_sliding", "hybrid_sliding", "hybrid"],
    final_logit_softcapping=30.0,
    eos_token_id=1,
)  # fmt: skip
# Released-style labels: streams entering layers 2 and 5 (not block boundaries), then the post-norm sentinel 7.
_MTP_CONFIG = {
    "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
    "intermediate_size": 48, "max_position_embeddings": 4096, "q_bias": False, "o_bias": False, "rms_norm_eps": 1e-6,
    "rope_theta": 1e6, "local_layer_ids": [0], "sliding_window_size": 4, "block_size": 4, "markov_rank": 8,
    "markov_head_type": "rnn_hidden", "target_layer_ids": [2, 5, 7],
}  # fmt: skip
_TARGET_LAYER_IDS = [1, 4, 6]
_SEQ = 12


def _config() -> Inkling2Config:
    return Inkling2Config(
        text_config=dict(_TEXT_CONFIG), vision_config=None, mtp_config=dict(_MTP_CONFIG), torch_dtype="float32"
    )


def _target() -> Inkling2ForConditionalGeneration:
    torch.manual_seed(0)
    backend = BackendConfig(attn="eager", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
    model = Inkling2ForConditionalGeneration(_config(), backend=backend)
    with torch.no_grad():
        for name, param in model.named_parameters():
            noise = torch.randn_like(param)
            param.copy_(1.0 + 0.2 * noise if name.endswith("norm.weight") else 0.15 * noise)
    return model.eval()


def _input_ids() -> torch.Tensor:
    """Return random int64 token ids of shape ``[1, _SEQ]``, avoiding ids 0 and 1."""
    return torch.randint(2, _TEXT_CONFIG["unpadded_vocab_size"], (1, _SEQ), generator=torch.Generator().manual_seed(3))


def test_released_labels_map_to_dspark_ids():
    released = Inkling2DSparkReleasedConfig.from_mtp_config(_MTP_CONFIG)
    assert released.automodel_target_layer_ids(_TEXT_CONFIG["num_hidden_layers"]) == tuple(_TARGET_LAYER_IDS)


def test_target_taps_are_the_streams_entering_the_labelled_layers():
    """Label L reads what target layer L receives (the running attention-residual block sum),
    and the sentinel reads the post-final-norm hidden state that feeds the LM head."""
    model = _target()
    text_model = model.model.language_model
    entering = {}
    handles = [
        text_model.layers[label].register_forward_pre_hook(
            lambda _module, args, label=label: entering.__setitem__(label, args[0].detach().clone())
        )
        for label in (2, 5)
    ]
    head_input = {}
    handles.append(
        model.lm_head.register_forward_pre_hook(lambda _module, args: head_input.__setitem__("x", args[0].detach()))
    )
    input_ids = _input_ids()
    batch = HFDSparkTargetModel(model, target_layer_ids=_TARGET_LAYER_IDS).generate_batch(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids), loss_mask=torch.ones_like(input_ids)
    )
    with torch.no_grad():
        model(input_ids=input_ids)
    for handle in handles:
        handle.remove()

    features = batch.target_hidden_states.unflatten(-1, (3, _TEXT_CONFIG["hidden_size"]))
    torch.testing.assert_close(features[..., 0, :], entering[2], rtol=0, atol=0)
    torch.testing.assert_close(features[..., 1, :], entering[5], rtol=0, atol=0)
    torch.testing.assert_close(features[..., 2, :], head_input["x"], rtol=0, atol=0)
    torch.testing.assert_close(batch.target_last_hidden_states, head_input["x"], rtol=0, atol=0)


def test_config_provides_the_draft_and_the_target():
    config = _config()
    assert isinstance(config, ModelOwnedDSparkProvider)
    assert isinstance(config, ModelOwnedDSparkTargetProvider)
    with pytest.raises(ValueError, match="no mtp_config"):
        Inkling2Config(text_config=dict(_TEXT_CONFIG), vision_config=None).build_dspark_draft(SimpleNamespace())


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


def test_recipe_trains_the_released_draft_against_the_native_target(monkeypatch, tmp_path):
    target = _target()
    built = {}

    def build_target(self, *, device, compute_dtype, distributed_setup, load_metadata_with_gloo=False):
        built.update(path=self.target_path, distributed_setup=distributed_setup)
        return target

    input_ids = _input_ids()
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "loss_mask": torch.cat((torch.zeros(1, 3), torch.ones(1, _SEQ - 3)), dim=1).long(),
    }
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
    )
    cfg = _CfgNode(
        recipe_args=recipe_args,
        optimizer=_CfgNode(lr=1e-3, warmup_ratio=0.0, min_lr_ratio=0.1),
        checkpoint=_CfgNode(enabled=False),
        raw_config={},
    )
    recipe = TrainDSparkRecipe(cfg)
    recipe.setup()

    assert built == {"path": "tiny-inkling2", "distributed_setup": "mesh"}
    draft = recipe.draft_model
    assert isinstance(draft, Inkling2DSparkModel)
    embedding = target.get_input_embeddings()
    torch.testing.assert_close(draft.embed_tokens.weight, embedding.weight, rtol=0, atol=0)
    torch.testing.assert_close(draft.embed_norm_weight, embedding.embed_norm.weight, rtol=0, atol=0)
    torch.testing.assert_close(draft.lm_head.weight, target.lm_head.weight, rtol=0, atol=0)

    torch.manual_seed(4)
    metrics = recipe._forward_batch(batch)
    metrics.loss.backward()
    assert torch.isfinite(metrics.loss)
    assert metrics.tau_den > 0 and metrics.accept_rate_per_pos_num.shape == (3,)
    assert all(p.grad is None for p in target.parameters())
    trainable = [p for p in draft.parameters() if p.requires_grad]
    assert trainable and all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
