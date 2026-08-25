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

"""CPU coverage for the dense Kimi K3 DFlash draft."""

import copy
from types import SimpleNamespace

import pytest
import torch

from nemo_automodel._transformers.models.kimi_k3.config import KimiK3Config, KimiK3TextConfig
from nemo_automodel.components.speculative.dflash.core import DFlashTrainerModule
from nemo_automodel.components.speculative.dflash.draft_kimi_k3 import (
    KimiK3DFlashDraftModel,
    build_kimi_k3_dflash_draft_config,
    build_kimi_k3_dflash_target_kwargs,
)
from nemo_automodel.components.speculative.dflash.registry import resolve_dflash_draft_spec
from nemo_automodel.recipes.llm import train_dflash
from nemo_automodel.recipes.llm.train_dflash import TrainDFlashRecipe

VOCAB = 64
HIDDEN = 32
TARGET_LAYERS = [1, 3]
BLOCK_SIZE = 3
MASK_TOKEN_ID = 5


def _text_config():
    return KimiK3TextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        num_experts=8,
        num_experts_per_token=2,
        num_shared_experts=1,
        linear_attn_config={"kda_layers": [1, 2, 3], "full_attn_layers": [4]},
    )


def _draft_config(**dflash_overrides):
    dflash_config = {"mask_token_id": MASK_TOKEN_ID, "target_layer_ids": TARGET_LAYERS}
    dflash_config.update(dflash_overrides)
    return build_kimi_k3_dflash_draft_config(
        _text_config(),
        num_draft_layers=2,
        num_target_layers=4,
        block_size=BLOCK_SIZE,
        dflash_config=dflash_config,
        attention_backend="sdpa",
    )


def test_config_builder_produces_a_dense_mla_draft():
    config = _draft_config()
    assert config.architectures == ["KimiK3DFlashDraftModel"]
    assert config.num_hidden_layers == 2
    assert config.num_target_layers == 4
    assert config.block_size == BLOCK_SIZE
    assert config.dflash_config == {"mask_token_id": MASK_TOKEN_ID, "target_layer_ids": TARGET_LAYERS}
    assert config._attn_implementation == "sdpa"
    # Everything the draft does not build is switched off explicitly, so the
    # serialized draft config describes the draft and not the target.
    assert config.num_experts is None
    assert config.num_shared_experts == 0
    assert config.linear_attn_config == {"kda_layers": [], "full_attn_layers": [1, 2]}
    assert config.num_nextn_predict_layers == 0
    assert config.attn_res_block_size is None
    assert config.tie_word_embeddings is False
    # The MLA geometry is inherited from the target unchanged.
    assert (config.hidden_size, config.vocab_size) == (HIDDEN, VOCAB)
    assert (config.q_lora_rank, config.kv_lora_rank, config.v_head_dim) == (16, 8, 4)


def test_config_builder_does_not_mutate_the_target_config():
    """The recipe builds the target from the same config object after this call."""
    target_config = _text_config()
    before = copy.deepcopy(target_config.to_dict())
    build_kimi_k3_dflash_draft_config(
        target_config,
        num_draft_layers=2,
        num_target_layers=4,
        block_size=BLOCK_SIZE,
        dflash_config={"mask_token_id": MASK_TOKEN_ID, "target_layer_ids": TARGET_LAYERS},
        attention_backend="sdpa",
    )
    assert target_config.to_dict() == before


def test_config_builder_rejects_the_multimodal_wrapper_config():
    """The recipe must unwrap ``text_config`` before calling the builder."""
    with pytest.raises(ValueError, match="expects a 'kimi_linear' text config"):
        build_kimi_k3_dflash_draft_config(
            KimiK3Config(text_config=_text_config()),
            num_draft_layers=2,
            num_target_layers=4,
            block_size=BLOCK_SIZE,
            dflash_config={"mask_token_id": MASK_TOKEN_ID, "target_layer_ids": TARGET_LAYERS},
            attention_backend="sdpa",
        )


@pytest.mark.parametrize("architecture", ["KimiK3ForCausalLM", "KimiK3ForConditionalGeneration"])
def test_registry_resolves_kimi_k3(architecture):
    """Both the text-only and the multimodal architecture map to the same draft."""
    spec = resolve_dflash_draft_spec([architecture])
    assert spec.draft_cls is KimiK3DFlashDraftModel
    assert spec.build_draft_config is build_kimi_k3_dflash_draft_config
    assert spec.build_target_kwargs is build_kimi_k3_dflash_target_kwargs
    # The draft has no FlexAttention path.
    assert spec.attention_backends == ("sdpa",)
    # K3 shards the sequence itself and bypasses the DFlash CP key/value-gather hook.
    assert spec.supports_context_parallel is False


def test_context_parallel_is_rejected_before_the_generic_cp_gates(monkeypatch):
    """The K3 CP rejection must win over the force_hf / sdpa CP gates.

    Those gates fire for every target with cp_size>1 and would otherwise send the
    user chasing ``target_force_hf=true``, which routes K3 onto the stock-HF path
    where a target-instance check could never be reached at all. The rejection
    must also land before any weight is loaded.
    """
    cp_mesh = SimpleNamespace(size=lambda: 2)
    dist_setup = SimpleNamespace(mesh_context=SimpleNamespace(device_mesh=object()))
    monkeypatch.setattr(train_dflash, "create_distributed_setup_from_config", lambda cfg, world_size: dist_setup)
    monkeypatch.setattr(train_dflash, "_submesh_or_none", lambda mesh, name: cp_mesh if name == "cp" else object())

    def _must_not_load(*args, **kwargs):
        raise AssertionError("the target must not be loaded before the CP gate rejects the config")

    monkeypatch.setattr(train_dflash, "NeMoAutoModelForCausalLM", SimpleNamespace(from_pretrained=_must_not_load))

    recipe = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    recipe.cfg = {"distributed": {"cp_size": 2}}
    recipe.dist_env = SimpleNamespace(world_size=2)
    recipe.device = torch.device("cpu")
    recipe.compute_dtype = torch.float32

    with pytest.raises(NotImplementedError, match="shards the sequence itself"):
        recipe._build_target_model({}, "moonshotai/Kimi-K3", resolve_dflash_draft_spec(["KimiK3ForCausalLM"]))


def test_draft_rejects_a_projector_head():
    with pytest.raises(ValueError, match="does not implement a draft projector"):
        KimiK3DFlashDraftModel(_draft_config(projector_type="domino"))


def test_draft_rejects_grouped_key_value_heads():
    config = _draft_config()
    config.num_key_value_heads = 2
    with pytest.raises(ValueError, match="one K/V head per query head"):
        KimiK3DFlashDraftModel(config)


def test_draft_rejects_rope():
    config = _draft_config()
    config.mla_use_nope = False
    with pytest.raises(ValueError, match="mla_use_nope"):
        KimiK3DFlashDraftModel(config)


def test_draft_defaults_target_layer_ids_from_the_target_depth():
    config = _draft_config()
    config.dflash_config = {"mask_token_id": MASK_TOKEN_ID}
    model = KimiK3DFlashDraftModel(config)
    assert len(model.target_layer_ids) == config.num_hidden_layers
    assert all(0 <= layer_id < config.num_target_layers for layer_id in model.target_layer_ids)


def test_draft_forward_shapes_and_context_isolation():
    torch.manual_seed(0)
    # The recipe casts the draft to the compute dtype once after construction.
    model = KimiK3DFlashDraftModel(_draft_config()).to(torch.float32).eval()
    assert model.target_layer_ids == TARGET_LAYERS

    batch, seq_len, blocks = 2, 8, 2
    noise = torch.randn(batch, blocks * BLOCK_SIZE, HIDDEN)
    target_hidden = torch.randn(batch, seq_len, len(TARGET_LAYERS) * HIDDEN)
    # Block b attends to the context strictly before its anchor and, bidirectionally,
    # to its own noise block; nothing else.
    mask = torch.full((batch, 1, blocks * BLOCK_SIZE, seq_len + blocks * BLOCK_SIZE), float("-inf"))
    anchors = [2, 5]
    for b, anchor in enumerate(anchors):
        rows = slice(b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE)
        mask[:, :, rows, :anchor] = 0.0
        mask[:, :, rows, seq_len + b * BLOCK_SIZE : seq_len + (b + 1) * BLOCK_SIZE] = 0.0

    with torch.no_grad():
        out = model(position_ids=None, attention_mask=mask, noise_embedding=noise, target_hidden=target_hidden)
    assert out.shape == (batch, blocks * BLOCK_SIZE, HIDDEN)
    assert torch.isfinite(out).all()

    # Masked-out context must not reach the first block: perturbing context
    # positions at or after its anchor leaves that block's output unchanged.
    perturbed = target_hidden.clone()
    perturbed[:, anchors[0] :, :] += 5.0
    with torch.no_grad():
        out_perturbed = model(position_ids=None, attention_mask=mask, noise_embedding=noise, target_hidden=perturbed)
    torch.testing.assert_close(out[:, :BLOCK_SIZE], out_perturbed[:, :BLOCK_SIZE])
    assert not torch.allclose(out[:, BLOCK_SIZE:], out_perturbed[:, BLOCK_SIZE:])


def test_trainer_module_trains_the_kimi_k3_draft():
    """The K3 draft runs end to end inside the DFlash trainer on the SDPA mask path."""
    torch.manual_seed(3)
    draft = KimiK3DFlashDraftModel(_draft_config()).to(torch.float32)
    embed_tokens = torch.nn.Embedding(VOCAB, HIDDEN)
    lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
    trainer = DFlashTrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed_tokens,
        mask_token_id=MASK_TOKEN_ID,
        block_size=BLOCK_SIZE,
        attention_backend="sdpa",
        num_anchors=4,
    )

    seq_len = 12
    generator = torch.Generator().manual_seed(5)
    metrics = trainer(
        input_ids=torch.randint(0, VOCAB, (2, seq_len), generator=generator),
        hidden_states=torch.randn(2, seq_len, len(TARGET_LAYERS) * HIDDEN, generator=generator),
        loss_mask=torch.ones(2, seq_len),
    )
    assert torch.isfinite(metrics.loss)
    assert metrics.valid_tokens > 0
    metrics.loss.backward()
    assert torch.isfinite(draft.layers[0].self_attn.q_a_proj.weight.grad).all()
    assert torch.isfinite(draft.fc.weight.grad).all()
    # The frozen target modules are non-registered references, so DDP (and the
    # optimizer) only ever see the draft's parameters.
    assert lm_head.weight not in set(trainer.parameters())
    assert embed_tokens.weight not in set(trainer.parameters())


def test_target_kwargs_defaults_and_overrides():
    kwargs = build_kimi_k3_dflash_target_kwargs({})
    # The multimodal checkpoint must load as the text-only causal LM.
    assert kwargs["config"] == {"architectures": ["KimiK3ForCausalLM"]}
    backend = kwargs["backend"]
    assert backend.dispatcher == "hybridep"
    assert backend.experts == "torch_mm"
    assert backend.enable_hf_state_dict_adapter is True
    assert backend.enable_fsdp_optimizations is True

    backend = build_kimi_k3_dflash_target_kwargs(
        {
            "target_dispatcher": "deepep",
            "target_experts": "gmm",
            "target_enable_fsdp_optimizations": False,
        }
    )["backend"]
    assert backend.dispatcher == "deepep"
    assert backend.experts == "gmm"
    assert backend.enable_fsdp_optimizations is False
