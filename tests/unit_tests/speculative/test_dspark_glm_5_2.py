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

"""CPU correctness tests for the GLM-5.2 DSpark draft model + its config/registry.

A tiny GLM-MLA draft (DeepSeek-V3-style Q-LoRA, a compressed KV latent, a separate
value up-projection, interleaved complex RoPE, dense non-causal attention) exercises the
anchor sampler, the block attention mask, the Markov + confidence heads, and the three-term
objective, mirroring ``test_dspark_draft_deepseek_v4.py``.
"""

import pytest
import torch
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

from nemo_automodel.components.speculative.dspark.config import build_glm_5_2_draft_config
from nemo_automodel.components.speculative.dspark.draft_glm_5_2 import Glm5_2DSparkModel
from nemo_automodel.components.speculative.dspark.loss import compute_dspark_loss
from nemo_automodel.components.speculative.dspark.registry import resolve_dspark_draft_spec

VOCAB = 128
HIDDEN = 64
TARGET_LAYER_IDS = [1, 3]
BLOCK_SIZE = 4
NUM_ANCHORS = 6
BATCH = 2
SEQ = 16


class _Args(dict):
    """Dict that also supports attribute access (the draft-config builder uses both)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(key) from exc


def _tiny_target_config() -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=24,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        intermediate_size=48,
        index_head_dim=16,
        index_n_heads=2,
        index_topk=8,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        hidden_act="silu",
    )


def _tiny_model_args(markov_head_type: str = "vanilla") -> _Args:
    return _Args(
        num_draft_layers=2,
        target_layer_ids=list(TARGET_LAYER_IDS),
        block_size=BLOCK_SIZE,
        mask_token_id=5,
        num_anchors=NUM_ANCHORS,
        markov_rank=16,
        markov_head_type=markov_head_type,
        confidence_head_alpha=1.0,
        confidence_head_with_markov=True,
    )


def _build_model(markov_head_type: str = "vanilla", **overrides) -> Glm5_2DSparkModel:
    model_args = _tiny_model_args(markov_head_type)
    model_args.update(overrides)
    draft_config = build_glm_5_2_draft_config(_tiny_target_config(), model_args)
    model = Glm5_2DSparkModel(draft_config).to(dtype=torch.float32).eval()
    model.initialize_embeddings_and_head(
        embed_tokens=torch.nn.Embedding(VOCAB, HIDDEN),
        lm_head=torch.nn.Linear(HIDDEN, VOCAB, bias=False),
        freeze=True,
    )
    return model


def _batch(seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    return {
        "input_ids": torch.randint(0, VOCAB, (BATCH, SEQ), generator=gen),
        "loss_mask": torch.ones(BATCH, SEQ, dtype=torch.uint8),
        "target_hidden_states": torch.randn(BATCH, SEQ, len(TARGET_LAYER_IDS) * HIDDEN, generator=gen),
        "target_last_hidden_states": torch.randn(BATCH, SEQ, HIDDEN, generator=gen),
    }


def _forward(model: Glm5_2DSparkModel, batch: dict, seed: int = 1234):
    torch.manual_seed(seed)
    return model(
        input_ids=batch["input_ids"],
        target_hidden_states=batch["target_hidden_states"],
        loss_mask=batch["loss_mask"],
        target_last_hidden_states=batch["target_last_hidden_states"],
    )


def test_forward_shapes():
    model = _build_model()
    with torch.no_grad():
        out = _forward(model, _batch())
    assert out.draft_logits.shape == (BATCH, NUM_ANCHORS, BLOCK_SIZE, VOCAB)
    assert out.target_ids.shape == (BATCH, NUM_ANCHORS, BLOCK_SIZE)
    assert out.eval_mask.shape == (BATCH, NUM_ANCHORS, BLOCK_SIZE)
    assert out.block_keep_mask.shape == (BATCH, NUM_ANCHORS)
    assert out.confidence_pred.shape == (BATCH, NUM_ANCHORS, BLOCK_SIZE)
    assert out.aligned_target_logits.shape == (BATCH, NUM_ANCHORS, BLOCK_SIZE, VOCAB)
    assert torch.isfinite(out.draft_logits).all()


@pytest.mark.parametrize("head", ["vanilla", "gated", "rnn"])
def test_markov_head_variants_forward(head):
    model = _build_model(markov_head_type=head)
    assert model.markov_head is not None and model.markov_head.markov_head_type == head
    with torch.no_grad():
        out = _forward(model, _batch())
    assert out.draft_logits.shape == (BATCH, NUM_ANCHORS, BLOCK_SIZE, VOCAB)
    assert torch.isfinite(out.draft_logits).all()


def test_embeddings_and_head_frozen():
    model = _build_model()
    assert model.embed_tokens.weight.requires_grad is False
    assert model.lm_head.weight.requires_grad is False
    assert model.fc.weight.requires_grad is True
    assert any(p.requires_grad for p in model.layers.parameters())


def test_loss_is_finite_scalar_and_backprops():
    model = _build_model()
    out = _forward(model, _batch())
    loss = compute_dspark_loss(
        outputs=out,
        loss_decay_gamma=4.0,
        ce_loss_alpha=0.1,
        l1_loss_alpha=0.9,
        confidence_head_alpha=1.0,
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    # Frozen tensors receive no gradient; the trainable backbone does.
    assert model.embed_tokens.weight.grad is None
    assert model.lm_head.weight.grad is None
    assert model.fc.weight.grad is not None and torch.isfinite(model.fc.weight.grad).all()
    attn_grad = model.layers[0].self_attn.q_a_proj.weight.grad
    assert attn_grad is not None and torch.isfinite(attn_grad).all()


def test_config_builder_fields():
    cfg = build_glm_5_2_draft_config(_tiny_target_config(), _tiny_model_args())
    assert cfg.architectures == ["Glm5_2DSparkModel"]
    assert cfg.num_hidden_layers == 2
    assert cfg.num_target_layers == 8
    # The draft is always dense: MoE and the DSA indexer / IndexShare are disabled.
    assert cfg.mlp_layer_types == ["dense", "dense"]
    assert cfg.indexer_types == ["full", "full"]
    assert cfg.tie_word_embeddings is False
    assert cfg._attn_implementation == "sdpa"
    assert cfg.block_size == BLOCK_SIZE
    assert cfg.mask_token_id == 5
    assert cfg.target_layer_ids == TARGET_LAYER_IDS
    assert cfg.num_anchors == NUM_ANCHORS
    assert cfg.markov_rank == 16


def test_registry_resolves_glm_5_2():
    spec = resolve_dspark_draft_spec(["GlmMoeDsaForCausalLM"])
    assert spec.draft_cls is Glm5_2DSparkModel


def test_freqs_stays_fp32_after_bf16_cast():
    """``model.to(bf16)`` must keep the GLM rotary ``freqs`` table in fp32."""
    model = _build_model()
    ref = model.freqs.detach().clone().float()
    model = model.to(torch.bfloat16)
    assert model.freqs.dtype == torch.float32
    # Recomputed fresh in fp32, not a bf16 round-trip (bf16 rel. error ~1e-2).
    assert torch.allclose(model.freqs, ref, rtol=1e-6, atol=1e-8)


def test_bf16_forward_runs():
    """The training build path casts the draft to bf16; a bf16 forward must stay finite."""
    model = _build_model().to(torch.bfloat16)
    batch = _batch()
    batch = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in batch.items()}
    with torch.no_grad():
        out = _forward(model, batch)
    assert out.draft_logits.dtype == torch.bfloat16
    assert torch.isfinite(out.draft_logits).all()


def test_predict_confidence_step_variants():
    hidden_states = torch.randn(BATCH, HIDDEN)
    prev_token_ids = torch.arange(BATCH)

    no_confidence = _build_model(markov_rank=0, confidence_head_alpha=0.0)
    assert no_confidence.predict_confidence_step(hidden_states) is None

    plain_confidence = _build_model(
        markov_rank=0,
        confidence_head_alpha=1.0,
        confidence_head_with_markov=False,
    )
    plain_pred = plain_confidence.predict_confidence_step(hidden_states)
    assert plain_pred.shape == (BATCH,)
    assert plain_pred.dtype == torch.float32

    markov_confidence = _build_model()
    with pytest.raises(AssertionError):
        markov_confidence.predict_confidence_step(hidden_states)
    markov_pred = markov_confidence.predict_confidence_step(hidden_states, prev_token_ids)
    assert markov_pred.shape == (BATCH,)
    assert torch.isfinite(markov_pred).all()


def test_sample_draft_tokens_variants():
    first_prev_token_ids = torch.arange(BATCH)
    empty_logits = torch.empty(BATCH, 0, VOCAB)
    model = _build_model()

    empty_tokens, returned_empty_logits = model.sample_draft_tokens(
        empty_logits,
        first_prev_token_ids=first_prev_token_ids,
    )
    assert empty_tokens.shape == (BATCH, 0)
    assert returned_empty_logits is empty_logits

    base_logits = torch.randn(BATCH, BLOCK_SIZE, VOCAB)
    no_markov = _build_model(markov_rank=0, confidence_head_alpha=0.0)
    plain_tokens, plain_logits = no_markov.sample_draft_tokens(
        base_logits,
        first_prev_token_ids=first_prev_token_ids,
    )
    assert plain_tokens.shape == (BATCH, BLOCK_SIZE)
    assert plain_logits is base_logits

    markov_tokens, markov_logits = model.sample_draft_tokens(
        base_logits,
        first_prev_token_ids=first_prev_token_ids,
    )
    assert markov_tokens.shape == (BATCH, BLOCK_SIZE)
    assert markov_logits.shape == base_logits.shape


@pytest.mark.parametrize("markov_rank", [0, 16])
def test_sample_draft_token_step(markov_rank):
    model = _build_model(
        markov_rank=markov_rank,
        confidence_head_alpha=1.0 if markov_rank else 0.0,
    )
    base_logits = torch.randn(BATCH, VOCAB)
    prev_token_ids = torch.arange(BATCH)

    sampled_token_ids, step_logits = model.sample_draft_token_step(
        base_logits,
        prev_token_ids=prev_token_ids,
    )
    assert sampled_token_ids.shape == (BATCH,)
    assert step_logits.shape == base_logits.shape
    assert torch.isfinite(step_logits).all()

    with pytest.raises(AssertionError, match="expects base_logits"):
        model.sample_draft_token_step(
            base_logits.unsqueeze(1),
            prev_token_ids=prev_token_ids,
        )


def test_forward_confidence_without_markov_features():
    model = _build_model(
        markov_rank=0,
        confidence_head_alpha=1.0,
        confidence_head_with_markov=False,
    )
    with torch.no_grad():
        out = _forward(model, _batch())
    assert out.confidence_pred.shape == (BATCH, NUM_ANCHORS, BLOCK_SIZE)
    assert torch.isfinite(out.confidence_pred).all()
