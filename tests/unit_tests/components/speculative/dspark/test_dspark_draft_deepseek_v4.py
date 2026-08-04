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

"""CPU correctness tests for the DeepSeek V4 DSpark draft model + its config/registry.

A tiny V4-attention draft (Q-LoRA, single shared K=V latent, grouped O-LoRA,
interleaved partial RoPE with output inverse-RoPE, attention sink) exercises the
anchor sampler, the block attention mask, the Markov + confidence heads, and the
three-term objective, mirroring ``test_dspark_draft.py``.
"""

import pytest
import torch

from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.speculative.dspark.config import build_deepseek_v4_draft_config
from nemo_automodel.components.speculative.dspark.draft_deepseek_v4 import DeepseekV4DSparkModel
from nemo_automodel.components.speculative.dspark.loss import compute_dspark_loss
from nemo_automodel.components.speculative.dspark.registry import resolve_dspark_draft_spec

VOCAB = 128
HIDDEN = 64
HEAD_DIM = 32
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


def _tiny_target_config() -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=HEAD_DIM,
        qk_rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=4,
        moe_intermediate_size=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        rope_theta=10000.0,
        rope_scaling=None,
        compress_ratios=None,
        num_hash_layers=0,
        num_nextn_predict_layers=0,
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


def _build_model(markov_head_type: str = "vanilla") -> DeepseekV4DSparkModel:
    draft_config = build_deepseek_v4_draft_config(_tiny_target_config(), _tiny_model_args(markov_head_type))
    model = DeepseekV4DSparkModel(draft_config).to(dtype=torch.float32).eval()
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


def _forward(model: DeepseekV4DSparkModel, batch: dict, seed: int = 1234):
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
    attn_grad = model.layers[0].self_attn.wq_a.weight.grad
    assert attn_grad is not None and torch.isfinite(attn_grad).all()


def test_config_builder_fields():
    cfg = build_deepseek_v4_draft_config(_tiny_target_config(), _tiny_model_args())
    assert cfg.architectures == ["DeepseekV4DSparkModel"]
    assert cfg.num_hidden_layers == 2
    assert cfg.num_target_layers == 8
    # The draft is always dense: DSA / hash routing / MTP are disabled.
    assert cfg.compress_ratios is None
    assert cfg.num_hash_layers == 0
    assert cfg.num_nextn_predict_layers == 0
    assert cfg.tie_word_embeddings is False
    assert cfg._attn_implementation == "sdpa"
    assert cfg.block_size == BLOCK_SIZE
    assert cfg.mask_token_id == 5
    assert cfg.target_layer_ids == TARGET_LAYER_IDS
    assert cfg.num_anchors == NUM_ANCHORS
    assert cfg.markov_rank == 16


def test_registry_resolves_deepseek_v4():
    spec = resolve_dspark_draft_spec(["DeepseekV4ForCausalLM"])
    assert spec.draft_cls is DeepseekV4DSparkModel


def test_rope_inv_freq_stays_fp32_after_bf16_cast():
    """``model.to(bf16)`` must keep the V4 rotary frequencies in fp32 (no DSA/YaRN here)."""
    model = _build_model()
    ref = model.rotary_emb.inv_freq.detach().clone().float()
    model = model.to(torch.bfloat16)
    inv_freq = model.rotary_emb.inv_freq
    assert inv_freq.dtype == torch.float32
    # Recomputed fresh in fp32, not a bf16 round-trip (bf16 rel. error ~1e-2).
    assert torch.allclose(inv_freq, ref, rtol=1e-6, atol=1e-8)
