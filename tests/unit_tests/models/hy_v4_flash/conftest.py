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

"""Tiny Hy4-flash configuration with every distinguishing feature enabled (CPU-sized), shared by the tests."""

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.hy_v4_flash.config import HYV4FlashConfig, HYV4FlashTextConfig
from nemo_automodel.components.models.hy_v4_flash.model import HYV4FlashForCausalLM

TINY_TEXT_KWARGS = dict(
    vocab_size=300,
    hidden_size=64,
    intermediate_size=96,
    moe_intermediate_size=32,
    routed_expert_hidden_size=24,
    num_hidden_layers=6,
    num_attention_heads=4,
    swa_num_attention_heads=8,
    n_routed_experts=8,
    num_experts_per_tok=2,
    q_lora_rank=32,
    kv_lora_rank=24,
    qk_nope_head_dim=16,
    qk_rope_head_dim=8,
    v_head_dim=12,
    swa_qk_nope_head_dim=16,
    swa_qk_rope_head_dim=8,
    swa_v_head_dim=12,
    sliding_window=6,
    index_topk=8,
    index_head_dim=24,
    # Sixteen heads make an exactly-zero group score (every head clipped by the ReLU) rare: such scores tie at the
    # selection boundary, where the reference's ``torch.topk`` leaves the choice unspecified.
    index_n_heads=16,
    compressed_indexer_compress_ratio=2,
    engram_layer_ids=[2, 4],
    engram_max_ngram_size=3,
    engram_n_head_per_ngram=2,
    engram_n_embed_per_ngram=16,
    engram_layer_head_vocab_sizes={str(i): [[1009, 1013], [1019, 1021]] for i in (2, 4)},
    engram_fp8_embedding=False,  # the PR config defaults to True; the released config.json sets False
    engram_pad_id=2,
    engram_lookup_table_size=290,
    rope_parameters={
        "sliding_attention": {"rope_type": "default", "rope_theta": 1.0e4, "mrope_section": [2, 1, 1]},
        "deepseek_sparse_attention": {"rope_type": "default", "rope_theta": 1.0e7, "mrope_section": [2, 1, 1]},
    },
    pad_token_id=2,
    bos_token_id=0,
    eos_token_id=1,
)

# Token ids the tests draw from (above the special ids, inside the tiny vocabulary) and the padding id.
TOKEN_RANGE = (3, 280)
PAD_ID = 2

CPU_BACKEND = BackendConfig(
    attn="sdpa", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch", rope_fusion=False
)


def tiny_text_config(**overrides: object) -> HYV4FlashTextConfig:
    """Six layers: sparse, sparse, swa, swa, swa, sparse; dense MLP first; Engram on layers 1 and 3 (ids 2, 4)."""
    return HYV4FlashTextConfig(**{**TINY_TEXT_KWARGS, **overrides})


def tiny_config(
    *, vision: dict[str, int] | None = None, image_token_id: int = 250, **overrides: object
) -> HYV4FlashConfig:
    """Composite config around :func:`tiny_text_config`; ``vision`` overrides the tiny vision tower."""
    vision_kwargs = dict(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        patch_size=4,
        spatial_merge_size=2,
        out_hidden_size=TINY_TEXT_KWARGS["hidden_size"],
        max_image_size=32,
        rms_norm_eps=1e-5,
    )
    vision_kwargs.update(vision or {})
    return HYV4FlashConfig(
        text_config=tiny_text_config(**overrides).to_dict(), vision_config=vision_kwargs, image_token_id=image_token_id
    )


def released_config() -> HYV4FlashConfig:
    """The fields the model reads from ``config.json`` of tencent/Hy4-flash at revision eccd3123, as released."""
    sparse, window = "deepseek_sparse_attention", "sliding_attention"
    rope = {"mrope_interleaved": True, "mrope_section": [12, 10, 10], "rope_type": "default"}
    text = dict(
        vocab_size=120832,
        hidden_size=3072,
        intermediate_size=13824,
        moe_intermediate_size=1536,
        routed_expert_hidden_size=1536,
        num_hidden_layers=42,
        num_attention_heads=32,
        swa_num_attention_heads=64,
        n_routed_experts=512,
        num_experts_per_tok=16,
        q_lora_rank=1024,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        swa_qk_nope_head_dim=192,
        swa_qk_rope_head_dim=64,
        swa_v_head_dim=256,
        sliding_window=512,
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=16,
        compressed_indexer_compress_ratio=4,
        hc_mult=4,
        mlp_layer_types=["dense"] + ["sparse"] * 41,
        layer_types=[sparse, *[sparse, window, window, window] * 10, sparse],
        rope_parameters={window: {**rope, "rope_theta": 10000}, sparse: {**rope, "rope_theta": 10000000}},
        engram_layer_ids=[2, 12, 22, 32],
        engram_max_ngram_size=3,
        engram_n_head_per_ngram=8,
        engram_n_embed_per_ngram=768,
        # The released moduli are 64 distinct primes just above 10,312,000; only their layout matters here.
        engram_layer_head_vocab_sizes={str(i): [[10312007] * 8, [10312177] * 8] for i in (2, 12, 22, 32)},
        engram_fp8_embedding=False,
    )
    return HYV4FlashConfig(text_config=text, image_token_id=120120)


# index_topk 16 with 4-token groups (the released ratio): three recalled groups plus the query's own group.
KPOOL, TOPK, WINDOW = 4, 16, 6
# Starts at every offset mod 4, documents of 1, 2 and 3 tokens, documents shorter and longer than window and top-k.
DOC_LENGTHS = (1, 2, 3, 21, 22, 23, 37, 5, 41)
TOLERANCE = dict(rtol=1e-4, atol=2e-5)  # fp32: same arithmetic, only the GEMM batch shapes (summation order) differ


def build_model() -> HYV4FlashForCausalLM:
    torch.manual_seed(0)
    config = tiny_config(
        vision={"num_hidden_layers": 0},
        index_topk=TOPK,
        compressed_indexer_compress_ratio=KPOOL,
        sliding_window=WINDOW,
    )
    model = HYV4FlashForCausalLM(config, backend=CPU_BACKEND).float()
    model.initialize_weights(dtype=torch.float32)
    return model


def build_plain_model() -> HYV4FlashForCausalLM:
    """The tiny model with its default top-k, pooling ratio and window."""
    torch.manual_seed(0)
    model = HYV4FlashForCausalLM(tiny_config(vision={"num_hidden_layers": 0}), backend=CPU_BACKEND).float().eval()
    model.initialize_weights(dtype=torch.float32)
    return model


def random_documents(lengths=DOC_LENGTHS, seed=1) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(*TOKEN_RANGE, (length,), generator=generator) for length in lengths]


def pack(documents: list[torch.Tensor], padding: int = 0) -> dict[str, torch.Tensor]:
    """One packed row: token ids, document ids (0 = padding) and document-local positions, each of shape [1, tokens]."""
    ids = torch.cat([*documents, torch.full((padding,), PAD_ID)])
    doc = torch.cat([*(torch.full((len(d),), i + 1) for i, d in enumerate(documents)), torch.zeros(padding).long()])
    positions = torch.cat([*(torch.arange(len(d)) for d in documents), torch.zeros(padding).long()])
    return dict(
        input_ids=ids[None], packed_seq_ids=doc[None], position_ids=positions[None], attention_mask=doc[None] > 0
    )


def run_packed(
    model: HYV4FlashForCausalLM, documents: list[torch.Tensor], padding: int = 0, **overrides: torch.Tensor
) -> list[torch.Tensor]:
    """Logits of every document of the pack, each of shape [document_tokens, vocab]."""
    batch = {**pack(documents, padding), **overrides}
    logits = model(**batch).logits[0]
    return list(logits[: sum(map(len, documents))].split([len(d) for d in documents]))


def run_alone(model: HYV4FlashForCausalLM, document: torch.Tensor) -> torch.Tensor:
    """Logits of one unpacked document, of shape [document_tokens, vocab] (the independent oracle)."""
    return model(document[None]).logits[0]


def assert_documents_match_alone(
    model: HYV4FlashForCausalLM, documents: list[torch.Tensor], logits: list[torch.Tensor]
) -> None:
    """Compare per-document logits (each [document_tokens, vocab]) with every document run alone and unpacked."""
    with torch.no_grad():
        for index, (document, got) in enumerate(zip(documents, logits, strict=True)):
            torch.testing.assert_close(
                got, run_alone(model, document), msg=lambda m, index=index: f"document {index}: {m}", **TOLERANCE
            )


def assert_packed_matches_alone(
    model: HYV4FlashForCausalLM, documents: list[torch.Tensor], padding: int = 0, **overrides: torch.Tensor
) -> None:
    """Run the documents as one pack and compare each with the same document run alone."""
    with torch.no_grad():
        packed = run_packed(model, documents, padding, **overrides)
    assert_documents_match_alone(model, documents, packed)


def complex_rotation(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Complex rotation of interleaved pairs, independent of the production helper.

    Args:
        x: Tensor of shape [tokens, rope_dim].
        positions: Integer tensor of shape [tokens].
        theta: Rotary base.

    Returns:
        Tensor of shape [tokens, rope_dim].
    """
    half = x.shape[-1] // 2
    angles = positions[:, None].double() * theta ** (-torch.arange(half).double() * 2 / x.shape[-1])
    rotated = torch.view_as_complex(x.double().reshape(-1, half, 2).contiguous()) * torch.polar(
        torch.ones_like(angles), angles
    )
    return torch.view_as_real(rotated).reshape(x.shape).float()
