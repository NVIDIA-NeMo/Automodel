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

"""Tiny Hy4-flash configuration with every distinguishing feature enabled (CPU-sized)."""

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.hy_v4_flash.config import HYV4FlashConfig, HYV4FlashTextConfig
from nemo_automodel.components.models.hy_v4_flash.model import HYV4FlashForCausalLM
from tests.unit_tests.models.hy_v4_flash.tiny_kwargs import TINY_TEXT_KWARGS

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
