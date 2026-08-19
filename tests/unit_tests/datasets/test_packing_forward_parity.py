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

"""Forward-pass parity for neat packing: no cross-document attention leakage.

The block-causal (sdpa) path is the reference behaviour the flash cu_seqlens path
must reproduce. This test proves the reference has no leakage -- a packed row's
per-document logits equal each document run alone -- which, combined with the
mask-equivalence test in ``test_packed_seq.py`` (flash cu_seqlens segmentation ==
sdpa block-causal mask), pins the flash path to the same isolation. It also
covers the sdpa 4D-mask path, where HuggingFace passes the prebuilt block-causal
mask through natively.
"""

import importlib.util

import pytest
import torch

transformers = pytest.importorskip("transformers")

from nemo_automodel.components.datasets.utils import neat_packed_collater  # noqa: E402

# Gating mirrors tests/functional_tests/speculative/test_eagle3_packing_fa2_parity.py
_HAS_FA = torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None


def _tiny_causal_lm():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attn_implementation="sdpa",
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).eval()
    return model


def test_sdpa_block_causal_has_no_cross_document_leakage():
    """Packed per-document logits must equal each document decoded alone."""
    model = _tiny_causal_lm()

    doc_a = [5, 9, 13, 21]  # length 4
    doc_b = [7, 3, 11]  # length 3
    packed_ids = doc_a + doc_b
    seq = len(packed_ids)

    # One packed row through the sdpa neat collater -> 4D block-causal mask +
    # per-document restarting position_ids (mirrors _build_packed_sample).
    packed_sample = {
        "input_ids": torch.tensor(packed_ids),
        "labels": torch.tensor(packed_ids),
        "attention_mask": torch.tensor([1, 1, 1, 1, 2, 2, 2]),
        "position_ids": torch.tensor([0, 1, 2, 3, 0, 1, 2]),
    }
    batch = neat_packed_collater([packed_sample], attn_implementation="sdpa")
    assert batch["attention_mask"].shape == (1, 1, seq, seq)

    with torch.no_grad():
        packed_logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
        ).logits[0]

        ref_a = model(input_ids=torch.tensor([doc_a])).logits[0]
        ref_b = model(input_ids=torch.tensor([doc_b])).logits[0]

    # Each document's packed logits must match decoding that document in isolation:
    # any cross-document attention would perturb these.
    torch.testing.assert_close(packed_logits[: len(doc_a)], ref_a, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(packed_logits[len(doc_a) :], ref_b, rtol=1e-4, atol=1e-4)


def test_sdpa_padded_pack_has_no_cross_document_leakage():
    """A pack with trailing padding must still isolate documents (padded packs)."""
    model = _tiny_causal_lm()

    doc_a = [5, 9, 13, 21]  # length 4
    doc_b = [7, 3, 11]  # length 3
    pad = 2  # trailing padding tokens
    packed_ids = doc_a + doc_b + [0] * pad
    packed_sample = {
        "input_ids": torch.tensor(packed_ids),
        "labels": torch.tensor(doc_a + doc_b + [-100] * pad),
        "attention_mask": torch.tensor([1, 1, 1, 1, 2, 2, 2, 0, 0]),
        "position_ids": torch.tensor([0, 1, 2, 3, 0, 1, 2, 0, 0]),
    }
    batch = neat_packed_collater([packed_sample], attn_implementation="sdpa")

    with torch.no_grad():
        packed_logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
        ).logits[0]
        ref_a = model(input_ids=torch.tensor([doc_a])).logits[0]
        ref_b = model(input_ids=torch.tensor([doc_b])).logits[0]

    # Real-document logits are unaffected by the trailing padding block.
    torch.testing.assert_close(packed_logits[: len(doc_a)], ref_a, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(packed_logits[len(doc_a) : len(doc_a) + len(doc_b)], ref_b, rtol=1e-4, atol=1e-4)


def test_sdpa_batch_gt_1_uneven_padding_has_no_cross_document_leakage():
    """Batch > 1 with uneven padding per row must isolate every document."""
    model = _tiny_causal_lm()

    doc_a = [5, 9, 13]  # row 0, doc 1
    doc_b = [7, 3]  # row 0, doc 2  -> row 0 has 1 padding token
    doc_c = [11, 4, 8, 2]  # row 1, doc 1 -> row 1 has 2 padding tokens
    # Rows are padded to a common length of 6 with uneven real content.
    row0 = {
        "input_ids": torch.tensor(doc_a + doc_b + [0]),
        "labels": torch.tensor(doc_a + doc_b + [-100]),
        "attention_mask": torch.tensor([1, 1, 1, 2, 2, 0]),
        "position_ids": torch.tensor([0, 1, 2, 0, 1, 0]),
    }
    row1 = {
        "input_ids": torch.tensor(doc_c + [0, 0]),
        "labels": torch.tensor(doc_c + [-100, -100]),
        "attention_mask": torch.tensor([1, 1, 1, 1, 0, 0]),
        "position_ids": torch.tensor([0, 1, 2, 3, 0, 0]),
    }
    batch = neat_packed_collater([row0, row1], attn_implementation="sdpa")
    assert batch["input_ids"].shape == (2, 6)

    with torch.no_grad():
        packed_logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
        ).logits
        ref_a = model(input_ids=torch.tensor([doc_a])).logits[0]
        ref_b = model(input_ids=torch.tensor([doc_b])).logits[0]
        ref_c = model(input_ids=torch.tensor([doc_c])).logits[0]

    torch.testing.assert_close(packed_logits[0, : len(doc_a)], ref_a, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(packed_logits[0, len(doc_a) : len(doc_a) + len(doc_b)], ref_b, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(packed_logits[1, : len(doc_c)], ref_c, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _HAS_FA, reason="requires flash-attn")
@pytest.mark.parametrize("pad", [0, 2], ids=["unpadded", "padded"])
def test_flash_varlen_has_no_cross_document_leakage_on_gpu(pad):
    """The flash cu_seqlens path must isolate documents in both padded and unpadded
    packs: packed per-doc logits equal each document decoded alone.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).to("cuda", torch.bfloat16).eval()
    model.config._attn_implementation = "flash_attention_2"

    doc_a = [5, 9, 13, 21]
    doc_b = [7, 3, 11]
    packed_sample = {
        "input_ids": torch.tensor(doc_a + doc_b + [0] * pad),
        "labels": torch.tensor(doc_a + doc_b + [-100] * pad),
        "attention_mask": torch.tensor([1, 1, 1, 1, 2, 2, 2] + [0] * pad),
        "position_ids": torch.tensor([0, 1, 2, 3, 0, 1, 2] + [0] * pad),
    }
    flash_batch = neat_packed_collater([packed_sample], attn_implementation="flash_attention_2")

    to_cuda = lambda t: t.to("cuda") if isinstance(t, torch.Tensor) else t
    with torch.no_grad():
        packed_logits = model(
            input_ids=to_cuda(flash_batch["input_ids"]),
            position_ids=to_cuda(flash_batch["position_ids"]),
            cu_seq_lens_q=to_cuda(flash_batch["cu_seq_lens_q"]),
            cu_seq_lens_k=to_cuda(flash_batch["cu_seq_lens_k"]),
            max_length_q=flash_batch["max_length_q"],
            max_length_k=flash_batch["max_length_k"],
        ).logits[0]

        # Reference: each document decoded alone as a plain causal flash forward.
        ref_a = model(input_ids=torch.tensor([doc_a], device="cuda")).logits[0]
        ref_b = model(input_ids=torch.tensor([doc_b], device="cuda")).logits[0]

    torch.testing.assert_close(packed_logits[: len(doc_a)].float(), ref_a.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        packed_logits[len(doc_a) : len(doc_a) + len(doc_b)].float(), ref_b.float(), rtol=2e-2, atol=2e-2
    )


@pytest.mark.skipif(not _HAS_FA, reason="requires flash-attn")
def test_flash_varlen_batch_gt_1_uneven_padding_no_leakage_on_gpu():
    """Flash varlen must isolate every document across a batch > 1 with uneven padding."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).to("cuda", torch.bfloat16).eval()
    model.config._attn_implementation = "flash_attention_2"

    doc_a = [5, 9, 13]  # row 0, doc 1
    doc_b = [7, 3]  # row 0, doc 2 -> row 0 keeps 1 padding token
    doc_c = [11, 4, 8, 2]  # row 1, doc 1 -> row 1 keeps 2 padding tokens
    row0 = {
        "input_ids": torch.tensor(doc_a + doc_b + [0]),
        "labels": torch.tensor(doc_a + doc_b + [-100]),
        "attention_mask": torch.tensor([1, 1, 1, 2, 2, 0]),
        "position_ids": torch.tensor([0, 1, 2, 0, 1, 0]),
    }
    row1 = {
        "input_ids": torch.tensor(doc_c + [0, 0]),
        "labels": torch.tensor(doc_c + [-100, -100]),
        "attention_mask": torch.tensor([1, 1, 1, 1, 0, 0]),
        "position_ids": torch.tensor([0, 1, 2, 3, 0, 0]),
    }
    flash_batch = neat_packed_collater([row0, row1], attn_implementation="flash_attention_2")

    to_cuda = lambda t: t.to("cuda") if isinstance(t, torch.Tensor) else t
    with torch.no_grad():
        packed_logits = model(
            input_ids=to_cuda(flash_batch["input_ids"]),
            position_ids=to_cuda(flash_batch["position_ids"]),
            cu_seq_lens_q=to_cuda(flash_batch["cu_seq_lens_q"]),
            cu_seq_lens_k=to_cuda(flash_batch["cu_seq_lens_k"]),
            max_length_q=flash_batch["max_length_q"],
            max_length_k=flash_batch["max_length_k"],
        ).logits
        ref_a = model(input_ids=torch.tensor([doc_a], device="cuda")).logits[0]
        ref_b = model(input_ids=torch.tensor([doc_b], device="cuda")).logits[0]
        ref_c = model(input_ids=torch.tensor([doc_c], device="cuda")).logits[0]

    torch.testing.assert_close(packed_logits[0, : len(doc_a)].float(), ref_a.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        packed_logits[0, len(doc_a) : len(doc_a) + len(doc_b)].float(), ref_b.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(packed_logits[1, : len(doc_c)].float(), ref_c.float(), rtol=2e-2, atol=2e-2)


def _build_tiny_qwen3_vl():
    """Build a tiny text-only Qwen3-VL for the packed FlashAttention parity check.

    Returns None if the Qwen3-VL classes are unavailable in the installed
    Transformers, so the release matrix (which auto-discovers the real config) is
    the source of truth for full-size coverage.
    """
    try:
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLTextConfig
    except ImportError:
        return None

    text_config = Qwen3VLTextConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rope_scaling={"type": "default", "mrope_section": [4, 2, 2]},
    )
    config = Qwen3VLConfig(text_config=text_config)
    for key in ("hidden_size", "intermediate_size", "depth", "num_heads", "out_hidden_size"):
        if hasattr(config.vision_config, key):
            setattr(config.vision_config, key, 2 if key == "depth" else 64)
    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration(config).to("cuda", torch.bfloat16).eval()
    model.config._attn_implementation = "flash_attention_2"
    return model


@pytest.mark.skipif(not _HAS_FA, reason="requires flash-attn")
def test_qwen3_vl_packed_flash_varlen_no_leakage_on_gpu():
    """The Qwen3-VL packed FlashAttention path (mRoPE, text-only) must isolate documents."""
    from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater

    model = _build_tiny_qwen3_vl()
    if model is None:
        pytest.skip("Qwen3-VL is unavailable in the installed transformers")

    doc_a = [5, 9, 13, 21]
    doc_b = [7, 3, 11]
    # Text-only mRoPE position ids: [3, sequence], restarting per document.
    packed_sample = {
        "input_ids": torch.tensor(doc_a + doc_b),
        "labels": torch.tensor(doc_a + doc_b),
        "attention_mask": torch.tensor([1, 1, 1, 1, 2, 2, 2]),
        "position_ids": torch.tensor([[0, 1, 2, 3, 0, 1, 2]] * 3),
    }
    flash_batch = neat_packed_vlm_collater([packed_sample], attn_implementation="flash_attention_2")
    assert flash_batch["position_ids"].shape == (3, 1, 7)  # mRoPE [3, batch, sequence]

    to_cuda = lambda t: t.to("cuda") if isinstance(t, torch.Tensor) else t
    with torch.no_grad():
        packed_logits = model(
            input_ids=to_cuda(flash_batch["input_ids"]),
            position_ids=to_cuda(flash_batch["position_ids"]),
            cu_seq_lens_q=to_cuda(flash_batch["cu_seq_lens_q"]),
            cu_seq_lens_k=to_cuda(flash_batch["cu_seq_lens_k"]),
            max_length_q=flash_batch["max_length_q"],
            max_length_k=flash_batch["max_length_k"],
        ).logits[0]
        # References let the model derive per-document mRoPE positions (0..len-1).
        ref_a = model(input_ids=torch.tensor([doc_a], device="cuda")).logits[0]
        ref_b = model(input_ids=torch.tensor([doc_b], device="cuda")).logits[0]

    torch.testing.assert_close(packed_logits[: len(doc_a)].float(), ref_a.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(packed_logits[len(doc_a) :].float(), ref_b.float(), rtol=2e-2, atol=2e-2)
