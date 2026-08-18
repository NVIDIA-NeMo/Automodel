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
guards that removing the global ``create_causal_mask`` monkeypatch did not break
the sdpa 4D-mask path (HuggingFace must pass a prebuilt 4D mask through natively).
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from nemo_automodel.components.datasets.utils import neat_packed_collater  # noqa: E402


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


def _flash_attention_available() -> bool:
    """Return True iff a flash-attention backend (flash_attn or kernels) can run FA2."""
    if not torch.cuda.is_available():
        return False
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
    )
    try:
        m = LlamaForCausalLM(cfg).to("cuda", torch.bfloat16).eval()
        m.config._attn_implementation = "flash_attention_2"
        with torch.no_grad():
            m(input_ids=torch.randint(0, 32, (1, 4), device="cuda"))
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _flash_attention_available(), reason="flash-attention backend unavailable")
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
