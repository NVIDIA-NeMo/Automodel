# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""A packed Gemma4-MoE batch must produce the same logits as its documents run alone.

Kept out of ``test_gemma4_model.py`` -- that module is CUDA-gated at module level and this contract
is observable on CPU -- but it reuses that module's config builder.
"""

import pytest
import torch

from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gemma4_moe.model import Gemma4ForConditionalGeneration
from tests.unit_tests.models.gemma4.test_gemma4_model import _make_gemma4_config

SEQ_LEN = 6


def _tiny_moe_model(attn_implementation: str) -> Gemma4ForConditionalGeneration:
    """Two-layer MoE Gemma4 -- one sliding, one full attention layer -- in fp32 on CPU."""
    config = _make_gemma4_config(
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=2,
        use_bidirectional_attention="vision",
        torch_dtype="float32",
    )
    config._attn_implementation = config.text_config._attn_implementation = attn_implementation
    backend = BackendConfig(linear="torch", attn="sdpa", rms_norm="torch", experts="torch", dispatcher="torch")
    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration(config, backend=backend)
    # Expert and router weights are allocated with ``torch.empty``; the production initializer fills them.
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    return model.eval()


def _reference_logits(model: Gemma4ForConditionalGeneration, input_ids: torch.Tensor) -> torch.Tensor:
    """Forward one document on its own, the result a packed forward must reproduce for it.

    Args:
        input_ids: Tensor of shape [1, document].

    Returns:
        Tensor of shape [1, document, vocab].
    """
    length = input_ids.shape[1]
    with torch.no_grad():
        return model(
            input_ids=input_ids,
            attention_mask=torch.ones(1, length, dtype=torch.long),
            position_ids=torch.arange(length)[None],
            mm_token_type_ids=torch.zeros(1, length, dtype=torch.long),
        ).logits


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("doc_lengths", [(SEQ_LEN,), (3, 3)], ids=["one-doc", "two-doc"])
def test_packed_batch_matches_documents_run_alone(attn_implementation, doc_lengths):
    """The compact-map batch the packed VLM collater emits must not change any document's logits.

    Under eager attention Transformers *adds* the mask to the logits, so the mask Gemma4 builds from
    ``_packed_seq_ids`` has to be additive; a bool mask is read as +1/+0 and masks nothing at all.
    """
    model = _tiny_moe_model(attn_implementation)
    torch.manual_seed(1)
    input_ids = torch.randint(1, 256, (1, SEQ_LEN))
    pack = {
        "input_ids": input_ids[0],
        "labels": input_ids[0],
        "attention_mask": torch.cat([torch.full((n,), i) for i, n in enumerate(doc_lengths, start=1)]),
        "position_ids": torch.cat([torch.arange(n) for n in doc_lengths]),
    }

    batch = neat_packed_vlm_collater([pack], max_length=SEQ_LEN, materialize_4d_mask=False)
    batch.pop("labels")
    assert batch["attention_mask"].shape == (1, SEQ_LEN), "expected the compact document map"

    with torch.no_grad():
        packed = model(**batch).logits
    reference = torch.cat([_reference_logits(model, doc) for doc in input_ids.split(doc_lengths, dim=1)], dim=1)

    torch.testing.assert_close(packed, reference, atol=1e-4, rtol=1e-4)
