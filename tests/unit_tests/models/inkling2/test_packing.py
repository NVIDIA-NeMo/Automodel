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

"""Packed documents must match separate forwards, including every gradient."""

import copy

import pytest
import torch

from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
from nemo_automodel.components.models.inkling2.context_parallel import unpack_row_layout
from nemo_automodel.components.models.inkling2.model import Inkling2ForCausalLM
from tests.unit_tests.models.inkling2.test_pipeline import partition
from tests.unit_tests.models.inkling2.test_precision import initialize_nonuniform_queries
from tests.unit_tests.models.inkling2.test_reference import tiny_config

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("pipeline", [False, True])
def test_packed_logits_gradients_and_document_isolation(pipeline: bool) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(909)
    reference = Inkling2ForCausalLM(tiny_config())
    initialize_nonuniform_queries(reference)
    model = copy.deepcopy(reference)
    parts = partition(model, (0, 3, 6, 10, 13), ac=True) if pipeline else [model]
    # Inter-document padding and adjacent documents; PAD=EOS ID remains real in docs.
    ids = torch.randint(1, 37, (2, 16))
    ids[0, 3] = 6
    ids[:, 4:7] = 6
    lengths = torch.tensor([[4, 3, 6], [4, 3, 6]])
    padded = torch.tensor([[7, 3, 6], [7, 3, 6]])
    labels = ids.clone()
    labels[:, 4:7] = -100
    batch = dict(input_ids=ids, labels=labels, seq_lens=lengths, seq_lens_padded=padded, qkv_format="thd")
    _, batch = ContextParallelSharder(parts[0], None, batch).shard(batch)
    assert batch["padding_mask"][0, 3].item() is False
    assert bool(batch["padding_mask"][:, 4:7].all())
    batch.pop("labels")
    actual = parts[0](**batch)
    for part in parts[1:]:
        actual = part(*actual)
    keep = ~batch["padding_mask"]
    expected = torch.cat([reference(ids[:, a:b]) for a, b in ((0, 4), (7, 10), (10, 16))], dim=1)
    actual_real = actual[keep].reshape(2, 13, -1)
    torch.testing.assert_close(actual_real, expected, rtol=2e-5, atol=2e-6)
    upstream = torch.randn_like(expected[..., :37])
    actual_real[..., :37].backward(upstream)
    expected[..., :37].backward(upstream)
    wanted = dict(reference.named_parameters())
    from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

    for part in parts:
        for name, parameter in part.named_parameters():
            torch.testing.assert_close(
                parameter.grad, wanted[canonical_parameter_fqn(name)].grad, rtol=2e-4, atol=2e-5, msg=name
            )
    changed = ids.clone()
    changed[:, :7] = 2
    with torch.no_grad():
        changed_result = model(changed, seq_lens=lengths, seq_lens_padded=padded)
    torch.testing.assert_close(changed_result[:, 7:], actual[:, 7:], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["missing", "cover", "sentinel", "position", "labels"])
def test_reject_malformed_shared_pack_metadata(kind: str) -> None:
    batch = dict(
        input_ids=torch.ones(1, 8, dtype=torch.long),
        seq_lens=torch.tensor([[2, 3]]),
        seq_lens_padded=torch.tensor([[4, 4]]),
    )
    if kind == "missing":
        batch.pop("seq_lens")
    elif kind == "cover":
        batch["seq_lens_padded"][0, 0] = 3
    elif kind == "sentinel":
        batch["seq_lens"][0, 0] = -1000
    elif kind == "position":
        batch["position_ids"] = torch.arange(8)[None]
    else:
        batch["labels"] = torch.ones(1, 8, dtype=torch.long)
    with pytest.raises(ValueError, match="Inkling2"):
        unpack_row_layout(batch)


def test_text_pipeline_mask_placeholder_contract() -> None:
    model = Inkling2ForCausalLM(tiny_config())
    ids = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(model(ids, causal_mask_mapping={"full_attention": None}), model(ids), rtol=0, atol=0)
    with pytest.raises(ValueError, match="constructs its own masks"):
        model(ids, causal_mask_mapping={"full_attention": torch.ones(3, 3)})
