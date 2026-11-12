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

"""Attention metadata conversion with document and padding boundaries."""

import pytest
import torch

from nemo_automodel.components.datasets.zephon.attention import adapt_model_inputs


def test_xdoc_flash_metadata_keeps_batch_documents_separate():
    from nemo_automodel.components.datasets.zephon.attention import (
        enable_xdoc_attention,
    )
    from nemo_automodel.components.models.common.packing import PackingCapabilities

    contract = PackingCapabilities(packed_mask_type="flash_varlen")
    ids = torch.tensor([[2, 3, 4, 5, 1], [6, 7, 8, 9, 1]])
    batch = adapt_model_inputs(
        {
            "input_ids": ids,
            "labels": ids.clone(),
            "position_ids": torch.tensor([[0, 1, 0, 1, 0], [0, 1, 0, 1, 0]]),
            "document_ids": torch.tensor([[1, 1, 2, 2, 0]] * 2),
            "padding_mask": torch.tensor([[False, False, False, False, True]]).expand(2, -1),
        },
    )
    assert batch["attention_document_ids"].tolist() == [[1, 1, 2, 2, 0]] * 2
    assert "cu_seq_lens_q" not in batch
    original_positions = batch["position_ids"]
    joined = enable_xdoc_attention(batch, packing_contract=contract)
    assert joined is batch
    assert joined["attention_document_ids"].tolist() == [[1, 1, 1, 1, 0]] * 2
    assert joined["document_ids"] is batch["document_ids"]
    assert joined["position_ids"].tolist() == [[0, 1, 2, 3, 0]] * 2
    assert original_positions.tolist() == [[0, 1, 0, 1, 0]] * 2


def test_xdoc_flash_allows_prefix_exclusions_but_rejects_internal_holes():
    from nemo_automodel.components.datasets.zephon.attention import (
        enable_xdoc_attention,
    )
    from nemo_automodel.components.models.common.packing import PackingCapabilities

    contract = PackingCapabilities("flash_varlen")
    batch = {
        "document_ids": torch.tensor([[0, 1, 1, 2, 2, 0]]),
        "position_ids": torch.tensor([[0, 0, 1, 0, 1, 0]]),
    }
    batch["attention_document_ids"] = batch["document_ids"]
    joined = enable_xdoc_attention(batch, packing_contract=contract)
    assert joined["attention_document_ids"].tolist() == [[0, 1, 1, 1, 1, 0]]
    batch["document_ids"] = torch.tensor([[0, 1, 1, 0, 2, 0]])
    batch["attention_document_ids"] = batch["document_ids"]
    with pytest.raises(ValueError, match="internal attention-excluded"):
        enable_xdoc_attention(batch, packing_contract=contract)


def test_xdoc_translates_all_rope_axes_together_without_counting_excluded_slots():
    from nemo_automodel.components.datasets.zephon.attention import (
        enable_xdoc_attention,
    )
    from nemo_automodel.components.models.common.packing import PackingCapabilities

    positions = torch.tensor([[[99, 0, 2, 77, 0, 1, 33]], [[99, 0, 0, 77, 0, 0, 33]], [[99, 0, 1, 77, 0, 1, 33]]])
    original = positions.clone()
    batch = {"document_ids": torch.tensor([[0, 1, 1, 0, 3, 3, 0]]), "position_ids": positions}
    batch["attention_document_ids"] = batch["document_ids"]
    joined = enable_xdoc_attention(batch, packing_contract=PackingCapabilities("block_causal"))
    expected = original.clone()
    expected[:, :, 4:6] += 3
    torch.testing.assert_close(joined["position_ids"], expected)
    torch.testing.assert_close(positions, original)
    assert joined["attention_document_ids"].tolist() == [[0, 1, 1, 0, 1, 1, 0]]


def test_dense_adapter_keeps_only_linear_metadata_until_model_entry():

    ids = torch.zeros(2, 1024, dtype=torch.long)
    positions = torch.arange(512).repeat(2).expand(2, -1)
    batch = adapt_model_inputs(
        {
            "input_ids": ids,
            "labels": ids,
            "position_ids": positions,
            "padding_mask": ids.bool(),
            "document_ids": torch.tensor([[1] * 512 + [2] * 512] * 2),
        },
    )
    assert "attention_mask" not in batch
    assert batch["attention_document_ids"].shape == ids.shape
    assert batch["attention_document_ids"] is batch["document_ids"]
    assert all(tensor.numel() <= ids.numel() for tensor in batch.values())


def test_xdoc_always_preserves_original_and_selected_boundaries():
    from nemo_automodel.components.datasets.zephon.attention import enable_xdoc_attention
    from nemo_automodel.components.models.common.packing import PackingCapabilities

    batch = {"document_ids": torch.tensor([[1, 1, 2, 2, 0]]), "position_ids": torch.tensor([[0, 1, 0, 1, 0]])}
    contract = PackingCapabilities("block_causal")
    batch["attention_document_ids"] = batch["document_ids"]
    assert enable_xdoc_attention(batch, packing_contract=contract)["attention_document_ids"].tolist() == [
        [1, 1, 1, 1, 0]
    ]
    batch["document_ids"] = torch.tensor([[0, 1, 1, 2, 0]])
    contract = PackingCapabilities("block_causal")
    batch["attention_document_ids"] = batch["document_ids"]
    assert enable_xdoc_attention(batch, packing_contract=contract)["attention_document_ids"].tolist() == [
        [0, 1, 1, 1, 0]
    ]


def test_processor_exclusions_do_not_destroy_original_document_membership():
    from nemo_automodel.components.datasets.zephon.attention import enable_xdoc_attention
    from nemo_automodel.components.models.common.packing import PackingCapabilities

    ids = torch.tensor([[1, 1, 2, 2, 3, 3, 0]])
    # Deliberately continuous: the adapter must use supplied membership, never
    # reconstruct it from model RoPE positions.
    positions = torch.arange(7)[None, :]
    batch = {"document_ids": ids, "position_ids": positions}
    eligible = torch.tensor([[False, False, True, True, True, True, False]])
    contract = PackingCapabilities("block_causal")
    result = adapt_model_inputs(batch, attention_eligibility=eligible)
    assert result is batch and result["document_ids"] is ids
    assert ids.tolist() == [[1, 1, 2, 2, 3, 3, 0]]
    assert result["attention_document_ids"].tolist() == [[0, 0, 2, 2, 3, 3, 0]]
    enable_xdoc_attention(result, packing_contract=contract)
    assert result["attention_document_ids"].tolist() == [[0, 0, 1, 1, 1, 1, 0]]
    assert result["document_ids"] is ids
    assert positions.tolist() == [list(range(7))]


def test_gathered_token_contract_is_supported():
    from nemo_automodel.components.datasets.zephon.attention import require_supported_packing_contract
    from nemo_automodel.components.models.common.packing import PackingCapabilities

    require_supported_packing_contract(PackingCapabilities("document_ids"))
    require_supported_packing_contract(PackingCapabilities("document_ids", return_packed_sequence_metadata=True))
