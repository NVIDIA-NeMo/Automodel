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

"""Native VLM packing preserves media and uses compact model-owned masks."""

import pytest
import torch

from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater
from nemo_automodel.components.datasets.vlm.neat_packing_vlm import _build_packed_vlm_sample, _shift_sample
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.packing import PackingCapabilities, configure_packing_for_models
from nemo_automodel.components.models.step5.configuration import Step5Config, Step5VLConfig
from nemo_automodel.components.models.step5.model import Step5ForConditionalGeneration


def test_native_vlm_packing_contract() -> None:
    text = Step5Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=192,
        vocab_size=128,
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=16,
        share_expert_dim=16,
        moe_layer_list=[3],
        num_nextn_predict_layers=0,
        sparse_config=dict(
            proxy_dim=16,
            sparse_indexer_rope_dim=4,
            sparse_indexer_num_heads=4,
            num_provider_groups=2,
            topk=2,
            region_block_size=8,
        ),
    )
    with torch.device("meta"):
        model = Step5ForConditionalGeneration(
            Step5VLConfig(
                text_config=text,
                vision_config=dict(width=16, layers=1, heads=2, image_size=112, patch_size=14, mlp_ratio=2),
                image_token_id=127,
            ),
            backend=BackendConfig(attn="fa4", linear="torch", dispatcher="torch", experts="torch_mm"),
        )
    packing = configure_packing_for_models([model])
    assert packing.packed_mask_type == "document_ids"
    assert not packing.requires_packed_sequence_metadata  # Model uses reset positions directly.


def test_native_vlm_packing_preserves_crops() -> None:
    samples = []
    for index, count in enumerate((2, 0, 1)):
        ids = torch.arange(13 + index)
        sample = dict(
            input_ids=ids,
            labels=ids.clone(),
            attention_mask=torch.ones_like(ids),
            pixel_values=torch.full((1, 3, 112, 112), float(index)),
            num_patches=torch.tensor([count]),
        )
        if count:
            sample["patch_pixel_values"] = torch.full((count, 3, 56, 56), float(index + 10))
        samples.append(sample)
    shifted = [_shift_sample(sample) for sample in samples]
    packed = _build_packed_vlm_sample(shifted, 64, 0)
    batch = neat_packed_vlm_collater([packed], packing=PackingCapabilities(packed_mask_type="document_ids"))
    assert "patch_pixel_values" in batch
    assert "num_patches" in batch
    assert batch["num_patches"].tolist() == [2, 0, 1]
    torch.testing.assert_close(batch["pixel_values"][:, 0, 0, 0].float(), torch.tensor([0.0, 1.0, 2.0]))
    torch.testing.assert_close(batch["patch_pixel_values"][:, 0, 0, 0].float(), torch.tensor([10.0, 10.0, 12.0]))
    offset = 0
    for sample in samples:
        length = sample["input_ids"].numel() - 1
        assert torch.equal(batch["labels"][0, offset : offset + length], sample["labels"][1:])
        assert torch.equal(batch["position_ids"][0, offset : offset + length], torch.arange(length))
        offset += length
    assert batch["attention_mask"].ndim == 2


@pytest.mark.runtime_budget(20, reason="Four-layer vision/text eager oracle and independent image-gradient isolation")
def test_vlm_packed_fp32_independent_documents_and_gradients() -> None:
    """Packing must match independent documents and isolate image gradients."""
    torch.manual_seed(197)
    text = Step5Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=12,
        vocab_size=128,
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=16,
        share_expert_dim=16,
        moe_layer_list=[],
        dtype="float32",
        num_nextn_predict_layers=0,
        sliding_window=13,
        sparse_config=dict(
            proxy_dim=16,
            sparse_indexer_rope_dim=4,
            sparse_indexer_num_heads=4,
            num_provider_groups=2,
            topk=2,
            region_block_size=8,
        ),
    )
    model = Step5ForConditionalGeneration(
        Step5VLConfig(
            text_config=text,
            vision_config=dict(width=16, layers=1, heads=2, image_size=112, patch_size=14, mlp_ratio=2),
            image_token_id=127,
        ),
        backend=BackendConfig(attn="eager", linear="torch", dispatcher="torch", experts="torch_mm"),
    )
    samples = []
    for length, crops in ((19, 1), (83, 0)):
        ids = torch.randint(1, 126, (length + 1,))
        ids[3 : 7 + crops] = 127
        labels = ids.clone()
        labels[labels == 127] = -100
        item = dict(
            input_ids=ids,
            labels=labels,
            attention_mask=torch.ones_like(ids),
            pixel_values=torch.randn(1, 3, 112, 112),
            num_patches=torch.tensor([crops]),
        )
        if crops:
            item["patch_pixel_values"] = torch.randn(crops, 3, 56, 56)
        samples.append(item)
    packing = configure_packing_for_models([model])

    def batch_for(documents):
        return neat_packed_vlm_collater(
            [_build_packed_vlm_sample([_shift_sample(s) for s in documents], 128, 0)], packing=packing
        )

    def evaluate(batch):
        return model(**{k: v for k, v in batch.items() if k != "labels"}).logits

    batch = batch_for(samples)
    batch["pixel_values"] = batch["pixel_values"].float().requires_grad_()
    batch["patch_pixel_values"] = batch["patch_pixel_values"].float().requires_grad_()
    packed = evaluate(batch)
    separate = torch.cat([evaluate(batch_for([s])) for s in samples], dim=1)
    torch.testing.assert_close(packed, separate, rtol=2e-5, atol=2e-6)
    packed[:, 19:].square().sum().backward()
    assert torch.count_nonzero(batch["pixel_values"].grad[0]) == 0
    assert torch.count_nonzero(batch["patch_pixel_values"].grad) == 0
    assert batch["pixel_values"].grad[1].abs().sum() > 0
    with torch.no_grad():
        changed = dict(batch)
        changed["pixel_values"] = batch["pixel_values"].detach().clone()
        changed["pixel_values"][0].mul_(-3)
        changed["patch_pixel_values"] = -batch["patch_pixel_values"].detach()
        torch.testing.assert_close(evaluate(changed)[:, 19:], packed[:, 19:], rtol=0, atol=0)
        missing = dict(batch)
        missing.pop("position_ids")
        with pytest.raises(ValueError, match="packed"):
            evaluate(missing)
        wrong = dict(batch)
        wrong["position_ids"] = torch.arange(102)[None]
        with pytest.raises(ValueError, match="document"):
            evaluate(wrong)
        duplicate = dict(batch)
        duplicate["position_ids"] = batch["position_ids"].clone()
        duplicate["position_ids"][0, 21] = duplicate["position_ids"][0, 20]
        with pytest.raises(ValueError, match="document"):
            evaluate(duplicate)
        # Removing both independent boundary sources simulates an upstream bug.
        wrong.pop("_packed_seq_ids")
        wrong["attention_mask"] = wrong["attention_mask"].bool()
        assert (evaluate(wrong)[:, 19:] - packed[:, 19:]).abs().max() > 1e-4
