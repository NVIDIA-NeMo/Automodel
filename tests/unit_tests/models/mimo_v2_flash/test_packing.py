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

"""Packed-sequence parity coverage for the MiMo V2.6 text and VLM paths."""

from __future__ import annotations

import pytest
import torch

from nemo_automodel.components.datasets.utils import (
    neat_packed_collater,
    pack_features_for_thd,
    packed_sequence_thd_collater,
)
from nemo_automodel.components.datasets.vlm.collate_fns import (
    neat_packed_vlm_collater,
    packed_sequence_thd_vlm_collater,
)
from nemo_automodel.components.datasets.vlm.neat_packing_vlm import (
    _build_packed_vlm_sample,
    _shift_sample,
)
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mimo_v2_flash.config import MiMoV2Config
from nemo_automodel.components.models.mimo_v2_flash.model import MiMoV2ForCausalLM

_DOCS = (
    torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long),
    torch.tensor([7, 8, 9, 10, 11, 12], dtype=torch.long),
)
_CHANGED_FIRST_DOC = torch.tensor([20, 21, 22, 23, 24, 25], dtype=torch.long)


def _vision_config() -> dict:
    return {
        "depth": 1,
        "fullatt_block_indexes": [0],
        "hidden_act": "silu",
        "hidden_size": 8,
        "in_chans": 3,
        "intermediate_size": 16,
        "num_heads": 2,
        "num_key_value_heads": 1,
        "qk_channels": 4,
        "out_hidden_size": 16,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 1,
        "use_sink": False,
        "visual_token_window_size": 2,
        "vit_window_attn_types": [-1],
    }


def _model_config(*, vision: bool) -> MiMoV2Config:
    return MiMoV2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        v_head_dim=4,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=4,
        swa_v_head_dim=4,
        hybrid_layer_pattern=[1],
        moe_layer_freq=[0],
        n_routed_experts=None,
        partial_rotary_factor=1.0,
        sliding_window=2,
        attention_dropout=0.0,
        add_full_attention_sink_bias=False,
        add_swa_attention_sink_bias=False,
        torch_dtype="float32",
        vision_config=_vision_config() if vision else None,
        image_token_id=62 if vision else None,
        video_token_id=63 if vision else None,
    )


def _backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )


def _make_model(*, vision: bool) -> MiMoV2ForCausalLM:
    torch.manual_seed(1234)
    model = MiMoV2ForCausalLM(_model_config(vision=vision), backend=_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    return model.train()


def _model_kwargs(batch: dict, *, image_embeds: torch.Tensor | None = None) -> dict:
    keys = (
        "input_ids",
        "position_ids",
        "attention_mask",
        "_packed_seq_ids",
        "seq_lens",
        "seq_lens_padded",
        "qkv_format",
    )
    kwargs = {key: batch[key] for key in keys if key in batch}
    if image_embeds is not None:
        kwargs["image_embeds"] = image_embeds
    return kwargs


def _packed_llm_batch(documents: tuple[torch.Tensor, torch.Tensor], packing_format: str) -> dict:
    if packing_format == "thd":
        record = pack_features_for_thd([{"input_ids": doc.tolist(), "labels": doc.tolist()} for doc in documents])
        return packed_sequence_thd_collater([record])

    lengths = [doc.numel() for doc in documents]
    item = {
        "input_ids": torch.cat(documents),
        "labels": torch.cat(documents),
        "position_ids": torch.cat([torch.arange(length) for length in lengths]),
        "attention_mask": torch.cat(
            [torch.full((length,), doc_id, dtype=torch.long) for doc_id, length in enumerate(lengths, start=1)]
        ),
    }
    return neat_packed_collater([item], attn_implementation="sdpa")


def _reference_text_logits(model: MiMoV2ForCausalLM, documents: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    outputs = []
    for doc in documents:
        outputs.append(
            model(
                input_ids=doc.unsqueeze(0),
                position_ids=torch.arange(doc.numel()).unsqueeze(0),
            ).logits
        )
    return torch.cat(outputs, dim=1)


def _q_proj_grad(model: MiMoV2ForCausalLM) -> torch.Tensor:
    grad = model.model.layers["0"].self_attn.q_proj.weight.grad
    assert grad is not None
    return grad.detach().clone()


@pytest.mark.parametrize("packing_format", ["neat"])
def test_llm_packed_cp1_matches_independent_forward_gradient_and_isolation(packing_format: str):
    """Both packed formats must preserve document-local SWA in forward and backward."""
    reference_model = _make_model(vision=False)
    packed_model = _make_model(vision=False)
    packed_model.load_state_dict(reference_model.state_dict())

    reference_logits = _reference_text_logits(reference_model, _DOCS)
    reference_logits.float().square().mean().backward()
    reference_grad = _q_proj_grad(reference_model)

    batch = _packed_llm_batch(_DOCS, packing_format)
    packed_logits = packed_model(**_model_kwargs(batch)).logits
    packed_logits.float().square().mean().backward()
    packed_grad = _q_proj_grad(packed_model)

    torch.testing.assert_close(packed_logits, reference_logits, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(packed_grad, reference_grad, atol=3e-6, rtol=3e-5)

    changed = _packed_llm_batch((_CHANGED_FIRST_DOC, _DOCS[1]), packing_format)
    with torch.no_grad():
        changed_logits = packed_model(**_model_kwargs(changed)).logits
    second_start = _DOCS[0].numel()
    torch.testing.assert_close(
        changed_logits[:, second_start:],
        packed_logits.detach()[:, second_start:],
        atol=0,
        rtol=0,
    )


def _raw_vlm_sample(token_ids: list[int], pixel_offset: float) -> dict:
    ids = torch.tensor(token_ids, dtype=torch.long)
    return {
        "input_ids": ids,
        "labels": ids.clone(),
        "attention_mask": torch.ones_like(ids),
        "pixel_values": torch.arange(4 * 12, dtype=torch.float32).reshape(4, 12) / 50 + pixel_offset,
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
    }


def _packed_vlm_batch(samples: tuple[dict, dict], packing_format: str) -> tuple[dict, tuple[dict, dict]]:
    shifted = tuple(_shift_sample(sample) for sample in samples)
    item = _build_packed_vlm_sample(list(shifted), pack_size=64, padding_idx=0)
    if packing_format == "thd":
        batch = packed_sequence_thd_vlm_collater([item])
    else:
        batch = neat_packed_vlm_collater([item], attn_implementation="sdpa")
    return batch, shifted


def _reference_vlm_logits(
    model: MiMoV2ForCausalLM,
    shifted: tuple[dict, dict],
    image_embeds: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    for index, sample in enumerate(shifted):
        ids = sample["input_ids"].unsqueeze(0)
        outputs.append(
            model(
                input_ids=ids,
                position_ids=torch.arange(ids.shape[1]).unsqueeze(0),
                image_embeds=image_embeds[index : index + 1],
            ).logits
        )
    return torch.cat(outputs, dim=1)


@pytest.mark.parametrize("packing_format", ["neat"])
def test_vlm_packed_cp1_matches_independent_forward_gradient_and_isolation(packing_format: str):
    """Two packed image documents must keep media order, boundaries, and the SWA window."""
    samples = (
        _raw_vlm_sample([1, 62, 3, 4, 5, 6, 7], 0.0),
        _raw_vlm_sample([8, 62, 9, 10, 11, 12, 13], 2.0),
    )
    batch, shifted = _packed_vlm_batch(samples, packing_format)
    assert int((batch["input_ids"] == 62).sum()) == 2
    assert batch["image_grid_thw"].shape == (2, 3)
    assert batch["pixel_values"].shape[0] == 8
    if packing_format == "neat":
        assert batch["n_images_per_sample"].tolist() == [2]

    reference_model = _make_model(vision=True)
    packed_model = _make_model(vision=True)
    packed_model.load_state_dict(reference_model.state_dict())
    image_values = torch.arange(2 * 16, dtype=torch.float32).reshape(2, 16) / 20
    reference_images = image_values.clone().requires_grad_()
    packed_images = image_values.clone().requires_grad_()

    reference_logits = _reference_vlm_logits(reference_model, shifted, reference_images)
    reference_logits.float().square().mean().backward()
    reference_grad = _q_proj_grad(reference_model)
    assert reference_images.grad is not None

    packed_logits = packed_model(**_model_kwargs(batch, image_embeds=packed_images)).logits
    packed_logits.float().square().mean().backward()
    packed_grad = _q_proj_grad(packed_model)
    assert packed_images.grad is not None

    torch.testing.assert_close(packed_logits, reference_logits, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(packed_grad, reference_grad, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(packed_images.grad, reference_images.grad, atol=3e-6, rtol=3e-5)

    changed_samples = (
        _raw_vlm_sample([20, 62, 21, 22, 23, 24, 25], 7.0),
        samples[1],
    )
    changed_batch, _ = _packed_vlm_batch(changed_samples, packing_format)
    changed_images = packed_images.detach().clone()
    changed_images[0].add_(4.0)
    with torch.no_grad():
        changed_logits = packed_model(**_model_kwargs(changed_batch, image_embeds=changed_images)).logits
    second_start = shifted[0]["input_ids"].numel()
    torch.testing.assert_close(
        changed_logits[:, second_start:],
        packed_logits.detach()[:, second_start:],
        atol=0,
        rtol=0,
    )
