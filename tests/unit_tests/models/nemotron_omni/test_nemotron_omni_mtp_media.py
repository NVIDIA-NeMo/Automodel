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

"""CPU regression for Omni's fused media inputs to context-parallel MTP.

The wrapper, backbone, MTP fusion/MLP, LM head, CE, and backward are real modules.
Synthetic encoder outputs isolate media-token placement from vision accuracy.
Only the TE partition lookup is replaced with an independent integer oracle;
these tests do not certify distributed attention or checkpoint loading.
"""

import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.nemotron_omni.model import NemotronOmniForConditionalGeneration
from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_mtp import MockNemotronV3Config

pytestmark = pytest.mark.cpu_only

IMAGE = 18
SOUND = 27
IGNORE = -100
HIDDEN = 64


@dataclass
class _CPMesh:
    rank: int

    def size(self) -> int:
        return 2

    def get_local_rank(self) -> int:
        return self.rank


class _FeatureEncoder(nn.Module):
    """Expose differentiable synthetic encoder outputs and count executions."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return features


class _TinyOmni(NemotronOmniForConditionalGeneration):
    """Use native pointwise decoder blocks so CPU tests isolate future inputs."""

    def __init__(self, depths: int, rank: int):
        nn.Module.__init__(self)
        torch.manual_seed(7)
        config = MockNemotronV3Config(
            num_hidden_layers=1,
            layers_block_type=["mlp"],
            torch_dtype="float32",
            num_nextn_predict_layers=depths,
            mtp_hybrid_override_pattern="-",
        )
        backend = BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            dispatcher="torch",
            experts="torch",
            fake_balanced_gate=False,
            enable_hf_state_dict_adapter=False,
        )
        self.language_model = NemotronHForCausalLM(config, backend=backend)
        self.language_model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
        self.config = SimpleNamespace(llm_config=config)
        self.img_context_token_id = IMAGE
        self.sound_context_token_id = SOUND
        self.sound_encoder = nn.Identity()
        self.feature_encoder = _FeatureEncoder()
        self.cp_mesh = _CPMesh(rank)
        self.train()

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.feature_encoder(pixel_values)

    def extract_feature_dynamic(self, pixel_values: torch.Tensor, imgs_sizes: torch.Tensor) -> torch.Tensor:
        return self.feature_encoder(pixel_values)

    def extract_video_feature(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        return self.feature_encoder(pixel_values_videos)

    def extract_sound_feature(
        self, sound_features: torch.Tensor, sound_attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        return self.feature_encoder(sound_features)


def _indices(layout: str, rank: int) -> torch.Tensor:
    """Mathematical head/tail partition; no production sharding helper."""
    spans = [(0, 16), (16, 32)] if layout == "thd" else [(0, 32)]
    result = []
    for start, end in spans:
        chunk = (end - start) // 4
        result.extend(range(start + rank * chunk, start + (rank + 1) * chunk))
        tail = start + (3 - rank) * chunk
        result.extend(range(tail, tail + chunk))
    return torch.tensor(result)


def _shift_documents(values: torch.Tensor, depth: int, fill: torch.Tensor | int) -> torch.Tensor:
    """Copy within physical document intervals, with no roll or ID equality."""
    result = torch.empty_like(values)
    result[...] = fill
    for start, end in [(0, 16), (16, values.shape[1])]:
        result[:, start : end - depth] = values[:, start + depth : end]
    return result


def _make_batch(media: str, layout: str, future_only: bool, sequence: int = 32):
    rows = 1 if layout == "thd" else 2
    ids = (torch.arange(rows * 32).reshape(rows, 32) * 7) % 10 + 3
    placeholder = SOUND if media == "audio" else IMAGE
    positions = [[4]] if future_only else [[3, 4, 7, 8, 12, 14, 16, 19, 20, 27, 28, 30], [1, 4, 16, 24, 30]]
    if media != "text":
        for row in range(rows):
            ids[row, positions[row]] = placeholder
    ids[:, [15, 31]] = 0
    labels = torch.full_like(ids, IGNORE)
    for start, end in [(0, 15), (16, 31)]:
        labels[:, start : end - 1] = ids[:, start + 1 : end]
    labels[labels == placeholder] = IGNORE
    batch = {
        "input_ids": ids,
        "labels": labels,
        "position_ids": torch.arange(16).repeat(2).expand(rows, -1),
        "seq_lens": torch.tensor([[15, 15]] * rows),
        "seq_lens_padded": torch.tensor([[16, 16]] * rows),
    }
    if layout == "thd":
        batch["qkv_format"] = "thd"
    if sequence != 32:
        assert layout == "bshd" and sequence == 31
        ids = ids[:, :sequence]
        batch["input_ids"] = ids
        batch["labels"] = labels[:, :sequence]
        batch["position_ids"] = batch["position_ids"][:, :sequence]
        batch["seq_lens_padded"][:, -1] = sequence - 16
    mask = ids == placeholder if media != "text" else torch.zeros_like(ids, dtype=torch.bool)
    count = int(mask.sum())
    features = torch.sin(torch.arange(count * HIDDEN).reshape(count, 1, HIDDEN) * 0.17)
    return batch, mask, features


def _loss(model, output, labels: torch.Tensor, targets: list[torch.Tensor]) -> torch.Tensor:
    def ce(logits, target):
        return F.cross_entropy(logits.reshape(-1, 64), target.reshape(-1), ignore_index=IGNORE, reduction="sum")

    total = ce(output.logits, labels)
    for hidden, target in zip(output.mtp_per_depth_h, targets):
        total = total + 0.3 / len(targets) * ce(model.language_model.lm_head(hidden), target)
    return total


@pytest.mark.parametrize("layout", ["bshd", "thd"])
@pytest.mark.parametrize("media", ["text", "image", "dynamic_image", "video", "audio"])
@pytest.mark.parametrize("depths,rank", [(1, 0), (2, 1)])
def test_mtp_media_futures_and_gradients_match_independent_oracle(monkeypatch, layout, media, depths, rank):
    _check_case(monkeypatch, layout, media, depths, rank, future_only=False)


@pytest.mark.parametrize("media", ["image", "audio"])
@pytest.mark.parametrize("depths", [1, 2])
def test_mtp_media_on_remote_rank_without_local_primary_placeholder(monkeypatch, media, depths):
    _check_case(monkeypatch, "thd", media, depths, rank=0, future_only=True)


def _check_case(
    monkeypatch,
    layout: str,
    media: str,
    depths: int,
    rank: int,
    future_only: bool,
    sequence: int = 32,
    cp_size: int = 2,
    padded_none: bool = False,
):
    model = _TinyOmni(depths, rank)
    if cp_size == 1:
        model.cp_mesh = None
    batch, media_mask, initial_features = _make_batch(media, layout, future_only, sequence)
    indices = _indices(layout, rank) if cp_size > 1 else torch.arange(sequence)

    def partition(cu_seqlens, total_tokens, cp_size, cp_rank):
        assert cu_seqlens.tolist() == [0, 16, 32]
        assert (total_tokens, cp_size, cp_rank) == (32, 2, rank)
        return indices

    monkeypatch.setitem(sys.modules, "transformer_engine_torch", SimpleNamespace(thd_get_partitioned_indices=partition))
    prepared = model.prepare_mtp_inputs_for_cp(batch)

    def local(tensor, fill=0):
        if tensor.shape[1] < int(indices.max()) + 1:
            shape = list(tensor.shape)
            shape[1] = int(indices.max()) + 1
            padded = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
            padded[...] = fill
            padded[:, : tensor.shape[1]] = tensor
            tensor = padded
        selected = tensor.index_select(1, indices)
        return selected.squeeze(0) if layout == "thd" else selected

    inputs = {
        "input_ids": local(batch["input_ids"]) if layout == "thd" else batch["input_ids"],
        "position_ids": local(batch["position_ids"]),
        "mtp_per_depth_input_ids": tuple(local(t) for t in prepared.input_ids),
        "mtp_per_depth_position_ids": tuple(local(t) for t in prepared.position_ids),
        "mtp_per_depth_valid_masks": tuple(local(t) for t in prepared.valid_masks),
        "cp_size": cp_size,
        "cp_rank": rank,
    }
    if layout == "thd":
        inputs.update(
            qkv_format="thd",
            cu_seqlens=torch.tensor([0, 16, 32], dtype=torch.int32),
            cu_seqlens_padded=torch.tensor([0, 16, 32], dtype=torch.int32),
            _thd_local_indices=indices,
            _nemotron_omni_global_image_mask=(batch["input_ids"] == IMAGE).flatten(),
            _nemotron_omni_global_sound_mask=(batch["input_ids"] == SOUND).flatten(),
        )
    if padded_none:
        inputs["cu_seqlens_padded"] = None
    actual_features = initial_features.clone().requires_grad_(True)
    if media in ("image", "dynamic_image"):
        inputs["pixel_values"] = actual_features
        if media == "dynamic_image":
            inputs["imgs_sizes"] = torch.tensor([[1, 1]] * actual_features.shape[0])
    elif media == "video":
        inputs["pixel_values_videos"] = actual_features
    elif media == "audio":
        inputs["sound_features"] = actual_features
    captured = []
    handles = [
        layer.enorm.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        for layer in model.mtp.layers
    ]
    actual = model(**inputs)
    for handle in handles:
        handle.remove()
    assert model.feature_encoder.calls == (0 if media == "text" else 1)

    oracle_features = initial_features.clone().requires_grad_(True)
    fused = model.language_model.get_input_embeddings()(batch["input_ids"]).clone()
    fused[media_mask] = oracle_features.reshape(-1, HIDDEN)
    padding_embedding = model.language_model.get_input_embeddings()(torch.tensor(0))
    expected_futures = [
        local(_shift_documents(fused, depth, padding_embedding), fill=padding_embedding)
        for depth in range(1, depths + 1)
    ]
    targets = [local(_shift_documents(batch["labels"], depth, IGNORE), fill=IGNORE) for depth in range(1, depths + 1)]
    labels = local(batch["labels"], fill=IGNORE)
    for observed, expected in zip(captured, expected_futures):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    for observed, expected in zip(prepared.targets, targets):
        assert torch.equal(local(observed, fill=IGNORE), expected)
    expected = model.language_model(
        None,
        *expected_futures,
        inputs_embeds=local(fused),
        position_ids=inputs["position_ids"],
        mtp_per_depth_position_ids=inputs["mtp_per_depth_position_ids"],
    )
    actual_loss = _loss(model, actual, labels, targets)
    expected_loss = _loss(model, expected, labels, targets)
    torch.testing.assert_close(actual.logits.reshape(-1, 64), expected.logits.reshape(-1, 64), rtol=0, atol=0)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-6, atol=1e-6)
    for actual_hidden, expected_hidden in zip(actual.mtp_per_depth_h, expected.mtp_per_depth_h):
        torch.testing.assert_close(actual_hidden, expected_hidden, rtol=1e-6, atol=1e-6)
    if media != "text":
        fusion_weight = model.mtp.layers[0].eh_proj.weight
        embedding_weight = model.language_model.get_input_embeddings().weight
        actual_gradient, actual_fusion_gradient, actual_embedding_gradient = torch.autograd.grad(
            actual_loss, (actual_features, fusion_weight, embedding_weight), allow_unused=True
        )
        expected_gradient, expected_fusion_gradient, expected_embedding_gradient = torch.autograd.grad(
            expected_loss, (oracle_features, fusion_weight, embedding_weight)
        )
        assert actual_gradient is not None
        assert torch.count_nonzero(expected_gradient) > 0
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual_fusion_gradient, expected_fusion_gradient, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual_embedding_gradient, expected_embedding_gradient, rtol=2e-5, atol=2e-6)
        placeholder = SOUND if media == "audio" else IMAGE
        assert torch.count_nonzero(actual_embedding_gradient[placeholder]) == 0
        if future_only:
            assert not media_mask.flatten()[indices].any()
            assert torch.count_nonzero(actual_gradient) > 0


@pytest.mark.parametrize("depths,rank", [(1, 0), (2, 1)])
def test_bshd_mtp_uses_original_batch_row_offsets_when_cp_pads_rows(monkeypatch, depths, rank):
    _check_case(monkeypatch, "bshd", "image", depths, rank, future_only=False, sequence=31)


@pytest.mark.parametrize("media", ["image", "audio"])
@pytest.mark.parametrize("depths", [1, 2])
def test_flat_thd_cp1_accepts_explicit_media_futures(monkeypatch, media, depths):
    _check_case(monkeypatch, "thd", media, depths, rank=0, future_only=False, cp_size=1)


def test_empty_mtp_future_tuple_preserves_fused_embedding_fallback():
    model = _TinyOmni(depths=1, rank=0)
    model.cp_mesh = None
    ids = torch.tensor([[3, IMAGE, 5, 6]])
    features = torch.sin(torch.arange(HIDDEN, dtype=torch.float32)).reshape(1, 1, HIDDEN)
    expected = model(input_ids=ids, pixel_values=features)
    actual = model(input_ids=ids, pixel_values=features, mtp_per_depth_input_ids=())
    assert model.feature_encoder.calls == 2
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.mtp_per_depth_h[0], expected.mtp_per_depth_h[0], rtol=0, atol=0)


def test_prebuilt_text_embeddings_preserve_explicit_mtp_future_support():
    model = _TinyOmni(depths=1, rank=0)
    model.cp_mesh = None
    ids = torch.tensor([[3, 4, 5, 6]])
    future_ids = torch.tensor([[4, 5, 6, 0]])
    positions = torch.tensor([[1, 2, 3, 0]])
    embed = model.language_model.get_input_embeddings()
    fused = embed(ids)
    actual = model(
        input_ids=ids,
        inputs_embeds=fused,
        mtp_per_depth_input_ids=(future_ids,),
        mtp_per_depth_position_ids=(positions,),
    )
    expected = model.language_model(
        None,
        embed(future_ids),
        inputs_embeds=fused,
        mtp_per_depth_position_ids=(positions,),
    )
    assert model.feature_encoder.calls == 0
    torch.testing.assert_close(actual.mtp_per_depth_h[0], expected.mtp_per_depth_h[0], rtol=0, atol=0)


def test_prebuilt_media_embeddings_cannot_silently_drop_future_media_features():
    model = _TinyOmni(depths=1, rank=0)
    model.cp_mesh = None
    ids = torch.tensor([[3, IMAGE, 5, 6]])
    fused = model.language_model.get_input_embeddings()(ids).clone()
    fused[:, 1] = torch.sin(torch.arange(HIDDEN, dtype=torch.float32))
    with pytest.raises(ValueError, match="precomputed inputs_embeds"):
        model(
            input_ids=ids,
            inputs_embeds=fused,
            mtp_per_depth_input_ids=(torch.tensor([[IMAGE, 5, 6, 0]]),),
            mtp_per_depth_position_ids=(torch.tensor([[1, 2, 3, 0]]),),
        )


def test_thd_media_futures_fall_back_to_physical_cu_when_optional_padded_cu_is_none(monkeypatch):
    _check_case(monkeypatch, "thd", "image", depths=1, rank=0, future_only=False, padded_none=True)
