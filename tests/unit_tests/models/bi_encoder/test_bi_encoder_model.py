# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from types import SimpleNamespace

import pytest
import torch

import nemo_automodel._transformers.auto_model as am
import nemo_automodel.recipes.retrieval.train_bi_encoder as tbe
from nemo_automodel._transformers.retrieval import BiEncoderModel, CrossEncoderModel
from nemo_automodel.recipes.retrieval.train_bi_encoder import (
    TrainBiEncoderRecipe,
    distributed_maxsim_scores_and_labels,
    maxsim_scores_and_labels,
)


class DummyModel:
    def __init__(self):
        self.config = {}
        self.marker = []


class DummyMesh:
    pass


class _ToyMultiVectorBiEncoder(torch.nn.Module):
    do_distributed_inbatch_negative = False
    l2_normalize = False
    pooling = "multi_vector"

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        return batch["input_ids"].float() * self.scale


def _apply_common_mocks(monkeypatch):
    """Mock CUDA-dependent infrastructure so tests run without a GPU."""
    monkeypatch.setattr(am, "instantiate_infrastructure", lambda **kwargs: (None, None, None, None))
    monkeypatch.setattr(
        am, "MeshContext", type("MeshContext", (), {"from_meshes": staticmethod(lambda *a, **k: DummyMesh())})
    )
    monkeypatch.setattr(am.torch.cuda, "current_device", lambda: 0)


def test_from_pretrained_happy_path(monkeypatch):
    calls = {"build": 0, "liger": 0, "sdpa": 0}
    last_kwargs = {}

    def fake_build(**kwargs):
        calls["build"] += 1
        nonlocal last_kwargs
        last_kwargs = kwargs
        return DummyModel()

    def fake_liger(model):
        calls["liger"] += 1
        model.marker.append("liger")
        return model

    def fake_sdpa(model, method):
        calls["sdpa"] += 1
        model.marker.append("sdpa")
        return model

    def fake_apply_infrastructure(model, **kwargs):
        return model

    _apply_common_mocks(monkeypatch)
    monkeypatch.setattr(BiEncoderModel, "build", staticmethod(fake_build))
    monkeypatch.setattr(am, "_patch_liger_kernel", fake_liger)
    monkeypatch.setattr(am, "_patch_attention", fake_sdpa)
    monkeypatch.setattr(am, "apply_model_infrastructure", fake_apply_infrastructure)

    model = am.NeMoAutoModelBiEncoder.from_pretrained(
        pretrained_model_name_or_path="some/path",
        pooling="avg",
        l2_normalize=True,
        do_distributed_inbatch_negative=True,
        detach_distributed_inbatch_negatives=False,
        use_liger_kernel=True,
        use_sdpa_patching=True,
        sdpa_method=None,
        some_other_kwarg="x",
    )
    assert isinstance(model, DummyModel)
    # Patches applied
    assert "liger" in model.marker and "sdpa" in model.marker
    # Ensure HF kwargs injected + passthrough of parameters to build
    assert last_kwargs["attn_implementation"] == am.DEFAULT_ATTN_IMPLEMENTATION
    assert last_kwargs["do_distributed_inbatch_negative"] is True
    assert last_kwargs["detach_distributed_inbatch_negatives"] is False
    assert last_kwargs["some_other_kwarg"] == "x"


def _assert_retries_without_liger(monkeypatch, build_model_cls, auto_model_cls):
    """Verify that when liger patching fails, from_pretrained retries without it."""
    calls = {"build": 0, "liger": 0, "sdpa": 0}

    def fake_build(**kwargs):
        calls["build"] += 1
        return DummyModel()

    def fake_liger(_):
        calls["liger"] += 1
        raise RuntimeError("liger failed")

    def fake_sdpa(model, _):
        calls["sdpa"] += 1
        return model

    _apply_common_mocks(monkeypatch)
    monkeypatch.setattr(build_model_cls, "build", staticmethod(fake_build))
    monkeypatch.setattr(am, "_patch_liger_kernel", fake_liger)
    monkeypatch.setattr(am, "_patch_attention", fake_sdpa)
    monkeypatch.setattr(am, "apply_model_infrastructure", lambda model, **kwargs: model)

    model = auto_model_cls.from_pretrained("x", use_liger_kernel=True, use_sdpa_patching=True)
    assert isinstance(model, DummyModel)
    assert calls["liger"] == 1
    assert calls["build"] == 2
    assert calls["sdpa"] == 1


def _assert_retries_without_sdpa(monkeypatch, build_model_cls, auto_model_cls):
    """Verify that when SDPA patching fails, from_pretrained retries without it."""
    calls = {"build": 0, "liger": 0, "sdpa": 0}

    def fake_build(**kwargs):
        calls["build"] += 1
        return DummyModel()

    def fake_liger(model):
        calls["liger"] += 1
        return model

    def fake_sdpa(_model, _method):
        calls["sdpa"] += 1
        raise Exception("sdpa failed")

    _apply_common_mocks(monkeypatch)
    monkeypatch.setattr(build_model_cls, "build", staticmethod(fake_build))
    monkeypatch.setattr(am, "_patch_liger_kernel", fake_liger)
    monkeypatch.setattr(am, "_patch_attention", fake_sdpa)
    monkeypatch.setattr(am, "apply_model_infrastructure", lambda model, **kwargs: model)

    model = auto_model_cls.from_pretrained("x", use_liger_kernel=True, use_sdpa_patching=True)
    assert isinstance(model, DummyModel)
    assert calls["sdpa"] == 1
    assert calls["build"] == 2
    assert calls["liger"] == 2


def test_from_pretrained_retries_without_liger(monkeypatch):
    _assert_retries_without_liger(monkeypatch, BiEncoderModel, am.NeMoAutoModelBiEncoder)


def test_from_pretrained_retries_without_sdpa(monkeypatch):
    _assert_retries_without_sdpa(monkeypatch, BiEncoderModel, am.NeMoAutoModelBiEncoder)


def test_cross_encoder_from_pretrained(monkeypatch):
    calls = {"build": 0}
    last_kwargs = {}

    def fake_build(**kwargs):
        calls["build"] += 1
        nonlocal last_kwargs
        last_kwargs = kwargs
        return DummyModel()

    def fake_apply_infrastructure(model, **kwargs):
        return model

    _apply_common_mocks(monkeypatch)
    monkeypatch.setattr(CrossEncoderModel, "build", staticmethod(fake_build))
    monkeypatch.setattr(am, "_patch_liger_kernel", lambda m: m)
    monkeypatch.setattr(am, "_patch_attention", lambda m, _: m)
    monkeypatch.setattr(am, "apply_model_infrastructure", fake_apply_infrastructure)

    model = am.NeMoAutoModelCrossEncoder.from_pretrained("mock-model")
    assert isinstance(model, DummyModel)
    assert calls["build"] == 1
    # CrossEncoder build should NOT receive pooling or l2_normalize
    assert "pooling" not in last_kwargs
    assert "l2_normalize" not in last_kwargs
    assert last_kwargs["model_name_or_path"] == "mock-model"


def test_cross_encoder_retries_without_liger(monkeypatch):
    _assert_retries_without_liger(monkeypatch, CrossEncoderModel, am.NeMoAutoModelCrossEncoder)


def test_cross_encoder_retries_without_sdpa(monkeypatch):
    _assert_retries_without_sdpa(monkeypatch, CrossEncoderModel, am.NeMoAutoModelCrossEncoder)


def test_maxsim_scores_and_labels_masks_padding_before_maxsim():
    query = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ]
    )
    key = torch.tensor(
        [
            [[-0.4, -0.4, 0.0, 0.0], [-0.6, -0.6, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.8, 0.0, 0.0, 0.0], [0.0, 0.7, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.2, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, -0.2, 0.0, 0.0], [0.0, -0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.6, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, -0.9, 0.0, 0.0], [0.0, -0.4, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ]
    )
    key_attention_mask = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]])

    scores, labels = maxsim_scores_and_labels(
        query,
        key,
        current_train_n_passages=3,
        key_attention_mask=key_attention_mask,
    )

    assert torch.allclose(scores, torch.tensor([[-0.8, 1.5, 0.3], [-0.2, 0.6, -0.4]]))
    assert torch.equal(labels, torch.tensor([0, 0]))


def test_distributed_maxsim_scores_and_labels_matches_all_at_once_scoring():
    torch.manual_seed(0)
    query = torch.randn(2, 3, 4, requires_grad=True)
    key = torch.randn(8, 5, 4, requires_grad=True)
    key_attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    query_ref = query.detach().clone().requires_grad_()
    key_ref = key.detach().clone().requires_grad_()

    scores, labels = distributed_maxsim_scores_and_labels(
        query,
        key,
        current_train_n_passages=2,
        key_attention_mask=key_attention_mask,
        rank=1,
    )

    ref_token_scores = torch.einsum("bqd,kpd->bkqp", query_ref, key_ref)
    ref_token_scores.masked_fill_(
        ~key_attention_mask[None, :, None, :].bool(),
        torch.finfo(ref_token_scores.dtype).min,
    )
    ref_scores = ref_token_scores.max(dim=3).values.sum(dim=2)
    ref_labels = torch.tensor([4, 6])

    assert scores.shape == (2, 8)
    assert torch.allclose(scores, ref_scores)
    assert torch.equal(labels, ref_labels)

    scores.sum().backward()
    ref_scores.sum().backward()
    assert torch.allclose(query.grad, query_ref.grad)
    assert torch.allclose(key.grad, key_ref.grad)


def test_forward_backward_step_supports_local_multi_vector_pooling():
    recipe = TrainBiEncoderRecipe.__new__(TrainBiEncoderRecipe)
    recipe.dist_env = SimpleNamespace(device="cpu")
    recipe.distributed_config = SimpleNamespace(defer_fsdp_grad_sync=True)
    recipe.model_parts = [_ToyMultiVectorBiEncoder()]
    recipe.temperature = 1.0
    recipe.train_n_passages = 2

    batch = {
        "q_input_ids": torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 1.0], [0.0, 0.0]],
            ]
        ),
        "q_attention_mask": torch.tensor([[1, 1], [1, 0]]),
        "d_input_ids": torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [1.0, 1.0]],
            ]
        ),
        "d_attention_mask": torch.tensor([[1, 1], [1, 0], [1, 0], [1, 1]]),
    }
    loss_buffer = []

    recipe._forward_backward_step(0, batch, loss_buffer=loss_buffer, num_batches=1, is_train=True)

    assert len(loss_buffer) == 1
    assert torch.isfinite(loss_buffer[0])
    assert recipe.model_parts[0].scale.grad is not None


def test_validation_epoch_supports_multi_vector_pooling():
    recipe = TrainBiEncoderRecipe.__new__(TrainBiEncoderRecipe)
    recipe.dist_env = SimpleNamespace(device="cpu")
    model = _ToyMultiVectorBiEncoder()
    model.do_distributed_inbatch_negative = False
    recipe.model_parts = [model]
    recipe.temperature = 1.0
    recipe.val_n_passages = 2
    recipe.step_scheduler = SimpleNamespace(step=3, epoch=1)
    recipe.device_mesh = None

    val_dataloader = [
        {
            "q_input_ids": torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[1.0, 1.0], [0.0, 0.0]],
                ]
            ),
            "q_attention_mask": torch.tensor([[1, 1], [1, 0]]),
            "d_input_ids": torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 1.0], [0.0, 0.0]],
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 1.0], [1.0, 1.0]],
                ]
            ),
            "d_attention_mask": torch.tensor([[1, 1], [1, 0], [1, 0], [1, 1]]),
        }
    ]

    metrics = recipe._run_validation_epoch(val_dataloader)

    assert metrics.step == 3
    assert metrics.epoch == 1
    assert torch.isfinite(torch.tensor(metrics.metrics["val_loss"]))
    assert 0.0 <= metrics.metrics["val_acc1"] <= 1.0
    assert 0.0 <= metrics.metrics["val_mrr"] <= 1.0
    assert recipe.model_parts[0].scale.grad is None


@pytest.mark.parametrize("detach_distributed_inbatch_negatives", [True, False])
def test_forward_backward_step_supports_distributed_multi_vector_inbatch_negatives(
    monkeypatch,
    detach_distributed_inbatch_negatives,
):
    """Exercise the trainer branch that gathers token embeddings across ranks."""
    import nemo_automodel.components.models.common.inbatch_neg_utils as inbatch_neg_utils

    recipe = TrainBiEncoderRecipe.__new__(TrainBiEncoderRecipe)
    recipe.dist_env = SimpleNamespace(device="cpu")
    recipe.distributed_config = SimpleNamespace(defer_fsdp_grad_sync=True)
    model = _ToyMultiVectorBiEncoder()
    model.do_distributed_inbatch_negative = True
    model.detach_distributed_inbatch_negatives = detach_distributed_inbatch_negatives
    recipe.model_parts = [model]
    recipe.temperature = 1.0
    recipe.train_n_passages = 2

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    gather_with_padding_calls = []

    def fake_gather_with_dim1_padding(tensor, padding_value=0, preserve_grad=False):
        gather_with_padding_calls.append((tuple(tensor.shape), padding_value, preserve_grad))
        return torch.cat([tensor.detach().clone(), tensor], dim=0)

    gather_tensor_calls = []

    def fake_gather_tensor(tensor, preserve_grad=False):
        gather_tensor_calls.append((tuple(tensor.shape), preserve_grad))
        remote_doc_ids = torch.tensor([500, 999, 600, 998], dtype=tensor.dtype, device=tensor.device)
        return torch.cat([remote_doc_ids, tensor], dim=0)

    captured = {}

    def fake_cross_entropy(scores, labels):
        captured["scores"] = scores.detach().clone()
        captured["labels"] = labels.detach().clone()
        return -scores.gather(1, labels.unsqueeze(1)).mean()

    monkeypatch.setattr(inbatch_neg_utils, "dist_gather_tensor_with_dim1_padding", fake_gather_with_dim1_padding)
    monkeypatch.setattr(inbatch_neg_utils, "dist_gather_tensor", fake_gather_tensor)
    monkeypatch.setattr(tbe.F, "cross_entropy", fake_cross_entropy)

    batch = {
        "q_input_ids": torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ]
        ),
        "q_attention_mask": torch.tensor([[1, 1], [1, 1]]),
        "d_input_ids": torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
            ]
        ),
        "d_attention_mask": torch.tensor([[1, 1], [1, 0], [1, 1], [1, 0]]),
        "passage_doc_ids": torch.tensor([500, 501, 600, 601], dtype=torch.long),
    }
    loss_buffer = []

    recipe._forward_backward_step(0, batch, loss_buffer=loss_buffer, num_batches=1, is_train=True)

    assert gather_with_padding_calls == [
        ((4, 2, 2), 0, not detach_distributed_inbatch_negatives),
        ((4, 2), False, False),
    ]
    assert gather_tensor_calls == [((4,), False)]
    assert torch.equal(captured["labels"], torch.tensor([4, 6]))
    assert captured["scores"].shape == (2, 8)
    assert captured["scores"][0, 0].item() == torch.finfo(captured["scores"].dtype).min
    assert captured["scores"][1, 2].item() == torch.finfo(captured["scores"].dtype).min
    assert captured["scores"][0, 4].item() > torch.finfo(captured["scores"].dtype).min
    assert captured["scores"][1, 6].item() > torch.finfo(captured["scores"].dtype).min
    assert len(loss_buffer) == 1
    assert torch.isfinite(loss_buffer[0])
    assert model.scale.grad is not None
