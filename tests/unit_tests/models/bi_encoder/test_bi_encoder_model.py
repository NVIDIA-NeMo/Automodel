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
from nemo_automodel._transformers.retrieval import BiEncoderModel, CrossEncoderModel, _dummy_vision_sum
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
        self.run_dummy_vision_flags = []

    def forward(self, batch):
        self.run_dummy_vision_flags.append(batch.get("run_dummy_vision"))
        return batch["input_ids"].float() * self.scale


class _ToyBackboneOutput:
    def __init__(self, hidden_state):
        self.last_hidden_state = hidden_state


class _ToyBackboneWithDummyFlag(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(name_or_path="")
        self.run_dummy_vision = None

    def forward(
        self,
        input_ids,
        attention_mask,
        return_dict=True,
        output_hidden_states=True,
        run_dummy_vision=None,
    ):
        self.run_dummy_vision = run_dummy_vision
        hidden_state = input_ids.float().unsqueeze(-1).expand(*input_ids.shape, 2)
        return _ToyBackboneOutput(hidden_state)


class _ToyBackboneWithoutDummyFlag(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(name_or_path="")
        self.received_kwargs = None

    def forward(self, input_ids, attention_mask, return_dict=True, output_hidden_states=True):
        self.received_kwargs = {
            "return_dict": return_dict,
            "output_hidden_states": output_hidden_states,
        }
        hidden_state = input_ids.float().unsqueeze(-1).expand(*input_ids.shape, 2)
        return _ToyBackboneOutput(hidden_state)


class _ToyVisionBackboneWithoutDummyFlag(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(name_or_path="")
        self.vision_tower = torch.nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.dummy_pixel_values = None
        self.dummy_image_sizes = None

    def get_image_features(self, pixel_values, image_sizes, return_dict=True):
        self.dummy_pixel_values = pixel_values
        self.dummy_image_sizes = image_sizes
        return SimpleNamespace(pooler_output=(self.vision_tower(pixel_values),))

    def forward(self, input_ids, attention_mask, return_dict=True, output_hidden_states=True):
        hidden_state = input_ids.float().unsqueeze(-1).expand(*input_ids.shape, 2)
        return _ToyBackboneOutput(hidden_state)


class _ToyMistralLikeVisionTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = (14, 16)


class _ToyMistralLikeVisionBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(name_or_path="", spatial_merge_size=2)
        self.vision_tower = _ToyMistralLikeVisionTower()
        self.dummy_pixel_values = None
        self.dummy_image_sizes = None
        self.merged_token_count = None

    def get_image_features(self, pixel_values, image_sizes, return_dict=True):
        self.dummy_pixel_values = pixel_values
        self.dummy_image_sizes = image_sizes
        patch_height, patch_width = self.vision_tower.patch_size
        merged_height = image_sizes[0, 0].item() // (patch_height * self.config.spatial_merge_size)
        merged_width = image_sizes[0, 1].item() // (patch_width * self.config.spatial_merge_size)
        self.merged_token_count = merged_height * merged_width
        if self.merged_token_count <= 1:
            raise RuntimeError("dummy image must produce more than one merged token")
        return SimpleNamespace(pooler_output=(pixel_values.sum().reshape(1),))


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


def test_from_pretrained_prefers_dtype_over_deprecated_torch_dtype_default(monkeypatch):
    last_kwargs = {}

    def fake_build(**kwargs):
        nonlocal last_kwargs
        last_kwargs = kwargs
        return DummyModel()

    _apply_common_mocks(monkeypatch)
    monkeypatch.setattr(BiEncoderModel, "build", staticmethod(fake_build))
    monkeypatch.setattr(am, "apply_model_infrastructure", lambda model, **kwargs: model)

    model = am.NeMoAutoModelBiEncoder.from_pretrained(
        pretrained_model_name_or_path="some/path",
        dtype="bfloat16",
        use_liger_kernel=False,
        use_sdpa_patching=False,
    )

    assert isinstance(model, DummyModel)
    assert last_kwargs["dtype"] == "bfloat16"
    assert "torch_dtype" not in last_kwargs


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


def test_bi_encoder_forwards_run_dummy_vision_when_backbone_supports_it():
    backbone = _ToyBackboneWithDummyFlag()
    model = BiEncoderModel(backbone, pooling="avg", l2_normalize=False)

    embeddings = model(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "run_dummy_vision": False,
        }
    )

    assert backbone.run_dummy_vision is False
    assert embeddings.shape == (1, 2)


def test_bi_encoder_drops_run_dummy_vision_when_backbone_does_not_support_it():
    backbone = _ToyBackboneWithoutDummyFlag()
    model = BiEncoderModel(backbone, pooling="avg", l2_normalize=False)

    embeddings = model(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "run_dummy_vision": True,
        }
    )

    assert backbone.received_kwargs == {"return_dict": True, "output_hidden_states": True}
    assert embeddings.shape == (1, 2)


def test_bi_encoder_runs_zero_contribution_dummy_vision_for_full_vlm_backbone():
    backbone = _ToyVisionBackboneWithoutDummyFlag()
    model = BiEncoderModel(backbone, pooling="avg", l2_normalize=False)
    model.train()

    embeddings = model(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "run_dummy_vision": True,
        }
    )
    embeddings.sum().backward()

    assert embeddings.shape == (1, 2)
    assert backbone.dummy_pixel_values.shape == (1, 3, 32, 32)
    assert torch.equal(backbone.dummy_image_sizes, torch.tensor([[32, 32]]))
    assert backbone.vision_tower.weight.grad is not None


def test_dummy_vision_sum_uses_multiple_merged_tokens_for_mistral3_like_backbone():
    backbone = _ToyMistralLikeVisionBackbone()

    dummy_sum = _dummy_vision_sum(backbone)

    assert dummy_sum is not None
    assert backbone.dummy_pixel_values.shape == (1, 3, 56, 64)
    assert torch.equal(backbone.dummy_image_sizes, torch.tensor([[56, 64]]))
    assert backbone.merged_token_count == 4


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


def test_forward_backward_step_disables_query_dummy_vision_but_keeps_passage_dummy_vision():
    recipe = TrainBiEncoderRecipe.__new__(TrainBiEncoderRecipe)
    recipe.dist_env = SimpleNamespace(device="cpu")
    recipe.distributed_config = SimpleNamespace(defer_fsdp_grad_sync=True)
    model = _ToyMultiVectorBiEncoder()
    recipe.model_parts = [model]
    recipe.temperature = 1.0
    recipe.train_n_passages = 2

    batch = {
        "q_input_ids": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        "q_attention_mask": torch.tensor([[1, 1]]),
        "d_input_ids": torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 0.0]],
            ]
        ),
        "d_attention_mask": torch.tensor([[1, 1], [1, 0]]),
    }

    recipe._forward_backward_step(0, batch, loss_buffer=[], num_batches=1, is_train=True)

    assert model.run_dummy_vision_flags == [False, True]


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
