# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Verify Super VL's checkpoint-owned norm before spatial pixel shuffle."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from nemo_automodel.components.models.nemotron_omni.model import (
    NemotronOmniForConditionalGeneration,
    VisionProjector,
)


class _PatchFeatures(nn.Module):
    """Expose deterministic patch features without constructing RADIO."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_size = 1
        self.features = nn.Parameter(torch.randn(1, 24, 4) + 3.0)
        self.radio_model = SimpleNamespace(
            model=SimpleNamespace(
                patch_generator=SimpleNamespace(patch_size=1, embedder=nn.Identity(), video_embedder=nn.Identity())
            )
        )

    def forward(self, images: torch.Tensor) -> SimpleNamespace:
        """Return test patch features.

        Args:
            images: Tensor of shape [batch, channels, height, width].

        Returns:
            Namespace with features of shape [batch, patches, hidden].
        """
        return SimpleNamespace(features=self.features)


@pytest.mark.parametrize("path", ["dense", "dynamic", "video"])
def test_final_vision_norm_matches_reference_before_shuffle_and_backward(path: str) -> None:
    """Non-square patch grids match HF's norm placement and its gradients."""
    torch.manual_seed(123)
    model = object.__new__(NemotronOmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.vision_model = _PatchFeatures()
    model.vision_projector = VisionProjector(4, 12, 8, vision_final_layernorm_eps=1e-6).to(torch.bfloat16)
    model.downsample_ratio = 0.5
    model.ps_version = "v2"
    model.patch_size = 1
    model.video_temporal_patch_dim = 2
    with torch.no_grad():
        model.vision_projector.vision_final_layernorm.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
        model.vision_projector.vision_final_layernorm.bias.copy_(torch.tensor([-0.5, 0.25, 0.0, 0.5]))
    ref_projector = copy.deepcopy(model.vision_projector)
    ref_features = model.vision_model.features.detach().clone().requires_grad_()
    norm = ref_projector.vision_final_layernorm
    normalized = F.layer_norm(ref_features.to(torch.bfloat16), (4,), norm.weight, norm.bias, norm.eps).reshape(
        1, 4, 6, 4
    )
    # Explicit spatial 2x2 neighborhoods, independent of production reshape logic.
    shuffled = torch.cat(
        [normalized[:, 0::2, 0::2], normalized[:, 0::2, 1::2], normalized[:, 1::2, 0::2], normalized[:, 1::2, 1::2]],
        dim=-1,
    ).reshape(1, 6, 16)
    expected = ref_projector(shuffled)
    images = torch.zeros(1, 3, 4, 6)
    if path == "dense":
        actual = model._extract_feature_dense(images)
    elif path == "dynamic":
        actual = model.extract_feature_dynamic(images, [(4, 6)])
    else:
        actual = model.extract_video_feature(images.repeat(2, 1, 1, 1))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(model.vision_model.features.grad, ref_features.grad, rtol=0, atol=0)
    for name, param in model.vision_projector.named_parameters():
        torch.testing.assert_close(param.grad, dict(ref_projector.named_parameters())[name].grad, rtol=0, atol=0)


def test_legacy_projector_has_no_additional_checkpoint_parameters() -> None:
    """Non-Super projectors retain their checkpoint schema."""
    projector = VisionProjector(4, 12, 8)
    assert projector.vision_final_layernorm is None
    assert set(projector.state_dict()) == {"norm.weight", "linear1.weight", "linear2.weight"}


class _VideoPatchFeatures(nn.Module):
    """Exercise native and legacy RADIO interfaces with a real video projection."""

    def __init__(self, *, legacy: bool) -> None:
        super().__init__()
        self.patch_size = 1
        self.video_projection = nn.Linear(6, 4, bias=False).to(torch.bfloat16)
        self.image_projection = nn.Linear(3, 4, bias=False).to(torch.bfloat16)
        self.fail = False
        if legacy:
            self.radio_model = SimpleNamespace(
                model=SimpleNamespace(
                    patch_generator=SimpleNamespace(
                        patch_size=1, embedder=self.image_projection, video_embedder=self.video_projection
                    )
                )
            )

    def forward(self, images: torch.Tensor, *, use_video_patch_projection: bool = False) -> SimpleNamespace:
        """Project spatial patches from consecutive RGB frame pairs.

        Args:
            images: Tensor of shape [groups, channels, height, width], where
                video channels contain two consecutive RGB frames.
            use_video_patch_projection: Select the native temporal projection.

        Returns:
            Namespace with features of shape [groups, height * width, 4].
        """
        if self.fail:
            raise RuntimeError("encoder failed")
        if hasattr(self, "radio_model"):
            projection = self.radio_model.model.patch_generator.embedder
        else:
            projection = self.video_projection if use_video_patch_projection else self.image_projection
        patches = images.permute(0, 2, 3, 1).flatten(1, 2)
        return SimpleNamespace(features=projection(patches))


def _video_model(*, legacy: bool) -> NemotronOmniForConditionalGeneration:
    model = object.__new__(NemotronOmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.vision_model = _VideoPatchFeatures(legacy=legacy)
    model.vision_projector = VisionProjector(4, 12, 8, vision_final_layernorm_eps=1e-6).to(torch.bfloat16)
    model.video_temporal_patch_dim = 2
    model.ps_version = "v2"
    return model


@pytest.mark.parametrize("legacy", [False, True], ids=["native", "legacy"])
@pytest.mark.parametrize("num_frames", [1, 2, 3, 4])
def test_video_temporal_projection_matches_frame_pairs_and_gradients(legacy: bool, num_frames: int) -> None:
    """Frame pairs, last-frame padding and their gradients preserve the video contract."""
    torch.manual_seed(129)
    model = _video_model(legacy=legacy)
    reference_projection = copy.deepcopy(model.vision_model.video_projection)
    reference_projector = copy.deepcopy(model.vision_projector)
    frames = torch.randn(num_frames, 3, 4, 6, requires_grad=True)
    reference_frames = frames.detach().clone().requires_grad_()

    # HF casts before padding, so repeated-frame gradients accumulate in BF16.
    reference_pixels = reference_frames.to(torch.bfloat16)
    # Enumerate pairs independently; an incomplete pair repeats its last frame.
    paired = torch.stack(
        [
            torch.cat([reference_pixels[index], reference_pixels[min(index + 1, num_frames - 1)]], dim=0)
            for index in range(0, num_frames, 2)
        ]
    ).to(torch.bfloat16)
    features = reference_projection(paired.permute(0, 2, 3, 1))
    norm = reference_projector.vision_final_layernorm
    normalized = F.layer_norm(features, (4,), norm.weight, norm.bias, norm.eps)
    shuffled = torch.cat(
        [normalized[:, 0::2, 0::2], normalized[:, 0::2, 1::2], normalized[:, 1::2, 0::2], normalized[:, 1::2, 1::2]],
        dim=-1,
    ).flatten(1, 2)
    expected = reference_projector(shuffled)
    actual = model.extract_video_feature(frames)
    assert actual.shape == ((num_frames + 1) // 2, 6, 8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(frames.grad, reference_frames.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        model.vision_model.video_projection.weight.grad, reference_projection.weight.grad, rtol=0, atol=0
    )
    assert model.vision_model.image_projection.weight.grad is None
    for name, param in model.vision_projector.named_parameters():
        torch.testing.assert_close(param.grad, dict(reference_projector.named_parameters())[name].grad, rtol=0, atol=0)
    assert model.vision_model.training
    if legacy:
        assert model.vision_model.radio_model.model.patch_generator.embedder is model.vision_model.image_projection


@pytest.mark.parametrize("legacy", [False, True], ids=["native", "legacy"])
@pytest.mark.parametrize("training", [False, True])
def test_video_encoder_error_restores_mode_and_image_projection(legacy: bool, training: bool) -> None:
    """A failed video encode leaves the image path and encoder mode usable."""
    model = _video_model(legacy=legacy)
    model.vision_model.train(training)
    model.vision_model.fail = True
    with pytest.raises(RuntimeError, match="encoder failed"):
        model.extract_video_feature(torch.randn(2, 3, 4, 6))
    assert model.vision_model.training is training
    if legacy:
        assert model.vision_model.radio_model.model.patch_generator.embedder is model.vision_model.image_projection
