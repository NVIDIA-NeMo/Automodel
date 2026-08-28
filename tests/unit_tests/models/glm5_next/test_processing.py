# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import subprocess
import sys

import pytest

from nemo_automodel.components.models.glm5_next.processing import (
    _MEDIA_REMINDER,
    _enable_image_placeholders,
)


def test_training_template_enables_images_but_keeps_other_media_disabled():
    template = "prefix " + _MEDIA_REMINDER + " suffix"

    patched = _enable_image_placeholders(template)

    assert "<|begin_of_image|><|image|><|end_of_image|>" in patched
    assert "media_type == 'image'" in patched
    assert "unable to process this" in patched


def test_existing_multimodal_template_is_not_rewritten():
    template = "{{ '<|image|>' }}"
    assert _enable_image_placeholders(template) == template


def test_unrecognized_text_only_template_fails_loudly():
    with pytest.raises(ValueError, match="no recognized media rendering branch"):
        _enable_image_placeholders("{{ messages }}")


def test_processor_modules_import_without_torchvision():
    script = r"""
import importlib
from unittest import mock

from nemo_automodel.shared.import_utils import is_unavailable

real_import_module = importlib.import_module

def import_without_torchvision(name, *args, **kwargs):
    if name == "torchvision.transforms.v2.functional":
        raise ModuleNotFoundError("No module named 'torchvision'", name="torchvision")
    if name == "transformers.models.glm46v.video_processing_glm46v":
        raise ModuleNotFoundError("No module named 'torchvision'", name="torchvision")
    return real_import_module(name, *args, **kwargs)

with mock.patch("importlib.import_module", side_effect=import_without_torchvision):
    image_processing = real_import_module("nemo_automodel.components.models.glm5_next.image_processing")
    processing = real_import_module("nemo_automodel.components.models.glm5_next.processing")

assert is_unavailable(image_processing.tvF)
assert is_unavailable(processing.Glm46VVideoProcessor)
"""
    subprocess.run([sys.executable, "-c", script], check=True)
