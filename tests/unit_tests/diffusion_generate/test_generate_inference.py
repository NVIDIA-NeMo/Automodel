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

"""CPU unit tests for diffusion generation inference inputs."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

import examples.diffusion.generate.generate as gen


def _make_cfg(tmp_path, prompts, input_images=None):
    """Build the minimal configuration consumed by ``run_inference``."""
    inference = SimpleNamespace(prompts=prompts)
    if input_images is not None:
        inference.input_images = input_images
    return SimpleNamespace(
        inference=inference,
        model=SimpleNamespace(),
        output=SimpleNamespace(output_dir=str(tmp_path)),
        seed=7,
    )


def _mock_cuda_generator():
    """Return a patched CUDA generator constructor and its seeded generator."""
    generator = MagicMock()
    generator.manual_seed.return_value = generator
    return patch.object(gen.torch, "Generator", return_value=generator), generator


class ImageEditPipeline:
    """Small image-conditioned pipeline double with an explicit ``image`` input."""

    def __init__(self):
        self.calls = []
        self.output_image = MagicMock()

    def __call__(self, prompt, generator, image):
        self.calls.append({"prompt": prompt, "generator": generator, "image": image})
        return SimpleNamespace(images=[self.output_image])


class TextToImagePipeline:
    """Small text-only pipeline double."""

    def __init__(self):
        self.calls = []
        self.output_image = MagicMock()

    def __call__(self, prompt, generator):
        self.calls.append({"prompt": prompt, "generator": generator})
        return SimpleNamespace(images=[self.output_image])


def test_input_images_are_loaded_and_passed_to_matching_prompts(tmp_path):
    pipe = ImageEditPipeline()
    source_paths = ["/data/source-0.png", "/data/source-1.png"]
    loaded_images = [object(), object()]
    cfg = _make_cfg(tmp_path, ["first edit", "second edit"], source_paths)
    generator_patch, generator = _mock_cuda_generator()

    with (
        generator_patch,
        patch.object(gen.torch.cuda, "is_available", return_value=False),
        patch("diffusers.utils.load_image", side_effect=loaded_images) as mock_load_image,
    ):
        gen.run_inference(pipe, cfg, is_rank0=True)

    assert mock_load_image.call_args_list == [call(source_paths[0]), call(source_paths[1])]
    assert pipe.calls == [
        {"prompt": "first edit", "generator": generator, "image": loaded_images[0]},
        {"prompt": "second edit", "generator": generator, "image": loaded_images[1]},
    ]


def test_max_samples_slices_prompts_and_input_images_together(tmp_path):
    pipe = ImageEditPipeline()
    cfg = _make_cfg(
        tmp_path,
        ["first edit", "second edit"],
        ["/data/source-0.png", "/data/source-1.png"],
    )
    cfg.inference.max_samples = 1
    loaded_image = object()
    generator_patch, generator = _mock_cuda_generator()

    with (
        generator_patch,
        patch.object(gen.torch.cuda, "is_available", return_value=False),
        patch("diffusers.utils.load_image", return_value=loaded_image) as mock_load_image,
    ):
        gen.run_inference(pipe, cfg, is_rank0=True)

    mock_load_image.assert_called_once_with("/data/source-0.png")
    assert pipe.calls == [{"prompt": "first edit", "generator": generator, "image": loaded_image}]


def test_input_image_count_must_match_prompt_count(tmp_path):
    cfg = _make_cfg(tmp_path, ["first edit", "second edit"], ["/data/source.png"])

    with pytest.raises(ValueError, match="one entry per prompt"):
        gen.run_inference(ImageEditPipeline(), cfg, is_rank0=True)


def test_input_images_must_be_a_list(tmp_path):
    cfg = _make_cfg(tmp_path, ["edit this"], "/data/source.png")

    with pytest.raises(TypeError, match="must be a list"):
        gen.run_inference(ImageEditPipeline(), cfg, is_rank0=True)


def test_input_images_cannot_conflict_with_pipeline_kwargs_image(tmp_path):
    cfg = _make_cfg(tmp_path, ["edit this"], ["/data/source.png"])
    cfg.inference.pipeline_kwargs = MagicMock()
    cfg.inference.pipeline_kwargs.to_dict.return_value = {"image": object()}

    with pytest.raises(ValueError, match="inference.input_images"):
        gen.run_inference(ImageEditPipeline(), cfg, is_rank0=True)


def test_input_images_reject_pipeline_without_image_parameter(tmp_path):
    cfg = _make_cfg(tmp_path, ["edit this"], ["/data/source.png"])

    with pytest.raises(ValueError, match="does not accept image inputs"):
        gen.run_inference(TextToImagePipeline(), cfg, is_rank0=True)


def test_absent_input_images_preserves_text_to_image_call(tmp_path):
    pipe = TextToImagePipeline()
    cfg = _make_cfg(tmp_path, ["create an image"])
    generator_patch, generator = _mock_cuda_generator()

    with (
        generator_patch,
        patch.object(gen.torch.cuda, "is_available", return_value=False),
        patch("diffusers.utils.load_image") as mock_load_image,
    ):
        gen.run_inference(pipe, cfg, is_rank0=True)

    mock_load_image.assert_not_called()
    assert pipe.calls == [{"prompt": "create an image", "generator": generator}]


def test_non_main_rank_still_runs_image_conditioned_pipeline_without_saving(tmp_path):
    pipe = ImageEditPipeline()
    cfg = _make_cfg(tmp_path, ["edit this"], ["/data/source.png"])
    loaded_image = object()
    generator_patch, generator = _mock_cuda_generator()

    with (
        generator_patch,
        patch.object(gen.torch.cuda, "is_available", return_value=False),
        patch("diffusers.utils.load_image", return_value=loaded_image) as mock_load_image,
    ):
        gen.run_inference(pipe, cfg, is_rank0=False)

    mock_load_image.assert_called_once_with("/data/source.png")
    assert pipe.calls == [{"prompt": "edit this", "generator": generator, "image": loaded_image}]
    pipe.output_image.save.assert_not_called()
