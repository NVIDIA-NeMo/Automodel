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

import pytest
import torch

from tools.diffusion.inference_dmd2_qwen_image import (
    QwenImageDMDInferencePipeline,
    QwenImageDMDOutput,
    _resolve_schedule,
)

# =============================================================================
# _resolve_schedule
# =============================================================================


def test_single_step_schedule_ignores_t_list():
    assert _resolve_schedule(num_inference_steps=1, max_t=0.999, t_list=None) == [0.999, 0.0]
    # Even if a caller passes an (invalid, ignored) t_list, single-step short-circuits.
    assert _resolve_schedule(num_inference_steps=1, max_t=0.999, t_list=[1.0, 0.5]) == [0.999, 0.0]


def test_default_multistep_schedule_is_linear():
    schedule = _resolve_schedule(num_inference_steps=4, max_t=1.0, t_list=None)
    assert len(schedule) == 5
    assert schedule[0] == pytest.approx(1.0)
    assert schedule[-1] == pytest.approx(0.0)
    assert schedule == sorted(schedule, reverse=True)


def test_explicit_t_list_used_verbatim():
    t_list = [1.0, 0.9, 0.75, 0.5, 0.0]
    schedule = _resolve_schedule(num_inference_steps=4, max_t=0.999, t_list=t_list)
    assert schedule == pytest.approx(t_list)


def test_t_list_wrong_length_raises():
    with pytest.raises(ValueError, match="num_inference_steps\\+1 entries"):
        _resolve_schedule(num_inference_steps=4, max_t=1.0, t_list=[1.0, 0.5, 0.0])


def test_t_list_not_ending_at_zero_raises():
    with pytest.raises(ValueError, match="must end at 0.0"):
        _resolve_schedule(num_inference_steps=2, max_t=1.0, t_list=[1.0, 0.5, 0.1])


# =============================================================================
# QwenImageDMDOutput
# =============================================================================


def test_output_holds_images_list():
    output = QwenImageDMDOutput(images=["a", "b"])
    assert output.images == ["a", "b"]


# =============================================================================
# QwenImageDMDInferencePipeline: construction / device / dtype (no diffusers needed)
# =============================================================================


class _FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, dtype=torch.bfloat16)

    @property
    def device(self):
        # Real diffusers ModelMixin subclasses expose this; plain nn.Module does not.
        return self.linear.weight.device


class _FakeBasePipeline:
    def __init__(self):
        self.transformer = _FakeTransformer()

    def to(self, device):
        self.transformer.to(device)
        return self


def test_device_and_dtype_reflect_transformer():
    fake_pipe = _FakeBasePipeline()
    pipe = QwenImageDMDInferencePipeline(base_pipeline=fake_pipe, max_t=0.999)

    assert pipe.dtype == torch.bfloat16
    assert pipe.device == fake_pipe.transformer.linear.weight.device


def test_to_returns_self_and_moves_pipeline():
    fake_pipe = _FakeBasePipeline()
    pipe = QwenImageDMDInferencePipeline(base_pipeline=fake_pipe)

    result = pipe.to("cpu")

    assert result is pipe


def test_from_pretrained_missing_student_path_raises_before_importing_diffusers(tmp_path):
    """Path validation must happen before the diffusers import, so a bad path fails fast
    with a clear FileNotFoundError instead of (or in addition to) an import error."""
    with pytest.raises(FileNotFoundError, match="student_path"):
        QwenImageDMDInferencePipeline.from_pretrained(
            student_path=tmp_path / "does_not_exist",
            base_pipeline_path=tmp_path,
        )


def test_from_pretrained_missing_base_pipeline_path_raises(tmp_path):
    student_dir = tmp_path / "student"
    student_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="base_pipeline_path"):
        QwenImageDMDInferencePipeline.from_pretrained(
            student_path=student_dir,
            base_pipeline_path=tmp_path / "does_not_exist",
        )


def test_call_rejects_invalid_sample_type():
    fake_pipe = _FakeBasePipeline()
    pipe = QwenImageDMDInferencePipeline(base_pipeline=fake_pipe)

    with pytest.raises(ValueError, match="sample_type must be"):
        pipe(prompt="a cat", sample_type="euler")
