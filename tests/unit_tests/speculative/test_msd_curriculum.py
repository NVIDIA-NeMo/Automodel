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

"""Unit tests for the MSD two-stage data curriculum."""

import pytest

from nemo_automodel.components.speculative.eagle.msd_curriculum import (
    MSDCurriculum,
    MSDCurriculumConfig,
    MSDCurriculumPhase,
    MSDDataSource,
)


def test_msd_curriculum_runs_text_only_then_ramps_to_multimodal() -> None:
    """The schedule follows the two-stage MSD progression."""
    curriculum = MSDCurriculum(
        MSDCurriculumConfig(text_only_steps=2, multimodal_ramp_steps=4, final_multimodal_ratio=1.0)
    )

    assert curriculum.multimodal_probability(0) == 0.0
    assert curriculum.multimodal_probability(1) == 0.0
    assert curriculum.multimodal_probability(2) == 0.25
    assert curriculum.multimodal_probability(4) == 0.75
    assert curriculum.multimodal_probability(5) == 1.0
    assert curriculum.phase_for_step(1) is MSDCurriculumPhase.TEXT_ONLY
    assert curriculum.phase_for_step(2) is MSDCurriculumPhase.MULTIMODAL_RAMP
    assert curriculum.phase_for_step(5) is MSDCurriculumPhase.MULTIMODAL
    assert curriculum.source_for_step(0) is MSDDataSource.TEXT
    assert curriculum.source_for_step(5) is MSDDataSource.MULTIMODAL


def test_msd_curriculum_is_deterministic_and_checkpointable() -> None:
    """The source choice is repeatable across resumed and distributed workers."""
    config = MSDCurriculumConfig(text_only_steps=0, multimodal_ramp_steps=0, final_multimodal_ratio=0.4, seed=7)
    first = MSDCurriculum(config)
    second = MSDCurriculum(config)

    first_choices = [first.next_source() for _ in range(12)]
    assert first_choices == [second.source_for_step(step) for step in range(12)]

    resumed = MSDCurriculum(config)
    resumed.load_state_dict(first.state_dict())
    assert resumed.next_source() is first.source_for_step(12)


def test_msd_curriculum_supports_an_immediate_partial_multimodal_mixture() -> None:
    """A zero-length ramp selects a configured post-stage-one mixture."""
    curriculum = MSDCurriculum(
        MSDCurriculumConfig(text_only_steps=3, multimodal_ramp_steps=0, final_multimodal_ratio=0.5)
    )

    assert curriculum.phase_for_step(2) is MSDCurriculumPhase.TEXT_ONLY
    assert curriculum.phase_for_step(3) is MSDCurriculumPhase.MULTIMODAL
    assert curriculum.multimodal_probability(3) == 0.5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"text_only_steps": -1, "multimodal_ramp_steps": 0}, "text_only_steps"),
        ({"text_only_steps": 0, "multimodal_ramp_steps": -1}, "multimodal_ramp_steps"),
        ({"text_only_steps": 0, "multimodal_ramp_steps": 0, "final_multimodal_ratio": 1.1}, "ratio"),
        ({"text_only_steps": 0, "multimodal_ramp_steps": 0, "seed": -1}, "seed"),
    ],
)
def test_msd_curriculum_rejects_invalid_configuration(kwargs, message) -> None:
    """Invalid schedule settings fail before training starts."""
    with pytest.raises(ValueError, match=message):
        MSDCurriculumConfig(**kwargs)


def test_msd_curriculum_rejects_invalid_steps_and_checkpoint_state() -> None:
    """Invalid caller or checkpoint state cannot silently alter the schedule."""
    curriculum = MSDCurriculum(MSDCurriculumConfig(text_only_steps=0, multimodal_ramp_steps=0))

    with pytest.raises(ValueError, match="global_step"):
        curriculum.multimodal_probability(-1)
    with pytest.raises(ValueError, match="global_step"):
        MSDCurriculum(curriculum.config, global_step=-1)
    with pytest.raises(ValueError, match="global_step"):
        curriculum.load_state_dict({"global_step": "3"})
