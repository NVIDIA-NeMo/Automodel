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

"""Tests for nemo_automodel.components.training.config — StepSchedulerConfig."""

from nemo_automodel.components.training.config import StepSchedulerConfig


class TestStepSchedulerConfig:
    def test_defaults(self):
        cfg = StepSchedulerConfig()
        assert cfg.global_batch_size == 32
        assert cfg.num_epochs == 10
        assert cfg.max_steps is None
        assert cfg.ckpt_every_steps == 100
        assert cfg.save_checkpoint_every_epoch is True
        assert cfg.val_every_steps is None
        assert cfg.log_remote_every_steps == 1
        assert cfg.gc_every_steps is None
        assert cfg.start_step == 0
        assert cfg.start_epoch == 0

    def test_custom_values(self):
        cfg = StepSchedulerConfig(
            global_batch_size=64,
            num_epochs=5,
            max_steps=1000,
            ckpt_every_steps=200,
            val_every_steps=50,
            gc_every_steps=10,
        )
        assert cfg.global_batch_size == 64
        assert cfg.num_epochs == 5
        assert cfg.max_steps == 1000
        assert cfg.ckpt_every_steps == 200
        assert cfg.val_every_steps == 50
        assert cfg.gc_every_steps == 10

    def test_resume_fields(self):
        cfg = StepSchedulerConfig(start_step=500, start_epoch=2)
        assert cfg.start_step == 500
        assert cfg.start_epoch == 2
