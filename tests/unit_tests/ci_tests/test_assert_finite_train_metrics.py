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

from tests.ci_tests.scripts.assert_finite_train_metrics import assert_finite_train_metrics


def test_assert_finite_train_metrics_accepts_finite_steps(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "step 0 | epoch 0 | loss 12.1606 | grad_norm 2.4152 | lr 1e-5\n"
        "step 1 | epoch 0 | loss 12.1000 | grad_norm 1.7500 | lr 1e-5\n"
    )

    assert assert_finite_train_metrics(log) == 0


def test_assert_finite_train_metrics_rejects_nan_grad_norm(tmp_path, capsys):
    log = tmp_path / "train.log"
    log.write_text("step 0 | epoch 0 | loss 12.2486 | grad_norm nan | lr 1e-5\n")

    assert assert_finite_train_metrics(log) == 1
    assert "step 0 grad_norm=nan" in capsys.readouterr().out


def test_assert_finite_train_metrics_requires_a_training_step(tmp_path, capsys):
    log = tmp_path / "train.log"
    log.write_text("setup complete\n")

    assert assert_finite_train_metrics(log) == 1
    assert "no training-step" in capsys.readouterr().out
