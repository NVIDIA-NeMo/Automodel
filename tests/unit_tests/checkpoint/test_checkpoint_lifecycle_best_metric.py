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

"""CheckpointLifecycle must honor the comparison direction of the selected
best-checkpoint metric, through the real reserve/publish/retention path.

Covers checkpoint.best_metric_key naming a higher-is-better custom evaluator
metric (e.g. tool_call_eval's tool_call/args_exact_match): the LOWEST_VAL
pointer, and checkpoint retention, must track the maximum rather than the
minimum, while the default (loss, lower-is-better) behavior is unchanged.
"""

import json

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.checkpoint.lifecycle import CheckpointLifecycle
from nemo_automodel.components.checkpoint.utils import read_checkpoint_pointer


def _make_checkpoint(lifecycle, tmp_path, step, metric_value, metric_key):
    """Reserve, populate, and publish one checkpoint directory."""
    path = tmp_path / f"epoch_0_step_{step}"
    lifecycle.reserve(str(path))
    (path / "losses.json").write_text(json.dumps({metric_key: metric_value}))
    lifecycle.publish(str(path), best_val_metric=metric_value, metric_key=metric_key)
    return path


def test_greater_is_better_metric_promotes_the_maximum(tmp_path):
    metric_key = "tool_call/args_exact_match"
    config = CheckpointingConfig(
        checkpoint_dir=str(tmp_path),
        save_consolidated=False,
        greater_is_better=True,
    )
    lifecycle = CheckpointLifecycle(config=config)

    _make_checkpoint(lifecycle, tmp_path, 1, 0.2, metric_key)
    best = _make_checkpoint(lifecycle, tmp_path, 2, 0.9, metric_key)
    _make_checkpoint(lifecycle, tmp_path, 3, 0.1, metric_key)

    lowest_val_target = read_checkpoint_pointer(tmp_path, "LOWEST_VAL")
    assert lowest_val_target == best.resolve()


def test_greater_is_better_metric_protects_best_checkpoint_from_retention(tmp_path):
    metric_key = "tool_call/args_exact_match"
    config = CheckpointingConfig(
        checkpoint_dir=str(tmp_path),
        save_consolidated=False,
        greater_is_better=True,
        max_recent_checkpoints=1,
    )
    lifecycle = CheckpointLifecycle(config=config)

    _make_checkpoint(lifecycle, tmp_path, 1, 0.2, metric_key)
    best = _make_checkpoint(lifecycle, tmp_path, 2, 0.9, metric_key)
    _make_checkpoint(lifecycle, tmp_path, 3, 0.1, metric_key)

    # max_recent_checkpoints=1 retains only the most recent window plus
    # pointer-protected checkpoints; the 0.9 checkpoint must survive pruning.
    assert best.exists()


def test_lower_is_better_default_is_unchanged(tmp_path):
    config = CheckpointingConfig(checkpoint_dir=str(tmp_path), save_consolidated=False)
    lifecycle = CheckpointLifecycle(config=config)

    best = _make_checkpoint(lifecycle, tmp_path, 1, 0.8, "val_loss")
    _make_checkpoint(lifecycle, tmp_path, 2, 2.0, "val_loss")

    lowest_val_target = read_checkpoint_pointer(tmp_path, "LOWEST_VAL")
    assert lowest_val_target == best.resolve()
