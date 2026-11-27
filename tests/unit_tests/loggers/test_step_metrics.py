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

"""The step metrics take their loss, training, and FLOPs helpers from the recipe, as train_ft passes them."""

import pytest
import torch

from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.loggers.step_metrics import StepMetricReducer, TrainingStepReporter
from nemo_automodel.components.loss.utils import _count_label_tokens
from nemo_automodel.components.training.utils import count_tail_padding
from nemo_automodel.components.utils.flops_utils import calculate_mfu


def _no_reduce(tensor, /, *, include_cp=False, keep_device=False):
    return tensor


def test_token_counts_use_the_recipe_counters():
    reducer = StepMetricReducer(
        device=torch.device("cpu"),
        dp_reduce=_no_reduce,
        count_label_tokens=_count_label_tokens,
        count_tail_padding=count_tail_padding,
    )
    labels = torch.tensor([[-100, 1, 2, -100], [3, -100, -100, -100]])

    counts = reducer.token_counts([{"labels": labels}], ignore_index=-100, pack_size=4)

    # Supervised labels 1, 2, 3; eight positions minus four trailing pads; two packs.
    assert (counts.labels.item(), counts.tokens.item(), counts.packs.item()) == (3, 4, 2)


class _TeraflopPerBatch:
    def get_flops(self, input_ids, /):
        return 1e12


def test_flops_count_context_parallel_ranks_once():
    """CP ranks hold their DP replica's unsharded batch, so its FLOPs count once, like its tokens."""
    dp_size, cp_size = 2, 2

    def group_sum(tensor, /, *, include_cp=False, keep_device=False):
        return tensor * (dp_size * cp_size if include_cp else dp_size)

    reducer = StepMetricReducer(
        device=torch.device("cpu"),
        dp_reduce=group_sum,
        count_label_tokens=_count_label_tokens,
        count_tail_padding=count_tail_padding,
    )
    batch = {"labels": torch.zeros(1, 4, dtype=torch.long), "input_ids": torch.zeros(1, 4, dtype=torch.long)}

    assert reducer.flops([batch], _TeraflopPerBatch()).item() == dp_size * 1e12


def test_training_reporter_computes_mfu_with_the_recipe_formula():
    reporter = TrainingStepReporter(
        torch.device("cpu"),
        world_size=2,
        pack_size=0,
        domain_names=(),
        reference_mfu=400.0,
        calculate_mfu=calculate_mfu,
    )
    sample = MetricsSample(step=1, epoch=0, metrics={"num_tokens_per_step": 100, "_step_flops": 3e12})
    reporter.enqueue(sample, start=0.0, end=0.5, remote_log=False)

    ((done, remote_log, seconds),) = reporter.poll(wait=True)
    reporter.close()

    assert (remote_log, seconds) == (False, 0.5)
    # 3 TFLOPs over 2 GPUs in 0.5 s is 3 TFLOP/s per GPU, 0.75% of a 400 TFLOP/s peak.
    assert done.metrics["mfu"] == pytest.approx(0.75)
    assert done.metrics["tps"] == 200
    assert "_step_flops" not in done.metrics
