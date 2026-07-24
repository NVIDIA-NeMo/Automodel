# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Tests for topology-independent per-sample corruption seeding.

The dLLM recipe seeds token corruption by each example's global index in the
shuffled data stream so the noise an example receives does not depend on the
parallel topology (dp_size / local_batch_size / grad-accum). This is what lets a
run at different node counts reproduce identical corruption (1-node vs 4-node
loss curves overlap) and makes corruption resume-safe.
"""

from nemo_automodel.recipes.dllm.train_ft import corruption_sample_seed


def _iter_step_samples(step, gbs, dp_size, local_batch_size):
    """Model StatefulDistributedSampler (strided) + grad-accum micro-batching.

    Yields ``(true_shuffled_pos, seed)`` for every sample processed at ``step``.
    Rank ``r`` owns shuffled positions ``r, r+dp_size, r+2*dp_size, ...`` and
    consumes them in order, grouped into micro-batches of ``local_batch_size``.
    """
    assert gbs % (local_batch_size * dp_size) == 0
    grad_acc = gbs // (local_batch_size * dp_size)
    for dp_rank in range(dp_size):
        for microbatch_idx in range(grad_acc):
            for offset in range(local_batch_size):
                rank_local_pos = microbatch_idx * local_batch_size + offset
                true_pos = step * gbs + dp_rank + rank_local_pos * dp_size
                seed = corruption_sample_seed(
                    base_seed=42,
                    step=step,
                    dp_rank=dp_rank,
                    dp_size=dp_size,
                    local_batch_size=local_batch_size,
                    grad_acc_steps=grad_acc,
                    microbatch_idx=microbatch_idx,
                    offset=offset,
                )
                yield true_pos, seed


# All (dp_size, local_batch_size) splits of a gbs=32 global batch.
_GBS = 32
_TOPOLOGIES = [(8, 1), (32, 1), (8, 4), (4, 8), (2, 16), (16, 2), (1, 32)]


def test_seed_is_topology_independent_per_sample():
    """The same example (same shuffled position) gets the same seed under any
    dp_size / local_batch_size / grad-accum split of the global batch."""
    for step in (0, 1, 7, 100):
        reference = None
        for dp_size, lbs in _TOPOLOGIES:
            by_pos = dict(_iter_step_samples(step, _GBS, dp_size, lbs))
            # A step must cover exactly its global-batch slice, once each.
            assert sorted(by_pos) == list(range(step * _GBS, step * _GBS + _GBS))
            if reference is None:
                reference = by_pos
            else:
                assert by_pos == reference, f"seed differs at dp_size={dp_size}, lbs={lbs}, step={step}"


def test_seed_is_deterministic():
    """Same coordinates -> same seed (resume reproduces corruption)."""
    kw = dict(
        base_seed=42, step=5, dp_rank=3, dp_size=8, local_batch_size=4, grad_acc_steps=1, microbatch_idx=0, offset=2
    )
    assert corruption_sample_seed(**kw) == corruption_sample_seed(**kw)


def test_distinct_samples_get_distinct_seeds():
    """Every example in a step draws a distinct seed (no accidental sharing)."""
    seeds = [seed for _, seed in _iter_step_samples(step=3, gbs=_GBS, dp_size=8, local_batch_size=1)]
    assert len(seeds) == _GBS
    assert len(set(seeds)) == _GBS


def test_seed_changes_across_steps():
    """The same shuffled position at different steps draws different noise."""
    common = dict(base_seed=42, dp_rank=0, dp_size=8, local_batch_size=1, grad_acc_steps=4, microbatch_idx=0, offset=0)
    assert corruption_sample_seed(step=0, **common) != corruption_sample_seed(step=1, **common)


def test_seed_within_torch_generator_range():
    """Seeds must be non-negative and fit the 63-bit mask for manual_seed."""
    for step in (0, 12345678):
        for _, seed in _iter_step_samples(step, _GBS, dp_size=8, local_batch_size=1):
            assert 0 <= seed <= (1 << 63) - 1
