# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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


"""Sequence bucketing of the Inkling2 flex attention."""

import pytest
import torch

from nemo_automodel.components.models.inkling2 import flex_attention
from nemo_automodel.components.models.inkling2.flex_attention import Inkling2FlexAttention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not flex_attention._HAVE_AUX,
    reason="Inkling2 flex attention needs CUDA and FlexAttention auxiliary outputs",
)

BATCH, HEADS, KV_HEADS, HEAD_DIM, EXTENT = 2, 8, 2, 64, 16


def _run(attention: Inkling2FlexAttention, sequence: int, seed: int = 0):
    """Attention output on valid tokens and the gradients of a fixed linear readout.

    Args:
        attention: Module under test.
        sequence: Number of tokens; the second row marks its last 100 tokens invalid.
        seed: Seed of the random inputs.

    Returns:
        Output of shape [valid tokens, heads, head_dim] and a dict of input gradients.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shapes = {
        "q": (BATCH, sequence, HEADS, HEAD_DIM),
        "k": (BATCH, sequence, KV_HEADS, HEAD_DIM),
        "v": (BATCH, sequence, KV_HEADS, HEAD_DIM),
        "profiles": (BATCH, HEADS, sequence, EXTENT),
        "sink": (BATCH, sequence, HEADS),
    }
    inputs = {
        name: torch.randn(shape, device="cuda", generator=generator).requires_grad_(True)
        for name, shape in shapes.items()
    }
    valid = torch.ones(BATCH, sequence, dtype=torch.bool, device="cuda")
    valid[1, sequence - 100 :] = False
    output = attention(
        inputs["q"], inputs["k"], inputs["v"], profiles=inputs["profiles"], sink=inputs["sink"], valid=valid
    )
    assert output.shape == (BATCH, sequence, HEADS, HEAD_DIM)
    readout = torch.linspace(-1, 1, HEAD_DIM, device="cuda")
    (output[valid] * readout).sum().backward()
    return output[valid].detach(), {name: tensor.grad for name, tensor in inputs.items()}


@pytest.mark.parametrize("window", [None, 128])
def test_bucketed_sequence_matches_the_unpadded_computation(monkeypatch, window):
    """A length between two buckets gives the values and gradients of the unpadded computation."""
    attention = Inkling2FlexAttention(head_dim=HEAD_DIM, window=window).cuda()
    output, gradients = _run(attention, 700)
    monkeypatch.setattr(flex_attention, "_SEQUENCE_BUCKET", 1)
    expected_output, expected_gradients = _run(attention, 700)
    # The padded kernel sums in another order: compare whole tensors, not single small elements.
    assert (output - expected_output).norm() <= 1e-5 * expected_output.norm()
    for name, expected in expected_gradients.items():
        assert (gradients[name] - expected).norm() <= 1e-4 * expected.norm(), name


def test_lengths_in_one_bucket_reuse_the_compiled_kernel():
    """A second length of the same bucket runs under a limit that any newly compiled shape would exceed."""
    attention = Inkling2FlexAttention(head_dim=HEAD_DIM, window=None).cuda()
    # Lengths no other test of this module compiles: the compiled-code cache is shared by the process.
    _run(attention, 600)
    with torch._dynamo.config.patch(recompile_limit=1, accumulated_recompile_limit=1):
        _run(attention, 650, seed=1)
