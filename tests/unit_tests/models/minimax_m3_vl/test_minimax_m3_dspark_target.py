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

"""DSpark target features captured from a MiniMax M3 VL target."""

import pytest
import torch

from nemo_automodel.components.speculative.dspark.target import HFDSparkTargetModel

HIDDEN = 64


def _capture(model, layer_ids):
    """Run the DSpark target capture and record each decoder block's output stream.

    Returns:
        The DSpark target batch and a mapping from layer index to a tensor of
        shape [batch, sequence, hidden] holding that block's residual output.
    """
    block_outputs = {}

    def record(key):
        def hook(_module, _inputs, output):
            """Store ``output``, a tensor of shape [batch, sequence, hidden]."""
            block_outputs[key] = output

        return hook

    handles = [block.register_forward_hook(record(int(key))) for key, block in model.model.layers.items()]
    input_ids = torch.randint(0, 100, (2, 16), generator=torch.Generator().manual_seed(0))
    ones = torch.ones_like(input_ids)
    try:
        batch = HFDSparkTargetModel(model, target_layer_ids=layer_ids).generate_batch(input_ids, ones, ones)
    finally:
        for handle in handles:
            handle.remove()
    return batch, block_outputs, input_ids


def test_last_layer_feature_is_the_unnormalized_residual_stream(vlm_model):
    batch, block_outputs, _ = _capture(vlm_model, [0, 2])
    first, last = batch.target_hidden_states.split(HIDDEN, dim=-1)

    torch.testing.assert_close(first, block_outputs[0], rtol=0, atol=0)
    torch.testing.assert_close(last, block_outputs[2], rtol=0, atol=0)
    # lm_head still reads the normalized state.
    torch.testing.assert_close(batch.target_last_hidden_states, vlm_model.model.norm(block_outputs[2]), rtol=0, atol=0)
    assert not torch.allclose(last, batch.target_last_hidden_states)


def test_embedding_feature_is_the_stream_entering_layer_zero(vlm_model):
    batch, block_outputs, input_ids = _capture(vlm_model, [-1, 1])
    embedding, middle = batch.target_hidden_states.split(HIDDEN, dim=-1)

    torch.testing.assert_close(embedding, vlm_model.model.embed_tokens(input_ids), rtol=0, atol=0)
    torch.testing.assert_close(middle, block_outputs[1], rtol=0, atol=0)


@pytest.mark.parametrize("layer_ids", [[1, 0], [0, 0], [-2, 1], [0, 3]])
def test_feature_modules_reject_invalid_layer_ids(vlm_model, layer_ids):
    with pytest.raises(ValueError, match="DSpark target layer IDs"):
        vlm_model.get_dspark_target_feature_modules(layer_ids)
