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

"""CUDA graph regressions for the DeepSeek-V4 text input boundary."""

import pytest
import torch

from tests.unit_tests.models.deepseek_v4.test_dsv4_model_smoke import _make_model, _tiny_config

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA graph capture"),
    # Native HC/norm compilation is warmed before graph capture.
    pytest.mark.timeout(180),
]


@pytest.mark.parametrize("provide_embeddings", [False, True])
def test_text_capture_replays_forward_backward(provide_embeddings: bool) -> None:
    """Capture the native input boundary and preserve eager outputs and gradients."""
    torch.manual_seed(71)
    # No transformer blocks: isolate the changed input boundary, HC head and norm.
    config = _tiny_config(num_hidden_layers=0, num_hash_layers=0, compress_ratios=[])
    model = _make_model(config).cuda()
    model.initialize_weights(buffer_device=torch.device("cuda"), dtype=torch.float32)
    input_ids = torch.arange(8, device="cuda").unsqueeze(0)
    embeddings = model.model.embed_tokens(input_ids).detach().requires_grad_(True) if provide_embeddings else None
    upstream = torch.randn(1, 8, config.vocab_size, device="cuda")

    expected = model(input_ids, inputs_embeds=embeddings).logits
    expected.backward(upstream)
    expected_output = expected.detach().clone()
    expected_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    expected_embedding_grad = embeddings.grad.detach().clone() if embeddings is not None else None
    del expected

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            if embeddings is not None:
                embeddings.grad = None
            model(input_ids, inputs_embeds=embeddings).logits.backward(upstream)
        model.zero_grad(set_to_none=True)
        if embeddings is not None:
            embeddings.grad = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            logits = model(input_ids, inputs_embeds=embeddings).logits
            logits.backward(upstream)

        graph.replay()
        actual_output = logits.detach().clone()
        actual_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        actual_embedding_grad = embeddings.grad.detach().clone() if embeddings is not None else None
        with torch.no_grad():
            input_ids.add_(1)
            if embeddings is not None:
                embeddings.copy_(model.model.embed_tokens(input_ids))
        graph.replay()
        changed_output = logits.detach().clone()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-6)
    assert actual_grads.keys() == expected_grads.keys()
    for name, expected_grad in expected_grads.items():
        torch.testing.assert_close(actual_grads[name], expected_grad, rtol=1e-5, atol=1e-6)
    if embeddings is not None:
        torch.testing.assert_close(actual_embedding_grad, expected_embedding_grad, rtol=1e-5, atol=1e-6)
    with torch.no_grad():
        changed_expected = model(input_ids, inputs_embeds=embeddings).logits
    torch.testing.assert_close(changed_output, changed_expected, rtol=1e-5, atol=1e-6)
    assert not torch.equal(changed_output, actual_output)
    graph.reset()


@pytest.mark.parametrize("metadata", ["vision_token_types", "pixel_values", "image_grid_hws"])
def test_text_capture_rejects_visual_inputs(metadata: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the Python metadata guard separately from actual capture/replay."""
    config = _tiny_config(num_hidden_layers=0, num_hash_layers=0, compress_ratios=[])
    model = _make_model(config).cuda()
    input_ids = torch.zeros(1, 8, device="cuda", dtype=torch.long)
    visual_inputs = {metadata: torch.full_like(input_ids, -1)}
    # A Python rejection would invalidate an active CUDA capture. Test the guard
    # outside capture; the preceding test exercises the real capture API.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(ValueError, match="Text-only CUDA graphs require no visual inputs"):
        model(input_ids, **visual_inputs)


def test_eager_text_input_validation_still_rejects_negative_ids() -> None:
    """The capture fast path must leave ordinary input validation in effect."""
    config = _tiny_config(num_hidden_layers=0, num_hash_layers=0, compress_ratios=[])
    model = _make_model(config).cuda()
    input_ids = torch.zeros(1, 8, device="cuda", dtype=torch.long)
    input_ids[0, 0] = -1
    with pytest.raises(ValueError, match="input_ids cannot be negative"):
        model(input_ids)
