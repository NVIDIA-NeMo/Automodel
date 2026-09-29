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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import triton
from triton.runtime.autotuner import Autotuner

from tests.functional_tests.checkpoint_robustness import test_checkpoint_robustness_llm as harness


class _Model(torch.nn.Module):
    def __init__(self, kernel: Autotuner) -> None:
        super().__init__()
        self.kernel = kernel

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None, use_cache: bool) -> torch.Tensor:
        """Expose the real autotuner's selected tile as toy logits.

        Args:
            input_ids: Integer tensor of shape [batch, sequence].
            attention_mask: Unused tensor of shape [batch, sequence].
            use_cache: Unused model-cache setting.

        Returns:
            Tensor of shape [batch, sequence, 1].
        """
        tile = self.kernel.run(128)
        return input_ids.float().unsqueeze(-1) * tile


@pytest.fixture
def kernel():
    # Run Triton's actual cache/config selection on CPU; only kernel execution
    # is replaced. A cached tile-4 choice stands in for an earlier training call.
    fn = SimpleNamespace(fn=lambda: None, run=Mock(side_effect=lambda *args, **kwargs: kwargs["BLOCK_SIZE_H"]))
    tuner = Autotuner(
        fn,
        arg_names=["chunk_size"],
        configs=[triton.Config({"BLOCK_SIZE_H": 1}), triton.Config({"BLOCK_SIZE_H": 4})],
        key=["chunk_size"],
        reset_to_zero=None,
        restore_value=None,
    )
    tuner.cache[(128,)] = tuner.configs[1]
    return tuner


@pytest.mark.parametrize("fail_forward", [False, True])
def test_mamba_parity_bypasses_cached_tile_and_restores_training(monkeypatch, kernel, fail_forward):
    module = SimpleNamespace(_chunk_cumsum_fwd_kernel=kernel)
    monkeypatch.setattr(harness, "safe_import", lambda name: (True, module))
    configs, cache = kernel.configs, kernel.cache.copy()
    model = _Model(kernel)
    if fail_forward:
        kernel.fn.run.side_effect = RuntimeError("forward failed")
        with pytest.raises(RuntimeError, match="forward failed"):
            harness._get_logits(model, [1, 2], torch.device("cpu"))
        assert kernel.fn.run.call_args.kwargs["BLOCK_SIZE_H"] == 1
        kernel.fn.run.side_effect = lambda *args, **kwargs: kwargs["BLOCK_SIZE_H"]
    else:
        logits = harness._get_logits(model, [1, 2], torch.device("cpu"))
        torch.testing.assert_close(logits, torch.tensor([[[1.0], [2.0]]]), rtol=0, atol=0)

    assert kernel.configs is configs
    assert kernel.cache == cache
    output = model(torch.tensor([[1, 2]]), None, False)
    torch.testing.assert_close(output, torch.tensor([[[4.0], [8.0]]]), rtol=0, atol=0)


def test_mamba_parity_allows_missing_optional_dependency(monkeypatch, kernel):
    monkeypatch.setattr(harness, "safe_import", lambda name: (False, None))
    logits = harness._get_logits(_Model(kernel), [1, 2], torch.device("cpu"))
    torch.testing.assert_close(logits, torch.tensor([[[4.0], [8.0]]]), rtol=0, atol=0)


def test_mamba_parity_rejects_missing_fixed_tile(monkeypatch, kernel):
    kernel.configs = kernel.configs[1:]
    module = SimpleNamespace(_chunk_cumsum_fwd_kernel=kernel)
    monkeypatch.setattr(harness, "safe_import", lambda name: (True, module))
    with pytest.raises(RuntimeError, match="BLOCK_SIZE_H=1"):
        harness._get_logits(_Model(kernel), [1, 2], torch.device("cpu"))
    kernel.fn.run.assert_not_called()
