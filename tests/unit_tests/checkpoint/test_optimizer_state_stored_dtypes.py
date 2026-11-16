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

"""TE FusedAdam with sub-FP32 moments must be serialized in its stored dtypes.

FusedAdam.state_dict() widens BF16 moments to fresh FP32 GPU tensors, which exhausts memory when a
very large model is saved. The context below swaps in PyTorch's base state_dict while saving or
loading. A stand-in class named ``FusedAdam`` is used so the test needs no GPU or Transformer Engine.
"""

import pytest
import torch

from nemo_automodel.components.checkpoint.stateful_wrappers import (
    _optimizer_state_as_stored,
    _stores_low_precision_adam_state,
)


class FusedAdam(torch.optim.Adam):
    """Stand-in with the two attributes the checkpoint code reads; its state_dict must not be used."""

    def __init__(self, params, exp_avg_dtype=torch.float32, exp_avg_sq_dtype=torch.float32):
        super().__init__(params)
        self.name_to_dtype_map = {"exp_avg": exp_avg_dtype, "exp_avg_sq": exp_avg_sq_dtype}

    def state_dict(self):
        raise AssertionError("the widening FusedAdam.state_dict must not run inside the context")


def _param():
    return torch.nn.Parameter(torch.zeros(4))


def _with_bf16_state(opt, p):
    opt.state[p]["exp_avg"] = torch.zeros(4, dtype=torch.bfloat16)
    opt.state[p]["exp_avg_sq"] = torch.zeros(4, dtype=torch.bfloat16)
    opt.state[p]["step"] = torch.tensor(1.0)


def test_detects_low_precision_moments_only():
    p = _param()
    assert _stores_low_precision_adam_state(FusedAdam([p], exp_avg_dtype=torch.bfloat16))
    assert _stores_low_precision_adam_state(FusedAdam([p], exp_avg_sq_dtype=torch.bfloat16))
    assert not _stores_low_precision_adam_state(FusedAdam([p]))
    assert not _stores_low_precision_adam_state(torch.optim.AdamW([p]))


def test_context_serializes_stored_tensors_without_copies_and_restores():
    p = _param()
    opt = FusedAdam([p], exp_avg_dtype=torch.bfloat16, exp_avg_sq_dtype=torch.bfloat16)
    _with_bf16_state(opt, p)
    with _optimizer_state_as_stored([opt]):
        state = opt.state_dict()["state"][0]
    assert state["exp_avg"].dtype == torch.bfloat16
    assert state["exp_avg"].data_ptr() == opt.state[p]["exp_avg"].data_ptr()
    assert "state_dict" not in vars(opt)
    with pytest.raises(AssertionError, match="widening"):
        opt.state_dict()


def test_context_restores_after_an_exception():
    p = _param()
    opt = FusedAdam([p], exp_avg_dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        with _optimizer_state_as_stored([opt]):
            raise RuntimeError("save failed")
    assert "state_dict" not in vars(opt)


def test_full_precision_optimizers_are_left_alone():
    p = _param()
    fp32 = FusedAdam([p])
    adamw = torch.optim.AdamW([p])
    with _optimizer_state_as_stored([fp32, adamw]):
        assert "state_dict" not in vars(fp32)
        assert "state_dict" not in vars(adamw)
