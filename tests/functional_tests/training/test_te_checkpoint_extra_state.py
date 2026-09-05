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

"""Restore trusted TE checkpoints without dropping serialized quantizer state."""

import pytest
import torch
import torch.distributed.checkpoint as dcp

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.shared.import_utils import safe_import

HAVE_TE, te = safe_import("transformer_engine.pytorch")
pytestmark = pytest.mark.skipif(not HAVE_TE or not torch.cuda.is_available(), reason="requires TE and CUDA")


@pytest.mark.parametrize("saved_recipe", ["delayed", "current", "nvfp4"])
def test_te_extra_state_restore_and_runtime_precision(tmp_path, saved_recipe):
    """Restore state into a fresh module and honor the next runtime precision choice."""
    from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling, NVFP4BlockScaling
    from transformer_engine.pytorch.quantization import autocast

    if saved_recipe == "nvfp4" and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 requires Blackwell or newer")
    recipe = {"delayed": DelayedScaling, "current": Float8CurrentScaling, "nvfp4": NVFP4BlockScaling}[saved_recipe]()
    torch.manual_seed(123)
    source = te.Linear(128, 128, params_dtype=torch.bfloat16, device="cuda")
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with autocast(enabled=True, recipe=recipe):
        source(x).float().square().mean().backward()
    assert source.get_extra_state().numel() > 0
    # Only locally generated trusted payloads: upstream TE interprets its own
    # serialized byte tensor via pickle in set_extra_state.
    dcp.save(source.state_dict(), checkpoint_id=tmp_path)
    destination = te.Linear(128, 128, params_dtype=torch.bfloat16, device="cuda")
    assert destination.get_extra_state().numel() == 0
    checkpointer = Checkpointer(
        CheckpointingConfig(
            enabled=True, checkpoint_dir=str(tmp_path), model_save_format="safetensors", save_consolidated=False
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )
    checkpointer.load_model(destination, model_path=str(tmp_path))
    torch.testing.assert_close(destination.weight, source.weight)
    torch.testing.assert_close(destination.bias, source.bias)
    assert type(destination.fp8_meta["recipe"]) is type(recipe)
    if saved_recipe == "delayed":
        for direction in ("scaling_fwd", "scaling_bwd"):
            for key in ("scale", "amax_history"):
                torch.testing.assert_close(
                    getattr(destination.fp8_meta[direction], key), getattr(source.fp8_meta[direction], key)
                )

    # Loaded quantizer metadata must not force quantization on a BF16 forward.
    with autocast(enabled=False):
        output = destination(x.detach())
        expected = source(x.detach())
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        output.float().square().mean().backward()
    assert destination.fp8 is False
    assert torch.isfinite(destination.weight.grad).all()

    # A subsequent enabled scope must select the requested recipe, not the saved one.
    destination.zero_grad(set_to_none=True)
    with autocast(enabled=True, recipe=Float8CurrentScaling()):
        output = destination(x.detach())
        output.float().square().mean().backward()
    assert type(destination.fp8_meta["recipe"]) is Float8CurrentScaling
    assert destination.fp8 is True
    assert torch.isfinite(output).all()
    assert torch.isfinite(destination.weight.grad).all()
