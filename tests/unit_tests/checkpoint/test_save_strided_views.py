# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file

from nemo_automodel.components.checkpoint.checkpointing import CheckpointingConfig


@pytest.mark.parametrize("save_format", ["torch_save", "safetensors"])
def test_save_model_strided_view_roundtrip(tmp_path, monkeypatch, save_format):
    """DCP preserves model aliases; safetensors receives independent contiguous tensors."""
    model = torch.nn.Linear(6, 4, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.arange(24).reshape(4, 6))
    view = model.weight.detach().t()
    expected = view.clone()
    checkpointer = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=str(tmp_path),
        model_save_format=save_format,
        is_async=False,
        save_consolidated=False,
        model_repo_id=None,
    ).build(dp_rank=0, tp_rank=0, pp_rank=0)
    # Isolate the adapter layout and offline HF metadata; use the real save_model and writer.
    monkeypatch.setattr(
        "nemo_automodel.components.checkpoint.checkpointing._maybe_adapt_state_dict_to_hf",
        lambda *args, **kwargs: {"weight": view},
    )
    monkeypatch.setattr(checkpointer, "_maybe_build_consolidated_index", lambda *args: {"weight": 1})
    monkeypatch.setattr(checkpointer, "_maybe_build_original_dtype_mapping", lambda *args: None)
    checkpointer._addons = []
    original_save = checkpointer._do_save
    observed = []

    def inspect_and_save(state_dict, *args, **kwargs):
        """Inspect exported [input, output] weight storage before invoking the real writer."""
        tensor = state_dict["weight"]
        aliases = tensor.untyped_storage().data_ptr() == model.weight.untyped_storage().data_ptr()
        assert aliases == (save_format == "torch_save")
        assert tensor.is_contiguous() == (save_format == "safetensors")
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
        observed.append(True)
        return original_save(state_dict, *args, **kwargs)

    monkeypatch.setattr(checkpointer, "_do_save", inspect_and_save)
    destination = tmp_path / "step_2"
    checkpointer.save_model(model, str(destination))
    assert observed == [True]
    if save_format == "torch_save":
        restored = {"weight": torch.empty_like(view)}
        dcp.load(restored, checkpoint_id=destination / "model")
    else:
        restored = {}
        for path in (destination / "model").glob("*.safetensors"):
            restored.update(load_file(path))
    torch.testing.assert_close(restored["weight"], expected, rtol=0, atol=0)
    torch.testing.assert_close(model.weight.t(), expected, rtol=0, atol=0)
