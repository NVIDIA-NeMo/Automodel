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
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from nemo_automodel.components.checkpoint._backports.hf_storage import _HuggingFaceStorageReader
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.finetune import FinetuneRecipeForNextTokenPrediction


def load_dcp(ckpt_dir: Path | str) -> tuple[dict, dict]:
    """Load a DCP checkpoint (model or optimizer shard) from *ckpt_dir*."""
    if not isinstance(ckpt_dir, Path):
        ckpt_dir = Path(ckpt_dir)
    if "model" in ckpt_dir.name:
        fs_reader = _HuggingFaceStorageReader(ckpt_dir)
    else:
        fs_reader = dcp.FileSystemReader(ckpt_dir)

    metadata = fs_reader.read_metadata()

    tensor_state_dict = {
        k: torch.empty(tp.size, dtype=tp.properties.dtype)
        for k, tp in metadata.state_dict_metadata.items()
        if type(tp).__name__ == "TensorStorageMetadata"
    }

    dcp.load(
        tensor_state_dict,
        storage_reader=fs_reader,
    )

    # Load scheduler data (if present)
    sched_keys = [k for k, tp in metadata.state_dict_metadata.items() if "sched" in k]
    sched_state_dict = {}
    if sched_keys:
        sched_state_dict = {k: None for k in sched_keys}
        try:
            dcp.load(sched_state_dict, storage_reader=fs_reader)
        except Exception:
            sched_state_dict = {}

    return tensor_state_dict, sched_state_dict


def load_safetensors(ckpt_file: Path | str) -> dict[str, torch.Tensor]:
    """Utility to load tensors from a single `.safetensors` file into a dict."""
    if not isinstance(ckpt_file, Path):
        ckpt_file = Path(ckpt_file)
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(ckpt_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


@torch.no_grad()
def _add_scalar_parameter(model: nn.Module, optimizer: torch.optim.Optimizer, value: float = 0.5):
    """Register an extra 0-dim (scalar) parameter on *model* and (optionally) add it to *optimizer*.

    Args:
        model: The model instance to modify.
        optimizer: The optimizer used during training. The new parameter will be
            appended as its own parameter group so that it is checkpointed.
        value: Initial value for the scalar parameter.
    Returns:
        str: The full name of the newly-added parameter.
    """

    param_name = "scalar_weight"
    # Avoid clashes if the test is rerun within the same Python interpreter
    if hasattr(model, param_name):
        raise RuntimeError(f"Model already has an attribute called {param_name}")

    scalar_param = nn.Parameter(torch.tensor(value, dtype=model.dtype))
    model.register_parameter(param_name, scalar_param)

    # Make sure the parameter gets an optimizer state so that it is part of the
    # checkpoint just like regular trainable weights. Using a dedicated param
    # group avoids altering existing hyper-parameters.
    if optimizer is not None:
        optimizer.add_param_group({"params": [scalar_param]})

    return param_name


def test_consolidated_llm_checkpoint_with_scalar_weight():
    """End-to-end test that verifies a scalar parameter survives consolidated HF safetensors checkpointing.

    The workflow mirrors *test_consolidated_llm_checkpoint* but injects an extra
    0-dim parameter (`scalar_weight`) before training. After checkpoint/restore we
    confirm that the tensor exists, has correct shape/dtype/device, and retains
    its original value in both the sharded DCP view and the fully consolidated
    safetensors file.
    """

    script_path = Path(__file__).parent.resolve()
    cfg = parse_args_and_load_config(script_path / "llama_3_2_1b_hellaswag.yaml")

    trainer = FinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    scalar_value = 3.14159
    scalar_param_local = _add_scalar_parameter(trainer.model.model, trainer.optimizer, scalar_value)
    scalar_param_name = f"model.{scalar_param_local}"

    trainer.run_train_validation_loop()

    ckpt_dir = Path(trainer.checkpoint_config.checkpoint_dir) / "epoch_0_step_10"

    restored_model_dict, _ = load_dcp(ckpt_dir / "model")
    restored_model_dict_consolidated = load_safetensors(
        ckpt_dir / "model" / "consolidated" / "model-00001-of-00001.safetensors"
    )

    model_state_dict = ModelState(trainer.model).state_dict()

    new_model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(cfg.model.pretrained_model_name_or_path))
    _add_scalar_parameter(new_model.model, None, 0.0)

    # assert that the scalar parameter is not the same as `scalar_value`
    assert new_model.model.scalar_weight.item() == 0.0 and scalar_value != 0.0

    new_model.load_state_dict(restored_model_dict)
    assert torch.allclose(new_model.model.scalar_weight, trainer.model.model.scalar_weight), (
        "Scalar parameter mismatch between in-memory and consolidated checkpoint"
    )

    assert scalar_param_name in model_state_dict, "Scalar parameter missing from in-memory state dict"
    assert scalar_param_name in restored_model_dict, "Scalar parameter missing from sharded safetensors checkpoint"
    assert scalar_param_name in restored_model_dict_consolidated, (
        "Scalar parameter missing from consolidated safetensors checkpoint"
    )

    # Retrieve tensors (DCP returns CPU tensors)
    orig_tensor = model_state_dict[scalar_param_name]

    restored_tensor = restored_model_dict[scalar_param_name]
    consolidated_tensor = restored_model_dict_consolidated[scalar_param_name]

    # All tensors must be 0-dimensional scalars
    expected_shape = torch.Size([])
    assert orig_tensor.shape == expected_shape
    assert restored_tensor.shape == expected_shape
    assert consolidated_tensor.shape == expected_shape

    # Dtype/device consistency (everything should reside on CPU after load)
    expected_dtype = trainer.model.dtype
    assert restored_tensor.dtype == expected_dtype
    assert consolidated_tensor.dtype == expected_dtype
    assert str(restored_tensor.device) == "cpu"

    # Numerical equality
    assert torch.allclose(orig_tensor.cpu(), restored_tensor), (
        "Mismatch between in-memory and sharded safetensors scalar value"
    )
    assert torch.allclose(orig_tensor.cpu(), consolidated_tensor), (
        "Mismatch between in-memory and consolidated safetensors scalar value"
    )

    if torch.distributed.get_rank() == 0:
        if ckpt_dir.parent.exists():
            shutil.rmtree(ckpt_dir.parent)
    torch.distributed.barrier()
