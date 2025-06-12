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
import os
import pathlib
import pytest
import torch
import torch.nn.functional as F
from nemo_automodel.loss.masked_ce import masked_cross_entropy
from recipes.llm.finetune import FinetuneRecipeForNextTokenPrediction
from nemo_automodel.config.cli import parse_args_and_load_config
from nemo_automodel.checkpoint.stateful_wrappers import ModelState, OptimizerState
from pathlib import Path


def test_masked_cross_entropy_no_mask():
    """
    Tests masked_cross_entropy with no mask against baseline
    """
    # Create dummy data
    batch_size = 4
    num_classes = 3
    torch.manual_seed(0)
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(high=num_classes, size=(batch_size,))

    # Compute loss with our function
    loss_custom = masked_cross_entropy(logits, targets, mask=None)

    # Compute baseline cross-entropy
    loss_ref = F.cross_entropy(logits, targets)

    # They should be very close
    assert torch.allclose(
        loss_custom, loss_ref
    ), f"Loss without mask expected {loss_ref.item():.4f}, but got {loss_custom.item():.4f}"


def load_dcp(ckpt_dir, torch_tensor=True):
    from pathlib import Path

    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    if not isinstance(ckpt_dir, Path):
        ckpt_dir = Path(ckpt_dir)
    fs_reader = FileSystemReader(ckpt_dir)
    metadata = fs_reader.read_metadata()

    state_dict = {
        k: torch.empty(tp.size, dtype=tp.properties.dtype)
        for k, tp in metadata.state_dict_metadata.items()
        if type(tp).__name__ == 'TensorStorageMetadata'
    }

    dcp.load(
        state_dict,
        storage_reader=fs_reader,
    )
    return state_dict

def test_dcp_checkpoint():
    """
    Tests DCP checkpoint
    """
    script_path = pathlib.Path(__file__).parent.resolve()
    cfg = parse_args_and_load_config(script_path / "llama_3_2_1b_hellaswag.yaml")
    trainer = FinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()

    # checkpoint is saved at this point
    # first extract the in-memory checkpoint
    model_state_dict = ModelState(trainer.model, cfg.checkpoint.model_save_format).state_dict()
    optimizer_state_dict = OptimizerState(trainer.model, trainer.optimizer, trainer.step_scheduler).state_dict()

    # assert the correct paths exist
    output_files = ["model", "optim", "step_scheduler.pt", "dataloader.pt", "model/__0_0.distcp", "model/.metadata", "optim/__0_0.distcp", "optim/.metadata", "step_scheduler.pt"]
    # TODO:add files for 2 GPUs
    for file in output_files:
        path = Path(cfg.checkpoint.checkpoint_dir / file)
        assert path.exists(), f"Expected {path} to exist"
        assert path.is_file(), f"Expected {path} to be a file"
        assert os.access(path, os.R_OK), f"Expected {path} to be readable"
        assert path.stat().st_size > 0, f"Expected {path} to be non-empty"

    breakpoint()

test_dcp_checkpoint()