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

"""Run the production recipe with topology-independent random test weights."""

import hashlib
import json
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor, distribute_tensor


@torch.no_grad()
def initialize_parity_weights(parts: list[torch.nn.Module]) -> dict[str, str]:
    """Assign identical global tensors to each named parameter across PP layouts.

    Args:
        parts: Local recipe model parts containing parameters of arbitrary shapes.
            DTensor parameters keep their device meshes and placements.

    Returns:
        SHA256 fingerprints of the global BF16/FP32 parameter values, by name.
    """
    fingerprints = {}
    for part in parts:
        for name, param in part.named_parameters():
            name = name.replace("_checkpoint_wrapped_module.", "")
            generator = torch.Generator().manual_seed(
                int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
            )
            full = torch.empty(param.shape, dtype=torch.float32)
            if param.ndim == 1:
                full.fill_(1.0 if "norm" in name else 0.0)
            else:
                full.normal_(mean=0.0, std=0.02, generator=generator)
            full = full.to(param.dtype)
            fingerprints[name] = hashlib.sha256(full.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            if isinstance(param, DTensor):
                # Every rank constructs identical global values; slicing needs
                # no cross-stage broadcast or DTensor random-number collective.
                value = distribute_tensor(
                    full.to(param.device), param.device_mesh, param.placements, src_data_rank=None
                )
            else:
                value = full.to(param.device)
            param.copy_(value)
    return fingerprints


def main() -> None:
    """Retain production initialization, optimizer, training and validation paths."""
    from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    cfg = parse_args_and_load_config("tests/functional_tests/parallelism_multigpu/llama.yaml")
    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    # Setup must run first: it exercises the real stage-specific initializers
    # whose DTensor RNG collectives previously hung on PP4 attention/MoE stages.
    fingerprints = initialize_parity_weights(recipe.model_parts)
    output = Path(cfg.checkpoint.checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"initial_weights.{torch.distributed.get_rank()}.json").write_text(
        json.dumps(fingerprints, sort_keys=True)
    )
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
