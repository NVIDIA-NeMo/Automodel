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

import gc
import hashlib
import json
from copy import deepcopy
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


def run_recipe_pair_in_process(config: Path, overrides: list[str], pp_size: int, output: Path) -> None:
    """Run PP and its reference in the same workers with equal global batches.

    The reference uses all ranks for data parallelism. Its local batch shrinks
    by PP size so the global batch and accumulation count remain unchanged.
    """
    from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
    from nemo_automodel.components.moe.megatron.fused_a2a import reset_hybrid_ep_buffer
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    cfg = parse_args_and_load_config(argv=["--config", str(config), *overrides])
    local_batch = int(cfg.step_scheduler.local_batch_size)
    assert local_batch % pp_size == 0
    for name, pp, batch in (("parallel", pp_size, local_batch), ("baseline", 1, local_batch // pp_size)):
        current = deepcopy(cfg)
        directory = output / name
        current.set_by_dotted("checkpoint.checkpoint_dir", str(directory))
        current.set_by_dotted("distributed.pp_size", pp)
        current.set_by_dotted("step_scheduler.local_batch_size", batch)
        recipe = TrainFinetuneRecipeForNextTokenPrediction(current)
        recipe.setup()
        # Retain production initialization before assigning the shared fixture.
        fingerprints = initialize_parity_weights(recipe.model_parts)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"initial_weights.{torch.distributed.get_rank()}.json").write_text(
            json.dumps(fingerprints, sort_keys=True)
        )
        recipe.run_train_validation_loop()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        del recipe
        reset_hybrid_ep_buffer()
        gc.collect()
