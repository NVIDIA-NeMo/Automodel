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

"""Train LoRA adapters on the ERNIE 5.5 proxy with expert parallelism and check what the run changed.

Launched by ``L2_Ernie5_5_MoE_EP2_LoRA.sh`` under torchrun, one process per GPU. Every process builds the
fine-tuning recipe from the proxy configuration and the LoRA overrides of the command line, copies its shard of
every parameter and buffer, runs the training loop, and compares the shards afterwards:

* only the adapter tensors are trainable;
* every adapter tensor has changed on this rank;
* every other parameter and every buffer holds the same bytes as before.

Rank 0 then checks the training log and the adapter checkpoint that the loop saved at its last step. The loss
bound was measured on two GPUs with seed 1234 and eager DSA: the loss starts at 8.349, close to the ln(4096) =
8.318 of a randomly initialized model, and ends at 6.425 after sixteen updates on one repeated batch, 0.77 of its
first value.

Usage (under torchrun):
    run_ernie5_5_lora_training.py --config tests/functional_tests/moe/ernie5_5_proxy.yaml --peft.<key> <value> ...
"""

from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction
from tests.functional_tests.moe.check_ernie5_5_training_log import check_training_log

DEFAULT_CONFIG = "tests/functional_tests/moe/ernie5_5_proxy.yaml"
# The loss has to end below this fraction of its first value. The measured run passes it at the seventh of its
# sixteen updates; see the module docstring.
MAX_FINAL_LOSS_FRACTION = 0.85


def _local_shard(tensor: torch.Tensor) -> torch.Tensor:
    """Copy the elements of a parameter or buffer that this rank holds.

    Args:
        tensor: Parameter or buffer of any shape. A sharded tensor is a ``DTensor`` whose local shard holds this
            rank's slice along the sharded axis.

    Returns:
        Detached CPU copy with the shape of the local shard; the whole tensor when it is not a ``DTensor``.
    """
    tensor = tensor.detach()
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.cpu().clone()


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy this rank's shard of every parameter and buffer of a model.

    Args:
        model: The parallelized model. Its FSDP units are returned to their sharded state first.

    Returns:
        CPU copies keyed by ``parameter:<name>`` and ``buffer:<name>``.
    """
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()
    state = {f"parameter:{name}": _local_shard(parameter) for name, parameter in model.named_parameters()}
    state.update({f"buffer:{name}": _local_shard(buffer) for name, buffer in model.named_buffers()})
    return state


def _check_saved_adapter(checkpoint_dir: Path, model: nn.Module, adapters: set[str], ep_size: int) -> None:
    """Assert that the checkpoint of the last step holds the whole trained adapter and nothing else.

    Args:
        checkpoint_dir: Checkpoint directory of the run.
        model: The parallelized model.
        adapters: Names of the adapter parameters of ``model``.
        ep_size: Number of expert groups of the run. Rank ``r`` of the expert-parallel axis owns the ``r``-th of
            ``ep_size`` equal parts of the first axis of every grouped factor of the routed experts.

    Raises:
        AssertionError: If there is not exactly one checkpoint, a tensor is missing, unexpected, misshaped or not
            finite, a second adapter factor is still zero, or the part of one expert group is still zero in a
            second factor of the routed experts.
    """
    saved = sorted(checkpoint_dir.glob("epoch_*_step_*/model/adapter_model.safetensors"))
    assert len(saved) == 1, f"Expected one adapter checkpoint under {checkpoint_dir}, found {saved}"
    state = load_file(saved[0])
    # Every rank holds a shard of each adapter tensor; the file holds the whole tensors.
    expected = {f"base_model.model.{name}": tuple(model.get_parameter(name).shape) for name in adapters}
    assert {name: tuple(value.shape) for name, value in state.items()} == expected
    assert all(torch.isfinite(value.float()).all() for value in state.values())
    second_factors = [name for name in state if name.endswith(("lora_B.weight", "_B"))]
    assert len(second_factors) == len(state) // 2
    assert all(torch.count_nonzero(state[name]) for name in second_factors), "A saved adapter is untrained"
    # The second factors of the routed experts start at zero, and every rank trains the part of its own expert
    # group. A part that is still zero means that the file lacks what one rank trained.
    grouped = {
        name: [int(torch.count_nonzero(part)) for part in state[name].chunk(ep_size, dim=0)]
        for name in second_factors
        if name.endswith("_B")
    }
    for name, trained in grouped.items():
        assert all(trained), (
            f"{name} holds {trained} non-zero elements in the parts of its {ep_size} expert groups; the part of a "
            "rank is missing"
        )
    assert (saved[0].parent / "adapter_config.json").is_file()
    print(f"saved adapter OK: {len(state)} tensors in {saved[0]}")
    print(f"non-zero elements per expert group in the second factors of the routed experts: {grouped}")


def main() -> None:
    """Build the recipe, train, and compare the state of this rank before and after."""
    config = parse_args_and_load_config(DEFAULT_CONFIG)
    recipe = TrainFinetuneRecipeForNextTokenPrediction(config)
    recipe.setup()
    model = recipe.model_parts[0]
    rank = torch.distributed.get_rank()

    adapters = {name for name, _ in model.named_parameters() if "lora_" in name}
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert adapters, "The run installed no LoRA adapter"
    assert trainable == adapters, f"Trainable parameters that are not adapters: {sorted(trainable - adapters)}"

    before = _snapshot(model)
    recipe.run_train_validation_loop()
    after = _snapshot(model)

    assert before.keys() == after.keys()
    changed = {name for name, value in before.items() if not torch.equal(value, after[name])}
    expected = {f"parameter:{name}" for name in adapters}
    assert changed - expected == set(), f"Tensors outside the adapters changed: {sorted(changed - expected)}"
    assert expected - changed == set(), f"Adapter tensors that did not train: {sorted(expected - changed)}"
    print(f"rank {rank}: {len(changed)} adapter tensors changed, {len(before) - len(changed)} other tensors unchanged")

    if rank == 0:
        checkpoint_dir = Path(recipe.checkpointer.config.checkpoint_dir)
        # The first loss of a randomly initialized model does not show that the adapters start neutral; the unit
        # tests compare the output with and without them.
        check_training_log(
            str(checkpoint_dir / "training.jsonl"),
            config.step_scheduler.max_steps,
            config.model.config.vocab_size,
            max_final_fraction=MAX_FINAL_LOSS_FRACTION,
        )
        _check_saved_adapter(checkpoint_dir, model, adapters, config.distributed.ep_size)
        print("ERNIE 5.5 EP2 LoRA training OK")


if __name__ == "__main__":
    main()
