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

"""Public two-GPU recipe: FP32 AdamW, BF16 compute, accumulation and exact resume."""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor


@dataclass
class TinyDatasetConfig:
    """Deterministic map-style fixture; every row has one next-token target shift."""

    def build(self) -> list[dict[str, list[int]]]:
        """Return 32 unpacked rows, including left/right padding and empty supervision."""
        generator = torch.Generator().manual_seed(922)
        examples = []
        for index in range(32):
            ids = torch.randint(1, 37, (11,), generator=generator)
            valid = torch.ones(11, dtype=torch.bool)
            if index % 4 == 0:
                valid[:2] = False
            elif index % 4 == 1:
                valid[-2:] = False
            labels = torch.cat((ids[1:], torch.tensor([-100])))
            labels[:-1].masked_fill_(~(valid[:-1] & valid[1:]), -100)
            if index % 8 == 3:
                labels.fill_(-100)
            examples.append({"input_ids": ids.tolist(), "attention_mask": valid.tolist(), "labels": labels.tolist()})
        return examples


def _worker(output: Path, restore: str) -> None:
    """Run the actual shared recipe and record independent continuation evidence."""
    from nemo_automodel.components.config.loader import load_yaml_config
    from nemo_automodel.components.models.inkling2.loss import Inkling2FusedLinearCrossEntropy
    from nemo_automodel.components.models.inkling2.model import Inkling2ForCausalLM
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = load_yaml_config(Path(__file__).with_name("tiny_train.yaml"))
    config.checkpoint.checkpoint_dir = str(output)
    if restore != "none":
        config.checkpoint.restore_from = restore
    recipe = TrainFinetuneRecipeForNextTokenPrediction(config)
    try:
        recipe.setup()
        model = recipe.model_parts[0]
        optimizer = recipe.optimizer[0]
        assert isinstance(model, Inkling2ForCausalLM) and isinstance(model, FSDPModule)
        assert isinstance(recipe.loss_fn, Inkling2FusedLinearCrossEntropy)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert recipe.step_scheduler.grad_acc_steps == 4
        assert recipe.step_scheduler.step == (0 if restore == "none" else 1)
        assert all(p.dtype == torch.float32 for p in model.parameters())
        experts = model.layers["0"].mlp.experts.gate_and_up_projs
        assert isinstance(experts, DTensor) and experts.to_local().shape[0] == 8
        batches, gradients = [], []

        def check_projection_dtype(module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            """Inspect the effective projection weight during its real forward.

            Args:
                module: Projection with gathered weight of shape [output, hidden].
                args: Hidden input tensor of shape [batch, sequence, hidden].
            """
            assert module.weight.dtype == torch.bfloat16

        def check_router_dtype(module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            """Inspect the FP32 router during its real forward.

            Args:
                module: Router with gathered weight of shape [experts, hidden].
                args: Hidden input tensor of shape [tokens, hidden].
            """
            assert module.weight.dtype == torch.float32

        def record_batch(module: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
            """Save each local forward's data for exact loader continuation checks.

            Args:
                module: Wrapped causal LM, unchanged by this hook.
                args: Optional positional input_ids tensor of shape [batch, sequence].
                kwargs: Named input_ids/attention_mask/position_ids tensors of shape
                    [batch, sequence], plus scalar forward options. Inputs are unchanged.
            """
            ids = kwargs.get("input_ids", args[0] if args else None)
            assert isinstance(ids, torch.Tensor)
            batches.append(ids.detach().cpu().clone())

        def record_gradients(optim: torch.optim.Optimizer, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
            """Copy local owning gradients after clipping; optimizer arguments are unchanged.

            Args:
                optim: AdamW owning FP32 DTensors. Gradients have arbitrary parameter
                    shapes; copies retain each rank's local DP/EP shard shape.
                args: Positional optimizer arguments (no tensor inputs).
                kwargs: Keyword optimizer arguments (no tensor inputs).
            """
            current = {}
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, name
                grad = parameter.grad
                grad = grad.to_local() if isinstance(grad, DTensor) else grad
                assert torch.isfinite(grad).all(), name
                current[name] = grad.detach().cpu().clone()
            gradients.append(current)

        forward_handle = model.register_forward_pre_hook(record_batch, with_kwargs=True)
        step_handle = optimizer.register_step_pre_hook(record_gradients)
        projection_handle = model.layers["0"].attn.qkvr.register_forward_pre_hook(check_projection_dtype)
        router_handle = model.layers["0"].mlp.gate.register_forward_pre_hook(check_router_dtype)
        recipe.run_train_validation_loop()
        projection_handle.remove()
        router_handle.remove()
        forward_handle.remove()
        step_handle.remove()
        assert recipe.step_scheduler.step == 3
        assert len(gradients) == (3 if restore == "none" else 2)
        assert len(batches) == 4 * len(gradients)
        for family in ("relative_table", "qkvr", "sconv", "mlp.gate", "experts", "output_query"):
            # Singleton query weights can leave one FSDP rank with no local
            # elements. Check the owning shards collectively, including that rank.
            nonzero = torch.tensor(
                int(
                    any(
                        torch.count_nonzero(value)
                        for step in gradients
                        for name, value in step.items()
                        if family in name
                    )
                ),
                device=recipe.dist_env.device,
            )
            dist.all_reduce(nonzero, op=dist.ReduceOp.MAX)
            assert nonzero.item(), family
        for state in optimizer.state.values():
            assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
            assert state["step"].item() == 3
        state, optim_state = get_state_dict(
            model, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        torch.save({"batches": batches, "gradients": gradients}, output / f"rank-{rank}.pt")
        if rank == 0:
            torch.save({"model": state, "optimizer": optim_state}, output / "final.pt")
            (output / "receipt.json").write_text(
                json.dumps(
                    {
                        "optimizer_steps": len(gradients),
                        "gradient_accumulation_steps": 4,
                        "parameter_tensors": len(list(model.parameters())),
                        "final_step": recipe.step_scheduler.step,
                        "master_dtype": "float32",
                        "compute_dtype": "bfloat16",
                        "optimizer": "AdamW",
                        "ep_size": 2,
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Inkling recipe qualification requires two CUDA devices")
def test_public_recipe_resume(tmp_path: Path) -> None:
    """Resume in fresh processes and require bitwise-equal data, gradients, weights and Adam moments."""
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(environment.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")[:2])
    baseline, resumed = tmp_path / "baseline", tmp_path / "resumed"
    for output, restore in ((baseline, "none"), (resumed, str(baseline / "epoch_0_step_0"))):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                __file__,
                "--worker",
                str(output),
                restore,
            ],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
        )
        (tmp_path / f"{output.name}.log").write_text(result.stdout)
        assert result.returncode == 0, result.stdout

    def load(path: Path) -> dict:
        return torch.load(path, map_location="cpu", weights_only=True)

    actual, expected = load(resumed / "final.pt"), load(baseline / "final.pt")
    assert actual.keys() == expected.keys() == {"model", "optimizer"}
    torch.testing.assert_close(actual["model"], expected["model"], rtol=0, atol=0)
    assert actual["optimizer"].keys() == expected["optimizer"].keys() == {"state", "param_groups"}
    torch.testing.assert_close(actual["optimizer"]["state"], expected["optimizer"]["state"], rtol=0, atol=0)
    # Parameter-group metadata includes FQN strings, which assert_close does
    # not support. Python equality still requires their exact values and order.
    assert actual["optimizer"]["param_groups"] == expected["optimizer"]["param_groups"]
    for rank in range(2):
        actual, expected = load(resumed / f"rank-{rank}.pt"), load(baseline / f"rank-{rank}.pt")
        torch.testing.assert_close(actual["batches"], expected["batches"][4:], rtol=0, atol=0)
        torch.testing.assert_close(actual["gradients"], expected["gradients"][1:], rtol=0, atol=0)
    assert list((baseline / "epoch_0_step_0" / "model").glob("*.safetensors"))


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker(Path(sys.argv[-2]), sys.argv[-1])
