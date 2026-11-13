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

"""Public PP2 recipe: stage ownership, FP32 AdamW, BF16 compute and fresh-process resume."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, get_optimizer_state_dict
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor


def _worker(
    schedule: str, output: Path, restore: str, *, dp_size: int = 1, ep_size: int = 1, attention: str = "eager"
) -> None:
    """Run the actual shared recipe and record independent continuation evidence."""
    from nemo_automodel.components.config.loader import load_yaml_config
    from nemo_automodel.components.models.inkling2.loss import Inkling2FusedLinearCrossEntropy
    from nemo_automodel.components.models.inkling2.model import Inkling2ForCausalLM
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    # Exact replay is tested under deterministic kernels; parity bounds are unchanged.
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = load_yaml_config(Path(__file__).with_name("tiny_pipeline.yaml"))
    config.distributed.pipeline.pp_schedule = schedule
    config.distributed.ep_size = ep_size
    assert attention in ("eager", "flex"), attention
    config.model.backend.attn = attention
    config.step_scheduler.global_batch_size = 8 * dp_size
    config.step_scheduler.num_epochs = dp_size
    config.checkpoint.checkpoint_dir = str(output)
    if restore != "none":
        config.checkpoint.restore_from = restore
    recipe = TrainFinetuneRecipeForNextTokenPrediction(config)
    try:
        recipe.setup()
        parts = recipe.model_parts
        optimizers = recipe.optimizer
        model = parts[0]
        # Capture every local stage's gradients before the trainer steps and
        # clears the first optimizer (each subsequent optimizer owns another stage).
        optimizer = optimizers[0]
        assert len(parts) == (2 if schedule == "interleaved1f1b" else 1)
        owners = {name: p for part in parts for name, p in part.named_parameters()}
        all_names = [None, None]
        dist.all_gather_object(all_names, list(owners), group=recipe.mesh_context.device_mesh["pp"].get_group())
        assert not set(all_names[0]) & set(all_names[1])
        assert len(set(all_names[0]) | set(all_names[1])) == 220
        assert isinstance(model, Inkling2ForCausalLM) and isinstance(model, FSDPModule)
        assert isinstance(recipe.loss_fn, Inkling2FusedLinearCrossEntropy)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert recipe.step_scheduler.grad_acc_steps == 2
        assert recipe.step_scheduler.step == (0 if restore == "none" else 1)
        assert all(p.dtype == torch.float32 for p in owners.values())
        assert all(
            (layer.attn.flex is not None) == (attention == "flex") for part in parts for layer in part.layers.values()
        )
        first_layer = next(iter(model.layers.values()))
        experts = first_layer.mlp.experts.gate_and_up_projs
        assert isinstance(experts, DTensor) and experts.to_local().shape[0] == 16 // ep_size
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
            for name, parameter in owners.items():
                assert parameter.grad is not None, name
                grad = parameter.grad
                grad = grad.to_local() if isinstance(grad, DTensor) else grad
                assert torch.isfinite(grad).all(), name
                current[name] = grad.detach().cpu().clone()
            gradients.append(current)

        forward_handle = (
            model.register_forward_pre_hook(record_batch, with_kwargs=True) if model.embed_tokens is not None else None
        )
        step_handle = optimizer.register_step_pre_hook(record_gradients)
        projection_handle = first_layer.attn.qkvr.register_forward_pre_hook(check_projection_dtype)
        router_handle = first_layer.mlp.gate.register_forward_pre_hook(check_router_dtype)
        recipe.run_train_validation_loop()
        projection_handle.remove()
        router_handle.remove()
        if forward_handle is not None:
            forward_handle.remove()
        step_handle.remove()
        assert recipe.step_scheduler.step == 3
        assert len(gradients) == (3 if restore == "none" else 2)
        assert len(batches) == (8 * len(gradients) if model.embed_tokens is not None else 0)
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
        for state in (state for optim in optimizers for state in optim.state.values()):
            assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
            assert state["step"].item() == 3
        states = []
        for part, optim in zip(parts, optimizers):
            options = StateDictOptions(full_state_dict=True, cpu_offload=False)
            states.append(
                {
                    "model": get_model_state_dict(part, options=options),
                    "optimizer": get_optimizer_state_dict(part, optim, options=options),
                }
            )
        torch.save({"batches": batches, "gradients": gradients, "states": states}, output / f"rank-{rank}.pt")
        (output / f"receipt-{rank}.json").write_text(
            json.dumps(
                {
                    "optimizer_steps": len(gradients),
                    "gradient_accumulation_steps": 2,
                    "pipeline_microbatches_per_step": 8,
                    "owned_parameter_tensors": len(owners),
                    "local_stages": len(parts),
                    "final_step": recipe.step_scheduler.step,
                    "master_dtype": "float32",
                    "compute_dtype": "bfloat16",
                    "optimizer": "AdamW",
                    "pp_size": 2,
                    "ep_size": ep_size,
                    "dp_size": dp_size,
                    "schedule": schedule,
                    "attention": attention,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Inkling recipe qualification requires two CUDA devices")
@pytest.mark.parametrize("schedule", ["1f1b", "interleaved1f1b"])
def test_public_pipeline_recipe_resume(schedule: str, tmp_path: Path) -> None:
    """Resume in fresh processes and require bitwise-equal data, gradients, weights and Adam moments."""
    run_recipe_resume(schedule, tmp_path)


def run_recipe_resume(
    schedule: str, tmp_path: Path, *, dp_size: int = 1, ep_size: int = 1, attention: str = "eager"
) -> None:
    """Run the shared recipe twice and compare continuation on every local shard.

    The default is the two-GPU PR fixture. Scheduled composition callers explicitly
    request DP2, requiring four GPUs for PP2 with optional EP2.

    Args:
        schedule: Shared pipeline schedule name.
        tmp_path: Directory for baseline and resumed run evidence.
        dp_size: Data-parallel degree; total rank count is twice this value.
        ep_size: Expert-parallel degree within each pipeline stage.
        attention: Native attention backend, eager or flex.
    """
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[4]) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(
        environment.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")[: 2 * dp_size]
    )
    baseline, resumed = tmp_path / "baseline", tmp_path / "resumed"
    for output, restore in ((baseline, "none"), (resumed, str(baseline / "epoch_0_step_0"))):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={2 * dp_size}",
                __file__,
                "--worker",
                schedule,
                str(output),
                restore,
                str(dp_size),
                str(ep_size),
                attention,
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

    for rank in range(2 * dp_size):
        actual, expected = load(resumed / f"rank-{rank}.pt"), load(baseline / f"rank-{rank}.pt")
        torch.testing.assert_close(actual["batches"], expected["batches"][8:], rtol=0, atol=0)
        torch.testing.assert_close(actual["gradients"], expected["gradients"][1:], rtol=0, atol=0)
        assert len(actual["states"]) == len(expected["states"])
        for state, reference in zip(actual["states"], expected["states"]):
            torch.testing.assert_close(state["model"], reference["model"], rtol=0, atol=0)
            torch.testing.assert_close(state["optimizer"]["state"], reference["optimizer"]["state"], rtol=0, atol=0)
            assert state["optimizer"]["param_groups"] == reference["optimizer"]["param_groups"]
    assert list((baseline / "epoch_0_step_0" / "model").glob("*.safetensors"))


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker(
        sys.argv[-6],
        Path(sys.argv[-5]),
        sys.argv[-4],
        dp_size=int(sys.argv[-3]),
        ep_size=int(sys.argv[-2]),
        attention=sys.argv[-1],
    )
