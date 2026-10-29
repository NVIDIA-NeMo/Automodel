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

"""Real HybridEP training, FP8 initialization, and native AdamW continuation.

Use torchrun on two GPUs for EP2/shard1, or four for EP2/shard2. No training
dataset is used: this is a bounded numerical correctness fixture.
"""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.fsdp import MixedPrecisionPolicy

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.distributed import DistributedSetup, FSDP2Config, ParallelismSizes
from nemo_automodel.components.distributed.model_parallelizer import parallelize_model
from nemo_automodel.components.models.mistral_large4.model import MistralLarge4ForCausalLM
from nemo_automodel.components.moe.experts import GroupedExpertsDeepEP
from nemo_automodel.components.training.utils import (
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn
from tests.functional_tests.models.mistral_large4.validate_pipeline import align_reference_weights, full_value
from tests.unit_tests.models.mistral_large4.math_reference import tiny_config
from tests.unit_tests.models.mistral_large4.test_hybridep import grouped_backend, use_transport_rounding_reference
from tests.unit_tests.models.mistral_large4.test_model import assert_bf16_error, tensor_error


def batch(rank: int, step: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local integer IDs/labels [1, tokens], with unequal rank lengths."""
    ids = ((torch.arange(32 + rank % 2, device=device) * 3 + rank + step) % 29 + 1)[None]
    labels = ids.clone()
    if rank == 0 and step == 1:
        labels.fill_(-100)
    return ids, labels


def main() -> None:
    """Exercise actual dispatch, router gradients, expert reductions and resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ac", action="store_true")
    parser.add_argument(
        "--full-index", action="store_true", help="Isolate smooth gradient parity by selecting all causal keys"
    )
    parser.add_argument(
        "--match-transport-rounding",
        action="store_true",
        help="Use a single-process reference with HybridEP BF16 weighted-output rounding",
    )
    parser.add_argument(
        "--align-step-weights", action="store_true", help="Compare each update from identical current weights"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world in (2, 4)
    context = DistributedSetup.build(
        strategy=FSDP2Config(
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=None, cast_forward_inputs=False
            ),
            defer_fsdp_grad_sync=False,
            reshard_after_forward=True,
        ),
        parallelism_sizes=ParallelismSizes(ep_size=2),
        activation_checkpointing=args.ac,
        world_size=world,
    ).mesh_context
    config = tiny_config(
        dtype=torch.bfloat16, hidden_size=256, intermediate_size=384, moe_intermediate_size=256, n_routed_experts=4
    )
    if args.full_index:
        config.index_topk = 64
    config.quantization_config = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    torch.manual_seed(17)
    reference = MistralLarge4ForCausalLM(config, grouped_backend()).to(device)
    reference.initialize_weights()
    if args.match_transport_rounding:
        use_transport_rounding_reference(reference)
    raw = {}
    source_weights = reference.state_dict_adapter.to_hf(reference.state_dict())
    for name, value in source_weights.items():
        if any(token in name for token in (".wq.", ".wo.", ".wq_b.", ".w1.", ".w2.", ".w3.")):
            raw[name] = value.to(torch.float8_e4m3fn)
            raw[name.removesuffix(".weight") + ".qscale_weight"] = torch.full(
                ((value.shape[0] + 127) // 128, (value.shape[1] + 127) // 128),
                0.5,
                dtype=torch.float32,
                device=value.device,
            )
        else:
            raw[name] = value

    expected = reference.state_dict_adapter.from_hf(raw)
    reference.load_state_dict(expected)
    source = args.output / "source"
    if rank == 0:
        source.mkdir(parents=True, exist_ok=True)
        config.save_pretrained(source)
        save_file({key: value.cpu().contiguous() for key, value in raw.items()}, source / "model.safetensors")
    dist.barrier()
    model = parallelize_model(MistralLarge4ForCausalLM(config, grouped_backend("hybridep")).to(device), context)
    experts = [module for module in model.modules() if isinstance(module, GroupedExpertsDeepEP)]
    assert experts and all(module.ep_size == 2 and module.dispatcher_backend == "hybridep" for module in experts)
    checkpointer = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=str(args.output),
        model_save_format="safetensors",
        save_consolidated=False,
        model_repo_id=str(source),
        dequantize_base_checkpoint=True,
        cpu_offload=True,
    ).build(dp_rank=rank, tp_rank=0, pp_rank=0, moe_mesh=context.moe_mesh)
    checkpointer.load_model(model, str(source), is_init_step=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(full_value(value), expected[canonical_parameter_fqn(name)].cpu(), atol=0, rtol=0)
    routes: dict[str, dict[str, torch.Tensor]] = {"actual": {}, "reference": {}}
    handles = []

    def capture_routes(owner: str, name: str):
        """Capture detached int64 [tokens, topk] DSA/router selections for diagnostics."""

        def capture(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            """Observe FP32 gate logits [tokens, experts] or integer DSA indices [tokens, keys]."""
            selected = output[1] if name.endswith("mlp.gate") else output
            routes[owner][name] = selected.detach().sort(dim=-1).values.cpu()

        return capture

    for owner, candidate in (("actual", model), ("reference", reference)):
        for name, module in candidate.named_modules():
            name = canonical_parameter_fqn(name)
            if name.endswith(".indexer") or name.endswith("mlp.gate"):
                handles.append(module.register_forward_hook(capture_routes(owner, name)))
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.003)
    ref_optimizer = torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr=0.003)
    maximum = {"logits": 0.0, "gradient_relative_l2": 0.0, "norm": 0.0, "update": 0.0}
    for iteration, accumulation in enumerate((1, 3, 2)):
        if args.align_step_weights:
            align_reference_weights([model], reference)
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)
        prepare_for_grad_accumulation([model])
        for micro in range(accumulation):
            if micro == accumulation - 1:
                prepare_for_final_backward([model])
            ids, labels = batch(rank, iteration + micro, device)
            actual = model(ids, labels=labels)
            expected_output = reference(ids, labels=labels)
            route_report = {}
            for name, selected in routes["actual"].items():
                ref_selected = routes["reference"][name]
                mismatch = (selected != ref_selected).any(dim=-1)
                route_report[name] = {
                    "rows": selected.shape[0],
                    "different_rows": int(mismatch.sum()),
                    "actual": selected[mismatch].tolist(),
                    "reference": ref_selected[mismatch].tolist(),
                }
            (args.output / f"routes-step{iteration}-micro{micro}-rank{rank}.json").write_text(
                json.dumps(route_report, indent=2)
            )
            assert_bf16_error(tensor_error("logits", actual.logits, expected_output.logits))
            maximum["logits"] = max(maximum["logits"], float((actual.logits - expected_output.logits).abs().max()))
            (actual.loss / accumulation).backward()
            for peer in range(world):
                ri, rl = batch(peer, iteration + micro, device)
                (reference(ri, labels=rl).loss / (world * accumulation)).backward()
        actual_norm = scale_grads_and_clip_grad_norm(
            float("inf"), [model], moe_mesh=context.moe_mesh, ep_axis_name="ep", dp_group_size=world, foreach=False
        )
        ref_params = dict(reference.named_parameters())
        squared_error, squared_norm = 0.0, 0.0
        gradient_errors = []
        for name, parameter in model.named_parameters():
            ref = ref_params[canonical_parameter_fqn(name)]
            if not parameter.requires_grad:
                assert parameter.grad is None and ref.grad is None
                continue
            actual_grad, ref_grad = full_value(parameter.grad).float(), ref.grad.detach().float().cpu()
            assert torch.isfinite(actual_grad).all(), name
            gradient_errors.append(tensor_error(name, actual_grad, ref_grad))
            squared_error += float((actual_grad - ref_grad).double().square().sum())
            squared_norm += float(ref_grad.double().square().sum())
        relative = (squared_error / squared_norm) ** 0.5
        (args.output / f"gradients-step{iteration}-rank{rank}.json").write_text(
            json.dumps(
                [
                    {
                        "name": error.name,
                        "relative_l2": error.relative_l2,
                        "rms_error": error.rms_error,
                        "reference_rms": error.reference_rms,
                        "max_absolute": error.max_absolute,
                    }
                    for error in gradient_errors
                ],
                indent=2,
            )
        )
        for error in gradient_errors:
            assert_bf16_error(error)
        assert relative < 0.005, relative
        maximum["gradient_relative_l2"] = max(maximum["gradient_relative_l2"], relative)
        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), float("inf"), foreach=False)
        torch.testing.assert_close(actual_norm.float(), expected_norm.float(), atol=0.003, rtol=0.02)
        maximum["norm"] = max(maximum["norm"], abs(float(actual_norm) - float(expected_norm)))
        factor = min(1.0, 0.25 / (float(actual_norm) + 1e-6)) if iteration == 1 else 1.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(factor)
        if iteration == 1:
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.25, foreach=False)
        optimizer.step()
        ref_optimizer.step()
        for name, parameter in model.named_parameters():
            actual_value, ref_value = full_value(parameter), ref_params[canonical_parameter_fqn(name)].detach().cpu()
            torch.testing.assert_close(actual_value, ref_value, atol=0.003, rtol=0.04, msg=name)
            maximum["update"] = max(maximum["update"], float((actual_value - ref_value).abs().max()))
    for handle in handles:
        handle.remove()
    # The native checkpoint contains real AdamW moments and survives continuation.
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    def update(step: int) -> torch.Tensor:
        """Perform an AdamW update and return the detached scalar local mean loss."""
        optimizer.zero_grad(set_to_none=True)
        prepare_for_final_backward([model])
        ids, labels = batch(rank, step, device)
        loss = model(ids, labels=labels).loss
        loss.backward()
        scale_grads_and_clip_grad_norm(
            0.5, [model], moe_mesh=context.moe_mesh, ep_axis_name="ep", dp_group_size=world, foreach=False
        )
        optimizer.step()
        scheduler.step()
        return loss.detach().clone()

    update(5)
    checkpoint = str(args.output / "native")
    checkpointer.save_model(model, checkpoint)
    checkpointer.save_optimizer(optimizer, model, checkpoint, scheduler)
    saved_parameters = {name: full_value(value).clone() for name, value in model.state_dict().items()}
    saved_moments = {
        canonical_parameter_fqn(name): {key: full_value(value).clone() for key, value in optimizer.state[p].items()}
        for name, p in model.named_parameters()
        if p in optimizer.state
    }
    expected_loss = update(6)
    expected_parameters = {name: full_value(value).clone() for name, value in model.state_dict().items()}
    expected_moments = {
        canonical_parameter_fqn(name): {key: full_value(value).clone() for key, value in optimizer.state[p].items()}
        for name, p in model.named_parameters()
        if p in optimizer.state
    }
    expected_scheduler = scheduler.state_dict()
    checkpointer.load_model(model, checkpoint + "/model", is_init_step=False)
    optimizer.state.clear()
    checkpointer.load_optimizer(optimizer, model, checkpoint, scheduler)
    restore_errors = {}
    for name, value in model.state_dict().items():
        restored, saved = full_value(value), saved_parameters[name]
        difference = float((restored - saved).abs().max())
        if difference:
            restore_errors[name] = difference
    (args.output / f"restore-errors-rank{rank}.json").write_text(json.dumps(restore_errors, indent=2))
    assert not restore_errors, restore_errors
    for name, parameter in model.named_parameters():
        name = canonical_parameter_fqn(name)
        if parameter.requires_grad:
            restored_state = optimizer.state[parameter]
            assert set(restored_state) == set(saved_moments[name]), name
            for key, value in restored_state.items():
                torch.testing.assert_close(
                    full_value(value), saved_moments[name][key], atol=0, rtol=0, msg=f"{name}.{key}"
                )
    actual_loss = update(6)
    torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(full_value(value), expected_parameters[name], atol=0, rtol=0, msg=name)
    actual_moments = {
        canonical_parameter_fqn(name): state
        for name, parameter in model.named_parameters()
        if (state := optimizer.state.get(parameter)) is not None
    }
    assert set(actual_moments) == set(expected_moments)
    for name, actual_state in actual_moments.items():
        expected_state = expected_moments[name]
        assert set(actual_state) == set(expected_state), name
        for key, value in actual_state.items():
            torch.testing.assert_close(full_value(value), expected_state[key], atol=0, rtol=0, msg=f"{name}.{key}")
    assert scheduler.state_dict() == expected_scheduler
    checkpointer.close()
    result = {
        "passed": True,
        "rank": rank,
        "world_size": world,
        "ep_size": 2,
        "full_index_fixture": args.full_index,
        "reference_matches_transport_rounding": args.match_transport_rounding,
        "reference_weights_aligned_each_step": args.align_step_weights,
        "expert_shard": world // 2,
        "activation_checkpointing": args.ac,
        "fp8_import_exact": True,
        "native_adamw_continuation_exact": True,
        "accumulation_windows": [1, 3, 2],
        "errors": maximum,
    }
    (args.output / f"rank{rank}.json").write_text(json.dumps(result, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
