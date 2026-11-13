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

"""Two-GPU Inkling text training through the shared model parallelizer and DCP."""

import copy
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import cast

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
from nemo_automodel.components.distributed.mesh import ParallelismSizes
from nemo_automodel.components.distributed.model_parallelizer import parallelize_model
from nemo_automodel.components.loss.utils import calculate_loss, prepare_lm_weight
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.loss import Inkling2FusedLinearCrossEntropy
from nemo_automodel.components.models.inkling2.model import Inkling2ForCausalLM
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from tests.functional_tests.models.inkling2.test_training_cuda import compare_gradients
from tests.unit_tests.models.inkling2.test_precision import initialize_nonuniform_queries
from tests.unit_tests.models.inkling2.test_reference import tiny_config


@dataclass(frozen=True)
class Case:
    """A qualified storage/compute/backend combination at a fixed two-rank size."""

    dtype: str
    experts: str
    ep: int = 2
    checkpoint: bool = False
    attention: str = "eager"
    fused_loss: bool = False
    master_weights: bool = False
    public_factory: bool = False


CASES = {
    "fp32-ep2": Case("float32", "torch"),
    "fp32-ep1-public": Case("float32", "torch", ep=1, public_factory=True),
    "master-bf16-ep1-public": Case("float32", "torch_mm", ep=1, master_weights=True, public_factory=True),
    "bf16-ep1": Case("bfloat16", "torch_mm", ep=1),
    "bf16-ep2-ac": Case("bfloat16", "torch_mm", checkpoint=True),
    "bf16-ep2-flex-loss": Case("bfloat16", "torch_mm", attention="flex", fused_loss=True),
    "master-bf16-ep2-ac": Case("float32", "torch_mm", checkpoint=True, fused_loss=True, master_weights=True),
}


def full(tensor: torch.Tensor) -> torch.Tensor:
    """Gather a parameter or gradient without changing its owning storage.

    Args:
        tensor: Plain tensor or global-shaped DTensor of arbitrary parameter shape.

    Returns:
        Detached dense tensor with the same global shape and dtype.
    """
    return (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach()


def compare_model(model: nn.Module, reference: nn.Module, *, bf16: bool, gradients: bool) -> float:
    """Check every canonical parameter or gradient against the reference owner."""
    from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

    expected = dict(reference.named_parameters())
    maximum = 0.0
    seen = set()
    for name, parameter in model.named_parameters():
        name = canonical_parameter_fqn(name)
        seen.add(name)
        other = expected[name]
        if gradients:
            if not other.requires_grad:
                assert not parameter.requires_grad and parameter.grad is None and other.grad is None, name
                continue
            assert parameter.grad is not None and other.grad is not None, name
            maximum = max(maximum, compare_gradients(full(parameter.grad), other.grad, bf16=bf16, name=name))
        else:
            maximum = max(maximum, (full(parameter).float() - other.float()).abs().max().item())
            torch.testing.assert_close(
                full(parameter),
                other,
                rtol=0.01 if bf16 else 4e-4,
                atol=1e-4 if bf16 else 2e-6,
                msg=name,
            )
    assert seen == set(expected)
    return maximum


def batches(device: torch.device, *, empty_peer: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a shared [3, 11] batch with uneven rank lengths and optional empty supervision."""
    generator = torch.Generator(device=device).manual_seed(901 + empty_peer)
    ids = torch.randint(1, 37, (3, 11), generator=generator, device=device)
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[0, :2] = False
    valid[1, 9:] = False
    valid[2] = False
    labels = torch.randint(37, (3, 11), generator=generator, device=device)
    labels[~valid] = -100
    if empty_peer:
        valid[1:] = False
        labels[1:] = -100
    return ids, valid, labels


def _worker(case_name: str, output: Path) -> None:
    """Run a real FSDP/EP optimizer lifecycle and compare every owning gradient."""
    case = CASES[case_name]
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    dist.init_process_group("nccl", timeout=timedelta(seconds=150))
    try:
        torch.manual_seed(481)
        cfg = tiny_config(dtype=case.dtype)
        cfg.hidden_size, cfg.intermediate_size, cfg.n_routed_experts = 64, 64, 16
        # The EP1 BF16 lifecycle oracle matches local batch shapes and the
        # expert kernel. BF16 grouped and loop GEMMs round gradients differently;
        # the independent loop-kernel comparison lives in test_training_cuda.
        split_reference = case.ep == 1 and (case.dtype == "bfloat16" or case.master_weights)
        reference_backend = (
            BackendConfig(attn="eager", linear="torch", rms_norm="torch", experts="torch_mm", dispatcher="torch")
            if split_reference
            else None
        )
        reference = Inkling2ForCausalLM(cfg, backend=reference_backend).to(device)
        initialize_nonuniform_queries(reference)
        # Force some layers entirely onto rank zero's expert partition. This
        # exercises empty local assignments without replacing the real router.
        with torch.no_grad():
            for index in (0, 6, 12):
                reference.layers[str(index)].mlp.gate.e_score_correction_bias[:8] += 2
        backend = BackendConfig(
            attn=case.attention,
            linear="torch",
            rms_norm="torch",
            experts=case.experts,
            dispatcher="torch",
        )
        model = Inkling2ForCausalLM(copy.deepcopy(cfg), backend=backend).to(device)
        model.load_state_dict(reference.state_dict())
        frozen_convolution = case.master_weights and not case.public_factory
        if frozen_convolution:
            reference.layers["0"].attn_sconv.weight.requires_grad_(False)
            model.layers["0"].attn_sconv.weight.requires_grad_(False)
        initial = {name: value.detach().cpu().clone() for name, value in reference.state_dict().items()}
        compute = torch.bfloat16 if case.dtype == "bfloat16" or case.master_weights else torch.float32
        setup = DistributedSetup.build(
            strategy=FSDP2Config(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=compute,
                    reduce_dtype=torch.float32,
                    output_dtype=None,
                    cast_forward_inputs=False,
                ),
                reshard_after_forward=True,
                enable_fsdp2_prefetch=False,
            ),
            parallelism_sizes=ParallelismSizes(ep_size=case.ep),
            activation_checkpointing=case.checkpoint,
            world_size=2,
        )
        mesh = setup.mesh_context

        def distribute(candidate: Inkling2ForCausalLM) -> Inkling2ForCausalLM:
            if not case.public_factory:
                return cast(Inkling2ForCausalLM, parallelize_model(candidate, mesh))
            from nemo_automodel import NeMoAutoModelForCausalLM

            constructed = NeMoAutoModelForCausalLM.from_config(
                copy.deepcopy(cfg),
                dtype=torch.float32,
                attn_implementation="eager",
                backend=backend,
                use_liger_kernel=False,
                use_sdpa_patching=False,
                distributed_setup=setup,
            )
            set_model_state_dict(
                constructed,
                candidate.state_dict(),
                options=StateDictOptions(full_state_dict=True),
            )
            return cast(Inkling2ForCausalLM, constructed)

        model = distribute(model)
        assert isinstance(model, FSDPModule)
        expert = model.layers["0"].mlp.experts.gate_and_up_projs
        assert isinstance(expert, DTensor)
        if case.ep == 2:
            assert expert.to_local().shape[0] == 8 and expert.placements[0].is_shard(0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, foreach=False)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9, foreach=False)
        loss_fn = Inkling2FusedLinearCrossEntropy(37)
        metrics = {"max_logit_error": 0.0, "max_gradient_relative_l2": 0.0, "max_norm_relative_error": 0.0}

        pinned = ("attn_query", "mlp_query", "output_query", "q_l2_clip", "rel_l2_clip", "mlp.gate")

        def reference_forward(ids: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            """Apply explicit BF16 casts while preserving FP32 optimizer owners.

            Args:
                ids: Token IDs of shape [batch, sequence] on this rank's GPU.
                valid: Boolean validity mask of shape [batch, sequence].

            Returns:
                FP32 logits of shape [batch, sequence, vocab]; owning parameters are unchanged.
            """
            if case.master_weights:
                effective = {
                    name: parameter if any(key in name for key in pinned) else parameter.bfloat16()
                    for name, parameter in reference.named_parameters()
                }
                return torch.func.functional_call(reference, effective, (ids,), {"attention_mask": valid})
            return reference(ids, attention_mask=valid)

        def step(empty_peer: bool) -> None:
            ids, valid, labels = batches(device, empty_peer=empty_peer)
            rows = slice(0, 1) if rank == 0 else slice(1, 3)
            length = 11 if rank == 0 else 9
            optimizer.zero_grad(set_to_none=True)
            expected_optimizer.zero_grad(set_to_none=True)
            if split_reference:
                # Independent weights/optimizer persist across both steps. Model
                # each rank's BF16 backward before the FP32 mean reduction; never
                # copy the distributed model's updated weights into this oracle.
                accumulated = {
                    name: torch.zeros_like(p, dtype=torch.float32) for name, p in reference.named_parameters()
                }
                logits, losses = [], []
                for reference_rows, reference_length in ((slice(0, 1), 11), (slice(1, 3), 9)):
                    reference.zero_grad(set_to_none=True)
                    values = reference_forward(
                        ids[reference_rows, :reference_length], valid[reference_rows, :reference_length]
                    )
                    local_loss = F.cross_entropy(
                        values.flatten(0, 1), labels[reference_rows, :reference_length].flatten(), reduction="sum"
                    )
                    local_loss = local_loss / (labels != -100).sum() * 2
                    local_loss.backward()
                    for name, parameter in reference.named_parameters():
                        accumulated[name].add_(parameter.grad.float() / 2)
                    logits.append(F.pad(values.detach(), (0, 0, 0, 11 - reference_length)))
                    losses.append(local_loss.detach() / 2)
                for name, parameter in reference.named_parameters():
                    parameter.grad = accumulated[name].to(parameter.dtype)
                expected_logits, expected_loss = torch.cat(logits), sum(losses)
            else:
                expected_logits = reference_forward(ids, valid)
                expected_loss = F.cross_entropy(expected_logits.flatten(0, 1), labels.flatten())
                expected_loss.backward()
            result = model(
                ids[rows, :length],
                attention_mask=valid[rows, :length],
                output_hidden_states=case.fused_loss,
                logits_to_keep=1 if case.fused_loss else 0,
            )
            actual_logits = result.logits if case.fused_loss else result
            wanted = expected_logits[rows, :length]
            if case.fused_loss:
                wanted = wanted[:, -1:]
            delta = (actual_logits[..., :37] - wanted[..., :37]).abs().max().item()
            tolerance = 0.01 if compute == torch.bfloat16 else 1e-5
            failed = torch.tensor(int(delta >= tolerance), device=device)
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
            if failed.item():
                # Separate forward implementation error from optimizer-trajectory
                # drift, using the distributed model's current weights as input.
                from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

                current = {canonical_parameter_fqn(name): full(p) for name, p in model.named_parameters()}
                if case.master_weights:
                    current = {
                        name: value if any(key in name for key in pinned) else value.bfloat16()
                        for name, value in current.items()
                    }
                with torch.no_grad():
                    same_weights = torch.func.functional_call(reference, current, (ids,), {"attention_mask": valid})
                same_weights = same_weights[rows, :length]
                if case.fused_loss:
                    same_weights = same_weights[:, -1:]
                current_error = (actual_logits[..., :37] - same_weights[..., :37]).abs().max().item()
                raise AssertionError(
                    f"logit error={delta}, same-current-weight error={current_error}, earlier={metrics}"
                )
            metrics["max_logit_error"] = max(metrics["max_logit_error"], delta)
            count = int((labels != -100).sum())
            if case.fused_loss:
                weight = prepare_lm_weight(loss_fn, model, grad_reduce_group=dist.group.WORLD)
                loss = calculate_loss(
                    loss_fn,
                    model=model,
                    hidden_states=result.hidden_states,
                    labels=labels[rows, :length],
                    lm_weight=weight,
                    num_label_tokens=count,
                )
            else:
                loss = (
                    F.cross_entropy(actual_logits.flatten(0, 1), labels[rows, :length].flatten(), reduction="sum")
                    / count
                )
            (loss * 2).backward()  # Shared recipe compensation for FSDP's mean reduction.
            total_loss = loss.detach().clone()
            dist.all_reduce(total_loss)
            torch.testing.assert_close(total_loss, expected_loss, rtol=0.002, atol=2e-5)
            if split_reference:
                metrics["max_raw_gradient_relative_l2"] = max(
                    metrics.get("max_raw_gradient_relative_l2", 0.0),
                    compare_model(model, reference, bf16=True, gradients=True),
                )
            norm = scale_grads_and_clip_grad_norm(
                0.25,
                [model],
                device_mesh=mesh.device_mesh,
                moe_mesh=mesh.moe_mesh,
                ep_axis_name="ep" if case.ep > 1 else None,
                dp_group_size=2,
            )
            expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.25)
            norm_delta = abs(float(norm) - float(expected_norm)) / max(float(expected_norm), 1e-6)
            assert norm_delta < (0.02 if compute == torch.bfloat16 else 2e-4), norm_delta
            metrics["max_norm_relative_error"] = max(metrics["max_norm_relative_error"], norm_delta)
            metrics["max_gradient_relative_l2"] = max(
                metrics["max_gradient_relative_l2"],
                compare_model(model, reference, bf16=compute == torch.bfloat16, gradients=True),
            )
            optimizer.step()
            expected_optimizer.step()
            parameter_error = compare_model(model, reference, bf16=compute == torch.bfloat16, gradients=False)
            metrics["max_parameter_error"] = max(metrics.get("max_parameter_error", 0.0), parameter_error)
            print("STEP", rank, empty_peer, json.dumps(metrics), flush=True)

        step(False)
        state, opt_state = get_state_dict(model, optimizer)
        dcp.save({"model": state, "optimizer": opt_state}, checkpoint_id=output / "checkpoint")
        step(True)
        expected_resume = {name: full(parameter).clone() for name, parameter in model.named_parameters()}
        # Recreate parameter identities and optimizer groups, then restore both.
        restored = Inkling2ForCausalLM(copy.deepcopy(cfg), backend=backend).to(device)
        restored.load_state_dict(initial)
        if frozen_convolution:
            restored.layers["0"].attn_sconv.weight.requires_grad_(False)
        restored = distribute(restored)
        restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.01, momentum=0.9, foreach=False)
        restored_state, restored_opt = get_state_dict(restored, restored_optimizer)
        dcp.load({"model": restored_state, "optimizer": restored_opt}, checkpoint_id=output / "checkpoint")
        set_state_dict(restored, restored_optimizer, model_state_dict=restored_state, optim_state_dict=restored_opt)
        # Replay only the distributed step; compare to the uninterrupted result.
        ids, valid, labels = batches(device, empty_peer=True)
        rows, length = (slice(0, 1), 11) if rank == 0 else (slice(1, 3), 9)
        result = restored(
            ids[rows, :length],
            attention_mask=valid[rows, :length],
            output_hidden_states=case.fused_loss,
            logits_to_keep=1 if case.fused_loss else 0,
        )
        count = int((labels != -100).sum())
        if case.fused_loss:
            weight = prepare_lm_weight(loss_fn, restored, grad_reduce_group=dist.group.WORLD)
            loss = calculate_loss(
                loss_fn,
                model=restored,
                hidden_states=result.hidden_states,
                labels=labels[rows, :length],
                lm_weight=weight,
                num_label_tokens=count,
            )
        else:
            loss = F.cross_entropy(result.flatten(0, 1), labels[rows, :length].flatten(), reduction="sum") / count
        (loss * 2).backward()
        scale_grads_and_clip_grad_norm(
            0.25,
            [restored],
            device_mesh=mesh.device_mesh,
            moe_mesh=mesh.moe_mesh,
            ep_axis_name="ep" if case.ep > 1 else None,
            dp_group_size=2,
        )
        restored_optimizer.step()
        for name, parameter in restored.named_parameters():
            torch.testing.assert_close(full(parameter), expected_resume[name], rtol=0, atol=0, msg=name)
        metrics["parameter_tensors"] = len(expected_resume)
        if rank == 0:
            (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print("INKLING_EP_PASS", case_name, json.dumps(metrics), flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize("case_name", list(CASES))
def test_expert_parallel_training(
    case_name: str, tmp_path: Path, record_property: Callable[[str, float], None]
) -> None:
    """Use the shared parallelizer, loss scaling, clipping and DCP on two ranks."""
    repo = Path(__file__).resolve().parents[4]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(environment.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")[:2])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            __file__,
            "--worker",
            case_name,
            str(tmp_path),
        ],
        cwd=repo,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=420,
    )
    assert result.returncode == 0, result.stdout
    assert "INKLING_EP_PASS" in result.stdout, result.stdout
    for name, value in json.loads((tmp_path / "metrics.json").read_text()).items():
        record_property(name, value)


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker(sys.argv[-2], Path(sys.argv[-1]))
