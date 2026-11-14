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

"""Two-GPU pipeline math, residual-source gradients and optimizer continuation."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

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
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
from nemo_automodel.components.distributed.mesh import ParallelismSizes
from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig
from nemo_automodel.components.loss.mtp import PipelineCausalLMLoss
from nemo_automodel.components.loss.utils import prepare_lm_weight
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.loss import Inkling2FusedLinearCrossEntropy
from nemo_automodel.components.models.inkling2.model import Inkling2ForCausalLM
from nemo_automodel.components.models.inkling2.parallelization import FP32_MODULES
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn
from tests.unit_tests.models.inkling2.test_pipeline import module_fqns
from tests.unit_tests.models.inkling2.test_precision import initialize_nonuniform_queries
from tests.unit_tests.models.inkling2.test_reference import tiny_config


@dataclass(frozen=True)
class Case:
    cuts: tuple[int, ...] | None
    bf16: bool = False
    ac: bool = False
    fused: bool = False
    flex: bool = False
    interleaved: bool = False
    dp: int = 1
    ep: int = 1
    ep_reference: bool = False


CASES = {
    "fp32-cut6": Case((0, 6, 13)),
    "fp32-auto-cut7": Case(None),
    "master-bf16-ac": Case(None, bf16=True, ac=True, fused=True),
    "master-bf16-interleaved": Case((0, 3, 6, 10, 13), bf16=True, ac=True, fused=True, interleaved=True),
    "master-bf16-flex": Case(None, bf16=True, ac=True, fused=True, flex=True),
}


def dense(tensor: torch.Tensor) -> torch.Tensor:
    """Gather one owning tensor for numerical assertions.

    Args:
        tensor: Parameter or gradient of arbitrary global shape, optionally a DTensor
            sharded on its data-parallel mesh; that mesh's ranks participate together.

    Returns:
        Detached dense tensor with the same global shape and dtype. It may alias input storage.
    """
    return (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach()


def _worker(name: str, output: Path, case: Case | None = None) -> None:
    from nemo_automodel import NeMoAutoModelForCausalLM

    case = CASES[name] if case is None else case
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.set_num_threads(1)
    # Exact replay is tested under deterministic kernels; parity bounds are unchanged.
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        torch.manual_seed(481)
        cfg = tiny_config()
        cfg.hidden_size = cfg.intermediate_size = 64
        cfg.n_routed_experts = 16
        backend = BackendConfig(
            attn="flex" if case.flex else "eager",
            linear="torch",
            rms_norm="torch",
            experts="torch_mm",
            dispatcher="torch",
        )
        reference = Inkling2ForCausalLM(
            cfg,
            backend=BackendConfig(
                attn="flex" if case.flex else "eager",
                linear="torch",
                rms_norm="torch",
                experts="torch_mm",
                dispatcher="torch",
            ),
        ).to(device)
        initialize_nonuniform_queries(reference)
        if case.ep > 1:
            # Exercise empty local expert assignments on both pipeline stages.
            with torch.no_grad():
                for index in (0, 6, 12):
                    reference.layers[str(index)].mlp.gate.e_score_correction_bias[:8] += 2
        initial = {name: tensor.detach().clone() for name, tensor in reference.state_dict().items()}
        # Lifecycle parity holds the attention kernel constant across independent
        # optimizer trajectories. Separately check the first step against eager;
        # cross-kernel BF16 updates can cross different rounding boundaries later.
        eager_oracle = None
        if case.flex:
            eager_oracle = Inkling2ForCausalLM(
                copy.deepcopy(cfg),
                backend=BackendConfig(
                    attn="eager", linear="torch", rms_norm="torch", experts="torch_mm", dispatcher="torch"
                ),
            ).to(device)
            eager_oracle.load_state_dict(initial)
        compute = torch.bfloat16 if case.bf16 else torch.float32

        def sum_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            """Sum CE over aligned targets.

            Args:
                logits: FP32 tensor of shape [batch, sequence, vocab].
                labels: Integer tensor of shape [batch, sequence], with -100 ignored.

            Returns:
                FP32 scalar loss, including differentiable zero for empty supervision.
            """
            return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum")

        config = PipelineConfig(
            pp_schedule="interleaved1f1b" if case.interleaved else "1f1b",
            pp_batch_size=4,
            pp_microbatch_size=1,
            dtype=compute,
            pp_seq_len=9,
            loss_fn=sum_ce,
            module_fqns_per_model_part=module_fqns(case.cuts) if case.cuts else None,
            patch_stage_backward_maybe_with_nosync=True,
        )
        setup = DistributedSetup.build(
            strategy=FSDP2Config(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=compute, reduce_dtype=torch.float32, output_dtype=None, cast_forward_inputs=False
                ),
                reshard_after_forward=True,
                enable_fsdp2_prefetch=False,
            ),
            parallelism_sizes=ParallelismSizes(pp_size=2, ep_size=case.ep),
            pipeline_config=config,
            activation_checkpointing=case.ac,
            world_size=2 * case.dp,
        )
        mesh = setup.mesh_context
        dp_rank = mesh.device_mesh["dp_cp"].get_local_rank()
        pp_rank = mesh.device_mesh["pp"].get_local_rank()
        pipeline = NeMoAutoModelForCausalLM.from_config(
            copy.deepcopy(cfg),
            dtype=torch.float32,
            backend=backend,
            attn_implementation="eager",
            use_liger_kernel=False,
            use_sdpa_patching=False,
            distributed_setup=setup,
        )
        assert isinstance(pipeline, AutoPipeline)
        parts = pipeline.parts
        for part in parts:
            state = {key: initial[key] for key in part.state_dict()}
            set_model_state_dict(part, state, options=StateDictOptions(full_state_dict=True))
        owners = {
            canonical_parameter_fqn(key): parameter for part in parts for key, parameter in part.named_parameters()
        }
        all_names = [None] * 2
        dist.all_gather_object(all_names, list(owners), group=mesh.device_mesh["pp"].get_group())
        assert not set(all_names[0]) & set(all_names[1])
        assert set(all_names[0]) | set(all_names[1]) == set(dict(reference.named_parameters()))
        assert all(p.dtype == torch.float32 for p in owners.values())
        if case.ep > 1:
            for part in parts:
                for layer in part.layers.values():
                    experts = layer.mlp.experts.gate_and_up_projs
                    assert isinstance(experts, DTensor)
                    assert experts.to_local().shape[0] == cfg.n_routed_experts // case.ep
                    assert "ep" in experts.device_mesh.mesh_dim_names
        metrics_topology = {
            "world_size": 2 * case.dp,
            "pp_rank": pp_rank,
            "dp_rank": dp_rank,
            "ep_size": case.ep,
            "attention": "flex" if case.flex else "eager",
        }
        fused = Inkling2FusedLinearCrossEntropy(37)
        captured = []
        if pipeline.info.has_last_stage:
            last = next(part for part in parts if part.lm_head is not None)
            last._pp_return_hidden_states = case.fused
            loss_module = (
                PipelineCausalLMLoss(
                    fused, last, grad_reduce_group=setup.mesh_context.device_mesh["dp_cp"].get_group()
                )
                if case.fused
                else None
            )

            def loss_fn(value: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                """Capture final output and delegate to the shared loss contract.

                Args:
                    value: Final-stage tensor [batch, sequence, hidden] in compute dtype
                        for fused CE, otherwise FP32 logits [batch, sequence, vocab].
                    labels: Aligned integer targets [batch, sequence], -100 ignored.

                Returns:
                    FP32 scalar summed loss; inputs remain unchanged.
                """
                # Capture the actual final-stage forward while weights are gathered.
                if case.fused:
                    with torch.no_grad():
                        logits = F.linear(value.float(), dense(last.lm_head.weight).to(compute).float())
                        logits = cfg.final_logit_softcapping * torch.tanh(logits / cfg.final_logit_softcapping)
                else:
                    logits = value
                captured.append(logits.detach())
                return loss_module(value, labels) if loss_module is not None else sum_ce(logits, labels)

            pipeline.info.schedule._loss_fn = loss_fn
        # Changing the final-output contract requires refreshing the shared metadata.
        pipeline.update_seq_len(11)
        wrapper = nn.ModuleList(parts)
        optimizer = torch.optim.SGD(wrapper.parameters(), lr=0.01, momentum=0.9, foreach=False)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9, foreach=False)
        expected_parameters = dict(reference.named_parameters())
        ep_reference = None
        ep_loss = None
        if case.ep_reference:
            from nemo_automodel.components.distributed.model_parallelizer import parallelize_model

            # Two independent non-pipeline EP2 replicas, one on each stage's DP
            # group. This isolates PP while preserving expert partitioning, GEMM
            # shapes and gradient accumulation. No weights are copied after init.
            ep_backend = copy.deepcopy(backend)
            ep_backend.enable_fsdp_optimizations = True
            ep_reference = Inkling2ForCausalLM(copy.deepcopy(cfg), backend=ep_backend).to(device)
            ep_reference.load_state_dict(initial)
            ep_reference = parallelize_model(ep_reference, mesh)
            expected_optimizer = torch.optim.SGD(ep_reference.parameters(), lr=0.01, momentum=0.9, foreach=False)
            expected_parameters = {canonical_parameter_fqn(key): p for key, p in ep_reference.named_parameters()}
            if case.fused:
                ep_loss = PipelineCausalLMLoss(
                    fused, ep_reference, grad_reduce_group=mesh.device_mesh["dp_cp"].get_group()
                )
        metrics = {
            "max_logit_error": 0.0,
            "max_gradient_relative_l2": 0.0,
            "max_norm_relative_error": 0.0,
            "max_parameter_error": 0.0,
            "max_loss_error": 0.0,
        }

        def reference_forward(
            ids: torch.Tensor, valid: torch.Tensor, *, reference_model: Inkling2ForCausalLM, hidden_states: bool = False
        ) -> torch.Tensor | CausalLMOutputWithPast:
            """Run independent FP32 master weights with matching compute casts.

            Args:
                ids: Integer token IDs [batch, sequence].
                valid: Boolean real-token mask [batch, sequence].
                reference_model: Independent unsharded model, unchanged except for gradients.
                hidden_states: Also return the final hidden states for the shared loss.

            Returns:
                FP32 logits [batch, sequence, vocab], or the output carrying them together with
                compute-dtype hidden states [batch, sequence, hidden]; inputs and owners are unchanged.
            """
            kwargs = {"attention_mask": valid, "output_hidden_states": hidden_states}
            if not case.bf16:
                return reference_model(ids, **kwargs)
            effective = {
                key: parameter if any(pin in key for pin in FP32_MODULES) else parameter.bfloat16()
                for key, parameter in reference_model.named_parameters()
            }
            return torch.func.functional_call(reference_model, effective, (ids,), kwargs)

        def step(index: int, *, compare: bool) -> None:
            length = 11 if index == 0 else 9
            ids = (
                (torch.arange(4 * case.dp * length, device=device).reshape(4 * case.dp, length) * 7 + 3 + index) % 36
            ) + 1
            valid = torch.ones_like(ids, dtype=torch.bool)
            valid[0, :2], valid[1, -2:] = False, False
            labels = (ids + 1) % 37
            labels[~valid] = -100
            labels[2] = -100  # One pipeline microbatch has no supervised tokens.
            if index == 1 and case.dp > 1:
                valid[4:] = False
                labels[4:] = -100  # One DP/EP lane has no tokens or supervision.
            rows = slice(dp_rank * 4, (dp_rank + 1) * 4)
            count = int((labels != -100).sum())
            optimizer.zero_grad(set_to_none=True)
            expected_optimizer.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            captured.clear()
            pipeline.update_seq_len(length)
            losses = [] if pipeline.info.has_last_stage else None
            pipeline.step(
                ids[rows],
                target=labels[rows] if pipeline.info.has_last_stage else None,
                losses=losses,
                attention_mask=valid[rows],
            )
            scale_grads_and_clip_grad_norm(
                None,
                parts,
                pp_enabled=True,
                device_mesh=setup.mesh_context.device_mesh,
                pp_axis_name="pp",
                num_label_tokens=count,
                dp_group_size=case.dp,
                moe_mesh=mesh.moe_mesh,
                ep_axis_name="ep" if case.ep > 1 else None,
            )
            if compare:
                # Keep a single-model mathematical check at the initial weights.
                # FP32 and the original PP-only cases retain it on every update.
                expected_logits = [None] * (4 * case.dp)
                if ep_reference is None or index == 0:
                    row_groups = (
                        [[micro + lane * 4 for lane in range(case.dp)] for micro in range(4)]
                        if case.ep > 1
                        else [[row] for row in range(4 * case.dp)]
                    )
                    for group in row_groups:
                        if case.fused:
                            # The same fused kernel on both sides keeps the trajectories comparable:
                            # its BF16 gradient rounding would otherwise split them after one update.
                            reference_output = reference_forward(
                                ids[group], valid[group], reference_model=reference, hidden_states=True
                            )
                            logits = reference_output.logits
                            loss = fused(
                                reference_output.hidden_states, labels[group], prepare_lm_weight(fused, reference)
                            )
                        else:
                            logits = reference_forward(ids[group], valid[group], reference_model=reference)
                            loss = sum_ce(logits, labels[group])
                        for row, value in zip(group, logits.detach().split(1)):
                            expected_logits[row] = value
                        loss.backward()
                    for parameter in reference.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(count)
                if ep_reference is not None:
                    ep_logits = []
                    ep_reference.prepare_for_grad_accumulation()
                    for micro in range(4):
                        if micro == 3:
                            ep_reference.prepare_for_final_backward()
                        row = dp_rank * 4 + micro
                        value = ep_reference(
                            ids[row : row + 1], attention_mask=valid[row : row + 1], output_hidden_states=case.fused
                        )
                        logits = value.logits if case.fused else value
                        ep_logits.append(logits.detach())
                        loss = (
                            ep_loss(value.hidden_states, labels[row : row + 1])
                            if ep_loss is not None
                            else sum_ce(logits, labels[row : row + 1])
                        )
                        loss.backward()
                    # PP normalization is also the required post-backward
                    # normalization of this independent summed-loss accumulator.
                    scale_grads_and_clip_grad_norm(
                        None,
                        [ep_reference],
                        pp_enabled=True,
                        device_mesh=mesh.device_mesh,
                        moe_mesh=mesh.moe_mesh,
                        ep_axis_name="ep",
                        pp_axis_name="pp",
                        num_label_tokens=count,
                        dp_group_size=case.dp,
                    )
                    if index == 0:
                        error = (
                            (torch.cat(ep_logits)[..., :37] - torch.cat(expected_logits)[rows, ..., :37])
                            .abs()
                            .max()
                            .item()
                        )
                        metrics["first_step_single_model_logit_error"] = error
                        assert error < 0.01, error
                        for key, parameter in reference.named_parameters():
                            expected = parameter.grad.float()
                            error = dense(expected_parameters[key].grad).float() - expected
                            relative = (error.norm() / expected.norm().clamp_min(1e-12)).item()
                            metrics["first_step_single_model_gradient_relative_l2"] = max(
                                metrics.get("first_step_single_model_gradient_relative_l2", 0.0), relative
                            )
                            assert relative <= 0.08 or error.abs().max() <= 5e-4, (
                                key,
                                relative,
                                error.abs().max().item(),
                            )
                    local_expected_logits = torch.cat(ep_logits)
                else:
                    local_expected_logits = torch.cat(expected_logits)[rows]
                if index == 0 and eager_oracle is not None:
                    # Match token grouping for the independent eager comparison,
                    # just as for the initial unsharded Flex reference above.
                    eager_logits = [None] * (4 * case.dp)
                    for group in row_groups:
                        logits = reference_forward(ids[group], valid[group], reference_model=eager_oracle)
                        for row, value in zip(group, logits.detach().split(1)):
                            eager_logits[row] = value
                        sum_ce(logits, labels[group]).backward()
                    error = (
                        (torch.cat(expected_logits)[..., :37] - torch.cat(eager_logits)[..., :37]).abs().max().item()
                    )
                    metrics["first_step_eager_logit_error"] = error
                    assert error < 0.01, error
                    for key, parameter in eager_oracle.named_parameters():
                        parameter.grad.div_(count)
                        expected = parameter.grad.float()
                        error = dense(expected_parameters[key].grad).float() - expected
                        relative = (error.norm() / expected.norm().clamp_min(1e-12)).item()
                        metrics["first_step_eager_gradient_relative_l2"] = max(
                            metrics.get("first_step_eager_gradient_relative_l2", 0.0), relative
                        )
                        assert relative <= 0.08 or error.abs().max() <= 5e-4, (key, relative, error.abs().max().item())
                if pipeline.info.has_last_stage:
                    # Compare the actual scheduled loss, including fused CE,
                    # with independent full-logit CE on this lane's targets.
                    actual_loss = torch.stack(losses).sum() / count
                    expected_loss = sum_ce(local_expected_logits, labels[rows]) / count
                    loss_error = (actual_loss - expected_loss).abs().item()
                    metrics["max_loss_error"] = max(metrics["max_loss_error"], loss_error)
                    assert loss_error < (0.02 if case.bf16 else 2e-5), (index, loss_error)
                    error = (torch.cat(captured)[..., :37] - local_expected_logits[..., :37]).abs().max().item()
                    metrics["max_logit_error"] = max(metrics["max_logit_error"], error)
                    assert error < (0.01 if case.bf16 else 1e-5), (index, error)
                for key, parameter in owners.items():
                    assert parameter.grad is not None, key
                    actual, expected = dense(parameter.grad).float(), dense(expected_parameters[key].grad).float()
                    assert torch.isfinite(actual).all(), key
                    error = actual - expected
                    relative = (error.norm() / expected.norm().clamp_min(1e-12)).item()
                    metrics["max_gradient_relative_l2"] = max(metrics["max_gradient_relative_l2"], relative)
                    if key.endswith(".attn.relative_table"):
                        assert actual.norm() > 0, key
                        metrics["max_relative_table_gradient_error"] = max(
                            metrics.get("max_relative_table_gradient_error", 0.0), error.abs().max().item()
                        )
                    if case.bf16:
                        assert relative <= 0.08 or error.abs().max() <= 5e-4, (key, relative, error.abs().max().item())
                    else:
                        torch.testing.assert_close(actual, expected, rtol=5e-5, atol=3e-6, msg=key)
                for key in (
                    "embed_tokens.weight",
                    "layers.0.attn.qkvr.weight",
                    "layers.6.attn.qkvr.weight",
                    "output_query.weight",
                ):
                    if key in owners:
                        assert dense(owners[key].grad).norm() > 0, key
            norm = scale_grads_and_clip_grad_norm(
                0.25,
                parts,
                pp_enabled=True,
                device_mesh=mesh.device_mesh,
                pp_axis_name="pp",
                moe_mesh=mesh.moe_mesh,
                ep_axis_name="ep" if case.ep > 1 else None,
            )
            if compare:
                expected_norm = (
                    scale_grads_and_clip_grad_norm(
                        0.25, [ep_reference], device_mesh=mesh.device_mesh, moe_mesh=mesh.moe_mesh, ep_axis_name="ep"
                    )
                    if ep_reference is not None
                    else torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.25, foreach=False)
                )
                norm_error = abs(float(norm) - float(expected_norm)) / max(float(expected_norm), 1e-12)
                metrics["max_norm_relative_error"] = max(metrics["max_norm_relative_error"], norm_error)
                assert norm_error < (0.02 if case.bf16 else 2e-4)
                expected_optimizer.step()
            optimizer.step()
            if compare:
                for key, parameter in owners.items():
                    error = (dense(parameter) - dense(expected_parameters[key])).abs().max().item()
                    metrics["max_parameter_error"] = max(metrics["max_parameter_error"], error)
                    torch.testing.assert_close(
                        dense(parameter),
                        dense(expected_parameters[key]),
                        rtol=0.02 if case.bf16 else 5e-5,
                        atol=2e-4 if case.bf16 else 3e-6,
                        msg=key,
                    )

            (output / f"step-{index}-rank-{rank}.json").write_text(json.dumps(metrics, indent=2) + "\n")

        step(0, compare=True)
        saved_parameters = {key: dense(p).clone() for key, p in owners.items()}
        saved_moments = {key: dense(optimizer.state[p]["momentum_buffer"]).clone() for key, p in owners.items()}
        state, opt_state = get_state_dict(wrapper, optimizer)
        key = f"pp_{pp_rank}"
        dcp.save({key: {"model": state, "optimizer": opt_state}}, checkpoint_id=output / "checkpoint")
        step(1, compare=True)
        expected_resume = {key: dense(parameter).clone() for key, parameter in owners.items()}
        expected_gradients = {key: dense(parameter.grad).clone() for key, parameter in owners.items()}
        state, opt_state = get_state_dict(wrapper, optimizer)
        dcp.load({key: {"model": state, "optimizer": opt_state}}, checkpoint_id=output / "checkpoint")
        set_state_dict(wrapper, optimizer, model_state_dict=state, optim_state_dict=opt_state)
        for key, parameter in owners.items():
            torch.testing.assert_close(dense(parameter), saved_parameters[key], rtol=0, atol=0, msg=f"restore {key}")
            torch.testing.assert_close(
                dense(optimizer.state[parameter]["momentum_buffer"]),
                saved_moments[key],
                rtol=0,
                atol=0,
                msg=f"moment {key}",
            )
        step(1, compare=False)
        for key, parameter in owners.items():
            torch.testing.assert_close(
                dense(parameter.grad),
                expected_gradients[key],
                rtol=0,
                atol=0,
                msg=lambda error: f"replay gradient {key}: {error}",
            )
        for key, parameter in owners.items():
            torch.testing.assert_close(
                dense(parameter), expected_resume[key], rtol=0, atol=0, msg=lambda error: f"{key}: {error}"
            )
        (output / f"metrics-{rank}.json").write_text(json.dumps({**metrics, **metrics_topology}, indent=2) + "\n")
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize("case", list(CASES))
def test_pipeline_training(case: str, tmp_path: Path) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(env.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")[:2])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            __file__,
            "--worker",
            case,
            str(tmp_path),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=600,
    )
    (tmp_path / "worker.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker(sys.argv[-2], Path(sys.argv[-1]))
