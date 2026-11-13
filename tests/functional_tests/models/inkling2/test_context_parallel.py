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

"""Real-process CP parity, image boundaries, LoRA and checkpoint continuation."""

import copy
import json
import os
import socket
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules
from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
from nemo_automodel.components.distributed.mesh import ParallelismSizes
from nemo_automodel.components.distributed.model_parallelizer import parallelize_model
from nemo_automodel.components.loss.utils import prepare_lm_weight
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.attention import Inkling2Attention
from nemo_automodel.components.models.inkling2.config import Inkling2Config
from nemo_automodel.components.models.inkling2.context_parallel import unpack_row_layout
from nemo_automodel.components.models.inkling2.flex_attention import Inkling2FlexAttention
from nemo_automodel.components.models.inkling2.layers import ShortConvolution
from nemo_automodel.components.models.inkling2.loss import Inkling2FusedLinearCrossEntropy
from nemo_automodel.components.models.inkling2.model import Inkling2ForCausalLM
from nemo_automodel.components.models.inkling2.multimodal import Inkling2ForConditionalGeneration
from nemo_automodel.components.models.inkling2.parallelization import FP32_MODULES
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn
from nemo_automodel.shared.peft_config import PeftConfig
from tests.unit_tests.models.inkling2.test_multimodal import vision_config
from tests.unit_tests.models.inkling2.test_precision import initialize_nonuniform_queries
from tests.unit_tests.models.inkling2.test_reference import tiny_config


def dense(value: torch.Tensor) -> torch.Tensor:
    """Materialize a parameter/gradient only inside the test process."""
    return (value.full_tensor() if isinstance(value, DTensor) else value).detach()


def worker(case: str, output: Path) -> None:
    """Compare CP against an independent whole-sequence model and replay an update."""
    cpu = case.startswith("cpu")
    bf16, image, lora = "bf16" in case, "image" in case, "lora" in case
    cp, ep = (4 if "cp4" in case else 2), (2 if "ep2" in case else 1)
    ac = "ac" in case
    local = int(os.environ["LOCAL_RANK"])
    if not cpu:
        torch.cuda.set_device(local)
    device = torch.device("cpu" if cpu else f"cuda:{local}")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    dist.init_process_group("gloo" if cpu else "nccl", timeout=timedelta(seconds=480))
    try:
        world = dist.get_world_size()
        torch.manual_seed(732)
        cfg = tiny_config()
        cfg.hidden_size = cfg.intermediate_size = 32 if cpu else 64
        cfg.n_routed_experts = 16
        # Width exceeds a CP4 shard in the short case; includes static and dynamic taps.
        cfg.sconv_kernel_size = 5
        text_cfg = cfg
        if image:
            text_cfg.vocab_size = text_cfg.unpadded_vocab_size = 200058
            cfg = Inkling2Config(text_config=text_cfg, vision_config=vision_config())
        cls = Inkling2ForConditionalGeneration if image else Inkling2ForCausalLM
        backend = BackendConfig(
            attn="eager" if cpu or not bf16 else "flex",
            linear="torch",
            rms_norm="torch",
            experts="torch" if cpu else "torch_mm",
            dispatcher="torch",
        )
        reference_backend = copy.deepcopy(backend)
        reference_backend.attn = "eager"
        reference = cls(copy.deepcopy(cfg), backend=reference_backend).to(device)
        initialize_nonuniform_queries(reference)
        if lora:
            reference.requires_grad_(False)
            apply_lora_to_linear_modules(
                reference,
                PeftConfig(
                    target_modules=["*.attn.qkvr", "*.attn.wo_ud"],
                    dim=3,
                    alpha=6,
                    dropout=0,
                    lora_dtype="float32",
                    use_memory_efficient_lora=False,
                ),
            )
            assert any("lora_A" in name and p.requires_grad for name, p in reference.named_parameters())
            with torch.no_grad():
                for name, parameter in reference.named_parameters():
                    if "lora_B" in name:
                        parameter.normal_(std=0.03)
        model = copy.deepcopy(reference)
        model.backend = copy.deepcopy(backend)
        if bf16 and not cpu:
            for module in model.modules():
                if isinstance(module, Inkling2Attention):
                    module.flex = Inkling2FlexAttention(
                        head_dim=module.head_dim,
                        window=module.window if module.is_local else None,
                    )
        if cpu:
            mesh = init_device_mesh("cpu", (world // cp, cp), mesh_dim_names=("dp", "cp"))
            model.cp_mesh = mesh["cp"]
            if image:
                model.visual.image_cp_mesh = mesh[("dp", "cp")]._flatten("dp_cp")
            for module in model.modules():
                if isinstance(module, (ShortConvolution, Inkling2Attention)):
                    module.cp_mesh = mesh["cp"]
            if ac:
                for name in model.layers:
                    model.layers[name] = checkpoint_wrapper(model.layers[name])
        else:
            setup = DistributedSetup.build(
                strategy=FSDP2Config(
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=torch.bfloat16 if bf16 else torch.float32,
                        reduce_dtype=torch.float32,
                        cast_forward_inputs=False,
                    ),
                    reshard_after_forward=True,
                    enable_fsdp2_prefetch=False,
                ),
                parallelism_sizes=ParallelismSizes(cp_size=cp, ep_size=ep),
                activation_checkpointing=ac,
                world_size=world,
            )
            model = parallelize_model(model, setup.mesh_context)
            mesh = setup.mesh_context.device_mesh
        # Reject unsupported layout semantics before any attention collective.
        ids_probe = torch.ones(1, 12, dtype=torch.long, device=device)
        for invalid in (
            {"qkv_format": "thd"},
            {"cu_seqlens": torch.tensor([0, 6, 12], device=device)},
            {"attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]], device=device)},
            {"position_ids": torch.tensor([[0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7]], device=device)},
        ):
            invalid_batch = {"input_ids": ids_probe, "labels": ids_probe, **invalid}
            with pytest.raises(ValueError, match="Inkling2"):
                ContextParallelSharder(model, mesh, invalid_batch).shard(invalid_batch)
        lane = mesh["dp"].get_local_rank()
        dp = world // cp
        owners = {canonical_parameter_fqn(n): p for n, p in model.named_parameters() if p.requires_grad}
        wanted = {n: p for n, p in reference.named_parameters() if p.requires_grad}
        assert owners.keys() == wanted.keys()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, foreach=False)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9, foreach=False)
        # The fused kernel takes BF16 operands only; FP32 and CPU runs keep the model's own FP32 projection.
        fused_loss = bf16 and not cpu
        loss_fn = Inkling2FusedLinearCrossEntropy(text_cfg.unpadded_vocab_size)
        # Expert selections of every forward, per side and layer, in call order.
        routing: dict[str, dict[str, list[torch.Tensor]]] = {"model": {}, "reference": {}}

        def record_routing(side: str, name: str):
            def hook(module: torch.nn.Module, args: tuple[torch.Tensor, ...], result: tuple[torch.Tensor, ...]) -> None:
                routing[side].setdefault(name, []).append(result[1].detach())

            return hook

        if bf16 and not cpu:
            for name, layer in model.layers.items():
                layer.mlp.gate.register_forward_hook(record_routing("model", name))
            for name, layer in reference.layers.items():
                layer.mlp.gate.register_forward_hook(record_routing("reference", name))
        # Labels and token count of each compared step, reused when the step is replayed.
        evaluated: dict[int, tuple[torch.Tensor, int]] = {}
        metrics = dict(
            case=case,
            comparison_mode="same_state_per_update" if bf16 else "independent_optimizer_trajectory",
            cp=cp,
            ep=ep,
            world=world,
            max_logit_error=0.0,
            max_gradient_relative_l2=0.0,
            max_gradient_abs=0.0,
            max_parameter_error=0.0,
            max_loss_error=0.0,
        )

        def step(index: int, compare: bool = True) -> None:
            length = 11 if index == 0 else 5
            ids = (torch.arange(2 * dp * length, device=device).reshape(2 * dp, length) + index) % 35 + 1
            valid = torch.ones_like(ids, dtype=torch.bool)
            valid[0, :2] = False
            valid[1, 3:] = False  # A CP peer has no valid queries/supervision in this row.
            if index:
                valid[1] = False
            if image and index:
                valid[0, :2] = True
            positions = (valid.long().cumsum(1) - 1).clamp_min(0) + 3
            labels = F.pad(ids[:, 1:], (0, 1), value=-100)
            labels.masked_fill_(~valid | ~F.pad(valid[:, 1:], (0, 1), value=False), -100)
            media = {}
            if image:
                body = torch.zeros_like(ids)
                # One complete image straddles CP0/CP1 on the long case.
                begin = 5 if index == 0 else 2
                ids[0, begin : begin + 2] = 200005
                body[0, begin : begin + 2] = 1
                labels[0, begin : begin + 2] = -100
                generator = torch.Generator(device=device).manual_seed(733 + index)
                media = dict(
                    pixel_values=torch.randn(2, 2, 40, 40, 3, device=device, dtype=torch.bfloat16, generator=generator),
                    image_grid_thw=torch.tensor([[1, 1, 1], [1, 1, 1]], device=device),
                    mm_token_type_ids=body,
                )
            packed_kwargs = {}
            if "packed" in case:
                boundary = 4 if index == 0 else 2
                lengths = torch.tensor([[boundary - 1, length - boundary]], device=device).expand(ids.shape[0], -1)
                padded = torch.tensor([[boundary, length - boundary]], device=device).expand_as(lengths)
                packed_kwargs = dict(seq_lens=lengths, seq_lens_padded=padded, qkv_format="thd")
                layout = dict(input_ids=ids, **packed_kwargs)
                unpack_row_layout(layout)
                valid, positions = ~layout["padding_mask"], layout["position_ids"]
                labels = ids.clone().masked_fill(~valid, -100)
            count = int((labels != -100).sum())
            assert count > 0
            rows = slice(lane * 2, (lane + 1) * 2)
            kwargs = dict(
                input_ids=ids[rows], attention_mask=valid[rows], position_ids=positions[rows], labels=labels[rows]
            )
            if packed_kwargs:
                kwargs.update({k: v[rows] if isinstance(v, torch.Tensor) else v for k, v in packed_kwargs.items()})
            if image:
                kwargs.update(media if lane == 0 else dict(mm_token_type_ids=media["mm_token_type_ids"][rows]))
                kwargs["mm_token_type_ids"] = media["mm_token_type_ids"][rows]
            sharder = ContextParallelSharder(model, mesh, kwargs)
            context, batch = sharder.shard(kwargs)
            local_labels = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            if compare:
                if bf16 and index:
                    # BF16 parity compares the same weights and optimizer input state.
                    # Independent topology trajectories can cross different rounding/
                    # routing boundaries (the saved diagnostic measured 3e-6 drift).
                    # FP32 retains an independent multi-update trajectory below;
                    # checkpoint replay separately requires bitwise identity.
                    with torch.no_grad():
                        for name, parameter in owners.items():
                            wanted[name].copy_(dense(parameter))
                            for state_name, state_value in optimizer.state[parameter].items():
                                reference_optimizer.state[wanted[name]][state_name] = (
                                    dense(state_value).clone()
                                    if isinstance(state_value, torch.Tensor)
                                    else copy.deepcopy(state_value)
                                )
                effective = {
                    name: parameter
                    if any(pin in name for pin in FP32_MODULES)
                    else parameter.to(torch.bfloat16 if bf16 else torch.float32)
                    for name, parameter in reference.named_parameters()
                }
                routing["reference"].clear()
                if packed_kwargs:
                    # Independent document forwards: no packed mask or CP transport.
                    pieces = []
                    for begin, end in ((0, boundary - 1), (boundary, length)):
                        pieces.append(torch.func.functional_call(reference, effective, (ids[:, begin:end],), {}))
                    expected = torch.cat((pieces[0], torch.zeros_like(pieces[0][:, :1]), pieces[1]), dim=1)
                else:
                    expected = torch.func.functional_call(
                        reference, effective, (ids,), dict(attention_mask=valid, position_ids=positions, **media)
                    )
                expected_routing = {}
                for name, calls in routing["reference"].items():
                    selections = [call.view(ids.shape[0], -1, call.shape[-1]) for call in calls]
                    if packed_kwargs:
                        selections.insert(1, torch.full_like(selections[0][:, :1], -1))
                    expected_routing[name] = torch.cat(selections, dim=1)
            routing["model"].clear()
            keep = ~batch["padding_mask"]
            same_routing = torch.ones_like(keep)
            with context():
                result = model(**batch, output_hidden_states=True)
                # Read the selections before backward: activation checkpointing recomputes the gates.
                local_routing = {
                    name: calls[-1].view(*result.hidden_states.shape[:2], -1)
                    for name, calls in routing["model"].items()
                }
                if compare and expected_routing:
                    # BF16 expert sums round differently for different batch compositions, and a
                    # near-tied top-k selection then flips for a token. Such tokens take different
                    # experts on the two sides: they are counted, and both sides train and compare
                    # on the remaining tokens.
                    for name, selection in expected_routing.items():
                        wanted_selection = sharder.shard_token_tensor(selection[rows], seq_dim=1, fill=-1)
                        same_routing &= (local_routing[name].sort(-1).values == wanted_selection.sort(-1).values).all(
                            -1
                        )
                    flipped = int((keep & ~same_routing).sum())
                    metrics["routing_flipped_tokens"] = max(metrics.get("routing_flipped_tokens", 0), flipped)
                    assert flipped <= max(1, int(keep.sum()) // 8), (case, index, "routing flips", flipped)
                    gathered = [None] * world
                    dist.all_gather_object(gathered, (lane, mesh["cp"].get_local_rank(), (keep & ~same_routing).cpu()))
                    global_flip = torch.zeros_like(labels, dtype=torch.bool)
                    for piece_lane, piece_cp, mask in gathered:
                        start = piece_cp * mask.shape[1]
                        width = min(mask.shape[1], labels.shape[1] - start)
                        if width > 0:
                            global_flip[piece_lane * 2 : (piece_lane + 1) * 2, start : start + width] = mask[
                                :, :width
                            ].to(labels.device)
                    labels = labels.masked_fill(global_flip, -100)
                    local_labels = local_labels.masked_fill(~same_routing, -100)
                    count = int((labels != -100).sum())
                    assert count > 0
                    evaluated[index] = (local_labels, count)
                elif not compare and index in evaluated:
                    local_labels, count = evaluated[index]
                if fused_loss:
                    weight = prepare_lm_weight(loss_fn, model, grad_reduce_group=mesh["dp_cp"].get_group())
                    loss = loss_fn(result.hidden_states, local_labels, weight) / count
                else:
                    loss = F.cross_entropy(result.logits.flatten(0, 1), local_labels.flatten(), reduction="sum") / count
                (loss * world).backward()
            if compare:
                expected_loss = F.cross_entropy(expected.flatten(0, 1), labels.flatten(), reduction="sum") / count
                expected_loss.backward()
            if cpu:
                for parameter in owners.values():
                    assert parameter.grad is not None
                    dist.all_reduce(parameter.grad)
                    parameter.grad.div_(world)
            if not cpu:
                scale_grads_and_clip_grad_norm(
                    None,
                    [model],
                    device_mesh=mesh,
                    moe_mesh=setup.mesh_context.moe_mesh,
                    ep_axis_name="ep" if ep > 1 else None,
                    dp_group_size=world,
                )
            if compare:
                expected_local = sharder.shard_token_tensor(expected[rows], seq_dim=1, fill=0)
                keep = keep & same_routing
                # Padding logits themselves are not a semantic output.
                delta = (
                    (result.logits[..., :37][keep] - expected_local[..., :37][keep]).abs().max().item()
                    if keep.any()
                    else 0.0
                )
                metrics["max_logit_error"] = max(metrics["max_logit_error"], delta)
                assert delta < (0.02 if bf16 else 2e-5), (case, index, "logits", delta)
                total = loss.detach().clone()
                dist.all_reduce(total)
                loss_error = abs(float(total - expected_loss.detach()))
                metrics["max_loss_error"] = max(metrics["max_loss_error"], loss_error)
                assert loss_error < (0.01 if bf16 else 2e-5), loss_error
                for name, parameter in owners.items():
                    actual, target = dense(parameter.grad).float(), wanted[name].grad.float()
                    assert torch.isfinite(actual).all(), name
                    error = actual - target
                    rel = float(error.norm() / target.norm().clamp_min(1e-12))
                    absolute = float(error.abs().max())
                    metrics["max_gradient_relative_l2"] = max(metrics["max_gradient_relative_l2"], rel)
                    metrics["max_gradient_abs"] = max(metrics["max_gradient_abs"], absolute)
                    if bf16:
                        assert rel <= 0.08 or absolute <= 5e-4, (name, rel, absolute)
                    else:
                        torch.testing.assert_close(
                            actual, target, rtol=1e-4, atol=3e-6, msg=lambda error: f"{name}: {error}"
                        )
            if cpu:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25, foreach=False)
            else:
                norm = scale_grads_and_clip_grad_norm(
                    0.25,
                    [model],
                    device_mesh=mesh,
                    moe_mesh=setup.mesh_context.moe_mesh,
                    ep_axis_name="ep" if ep > 1 else None,
                )
            if compare:
                expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.25, foreach=False)
                relative = abs(float(norm) - float(expected_norm)) / max(float(expected_norm), 1e-12)
                metrics["max_norm_relative_error"] = max(metrics.get("max_norm_relative_error", 0.0), relative)
                assert relative < (0.03 if bf16 else 2e-4), relative
                reference_optimizer.step()
            optimizer.step()
            if compare:
                for name, parameter in owners.items():
                    actual, target = dense(parameter), wanted[name].detach()
                    delta = float((actual - target).abs().max())
                    metrics["max_parameter_error"] = max(metrics["max_parameter_error"], delta)
                    torch.testing.assert_close(
                        actual, target, rtol=0.02 if bf16 else 1e-4, atol=2e-4 if bf16 else 3e-6, msg=name
                    )
            (output / f"step-{index}-rank-{dist.get_rank()}.json").write_text(json.dumps(metrics, indent=2))
            print(f"rank={dist.get_rank()} {case} step={index} compare={compare} passed", flush=True)

        step(0)
        state, optim_state = get_state_dict(model, optimizer)
        dcp.save({"model": state, "optimizer": optim_state}, checkpoint_id=output / "checkpoint")
        step(1)
        replay = {
            name: (dense(p).clone(), dense(p.grad).clone(), dense(optimizer.state[p]["momentum_buffer"]).clone())
            for name, p in owners.items()
        }
        state, optim_state = get_state_dict(model, optimizer)
        dcp.load({"model": state, "optimizer": optim_state}, checkpoint_id=output / "checkpoint")
        set_state_dict(model, optimizer, model_state_dict=state, optim_state_dict=optim_state)
        step(1, compare=False)

        def replay_tolerance(kind: int, target: torch.Tensor) -> dict[str, float]:
            # The fused kernel accumulates its backward with atomics: a replayed step reproduces the
            # BF16 gradients and momenta to a few BF16 ulps of their largest magnitude (3 percent)
            # and the FP32 masters to the resulting update.
            if not fused_loss:
                return dict(rtol=0.0, atol=0.0)
            if kind == 0:
                return dict(rtol=0.0, atol=1e-5)
            return dict(rtol=0.0, atol=float(target.abs().max()) * 2**-5)

        for name, parameter in owners.items():
            current = (dense(parameter), dense(parameter.grad), dense(optimizer.state[parameter]["momentum_buffer"]))
            for kind, (actual, target) in enumerate(zip(current, replay[name])):
                torch.testing.assert_close(
                    actual,
                    target,
                    **replay_tolerance(kind, target),
                    msg=lambda error, name=name: f"replay {name}: {error}",
                )
        metrics["checkpoint_replay_exact"] = not fused_loss
        (output / f"metrics-{dist.get_rank()}.json").write_text(json.dumps(metrics, indent=2) + "\n")
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("case", ["cpu-ac", "cpu-lora", "cpu-image", "cpu-image-dp"])
def test_context_parallel_cpu(case: str, tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[4]) + os.pathsep + env.get("PYTHONPATH", "")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--master_addr=127.0.0.1",
            f"--master_port={port}",
            f"--nproc_per_node={4 if case.endswith('-dp') else 2}",
            __file__,
            "--worker",
            case,
            str(tmp_path),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA GPUs")
@pytest.mark.parametrize(
    "case",
    [
        "gpu-ac",
        "gpu-cp4-ac",
        "gpu-ep2-ac",
        "gpu-bf16-ep2-ac",
        "gpu-lora-ac",
        "gpu-image-bf16-ac",
        "gpu-packed-bf16-ep2-ac",
        "gpu-packed-cp4-ac",
    ],
)
def test_context_parallel_cuda(case: str, tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[4]) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--master_addr=127.0.0.1",
            f"--master_port={port}",
            "--nproc_per_node=4",
            __file__,
            "--worker",
            case,
            str(tmp_path),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA GPUs")
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_context_pipeline_cuda(precision: str, tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[4]) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--master_addr=127.0.0.1",
            f"--master_port={port}",
            "--nproc_per_node=4",
            str(Path(__file__).with_name("context_pipeline.py")),
            precision,
            str(tmp_path),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout


if __name__ == "__main__" and "--worker" in sys.argv:
    worker(sys.argv[-2], Path(sys.argv[-1]))
