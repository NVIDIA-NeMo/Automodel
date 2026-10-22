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

"""Two-GPU FSDP, EP/PP and expert-LoRA training gates against independent global objectives."""

from __future__ import annotations

import hashlib
import json
import os
import socket
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components._peft.lora_experts import GroupedExpertsDeepEPLoRA, GroupedExpertsLoRA
from nemo_automodel.components.distributed import DistributedSetup, FSDP2Config, ParallelismSizes
from nemo_automodel.components.distributed.config import MoEParallelizerConfig
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loss.loss import MaskedCrossEntropyConfig
from nemo_automodel.components.loss.utils import calculate_loss
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.flint.config import FlintConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP
from nemo_automodel.components.optim.optimizer import AdamWConfig
from nemo_automodel.components.training.utils import (
    clip_grad_norm,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from nemo_automodel.components.utils.model_utils import FreezeConfig, ModuleSelector
from tests.functional_tests.models.flint.reference import (
    full_native_tensors,
    reference_logits,
    tiny_config,
    torch_backend,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs"),
]


def _batch(rank: int, microbatch: int, *, empty: bool, device: torch.device) -> dict[str, torch.Tensor]:
    """Construct uneven already-shifted CUDA supervision without external data.

    Args:
        rank: Data-parallel rank, 0 or 1.
        microbatch: Deterministic sample variant.
        empty: Exclude all queries/targets on this rank while peers still train.
        device: This rank's CUDA device.

    Returns:
        Independent int64 input_ids/labels and boolean attention_mask [1,5].
        Labels are shifted once, with -100 at invalid targets. A valid EOS=2
        remains a real input even though it is also the padding token value.
    """
    tokens = (torch.tensor([5, 2, 9, 4, 7, 3], device=device) + microbatch + 3 * rank) % 31
    tokens[1] = 2
    length = 0 if empty else max(1, 5 - rank - microbatch % 2)
    valid = torch.arange(5, device=device) < length
    ids, labels = tokens[:-1].clone(), tokens[1:].clone()
    ids[~valid], labels[~valid] = 2, -100
    return {"input_ids": ids[None], "labels": labels[None], "attention_mask": valid[None]}


def _worker(
    rank: int, port: int, accumulation: int, clipping: float, ac: bool, bf16: bool, frozen_head: bool, evidence: str
) -> None:
    """Run one real two-rank topology; all tensor collectives are rank symmetric."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank)
    )
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", rank=rank, world_size=2, timeout=timedelta(minutes=5))
    handles = []
    try:
        torch.manual_seed(41)
        compute = torch.bfloat16 if bf16 else torch.float32
        policy = FSDP2Config(
            mp_policy=MixedPrecisionPolicy(
                param_dtype=compute,
                reduce_dtype=torch.float32,
                output_dtype=torch.bfloat16 if bf16 else None,
                cast_forward_inputs=False,
            ),
            defer_fsdp_grad_sync=True,
            enable_fsdp2_prefetch=False,
        )
        setup = DistributedSetup.build(
            strategy=policy,
            parallelism_sizes=ParallelismSizes(dp_size=2, tp_size=1, cp_size=1, pp_size=1, ep_size=1),
            activation_checkpointing=ac,
            world_size=2,
        )
        config = tiny_config(torch.float32)
        model = NeMoAutoModelForCausalLM.from_config(
            config,
            dtype=torch.float32,
            attn_implementation="eager",
            backend=torch_backend(),
            use_liger_kernel=False,
            use_sdpa_patching=False,
            distributed_setup=setup,
            freeze_config=FreezeConfig(freeze_modules=[ModuleSelector(path="lm_head")]) if frozen_head else None,
        ).train()
        assert all(isinstance(p, DTensor) and p.dtype == torch.float32 for p in model.parameters())
        original = model.state_dict_adapter.to_hf(full_native_tensors(model))
        reference = {
            name: value.clone().requires_grad_(not (frozen_head and name == "lm_head.weight"))
            for name, value in original.items()
        }
        assert model.lm_head.weight.requires_grad is not frozen_head
        optimizer = AdamWConfig(lr=0.003, weight_decay=0.01, eps=1e-6).build(model)[0]
        reference_optimizer = torch.optim.AdamW(list(reference.values()), lr=0.003, weight_decay=0.01, eps=1e-6)
        observations = {"sink": [], "head": []}

        def capture_sink(module: torch.nn.Module, inputs: tuple, output: torch.Tensor) -> None:
            """Observe gathered sink precision without touching its parameter.

            Args:
                module: Nested FP32 FSDP sink holder.
                inputs: Empty input tuple.
                output: Trainable FP32 CUDA sink tensor [query_heads].
            """
            observations["sink"].append(output.dtype == torch.float32 and not torch.is_autocast_enabled("cuda"))

        def capture_head(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            """Check the gathered output projection at its actual compute boundary.

            Args:
                module: FSDP head owning global [vocab,hidden] weights.
                inputs: One FP32 CUDA tensor [batch,sequence,hidden].
                output: FP32 CUDA logits [batch,sequence,vocab].
            """
            observations["head"].append(module.weight.dtype == inputs[0].dtype == output.dtype == torch.float32)

        handles.append(model.lm_head.register_forward_hook(capture_head))
        for layer in model.model.layers:
            handles.append(layer.self_attn._fp32_params.register_forward_hook(capture_sink))
        windows = [accumulation] if accumulation == 1 else [accumulation, 1]
        results = []
        for window_index, count in enumerate(windows):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            local_batches = [
                _batch(rank, index, empty=window_index > 0 and rank == 1, device=device) for index in range(count)
            ]
            total = torch.tensor(sum(int((b["labels"] != -100).sum()) for b in local_batches), device=device)
            dist.all_reduce(total)
            assert total > 0
            prepare_for_grad_accumulation([model])
            native_loss_sum = torch.zeros((), device=device)
            for index, batch in enumerate(local_batches):
                final = index == count - 1
                if final:
                    prepare_for_final_backward([model])
                with get_sync_ctx(model, final, defer_fsdp_grad_sync=True):
                    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                    assert logits.dtype == torch.float32
                    loss = calculate_loss(
                        MaskedCrossEntropyConfig().build(),
                        logits=logits,
                        labels=batch["labels"],
                        num_label_tokens=total,
                    )
                    native_loss_sum += loss.detach()
                    # Match the actual recipe: FSDP averages ranks after local
                    # token sums were divided by the GLOBAL supervised count.
                    (loss * 2).backward()
            expected_loss = torch.zeros((), device=device)
            for reference_rank in range(2):
                for index in range(count):
                    batch = _batch(reference_rank, index, empty=window_index > 0 and reference_rank == 1, device=device)
                    logits, _ = reference_logits(
                        reference,
                        config,
                        batch,
                        compute_dtype=compute,
                        parameter_dtype=compute if bf16 else None,
                    )
                    expected_loss = (
                        expected_loss
                        + F.cross_entropy(
                            logits.reshape(-1, config.vocab_size),
                            batch["labels"].reshape(-1),
                            reduction="sum",
                        )
                        / total
                    )
            expected_loss.backward()
            dist.all_reduce(native_loss_sum)
            atol, rtol = (4e-6, 5e-4) if not bf16 else (5e-4, 1e-1)
            torch.testing.assert_close(native_loss_sum, expected_loss, atol=atol, rtol=rtol)
            gradients = model.state_dict_adapter.to_hf(full_native_tensors(model, gradients=True))
            errors = {}
            for name, value in reference.items():
                if not value.requires_grad:
                    assert name == "lm_head.weight" and model.lm_head.weight.grad is None
                    continue
                assert value.grad is not None and torch.isfinite(gradients[name]).all(), name
                torch.testing.assert_close(
                    gradients[name], value.grad, atol=atol, rtol=rtol, msg=lambda message, key=name: f"{key}\n{message}"
                )
                errors[name] = float((gradients[name] - value.grad).abs().max())
            norm = scale_grads_and_clip_grad_norm(
                clipping,
                [model],
                device_mesh=setup.mesh_context.device_mesh,
                foreach=True,
                grad_norm_backend="triton",
            )
            if isinstance(norm, DTensor):
                norm = norm.full_tensor()
            expected_norm = torch.nn.utils.clip_grad_norm_(list(reference.values()), clipping)
            torch.testing.assert_close(norm.double(), expected_norm.double(), atol=atol, rtol=rtol)
            optimizer.step()
            reference_optimizer.step()
            updated = model.state_dict_adapter.to_hf(full_native_tensors(model))
            for name, value in reference.items():
                torch.testing.assert_close(updated[name], value, atol=2e-6 if not bf16 else 5e-4, rtol=rtol, msg=name)
            assert all(state["exp_avg"].dtype == torch.float32 for state in optimizer.state.values())
            results.append(
                {
                    "microbatches": count,
                    "global_supervised_tokens": total.item(),
                    "empty_rank": window_index > 0,
                    "loss": native_loss_sum.item(),
                    "reference_loss": expected_loss.item(),
                    "gradient_max_abs_error": max(errors.values()),
                    "gradient_error_by_name": errors,
                    "global_norm": norm.item(),
                    "reference_global_norm": expected_norm.item(),
                    "updated_parameter_max_abs_error": max(
                        float((updated[n] - v).detach().abs().max()) for n, v in reference.items()
                    ),
                }
            )
        assert observations["sink"] and all(observations["sink"])
        assert observations["head"] and all(observations["head"])
        if rank == 0:
            Path(evidence).write_text(
                json.dumps(
                    {
                        "world_size": 2,
                        "compute_dtype": str(compute),
                        "master_dtype": "float32",
                        "activation_checkpointing": ac,
                        "frozen_head": frozen_head,
                        "accumulation": accumulation,
                        "clipping": "infinity" if clipping == float("inf") else clipping,
                        "windows": results,
                        "protected_fp32_compute_observed": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
    finally:
        for handle in handles:
            handle.remove()
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "accumulation,clipping,ac,bf16,frozen_head",
    [
        (1, float("inf"), False, False, False),
        (2, 0.2, False, False, False),
        (2, 0.2, True, False, False),
        (2, 0.2, False, True, False),
        (2, 0.2, False, True, True),
    ],
    ids=[
        "fp32_single",
        "fp32_accum_partial_empty",
        "fp32_ac_partial_empty",
        "bf16_compute_fp32_masters",
        "bf16_frozen_fp32_head",
    ],
)
def test_two_rank_fsdp_matches_independent_global_objective(
    accumulation: int,
    clipping: float,
    ac: bool,
    bf16: bool,
    frozen_head: bool,
    tmp_path: Path,
) -> None:
    """Spawn exactly two allocated GPUs; no TP/CP/PP/EP capability is implied."""
    if bf16 and not torch.cuda.is_bf16_supported():
        pytest.skip("requires CUDA BF16 support")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    evidence = tmp_path / "fsdp_metrics.json"
    mp.spawn(_worker, args=(port, accumulation, clipping, ac, bf16, frozen_head, str(evidence)), nprocs=2, join=True)
    assert evidence.is_file()
    print(f"FLINT_FSDP_EVIDENCE={evidence}")


def _runtime_receipt() -> dict:
    """Fingerprint the actual process runtime and source before model construction."""
    root = Path(__file__).resolve().parents[4]
    files = sorted((root / "nemo_automodel/components/models/flint").glob("*.py"))
    files += [
        root / "nemo_automodel" / relative
        for relative in (
            "components/moe/experts.py",
            "components/training/utils.py",
            "components/distributed/parallelizer_utils.py",
            "components/_peft/lora_experts.py",
            "components/distributed/context_parallel/utils.py",
            "components/checkpoint/stateful_wrappers.py",
        )
    ]
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }


def _config() -> FlintConfig:
    config = tiny_config()
    return FlintConfig(
        **{
            **config.to_dict(),
            "num_hidden_layers": 4,
            "schedule_num_hidden_layers": 4,
            "layer_types": ["full_attention", "sliding_attention", "full_attention", "sliding_attention"],
            "no_rope_layer_indices": [0, 2],
        }
    )


def _write_source(path: Path) -> None:
    torch.manual_seed(4375)
    model = _config().build(backend=torch_backend())
    path.mkdir()
    model.config.save_pretrained(path)
    save_file(model.state_dict_adapter.to_hf(model.state_dict()), path / "model.safetensors")


def _ep_pp_batch(data_rank: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Create shifted targets with uneven padding and a fully ignored microbatch.

    Args:
        data_rank: Data-parallel sample partition, independently of PP rank.
        device: Runtime CUDA device.

    Returns:
        Integer input_ids/labels and binary attention_mask of shape [2, 5].
        Labels are already shifted once. EOS=2 also occurs as valid input.
    """
    tokens = (torch.tensor([[5, 2, 9, 4, 7, 3], [8, 6, 2, 5, 4, 1]], device=device) + 3 * data_rank) % 31
    tokens[0, 1] = 2
    lengths = torch.tensor([5 - data_rank, 0], device=device)
    valid = torch.arange(5, device=device)[None] < lengths[:, None]
    ids, labels = tokens[:, :-1].clone(), tokens[:, 1:].clone()
    ids[~valid], labels[~valid] = 2, -100
    return {"input_ids": ids, "labels": labels, "attention_mask": valid}


def _union(parts: list[torch.nn.Module], *, gradients: bool = False) -> dict[str, torch.Tensor]:
    """Gather stage-local state without mixing independent FSDP/EP collectives.

    Args:
        parts: Local PP model parts. Native projection and grouped expert tensor
            layouts follow full_native_tensors; DTensors gather on their own mesh.
        gradients: Select gradients instead of parameter values.

    Returns:
        Global HF tensors on CPU, with split experts [out,in] and vector norms.
        Every world rank enters the final bounded test-only object collective.
    """
    local = {}
    for part in parts:
        native = full_native_tensors(part, gradients=gradients)
        local.update({name: tensor.cpu() for name, tensor in part.state_dict_adapter.to_hf(native).items()})
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    result = {}
    for mapping in gathered:
        for name, value in mapping.items():
            if name in result:
                torch.testing.assert_close(value, result[name], rtol=0, atol=0)
            result[name] = value
    return result


def _parallel_worker(rank: int, port: int, source: str, evidence: str, pp_size: int, ep_size: int) -> None:
    """Run actual public checkpoint load, full-vocabulary forward, backward and SGD."""
    world = max(ep_size, 1) * pp_size
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank)
    )
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(minutes=8))
    try:
        provenance = _runtime_receipt()
        dp_size, data_rank = world // pp_size, rank % (world // pp_size)
        total = sum(int((_ep_pp_batch(r, torch.device("cpu"))["labels"] != -100).sum()) for r in range(dp_size))
        captured = []
        loss_values = []

        def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Compute the recipe-scaled mean objective from stage logits [1,S,V].

            Args:
                prediction: Full FP32 vocabulary logits [microbatch, sequence, vocab].
                target: Already shifted labels [microbatch, sequence], with -100 ignored.

            Returns:
                Scalar objective scaled once to cancel FSDP data-rank averaging.
            """
            logits = prediction.logits if hasattr(prediction, "logits") else prediction
            captured.append(logits.detach().cpu())
            loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="sum") / total
            loss_values.append(loss.detach())
            return loss * dp_size

        pipeline = (
            PipelineConfig(
                pp_schedule="1f1b",
                pp_microbatch_size=1,
                pp_batch_size=2,
                layers_per_stage=2,
                scale_grads_in_schedule=False,
                patch_inner_model=False,
                patch_causal_lm_model=False,
                loss_fn=loss_fn,
            )
            if pp_size > 1
            else None
        )
        setup = DistributedSetup.build(
            strategy=FSDP2Config(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.float32, reduce_dtype=torch.float32, output_dtype=None
                ),
                defer_fsdp_grad_sync=False,
                enable_fsdp2_prefetch=False,
            ),
            parallelism_sizes=ParallelismSizes(dp_size=dp_size, pp_size=pp_size, ep_size=ep_size),
            pipeline_config=pipeline,
            moe_parallel_config=MoEParallelizerConfig(reshard_after_forward=True) if ep_size > 1 else None,
            world_size=world,
        )
        native = NeMoAutoModelForCausalLM.from_pretrained(
            source,
            dtype=torch.float32,
            backend=torch_backend(),
            distributed_setup=setup,
            attn_implementation="eager",
            use_liger_kernel=False,
            use_sdpa_patching=False,
            local_files_only=True,
        )
        parts = native.parts if isinstance(native, AutoPipeline) else [native]
        reference = {name: value.to(device).requires_grad_() for name, value in _union(parts).items()}
        parameters = [parameter for part in parts for parameter in part.parameters() if parameter.requires_grad]
        optimizer = torch.optim.SGD(parameters, lr=0.003)
        reference_optimizer = torch.optim.SGD(reference.values(), lr=0.003)
        batch = _ep_pp_batch(data_rank, device)
        reference_loss = torch.zeros((), device=device)
        reference_local_logits = None
        for r in range(dp_size):
            sample = _ep_pp_batch(r, device)
            logits, _ = reference_logits(reference, _config(), sample, compute_dtype=torch.float32)
            if r == data_rank:
                reference_local_logits = logits.detach().cpu()
            reference_loss += F.cross_entropy(logits.flatten(0, 1), sample["labels"].flatten(), reduction="sum") / total
        reference_loss.backward()
        prepare_for_grad_accumulation(parts)
        prepare_for_final_backward(parts)
        if isinstance(native, AutoPipeline):
            native.update_seq_len(batch["input_ids"].shape[1])
            native.step(
                batch["input_ids"],
                target=batch["labels"] if native.info.has_last_stage else None,
                losses=[] if native.info.has_last_stage else None,
                attention_mask=batch["attention_mask"],
            )
        else:
            logits = native(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss_fn(logits, batch["labels"]).backward()
        if captured:
            valid_queries = batch["attention_mask"].bool().cpu()
            torch.testing.assert_close(
                torch.cat(captured)[valid_queries], reference_local_logits[valid_queries], atol=1e-5, rtol=1e-5
            )
        actual_loss = sum(loss_values, torch.zeros((), device=device))
        dist.all_reduce(actual_loss)
        torch.testing.assert_close(actual_loss, reference_loss, atol=2e-6, rtol=2e-5)
        # The public recipe scales EP expert sums separately from dense FSDP
        # averaging. Apply that shared policy once before inspecting gradients.
        norm_options = dict(
            pp_enabled=pp_size > 1,
            device_mesh=setup.mesh_context.device_mesh,
            moe_mesh=setup.mesh_context.moe_mesh,
            ep_axis_name="ep" if setup.mesh_context.moe_mesh is not None else None,
            pp_axis_name="pp" if pp_size > 1 else None,
            foreach=True,
            grad_norm_backend="triton",
        )
        norm = scale_grads_and_clip_grad_norm(float("inf"), parts, dp_group_size=dp_size, **norm_options)
        if isinstance(norm, DTensor):
            norm = norm.full_tensor()
        gradients = _union(parts, gradients=True)
        assert gradients.keys() == reference.keys()
        for name, expected in reference.items():
            assert expected.grad is not None and torch.isfinite(expected.grad).all(), name
            assert torch.isfinite(gradients[name]).all(), name
            torch.testing.assert_close(gradients[name], expected.grad.cpu(), atol=4e-6, rtol=5e-4, msg=name)
        gradient_max_error = max(
            float((gradients[name] - parameter.grad.cpu()).abs().max()) for name, parameter in reference.items()
        )
        clipped_norm = clip_grad_norm(0.2, parts, **norm_options)
        if isinstance(clipped_norm, DTensor):
            clipped_norm = clipped_norm.full_tensor()
        torch.testing.assert_close(clipped_norm.double(), norm.double(), atol=1e-8, rtol=1e-8)
        expected_norm = torch.nn.utils.clip_grad_norm_(list(reference.values()), 0.2)
        torch.testing.assert_close(norm.double(), expected_norm.double(), atol=4e-6, rtol=5e-4)
        optimizer.step()
        reference_optimizer.step()
        updated = _union(parts)
        for name, expected in reference.items():
            torch.testing.assert_close(updated[name], expected.detach().cpu(), atol=2e-6, rtol=5e-4, msg=name)
        if rank == 0:
            Path(evidence).write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "provenance": provenance,
                        "world_size": world,
                        "ep_size": ep_size,
                        "pp_size": pp_size,
                        "compute_dtype": "float32",
                        "all_vocabulary_logits": True,
                        "query_scope": "all valid queries; padding query outputs have no objective contract",
                        "global_supervised_target_count": total,
                        "uneven_padding_and_empty_microbatch": True,
                        "loss": actual_loss.item(),
                        "reference_loss": reference_loss.item(),
                        "global_norm": norm.item(),
                        "reference_global_norm": expected_norm.item(),
                        "gradient_max_abs_error": gradient_max_error,
                        "update_max_abs_error": max(
                            float((updated[n] - p.detach().cpu()).abs().max()) for n, p in reference.items()
                        ),
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("pp_size,ep_size", [(1, 2), (2, 1)], ids=["ep2", "pp2"])
def test_two_gpu_parallel_training(tmp_path: Path, pp_size: int, ep_size: int) -> None:
    """Keep PR functional tests within two GPUs; composed EP2/PP2 is a scheduled run."""
    source = tmp_path / "source"
    _write_source(source)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    evidence = Path(os.environ.get("FLINT_PARALLEL_EVIDENCE", str(tmp_path)))
    evidence.mkdir(parents=True, exist_ok=True)
    mp.spawn(
        _parallel_worker,
        args=(port, str(source), str(evidence / f"parallel_ep{ep_size}_pp{pp_size}.json"), pp_size, ep_size),
        nprocs=2,
        join=True,
    )


def _mixed_te_ep_worker(rank: int, port: int, source: str, evidence: str, peft_enabled: bool) -> None:
    """Exercise EP2 FP32 masters/BF16 TE compute against a same-kernel storage control."""
    from safetensors.torch import load_file
    from torch.distributed.tensor import distribute_tensor

    from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
    from nemo_automodel.components.models.flint.model import FlintForCausalLM
    from nemo_automodel.shared.parameter_names import canonical_parameter_fqn
    from tests.functional_tests.models.flint.test_attention import attention_backend, te_config

    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank)
    )
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, timeout=timedelta(minutes=8))
    try:
        provenance = _runtime_receipt()
        setup = DistributedSetup.build(
            strategy=FSDP2Config(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    output_dtype=torch.bfloat16,
                    cast_forward_inputs=False,
                ),
                enable_fsdp2_prefetch=False,
                defer_fsdp_grad_sync=False,
            ),
            parallelism_sizes=ParallelismSizes(dp_size=2, ep_size=2),
            moe_parallel_config=MoEParallelizerConfig(reshard_after_forward=True),
            world_size=2,
        )
        peft = (
            PeftConfig(
                target_modules=["model.layers.*.self_attn.*_proj", "model.layers.*.mlp.experts"],
                dim=4,
                alpha=8,
                lora_dtype=torch.float32,
                use_memory_efficient_lora=False,
            )
            if peft_enabled
            else None
        )
        initialization_dtypes = []
        original_initialize = FlintForCausalLM.initialize_weights

        def adapter_storage(model: torch.nn.Module) -> dict:
            """Observe declared adapter master storage without changing or gathering it."""
            return {
                name: {
                    "dtype": str(value.dtype),
                    "local_dtype": str(value.to_local().dtype) if isinstance(value, DTensor) else str(value.dtype),
                    "shape": list(value.shape),
                    "grad_dtype": None if value.grad is None else str(value.grad.dtype),
                }
                for name, value in model.named_parameters()
                if "lora_" in name
            }

        def observe_initialize(model: torch.nn.Module, *, dtype: torch.dtype | None = None) -> None:
            """Trace family initialization's effect on explicit PEFT master dtypes."""
            before = adapter_storage(model)
            original_initialize(model, dtype=dtype)
            initialization_dtypes.append(
                {"requested_dtype": str(dtype), "before": before, "after": adapter_storage(model)}
            )

        FlintForCausalLM.initialize_weights = observe_initialize
        try:
            native = NeMoAutoModelForCausalLM.from_pretrained(
                source,
                dtype=torch.bfloat16 if peft_enabled else torch.float32,
                peft_config=peft,
                backend=attention_backend("te"),
                distributed_setup=setup,
                attn_implementation="eager",
                use_liger_kernel=False,
                use_sdpa_patching=False,
                local_files_only=True,
            )
        finally:
            FlintForCausalLM.initialize_weights = original_initialize
        from torch.distributed.fsdp import FSDPModule

        protected_storage = []
        for index, layer in enumerate(native.model.layers):
            module = layer.self_attn.attn_module
            state_getter = getattr(module, "_get_fsdp_state", None)
            state = state_getter() if isinstance(module, FSDPModule) and callable(state_getter) else None
            group = getattr(state, "_fsdp_param_group", None)
            policy = getattr(group, "mp_policy", None)
            protected_storage.append(
                {
                    "layer": index,
                    "is_fsdp": isinstance(module, FSDPModule),
                    "sink_storage_dtype": str(module.softmax_offset.dtype),
                    "compute_policy": str(getattr(policy, "param_dtype", None)),
                }
            )
        if rank == 0:
            Path(evidence).with_suffix(".pre_forward.json").write_text(
                json.dumps(
                    {
                        "status": "observed",
                        "peft_enabled": peft_enabled,
                        "provenance": provenance,
                        "protected_attention_units": protected_storage,
                        "adapter_initialization_dtypes": initialization_dtypes,
                        "adapter_storage_after_public_loader": adapter_storage(native),
                    },
                    indent=2,
                )
                + "\n"
            )
        if os.environ.get("FLINT_DTYPE_PROBE_ONLY") == "1":
            return
        source_weights = load_file(str(Path(source) / "model.safetensors"))
        reference = te_config(dtype=torch.bfloat16).build(backend=attention_backend("te")).to(device)
        missing = reference.load_state_dict(reference.state_dict_adapter.from_hf(dict(source_weights)), strict=False)
        assert not missing.unexpected_keys and all(name.endswith("._extra_state") for name in missing.missing_keys)
        if peft_enabled:
            apply_lora_to_linear_modules(reference, peft)
            # GroupedExpertsLoRA constructs its replacement from the config;
            # materialize that new subtree before using the CUDA control.
            reference.to(device)
            for model in (native, reference):
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if "lora_" not in name:
                            continue
                        canonical = canonical_parameter_fqn(name)
                        seed = int(hashlib.sha256(canonical.encode()).hexdigest()[:8], 16)
                        generator = torch.Generator().manual_seed(seed)
                        value = torch.randn(tuple(parameter.shape), generator=generator, dtype=torch.float32) * 0.02
                        if isinstance(parameter, DTensor):
                            value = distribute_tensor(value.to(device), parameter.device_mesh, parameter.placements)
                        else:
                            value = value.to(device)
                        parameter.copy_(value)
            adapter_values = {
                name: parameter.detach() for name, parameter in reference.named_parameters() if "lora_" in name
            }
            source_weights.update(
                {name: value.cpu() for name, value in reference.state_dict_adapter.to_hf(adapter_values).items()}
            )
        observations = {"head": [], "attention": [], "sink": [], "residual": [], "backbone": []}

        def observe_head(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            """Observe actual head input [B,S,H] and protected [V,H] weight inside FSDP."""
            observations["head"].append({"input": str(args[0].dtype), "weight": str(module.weight.dtype)})
            assert args[0].dtype == module.weight.dtype == torch.float32

        def observe_attention(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            """Observe real DPA Q/K/V tensors and learned-offset vector during compute."""
            observations["attention"].append([str(value.dtype) for value in args[:3]])
            observations["sink"].append(str(module.softmax_offset.dtype))
            assert all(value.dtype == torch.bfloat16 for value in args[:3])
            assert module.softmax_offset.dtype == torch.float32

        def observe_residual(module: torch.nn.Module, args: tuple, output: torch.Tensor) -> None:
            """Observe the decoder's residual [B,S,H] after its FSDP output boundary."""
            observations["residual"].append(str(output.dtype))
            assert output.dtype == torch.float32

        def observe_backbone(module: torch.nn.Module, args: tuple, output: object) -> None:
            """Observe final hidden [B,S,H] after the inner backbone's FSDP boundary."""
            hidden = output.last_hidden_state
            observations["backbone"].append(str(hidden.dtype))
            assert hidden.dtype == torch.float32

        handles = [
            native.lm_head.register_forward_pre_hook(observe_head),
            native.model.register_forward_hook(observe_backbone),
        ]
        for layer in native.model.layers:
            handles.append(layer.self_attn.attn_module.register_forward_pre_hook(observe_attention))
            handles.append(layer.register_forward_hook(observe_residual))
        total = sum(int((_ep_pp_batch(index, torch.device("cpu"))["labels"] != -100).sum()) for index in range(2))
        samples = [_ep_pp_batch(index, device) for index in range(2)]
        reference_batch = {name: torch.cat([sample[name] for sample in samples]) for name in samples[0]}
        labels = reference_batch.pop("labels")
        # FSDP keeps FP32 adapter masters while its declared bulk policy
        # computes their unhinted linears in BF16. The plain control needs the
        # equivalent compute context; the model's protected head disables AMP.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=peft_enabled):
            reference_logits_value = reference(**reference_batch).logits
        reference_loss = (
            F.cross_entropy(reference_logits_value.flatten(0, 1), labels.flatten(), reduction="sum") / total
        )
        reference_loss.backward()
        batch = samples[rank]
        prepare_for_grad_accumulation([native])
        prepare_for_final_backward([native])
        logits = native(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        expected_logits = reference_logits_value[rank * 2 : (rank + 1) * 2].detach()
        valid = batch["attention_mask"].bool()
        actual_valid, expected_valid = logits[valid], expected_logits[valid]
        logit_error = float((actual_valid - expected_valid).abs().max())
        cosine = float(F.cosine_similarity(actual_valid.flatten(), expected_valid.flatten(), dim=0))
        assert logit_error < 1e-2 and cosine > 0.9999
        loss = F.cross_entropy(logits.flatten(0, 1), batch["labels"].flatten(), reduction="sum") / total
        (loss * 2).backward()
        reduced_loss = loss.detach().clone()
        dist.all_reduce(reduced_loss)
        torch.testing.assert_close(reduced_loss, reference_loss, atol=1e-3, rtol=1e-3)
        norm_options = dict(
            device_mesh=setup.mesh_context.device_mesh,
            moe_mesh=setup.mesh_context.moe_mesh,
            ep_axis_name="ep",
            foreach=True,
            grad_norm_backend="triton",
        )
        norm = scale_grads_and_clip_grad_norm(float("inf"), [native], dp_group_size=2, **norm_options)
        if isinstance(norm, DTensor):
            norm = norm.full_tensor()
        if rank == 0:
            Path(evidence).with_suffix(".raw_adapter_gradients.json").write_text(
                json.dumps({"adapter_storage_after_backward": adapter_storage(native)}, indent=2) + "\n"
            )
        gradients = _union([native], gradients=True)
        reference_native = {
            name: parameter.grad.detach()
            for name, parameter in reference.named_parameters()
            if parameter.grad is not None
        }
        expected_gradients = reference.state_dict_adapter.to_hf(reference_native)
        assert gradients.keys() == expected_gradients.keys()
        errors = {}
        failures = []
        for name, gradient in gradients.items():
            actual_dtype = str(gradient.dtype)
            gradient = gradient.float()
            expected = expected_gradients[name].float().cpu()
            assert torch.isfinite(gradient).all() and torch.isfinite(expected).all(), name
            relative = float((gradient - expected).norm() / expected.norm().clamp_min(1e-20))
            closest = bool(torch.allclose(gradient, expected, atol=3e-3, rtol=0.15))
            difference = (gradient - expected).abs()
            worst = int(difference.flatten().argmax())
            errors[name] = {
                "max_abs": float(difference.max()),
                "actual_storage_dtype": actual_dtype,
                "relative_l2": relative,
                "actual_norm": float(gradient.norm()),
                "reference_norm": float(expected.norm()),
                "norm_ratio": float(gradient.norm() / expected.norm().clamp_min(1e-20)),
                "cosine": float(F.cosine_similarity(gradient.flatten(), expected.flatten(), dim=0)),
                "elementwise_gate_passed": closest,
                "worst_actual": float(gradient.flatten()[worst]),
                "worst_reference": float(expected.flatten()[worst]),
                "worst_flat_index": worst,
            }
            if not closest or relative >= 0.12:
                failures.append(name)
            if peft_enabled:
                if "lora_" not in name or actual_dtype != "torch.float32" or not torch.count_nonzero(gradient):
                    failures.append(name)
            if name.endswith("softmax_offset"):
                assert gradient.dtype == torch.float32 and torch.count_nonzero(gradient) == 4
        expected_norm = (
            torch.stack(
                [
                    parameter.grad.double().square().sum()
                    for parameter in reference.parameters()
                    if parameter.grad is not None
                ]
            )
            .sum()
            .sqrt()
        )
        if rank == 0:
            Path(evidence).with_suffix(".gradient_diagnostics.json").write_text(
                json.dumps(
                    {
                        "status": "observed",
                        "numerical_gradient_gate_passed": not failures,
                        "provenance": provenance,
                        "peft_enabled": peft_enabled,
                        "loss": float(reduced_loss),
                        "reference_loss": float(reference_loss),
                        "logit_max_abs": logit_error,
                        "logit_cosine": cosine,
                        "global_norm": float(norm),
                        "reference_global_norm": float(expected_norm),
                        "observed_dtypes": observations,
                        "failures": failures,
                        "gradient_errors": errors,
                        "fixed_thresholds": {"atol": 0.003, "rtol": 0.15, "tensor_relative_l2_lt": 0.12},
                    },
                    indent=2,
                )
                + "\n"
            )
        assert not failures, {name: errors[name] for name in failures}
        torch.testing.assert_close(norm.double(), expected_norm.double(), atol=1e-5, rtol=0.02)
        clip_grad_norm(0.2, [native], **norm_options)
        optimizer = torch.optim.SGD(native.parameters(), lr=0.003)
        optimizer.step()
        updated = _union([native])
        coefficient = min(1.0, 0.2 / (float(expected_norm) + 1e-6))
        update_error = 0.0
        for name, value in updated.items():
            if name in expected_gradients:
                expected = source_weights[name].float() - 0.003 * coefficient * expected_gradients[name].float().cpu()
                torch.testing.assert_close(value, expected, atol=3e-6, rtol=0, msg=name)
            else:
                assert peft_enabled
                expected = source_weights[name].to(value.dtype)
                torch.testing.assert_close(value, expected, atol=0, rtol=0, msg=name)
            update_error = max(update_error, float((value - expected).abs().max()))
        if rank == 0:
            Path(evidence).with_suffix(".training_gate.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "scope": "actual EP2 BF16 compute, FP32 masters/protected units, all-logit/gradient/norm/SGD parity",
                        "checkpoint_continuation_status": "not_yet_tested",
                        "peft_enabled": peft_enabled,
                        "provenance": provenance,
                        "loss": float(reduced_loss),
                        "reference_loss": float(reference_loss),
                        "logit_max_abs": logit_error,
                        "logit_cosine": cosine,
                        "global_norm": float(norm),
                        "reference_global_norm": float(expected_norm),
                        "update_max_abs": update_error,
                        "observed_dtypes": observations,
                        "supervised_target_count": total,
                        "adapter_storage_after_backward": adapter_storage(native),
                    },
                    indent=2,
                )
                + "\n"
            )
        restart = None
        if peft_enabled:
            from nemo_automodel.components.checkpoint.config import CheckpointingConfig

            def adam_step(model: torch.nn.Module, adam: torch.optim.Optimizer) -> None:
                """Apply the same EP2 global objective and optimizer update once."""
                adam.zero_grad(set_to_none=True)
                prepare_for_grad_accumulation([model])
                prepare_for_final_backward([model])
                prediction = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                objective = F.cross_entropy(prediction.flatten(0, 1), batch["labels"].flatten(), reduction="sum")
                (objective * 2 / total).backward()
                scale_grads_and_clip_grad_norm(0.2, [model], dp_group_size=2, **norm_options)
                adam.step()

            adam = torch.optim.AdamW([p for p in native.parameters() if p.requires_grad], lr=0.001, fused=True)
            adam_step(native, adam)
            checkpoint = Path(evidence).parent / "actual_ep2_lora_checkpoint"
            checkpointer = CheckpointingConfig(
                checkpoint_dir=checkpoint,
                model_cache_dir=str(Path(source).parent),
                model_repo_id=source,
                model_save_format="safetensors",
                save_consolidated=False,
                is_peft=True,
                v4_compatible=False,
            ).build(dp_rank=rank, tp_rank=0, pp_rank=0, moe_mesh=setup.mesh_context.moe_mesh)
            try:
                checkpointer.save_model(native, str(checkpoint), peft_config=peft)
                checkpointer.save_optimizer(adam, native, str(checkpoint))
                checkpointer.async_wait()
                checkpoint_values = _union([native])
                adam_step(native, adam)
                continuation_values = _union([native])
                restored = NeMoAutoModelForCausalLM.from_pretrained(
                    source,
                    dtype=torch.bfloat16,
                    peft_config=peft,
                    backend=attention_backend("te"),
                    distributed_setup=setup,
                    attn_implementation="eager",
                    use_liger_kernel=False,
                    use_sdpa_patching=False,
                    local_files_only=True,
                )
                from nemo_automodel.components.checkpoint.stateful_wrappers import _has_expert_parallelism

                adapter_class = type(restored.state_dict_adapter)
                original_to_hf = adapter_class.to_hf

                def observe_restore_destinations(adapter, state_dict: dict, *args, **kwargs) -> dict:
                    """Capture the exact public restore destination inventory before strict validation."""
                    if kwargs.get("for_checkpoint_load", False):
                        Path(evidence).with_suffix(f".restore_destinations.rank{rank}.json").write_text(
                            json.dumps(
                                {
                                    "rank": rank,
                                    "checkpointer_has_moe_mesh": checkpointer.moe_mesh is not None,
                                    "module_ep_marker_detected": _has_expert_parallelism(restored),
                                    "destination_count": len(state_dict),
                                    "destinations": {
                                        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                        if isinstance(value, torch.Tensor)
                                        else {"type": type(value).__name__}
                                        for name, value in state_dict.items()
                                    },
                                },
                                indent=2,
                            )
                            + "\n"
                        )
                    return original_to_hf(adapter, state_dict, *args, **kwargs)

                adapter_class.to_hf = observe_restore_destinations
                try:
                    checkpointer.load_model(restored, str(checkpoint / "model"))
                finally:
                    adapter_class.to_hf = original_to_hf
                restored_adam = torch.optim.AdamW(
                    [p for p in restored.parameters() if p.requires_grad], lr=0.001, fused=True
                )
                checkpointer.load_optimizer(restored_adam, restored, str(checkpoint))
                restored_values = _union([restored])
                assert restored_values.keys() == checkpoint_values.keys()
                for name in checkpoint_values:
                    torch.testing.assert_close(restored_values[name], checkpoint_values[name], atol=0, rtol=0, msg=name)
                adam_step(restored, restored_adam)
                resumed_values = _union([restored])
                for name in continuation_values:
                    torch.testing.assert_close(
                        resumed_values[name], continuation_values[name], atol=0, rtol=0, msg=name
                    )
                restart = {
                    "status": "passed",
                    "public_checkpointer": True,
                    "model_tensors_exact": len(checkpoint_values),
                    "continuation_max_abs": 0.0,
                    "optimizer": "torch.optim.AdamW(fused=True), actual DCP model+optimizer save/load",
                    "model_artifact": "actual_ep2_lora_checkpoint/model/adapter_model.safetensors",
                }
            finally:
                checkpointer.close()
        assert observations["head"] and observations["attention"] and observations["residual"]
        after = _runtime_receipt()
        assert after["source_sha256"] == provenance["source_sha256"], "Source changed during mixed EP gate"
        if rank == 0:
            Path(evidence).write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "scope": "actual public EP2 with BF16 TE compute and protected FP32 units; same-kernel storage control",
                        "provenance": provenance,
                        "source_sha256_after": after["source_sha256"],
                        "peft_enabled": peft_enabled,
                        "resident_base_dtype": "bfloat16" if peft_enabled else "float32",
                        "trainable_dtype": "float32",
                        "trainable_family": "attention/expert LoRA" if peft_enabled else "full model",
                        "ep_size": 2,
                        "dp_size": 2,
                        "cp_size": 1,
                        "source_dtype": "float32",
                        "compute_dtype": "bfloat16",
                        "global_supervised_target_count": total,
                        "dense_backward_multiplier": 2,
                        "expert_gradient_divisor": "shared dp_group_size/ep_shard_size = 2",
                        "observed_dtypes": observations,
                        "loss": float(reduced_loss),
                        "reference_loss": float(reference_loss),
                        "logit_max_abs": logit_error,
                        "logit_cosine": cosine,
                        "gradient_errors": errors,
                        "global_norm": float(norm),
                        "reference_global_norm": float(expected_norm),
                        "update_max_abs": update_error,
                        "checkpoint_restart": restart,
                        "independent_fp32_oracle": "separate unchanged FP32 EP/PP tests",
                    },
                    indent=2,
                )
                + "\n"
            )
        for handle in handles:
            handle.remove()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("peft_enabled", [False, True], ids=["full_sft", "attention_expert_lora"])
def test_two_gpu_mixed_precision_te_expert_parallel(tmp_path: Path, peft_enabled: bool) -> None:
    """Guard actual EP2 protected head/sink lifetime under BF16 bulk FSDP compute."""
    from tests.functional_tests.models.flint.test_attention import attention_backend, te_config

    torch.manual_seed(54375)
    source = tmp_path / "source"
    source.mkdir()
    model = te_config(dtype=torch.float32).build(backend=attention_backend("eager"))
    model.config.save_pretrained(source)
    save_file(model.state_dict_adapter.to_hf(model.state_dict()), source / "model.safetensors")
    directory = Path(os.environ.get("FLINT_PARALLEL_EVIDENCE", str(tmp_path)))
    directory.mkdir(parents=True, exist_ok=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(
        _mixed_te_ep_worker,
        args=(port, str(source), str(directory / f"mixed_te_ep2_lora_{peft_enabled}.json"), peft_enabled),
        nprocs=2,
        join=True,
    )


_ADAPTERS = ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B")


class _SwiGLU(torch.autograd.Function):
    """Independent BF16 projection-store activation with one FP32 analytic adjoint."""

    @staticmethod
    def forward(ctx, projection: torch.Tensor) -> torch.Tensor:
        """Map fused projection [rows,2*intermediate] to [rows,intermediate]."""
        ctx.save_for_backward(projection)
        gate, up = projection.float().chunk(2, -1)
        return (gate * gate.sigmoid() * up).to(projection.dtype)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        """Return one rounded concatenated adjoint [rows,2*intermediate]."""
        (projection,) = ctx.saved_tensors
        gate, up = projection.float().chunk(2, -1)
        sigmoid = gate.sigmoid()
        grad_gate = gradient.float() * up * sigmoid * (1 + gate * (1 - sigmoid))
        grad_up = gradient.float() * gate * sigmoid
        return (torch.cat((grad_gate, grad_up), -1).to(projection.dtype),)


def _lora_config() -> MoEConfig:
    return MoEConfig(
        dim=128,
        inter_dim=256,
        moe_inter_dim=128,
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0,
        aux_loss_coeff=0,
        score_func="sigmoid",
        route_scale=1,
        norm_topk_prob=True,
        expert_activation="swiglu",
        apply_router_weight_after_down=True,
        dtype=torch.bfloat16,
    )


def _sample(rank: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Return ragged BF16 x/cotangent [T,128], mask[T], FP32 p/IDs[T,2]."""
    generator = torch.Generator(device=device).manual_seed(14375 + rank)
    count = 11 + 2 * rank
    values = (torch.randn(count, 128, generator=generator, device=device) * 0.2).to(torch.bfloat16)
    probabilities = torch.rand(count, 2, generator=generator, device=device) * 0.4 + 0.1
    routes = torch.stack((torch.arange(count, device=device) % 4, (torch.arange(count, device=device) + 2) % 4), -1)
    mask = torch.arange(count, device=device) < count - 2
    cotangent = torch.randn(count, 128, generator=generator, device=device) * 0.2
    return values, mask, probabilities, routes, cotangent


def _independent(
    states: dict[str, torch.Tensor],
    values: torch.Tensor,
    mask: torch.Tensor,
    probabilities: torch.Tensor,
    routes: torch.Tensor,
    *,
    dispatched: bool,
) -> torch.Tensor:
    """Compute explicit expert-selected LoRA branches without production helpers.

    Args:
        states: Full expert base [E,H,2I]/[E,I,H] and LoRA A/B tensors.
        values: BF16 inputs [T,H].
        mask: Valid tokens [T].
        probabilities: FP32 route probabilities [T,K].
        routes: Global integer expert IDs [T,K].
        dispatched: DeepEP stores weighted per-slot output BF16 before combine.

    Returns:
        BF16 combined output [T,H], preserving after-down routing placement.
    """
    slots = torch.zeros(*routes.shape, values.shape[-1], dtype=torch.float32, device=values.device)
    for expert in range(4):
        token, slot = torch.where((routes == expert) & mask[:, None])
        selected = values[token]
        base_up = selected @ states["gate_and_up_projs"][expert].to(values.dtype)
        adapter_up = (selected @ states[_ADAPTERS[0]][expert].to(values.dtype)) @ states[_ADAPTERS[1]][expert].to(
            values.dtype
        )
        activated = _SwiGLU.apply(base_up + adapter_up * 2)
        base_down = activated @ states["down_projs"][expert].to(values.dtype)
        adapter_down = (activated @ states[_ADAPTERS[2]][expert].to(values.dtype)) @ states[_ADAPTERS[3]][expert].to(
            values.dtype
        )
        weighted = (base_down + adapter_down * 2).float() * probabilities[token, slot, None]
        slots[token, slot] = weighted.to(values.dtype).float() if dispatched else weighted
    return slots.sum(1).to(values.dtype)


def _lora_worker(rank: int, port: int, dispatcher: str, evidence: str) -> None:
    """Compare real EP2 kernels, every adapter/input/router gradient and one SGD update."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank)
    )
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, timeout=timedelta(minutes=8))
    try:
        torch.manual_seed(24375)
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("ep",))
        backend = BackendConfig(experts="torch_mm", dispatcher=dispatcher)
        base = (
            GroupedExpertsDeepEP(_lora_config(), backend=backend)
            if dispatcher == "deepep"
            else GroupedExperts(_lora_config(), backend=backend)
        ).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            for parameter in base.parameters():
                parameter.normal_(std=0.05)
        module = (
            GroupedExpertsDeepEPLoRA(base, lora_dim=8, alpha=16, lora_dtype=torch.float32)
            if dispatcher == "deepep"
            else GroupedExpertsLoRA(base, lora_dim=8, alpha=16, lora_dtype=torch.float32)
        )
        module = module.to(device)
        with torch.no_grad():
            for parameter in module.parameters():
                if not parameter.requires_grad:
                    parameter.data = parameter.data.to(torch.bfloat16)
            for name in _ADAPTERS:
                getattr(module, name).normal_(std=0.03)
        states = {
            name: parameter.detach().clone().requires_grad_(parameter.requires_grad)
            for name, parameter in module.named_parameters()
        }
        for name, parameter in list(module.named_parameters()):
            setattr(
                module,
                name,
                torch.nn.Parameter(
                    distribute_tensor(parameter.detach(), mesh, [Shard(0)]), requires_grad=parameter.requires_grad
                ),
            )
        if dispatcher == "deepep":
            module.init_token_dispatcher(mesh)
        values, mask, probabilities, routes, cotangent = _sample(rank, device)
        values.requires_grad_()
        probabilities.requires_grad_()
        actual = module(values, mask, probabilities, routes)
        (actual.float() * cotangent).sum().backward()
        references = []
        reference_loss = torch.zeros((), device=device)
        for data_rank in range(2):
            x, valid, probs, ids, upstream = _sample(data_rank, device)
            x.requires_grad_()
            probs.requires_grad_()
            output = _independent(states, x, valid, probs, ids, dispatched=dispatcher == "deepep")
            reference_loss = reference_loss + (output.float() * upstream).sum()
            references.append((output, x, probs))
        reference_loss.backward()
        expected, expected_x, expected_probabilities = references[rank]
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0.025)
        assert torch.count_nonzero(actual[~mask]) == 0
        errors = {}
        comparisons = {
            "input": (values.grad, expected_x.grad),
            "probabilities": (probabilities.grad, expected_probabilities.grad),
        }
        for name in _ADAPTERS:
            parameter = getattr(module, name)
            comparisons[name] = (parameter.grad.full_tensor(), states[name].grad)
        for name, (observed, target) in comparisons.items():
            assert torch.isfinite(observed).all() and torch.count_nonzero(observed)
            torch.testing.assert_close(observed, target, atol=1e-4, rtol=0.1, msg=name)
            relative = float((observed.float() - target.float()).norm() / target.float().norm().clamp_min(1e-20))
            assert relative < 0.04, (name, relative)
            errors[name] = {"max_abs": float((observed - target).abs().max()), "relative_l2": relative}
        assert module.gate_and_up_projs.grad is None and module.down_projs.grad is None
        optimizer = torch.optim.SGD([getattr(module, name) for name in _ADAPTERS], lr=0.01)
        optimizer.step()
        for name in _ADAPTERS:
            expected_update = states[name].detach() - 0.01 * states[name].grad
            torch.testing.assert_close(
                getattr(module, name).full_tensor(), expected_update, atol=1e-6, rtol=1e-4, msg=name
            )
        if rank == 0:
            Path(evidence).write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "dispatcher": dispatcher,
                        "experts": "actual torch._grouped_mm",
                        "ep_size": 2,
                        "mock_dispatcher": False,
                        "source_dtype": "bfloat16",
                        "lora_dtype": "float32",
                        "apply_router_weight_after_down": True,
                        "output_max_abs": float((actual - expected).abs().max()),
                        "gradient_errors": errors,
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("dispatcher", ["torch", "deepep"])
def test_actual_expert_parallel_lora_kernels(tmp_path: Path, dispatcher: str) -> None:
    """Exercise optional real CUDA communication and grouped GEMMs, never CPU mocks."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    directory = Path(os.environ.get("FLINT_EXPERT_LORA_EVIDENCE", str(tmp_path)))
    directory.mkdir(parents=True, exist_ok=True)
    mp.spawn(
        _lora_worker, args=(port, dispatcher, str(directory / f"expert_lora_{dispatcher}.json")), nprocs=2, join=True
    )
