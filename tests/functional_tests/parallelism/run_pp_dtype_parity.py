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

"""Check Nemotron PP dtypes, losses, gradients and updates on two to eight GPUs.

Compose PP2 or PP4 with FSDP over the remaining ranks. All models use four tiny attention/MLP blocks;
no pretrained weights, dataset downloads or Mamba kernels are needed.
"""

import argparse
import os
from copy import deepcopy
from datetime import timedelta
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM


def _model(device: torch.device, *, fp32_residual: bool, mtp_depth: int) -> NemotronHForCausalLM:
    config = NemotronHConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layers_block_type=["attention", "mlp", "attention", "mlp"],
        residual_in_fp32=fp32_residual,
        dtype="bfloat16",
        use_cache=False,
        num_nextn_predict_layers=mtp_depth,
        mtp_layers_block_type=["attention"],
        attention_dropout=0.0,
    )
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    torch.manual_seed(17)
    return NemotronHForCausalLM(config, backend=backend).to(device)


def _shard(
    model: torch.nn.Module,
    world_mesh: DeviceMesh,
    moe_mesh: DeviceMesh | None,
    *,
    activation_checkpointing: bool,
    dp_axis_names: tuple[str, ...],
    **kwargs,
) -> None:
    del moe_mesh, kwargs
    policy = FSDP2Config().mp_policy
    for name, layer in model.model.layers.items():
        if activation_checkpointing:
            layer = checkpoint_wrapper(layer)
            model.model.layers[name] = layer
        fully_shard(layer, mesh=world_mesh[dp_axis_names[0]], mp_policy=policy)
    fully_shard(model, mesh=world_mesh[dp_axis_names[0]], mp_policy=policy)


def _loss(logits: torch.Tensor | tuple[torch.Tensor, ...], labels: torch.Tensor) -> torch.Tensor:
    """Compute a summed token loss.

    Args:
        logits: BF16 logits [batch, sequence, vocab], or a tuple of logits,
            MTP states [batch, sequence, hidden] and optional int32 document
            IDs [batch, sequence]. MTP states can be BF16 or FP32.
        labels: Int64 token IDs [batch, sequence].

    Returns:
        Scalar FP32 cross entropy plus a differentiable MTP-state penalty.
    """
    auxiliary = ()
    if isinstance(logits, tuple):
        logits, *auxiliary = logits
    loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="sum")
    return loss + sum(0.01 * state.float().square().sum() for state in auxiliary if state.is_floating_point())


def _full(tensor: torch.Tensor) -> torch.Tensor:
    """Read a complete parameter or gradient without mutating it.

    Args:
        tensor: Parameter-shaped tensor, potentially sharded over the DP mesh.

    Returns:
        FP32 tensor with the global parameter shape on the same device.
    """
    return (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach().float()


def _run_case(
    mesh: DeviceMesh, device: torch.device, *, fp32_residual: bool, checkpointing: bool, mtp_depth: int
) -> None:
    reference = _model(device, fp32_residual=fp32_residual, mtp_depth=mtp_depth)
    candidate = deepcopy(reference)
    # Terminal MTP states do not cross a PP boundary. CUDA autocast can
    # promote their final RMSNorm to FP32 even for BF16 residuals; use the
    # unpartitioned model as the dtype oracle for these backend-owned outputs.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        probe = reference(torch.zeros((1, 16), device=device, dtype=torch.long))
    final_dtypes = (probe.logits.dtype, *(state.dtype for state in (probe.mtp_per_depth_h or ())))
    del probe
    batch_size = mesh["pp"].size()
    pp = AutoPipeline(
        world_mesh=mesh,
        moe_mesh=None,
        pp_axis_name="pp",
        dp_axis_names=("dp",),
        pp_schedule="1f1b",
        pp_microbatch_size=1,
        pp_batch_size=batch_size,
        device=device,
        dtype=torch.bfloat16,
        pp_seq_len=16,
        patch_inner_model=False,
        patch_causal_lm_model=False,
        defer_fsdp_grad_sync=False,
    ).build(candidate, loss_fn=_loss, parallelize_fn=partial(_shard, activation_checkpointing=checkpointing))
    part = pp.parts[0]
    hidden_dtype = torch.float32 if fp32_residual else torch.bfloat16

    def check_output_dtypes(
        module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> None:
        """Check actual wire dtypes, including each tensor of a mixed MTP payload.

        Args:
            module: Pipeline stage whose forward just completed.
            inputs: Int64 token IDs [batch, sequence] or hidden states and
                optional MTP embeddings [batch, sequence, hidden].
            output: Hidden states [batch, sequence, hidden] or final-stage
                logits [batch, sequence, vocab], followed for MTP by carries
                [batch, sequence, hidden] and final int32 IDs [batch, sequence].
        """
        del module, inputs
        outputs = output if isinstance(output, tuple) else (output,)
        if pp.info.has_last_stage:
            expected = final_dtypes + ((torch.int32,) if mtp_depth else ())
        else:
            expected = (hidden_dtype,) + (torch.bfloat16,) * mtp_depth
        actual = tuple(t.dtype for t in outputs)
        assert actual == expected, f"stage output dtypes {actual}, expected {expected}"

    part.register_forward_hook(check_output_dtypes)
    optimizer = torch.optim.SGD(part.parameters(), lr=0.01)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    ref_params = dict(reference.named_parameters())
    for seq_len in (16, 24):
        # Distinct DP batches make a missing/incorrect gradient reduction observable.
        tokens = (
            torch.arange(batch_size * seq_len, device=device).reshape(batch_size, seq_len)
            + 3
            + 7 * mesh["dp"].get_local_rank()
        ) % 64
        labels = (tokens + 1) % 64
        optimizer.zero_grad(set_to_none=True)
        reference_grads = [torch.zeros_like(param) for param in reference.parameters()]
        pp.update_seq_len(seq_len)
        losses = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if pp.info.has_first_stage:
                pp.info.schedule.step(tokens, target=labels, losses=losses)
            else:
                pp.info.schedule.step(target=labels, losses=losses)
            reference_loss = torch.zeros((), device=device)
            for sample, target in zip(tokens.split(1), labels.split(1)):
                ref_optimizer.zero_grad(set_to_none=True)
                ref_output = reference(sample)
                ref_logits = (ref_output.logits, *(ref_output.mtp_per_depth_h or ()))
                loss = _loss(ref_logits, target)
                loss.backward()
                # Match the configured per-microbatch FP32 reduction followed
                # by BF16 gradient accumulation, without using FSDP in the oracle.
                for param, accumulated in zip(reference.parameters(), reference_grads):
                    assert param.grad is not None
                    reduced_grad = param.grad.float()
                    dist.all_reduce(reduced_grad, group=mesh["dp"].get_group())
                    accumulated.add_((reduced_grad / mesh["dp"].size()).to(param.dtype))
                reference_loss += loss.detach()
        if pp.info.has_last_stage:
            torch.testing.assert_close(torch.stack(losses).sum(), reference_loss, rtol=1e-6, atol=1e-5)
        for param, accumulated in zip(reference.parameters(), reference_grads):
            param.grad = accumulated
        checked = 0
        for name, param in part.named_parameters():
            name = name.replace("_checkpoint_wrapped_module.", "")
            expected = ref_params[name]
            assert param.grad is not None and expected.grad is not None, name
            torch.testing.assert_close(_full(param.grad), _full(expected.grad), rtol=0.02, atol=0.002, msg=name)
            assert torch.isfinite(_full(param.grad)).all(), name
            checked += 1
        assert checked > 0
        optimizer.step()
        ref_optimizer.step()
        for name, param in part.named_parameters():
            name = name.replace("_checkpoint_wrapped_module.", "")
            torch.testing.assert_close(_full(param), _full(ref_params[name]), rtol=0.02, atol=0.002, msg=name)
        print(
            f"PASS rank={dist.get_rank()} pp={mesh['pp'].size()} dp={mesh['dp'].size()} fp32_residual={fp32_residual} "
            f"checkpointing={checkpointing} mtp_depth={mtp_depth} seq_len={seq_len} parameters={checked} "
            f"loss={reference_loss.item():.8f}",
            flush=True,
        )
    dist.barrier()


def main() -> None:
    """Run the smallest real PP/FSDP precision regression matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-residual-only", action="store_true")
    parser.add_argument("--pp-size", type=int, choices=(2, 4), default=2)
    args = parser.parse_args()
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    world = dist.get_world_size()
    if world not in (2, 4, 8) or world % args.pp_size:
        raise ValueError("Run with two, four or eight GPUs, divisible by --pp-size")
    mesh = init_device_mesh("cuda", (args.pp_size, world // args.pp_size), mesh_dim_names=("pp", "dp"))
    for fp32_residual in (True,) if args.fp32_residual_only else (False, True):
        for checkpointing in (False, True):
            for mtp_depth in (0, 1):
                _run_case(mesh, device, fp32_residual=fp32_residual, checkpointing=checkpointing, mtp_depth=mtp_depth)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
