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

"""Check Nemotron PP dtypes, losses, gradients and updates on two or four GPUs.

Four GPUs compose PP2 with FSDP2. All models use four tiny attention/MLP blocks;
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
    pp = AutoPipeline(
        world_mesh=mesh,
        moe_mesh=None,
        pp_axis_name="pp",
        dp_axis_names=("dp",),
        pp_schedule="1f1b",
        pp_microbatch_size=1,
        pp_batch_size=2,
        device=device,
        dtype=torch.bfloat16,
        pp_seq_len=16,
        patch_inner_model=False,
        patch_causal_lm_model=False,
    ).build(candidate, loss_fn=_loss, parallelize_fn=partial(_shard, activation_checkpointing=checkpointing))
    part = pp.parts[0]
    optimizer = torch.optim.SGD(part.parameters(), lr=0.01)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    ref_params = dict(reference.named_parameters())
    for seq_len in (16, 24):
        tokens = (torch.arange(2 * seq_len, device=device).reshape(2, seq_len) + 3) % 64
        labels = (tokens + 1) % 64
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)
        pp.update_seq_len(seq_len)
        losses = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if pp.info.has_first_stage:
                pp.info.schedule.step(tokens, target=labels, losses=losses)
            else:
                pp.info.schedule.step(target=labels, losses=losses)
            reference_loss = torch.zeros((), device=device)
            for sample, target in zip(tokens.split(1), labels.split(1)):
                ref_output = reference(sample)
                ref_logits = (ref_output.logits, *(ref_output.mtp_per_depth_h or ()))
                loss = _loss(ref_logits, target)
                loss.backward()
                reference_loss += loss.detach()
        if pp.info.has_last_stage:
            torch.testing.assert_close(torch.stack(losses).sum(), reference_loss, rtol=1e-6, atol=1e-5)
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
            f"PASS rank={dist.get_rank()} pp=2 dp={mesh['dp'].size()} fp32_residual={fp32_residual} "
            f"checkpointing={checkpointing} mtp_depth={mtp_depth} seq_len={seq_len} parameters={checked} "
            f"loss={reference_loss.item():.8f}",
            flush=True,
        )
    dist.barrier()


def main() -> None:
    """Run the smallest real PP/FSDP precision regression matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-residual-only", action="store_true")
    args = parser.parse_args()
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    world = dist.get_world_size()
    if world not in (2, 4):
        raise ValueError("Run with two GPUs (PP2) or four GPUs (PP2 x FSDP2)")
    mesh = init_device_mesh("cuda", (2, world // 2), mesh_dim_names=("pp", "dp"))
    for fp32_residual in (True,) if args.fp32_residual_only else (False, True):
        for checkpointing in (False, True):
            for mtp_depth in (0, 1):
                _run_case(mesh, device, fp32_residual=fp32_residual, checkpointing=checkpointing, mtp_depth=mtp_depth)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
