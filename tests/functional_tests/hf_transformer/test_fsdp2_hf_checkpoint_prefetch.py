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

"""FSDP2 prefetch regression coverage for Hugging Face activation checkpointing."""

import copy
import functools
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint
from transformers.modeling_layers import GradientCheckpointingLayer

from nemo_automodel.components.distributed.parallelizer import DefaultParallelizationStrategy
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

_WORLD_SIZE = 2
_NUM_LAYERS = 4
_HIDDEN_SIZE = 8


class _HFCheckpointLayer(GradientCheckpointingLayer):
    """Tiny HF-style decoder layer using the upstream checkpointing ``__call__`` contract."""

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(_HIDDEN_SIZE, 2 * _HIDDEN_SIZE)
        self.down = nn.Linear(2 * _HIDDEN_SIZE, _HIDDEN_SIZE)
        self.observed_dtypes: list[torch.dtype] = []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply a residual MLP.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        self.observed_dtypes.append(hidden_states.dtype)
        return hidden_states + self.down(torch.nn.functional.silu(self.up(hidden_states)))


# AutoModel deliberately restricts its HF-native path to upstream transformers modules.
_HFCheckpointLayer.__module__ = "transformers.models.test_fsdp2.modeling_test_fsdp2"


class _TinyBackbone(nn.Module):
    """Backbone with root-owned input parameters and independently sharded layers."""

    def __init__(self) -> None:
        super().__init__()
        self.input_proj = nn.Linear(_HIDDEN_SIZE, _HIDDEN_SIZE)
        self.layers = nn.ModuleList([_HFCheckpointLayer() for _ in range(_NUM_LAYERS)])


class _TinyHFModel(nn.Module):
    """Minimal model exposing the Hugging Face gradient-checkpointing API."""

    supports_gradient_checkpointing = True

    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyBackbone()
        self.lm_head = nn.Linear(_HIDDEN_SIZE, _HIDDEN_SIZE)
        self.config = SimpleNamespace(use_cache=False, num_kv_shared_layers=0)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, object] | None = None) -> None:
        """Enable the upstream HF layer checkpointing behavior."""
        kwargs = gradient_checkpointing_kwargs or {"use_reentrant": True}
        checkpoint_fn = functools.partial(checkpoint, **kwargs)
        for layer in self.model.layers:
            layer.gradient_checkpointing = True
            layer._gradient_checkpointing_func = checkpoint_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the tiny language model.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        hidden_states = self.model.input_proj(hidden_states)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return self.lm_head(hidden_states)


def _free_port() -> int:
    """Return an available localhost TCP port for the spawned process group."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _full_gradient(parameter: nn.Parameter) -> torch.Tensor:
    """Materialize an FSDP2 parameter gradient on every rank.

    Args:
        parameter: Parameter of arbitrary shape. Its gradient may be a DTensor
            sharded on the FSDP mesh or an ordinary local tensor.

    Returns:
        Full gradient tensor with the parameter's global shape, replicated on
        every rank. An ordinary gradient is returned without copying.
    """
    if parameter.grad is None:
        raise AssertionError("expected every parameter to receive a gradient")
    if isinstance(parameter.grad, DTensor):
        return parameter.grad.full_tensor()
    return parameter.grad


def _worker(rank: int, port: int) -> None:
    """Run one rank of the real FSDP2 checkpoint/prefetch regression."""
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=_WORLD_SIZE,
    )

    original_unshard = FSDPParamGroup.unshard
    try:
        torch.manual_seed(1234)
        model = _TinyHFModel().cuda(rank)
        reference = copy.deepcopy(model).to(torch.bfloat16)
        mesh = init_device_mesh(
            "cuda",
            (1, _WORLD_SIZE, 1),
            mesh_dim_names=("dp_replicate", "dp_shard_cp", "tp"),
        )

        phase = {"name": "setup"}
        all_gather_counts = {"forward": 0, "backward": 0}

        def counted_unshard(param_group: FSDPParamGroup, async_op: bool = False):
            should_all_gather = param_group._all_gather_result is None and not param_group.is_unsharded
            if should_all_gather and phase["name"] in all_gather_counts:
                all_gather_counts[phase["name"]] += 1
            return original_unshard(param_group, async_op)

        FSDPParamGroup.unshard = counted_unshard

        model = DefaultParallelizationStrategy().parallelize(
            model=model,
            device_mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            ),
            activation_checkpointing=True,
            enable_fsdp2_prefetch=True,
            fsdp2_backward_prefetch_depth=2,
            fsdp2_forward_prefetch_depth=1,
            reshard_after_forward=True,
        )

        assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
        assert set(model.state_dict()) == set(reference.state_dict())
        for layer in model.model.layers:
            assert hasattr(layer, "_checkpoint_wrapped_module")
            assert not hasattr(layer._checkpoint_wrapped_module, "_gradient_checkpointing_func")

        torch.manual_seed(5678)
        inputs = torch.randn(2, 4, _HIDDEN_SIZE, device=rank)

        phase["name"] = "forward"
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(inputs)
        phase["name"] = "backward"
        output.sum().backward()
        phase["name"] = "done"

        assert all(
            layer._checkpoint_wrapped_module.observed_dtypes == [torch.bfloat16, torch.bfloat16]
            for layer in model.model.layers
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reference_output = reference(inputs.to(torch.bfloat16)).float()
        reference_output.sum().backward()

        assert all_gather_counts == {"forward": _NUM_LAYERS + 1, "backward": _NUM_LAYERS}
        torch.testing.assert_close(output, reference_output, rtol=1e-2, atol=1e-2)
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            reference_name = canonical_parameter_fqn(name)
            torch.testing.assert_close(
                _full_gradient(parameter),
                reference_parameters[reference_name].grad,
                rtol=1e-2,
                atol=1e-2,
                check_dtype=False,
            )
    finally:
        FSDPParamGroup.unshard = original_unshard
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE,
    reason="requires two CUDA GPUs",
)
def test_hf_non_reentrant_checkpointing_avoids_duplicate_fsdp2_all_gathers() -> None:
    """HF recomputation must preserve gradients without rerunning forward-prefetch collectives."""
    mp.spawn(_worker, args=(_free_port(),), nprocs=_WORLD_SIZE, join=True)
