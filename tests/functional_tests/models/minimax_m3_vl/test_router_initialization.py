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

"""Preserve M3 router shards through initialization after FSDP wrapping."""

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.minimax_m3_vl import model as model_module
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLConfig, MiniMaxM3VLTextConfig
from nemo_automodel.components.models.minimax_m3_vl.model import (
    MiniMaxM3SparseForCausalLM,
    MiniMaxM3SparseForConditionalGeneration,
)
from nemo_automodel.components.moe.layers import Gate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FSDP2")
@pytest.mark.parametrize("vision", [False, True])
def test_initialize_after_sharding_preserves_fp32_router(vision: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Actual local shards, projection and gradients must preserve the FP32 contract."""
    torch.cuda.set_device(0)
    torch.distributed.init_process_group("gloo", store=torch.distributed.HashStore(), rank=0, world_size=1)
    try:
        config = MiniMaxM3VLTextConfig(
            hidden_size=64,
            intermediate_size=32,
            dense_intermediate_size=48,
            shared_intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            rotary_dim=8,
            partial_rotary_factor=0.5,
            vocab_size=128,
            num_local_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            moe_layer_freq=[1],
            num_mtp_modules=0,
            use_routing_bias=True,
        )
        backend = BackendConfig(linear="torch", attn="sdpa", experts="torch", dispatcher="torch", rope_fusion=False)
        with torch.device("cuda"):
            if vision:
                vl_config = MiniMaxM3VLConfig(
                    text_config=config.to_dict(),
                    vision_config={
                        "hidden_size": 32,
                        "intermediate_size": 64,
                        "num_attention_heads": 4,
                        "num_hidden_layers": 1,
                        "patch_size": 2,
                        "img_token_compression_config": {"spatial_merge_size": 2, "temporal_patch_size": 2},
                    },
                    projector_hidden_size=64,
                )
                model = MiniMaxM3SparseForConditionalGeneration(vl_config, backend=backend)
            else:
                model = MiniMaxM3SparseForCausalLM(config, backend=backend)
        # Shard the actual router as its own FP32 unit, as the model parallelizer
        # does. Initializing the containing model must not mutate its local dtype.
        mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("dp",))
        gates = [module for module in model.modules() if isinstance(module, Gate)]
        assert len(gates) == 1
        for gate in gates:
            fully_shard_by_dtype(
                gate,
                mesh=mesh,
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32),
                offload_policy=None,
                reshard_after_forward=True,
            )
        originals: dict[int, torch.Tensor] = {}

        def capture_before_cast(
            module: torch.nn.Module, dtype: torch.dtype, *, skip_modules: tuple[str, ...] = ()
        ) -> None:
            for gate in gates:
                originals[id(gate)] = gate.weight.to_local().detach().clone()
            cast_model_to_dtype(module, dtype, skip_modules=skip_modules)

        monkeypatch.setattr(model_module, "cast_model_to_dtype", capture_before_cast)
        torch.manual_seed(29)
        model.initialize_weights(dtype=torch.bfloat16)
        for gate in gates:
            assert isinstance(gate.weight, DTensor)
            assert gate.weight.dtype == torch.float32
            assert gate.weight.to_local().dtype == torch.float32
            torch.testing.assert_close(gate.weight.to_local(), originals[id(gate)], rtol=0, atol=0)
            assert gate.e_score_correction_bias.dtype == torch.float32
            assert gate.weight.to_local().float().std() > 0
            # A cast back to FP32 cannot repair already-rounded values.
            assert not torch.equal(gate.weight.to_local(), gate.weight.to_local().bfloat16().float())
            observed: list[torch.Tensor] = []

            def check_forward_precision(module: Gate, args: tuple[torch.Tensor | None, ...]) -> None:
                assert module.weight.dtype == torch.float32
                observed.append(module.weight.detach().clone())

            handle = gate.register_forward_pre_hook(check_forward_precision)
            optimizer = torch.optim.AdamW(gate.parameters(), lr=0.01)
            for step in range(2):
                x = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                result = gate(x, torch.ones(7, device="cuda", dtype=torch.bool), None)
                weights = result[0]
                if step == 0:
                    torch.testing.assert_close(observed[-1], originals[id(gate)], rtol=0, atol=0)
                (weights.float().square().sum()).backward()
                assert gate.weight.to_local().dtype == torch.float32
                assert gate.weight.grad.to_local().dtype == torch.float32
                assert torch.isfinite(gate.weight.grad.to_local()).all()
                optimizer.step()
                assert gate.weight.dtype == torch.float32
                assert gate.weight.to_local().dtype == torch.float32
                state = optimizer.state[gate.weight]
                assert state["exp_avg"].to_local().dtype == torch.float32
                assert state["exp_avg_sq"].to_local().dtype == torch.float32
                gate.zero_grad(set_to_none=True)
            assert len(observed) == 2
            handle.remove()
        assert model.model.embed_tokens.weight.dtype == torch.bfloat16
        assert model.model.norm.weight.dtype == torch.bfloat16
        assert torch.count_nonzero(model.model.norm.weight) == 0
    finally:
        torch.distributed.destroy_process_group()
