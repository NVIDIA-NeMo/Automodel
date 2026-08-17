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

"""NeMo RL mixed-precision LM-head regressions under real FSDP2 hooks.

NeMo RL keeps FP32 master parameters, executes with BF16 parameters, and asks
FSDP2 to return FP32 module outputs. This exercises the two model layouts from
PR #3491:

* Gemma4 ties the output head to the input embedding, so the head is owned by
  the outer FSDP root rather than wrapped independently.
* Nemotron Omni exposes an untied nested head, allowing the parallelizer to
  wrap that head as its own FSDP unit.
"""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    """Reserve a free localhost port for process-group rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rendezvous_socket:
        rendezvous_socket.bind(("127.0.0.1", 0))
        return int(rendezvous_socket.getsockname()[1])


def _backend():
    from nemo_automodel._transformers.models.common import BackendConfig

    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def _build_gemma4():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig

    from nemo_automodel._transformers.models.gemma4_moe.model import Gemma4ForConditionalGeneration

    text_config = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=64,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=32,
        hidden_activation="gelu_pytorch_tanh",
        torch_dtype="float32",
        tie_word_embeddings=True,
    )
    config = Gemma4Config(text_config=text_config, tie_word_embeddings=True)
    return Gemma4ForConditionalGeneration(config, backend=_backend())


class _NemotronConfig:
    """Small attention/MLP-only Nemotron-H configuration."""

    def __init__(self) -> None:
        self.num_attention_heads = 4
        self.num_key_value_heads = 2
        self.head_dim = 16
        self.hidden_size = 64
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.intermediate_size = 128
        self.mlp_bias = False
        self.mlp_hidden_act = "relu2"
        self.layer_norm_epsilon = 1e-5
        self.num_hidden_layers = 2
        self.vocab_size = 128
        self.torch_dtype = "float32"
        self.initializer_range = 0.02
        self.rescale_prenorm_residual = True
        self.residual_in_fp32 = False
        self.layers_block_type = ["attention", "mlp"]
        self.n_routed_experts = None
        self.num_nextn_predict_layers = 0
        self.tie_word_embeddings = False
        self.output_hidden_states = True

    def to_dict(self) -> dict[str, object]:
        return vars(self)


def _build_nemotron_omni():
    from nemo_automodel._transformers.models.nemotron_omni.model import (
        NemotronOmniForConditionalGeneration,
        _ModelProxy,
    )
    from nemo_automodel._transformers.models.nemotron_v3.model import NemotronHForCausalLM

    llm_config = _NemotronConfig()
    language_model = NemotronHForCausalLM(llm_config, backend=_backend())

    # The dtype regression only needs Omni's real text-forward and lm_head
    # ownership contract; omit the large RADIO/Parakeet towers.
    model = object.__new__(NemotronOmniForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(llm_config=llm_config, tie_word_embeddings=False)
    model.language_model = language_model
    model.model = _ModelProxy(language_model)
    model.cp_mesh = None
    model.sound_encoder = None
    return model


def _run_case(rank: int, case: str, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy

        from nemo_automodel.components.moe.parallelizer import apply_fsdp

        torch.manual_seed(1234)
        model = (_build_gemma4() if case == "gemma4" else _build_nemotron_omni()).to(device)
        assert model.lm_head.weight.dtype == torch.float32

        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
            cast_forward_inputs=True,
        )
        apply_fsdp(
            model,
            fsdp_mesh=mesh,
            ep_enabled=False,
            ep_shard_enabled=False,
            mp_policy=policy,
            reshard_after_forward=True,
        )

        observed: dict[str, torch.dtype] = {}

        def record_head_input(module, args):
            observed["head_input"] = args[0].dtype
            observed["head_weight"] = module.weight.dtype

        if case == "gemma4":
            assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
            assert not isinstance(model.lm_head, FSDPModule)

            def record_hidden(_module, _args, output):
                observed["hidden"] = output.last_hidden_state.dtype

            model.model.language_model.register_forward_hook(record_hidden)
        else:
            assert model.lm_head is model.language_model.lm_head
            assert isinstance(model.lm_head, FSDPModule)

            def record_hidden(_module, _args, output):
                observed["hidden"] = output.dtype

            model.language_model.model.register_forward_hook(record_hidden)

        model.lm_head.register_forward_pre_hook(record_head_input)
        input_ids = torch.randint(0, 128, (2, 8), device=device)
        outputs = model(input_ids=input_ids, use_cache=False, logits_to_keep=2)
        outputs.logits.float().square().mean().backward()

        assert observed == {
            "hidden": torch.float32,
            "head_input": torch.bfloat16,
            "head_weight": torch.bfloat16,
        }
        assert torch.isfinite(outputs.logits).all()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("case", ["gemma4", "nemotron_omni"])
def test_nemo_rl_lm_head_dtype_boundary(case: str) -> None:
    """Exercise NeMo RL's FP32-output/BF16-compute policy on both model layouts."""
    world_size = 1
    mp.spawn(_run_case, args=(case, world_size, _free_port()), nprocs=world_size, join=True)
