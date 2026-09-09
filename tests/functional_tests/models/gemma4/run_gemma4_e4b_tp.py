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

"""Two-rank tensor-parallel smoke test for dense Gemma4 E-series models.

Run with::

    torchrun --standalone --nproc-per-node=2 \
        tests/functional_tests/models/gemma4/run_gemma4_e4b_tp.py
"""

import copy
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from nemo_automodel.components.distributed.parallelizer import fsdp2_strategy_parallelize
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gemma4_moe.loss import Gemma4TensorParallelFusedLinearCrossEntropy
from nemo_automodel.components.models.gemma4_moe.model import (
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
)


def _tiny_e4b_config() -> Gemma4Config:
    """Build an E4B-shaped config with per-layer embeddings and shared KV."""
    text_config = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=16,
        num_hidden_layers=4,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        enable_moe_block=False,
        layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=32,
        hidden_activation="gelu_pytorch_tanh",
        dtype="float32",
        num_kv_shared_layers=2,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=128,
        use_double_wide_mlp=False,
        pad_token_id=0,
        output_hidden_states=True,
    )
    config = Gemma4Config(
        text_config=text_config,
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "position_embedding_size": 16,
            "patch_size": 2,
            "pooling_kernel_size": 1,
            "dtype": "float32",
        },
        audio_config=None,
        image_token_id=127,
        tie_word_embeddings=True,
        dtype="float32",
    )
    return config


def _backend() -> BackendConfig:
    """Return the plain PyTorch backend used by the TP parity test."""
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def main() -> None:
    """Run the real two-rank Gemma4 E-series TP forward."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    device_mesh = init_device_mesh(
        "cuda",
        (1, 1, dist.get_world_size()),
        mesh_dim_names=("dp_replicate", "dp_shard_cp", "tp"),
    )

    torch.manual_seed(1234)
    model = (
        Gemma4ForConditionalGeneration(_tiny_e4b_config(), backend=_backend())
        .to(device=device, dtype=torch.float32)
        .eval()
    )
    for parameter in model.parameters():
        dist.broadcast(parameter.data, src=0)
    for buffer in model.buffers():
        dist.broadcast(buffer.data, src=0)
    torch.cuda.synchronize(device)

    reference = copy.deepcopy(model)
    input_ids = torch.tensor([[1, 7, 11, 19, 23, 29, 31, 37]], device=device)
    reference_output = reference(input_ids=input_ids)
    reference_logits = reference_output.logits

    model = fsdp2_strategy_parallelize(
        model,
        device_mesh,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
        ),
        activation_checkpointing=False,
        enable_fsdp2_prefetch=False,
    )
    torch.cuda.synchronize(device)

    reference_parameters = dict(reference.named_parameters())
    tp_parameters = dict(model.named_parameters())
    for name, parameter in tp_parameters.items():
        if name not in reference_parameters:
            continue
        full_parameter = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
        trigger_wait = getattr(full_parameter, "trigger_wait", None)
        if trigger_wait is not None:
            full_parameter = trigger_wait()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(full_parameter, reference_parameters[name], msg=lambda msg: f"{name}: {msg}")

    tp_output = model(input_ids=input_ids)
    tp_logits = tp_output.logits
    for layer_index, (tp_hidden, reference_hidden) in enumerate(
        zip(tp_output.hidden_states, reference_output.hidden_states)
    ):
        torch.testing.assert_close(
            tp_hidden,
            reference_hidden,
            rtol=1e-5,
            atol=1e-5,
            msg=lambda msg, index=layer_index: f"hidden state {index}: {msg}",
        )
    if not isinstance(tp_logits, DTensor):
        raise AssertionError("Expected Gemma4 E-series TP logits to remain vocabulary-sharded")
    gathered_tp_logits = tp_logits.detach().full_tensor()
    trigger_wait = getattr(gathered_tp_logits, "trigger_wait", None)
    if trigger_wait is not None:
        gathered_tp_logits = trigger_wait()
    torch.cuda.synchronize(device)
    if not torch.isfinite(gathered_tp_logits).all():
        raise AssertionError("Gemma4 E-series TP forward produced non-finite logits")
    torch.testing.assert_close(gathered_tp_logits, reference_logits, rtol=1e-5, atol=1e-5)

    torch.manual_seed(2026)
    logits_grad = torch.randn_like(reference_logits)
    reference_logits.backward(logits_grad)
    gathered_tp_logits_for_backward = tp_logits.full_tensor()
    trigger_wait = getattr(gathered_tp_logits_for_backward, "trigger_wait", None)
    if trigger_wait is not None:
        gathered_tp_logits_for_backward = trigger_wait()
    torch.cuda.synchronize(device)
    gathered_tp_logits_for_backward.backward(logits_grad)

    parameters_to_compare = (
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.o_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.0.per_layer_input_gate.weight",
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "model.language_model.embed_tokens_per_layer.weight",
    )
    for name in parameters_to_compare:
        reference_grad = reference_parameters[name].grad
        tp_grad = tp_parameters[name].grad
        if reference_grad is None or tp_grad is None:
            raise AssertionError(f"Expected a gradient for {name}")
        if isinstance(tp_grad, DTensor):
            tp_grad = tp_grad.full_tensor()
        relative_l2_error = (tp_grad - reference_grad).float().norm() / reference_grad.float().norm().clamp_min(1e-12)
        if relative_l2_error > 1e-3:
            raise AssertionError(f"{name} gradient relative L2 error {relative_l2_error.item():.6g} exceeds 1e-3")

    for name in (
        "model.language_model.embed_tokens.weight",
        "model.language_model.embed_tokens_per_layer.weight",
    ):
        weight = tp_parameters[name]
        if not isinstance(weight, DTensor) or weight.to_local().shape[0] * dist.get_world_size() != weight.shape[0]:
            raise AssertionError(f"Expected {name} to be vocabulary-sharded across TP ranks")

    # The fused-CE adapter must gather a full vocabulary for Cut CE while
    # slicing, rather than summing, the duplicate TP gradients in backward.
    torch.manual_seed(31415)
    full_loss_weight = torch.randn(32, 16, device=device, requires_grad=True)
    tp_loss_weight = torch.nn.Parameter(
        distribute_tensor(full_loss_weight.detach().clone(), device_mesh["tp"], (Shard(0),))
    )
    loss_hidden = torch.randn(2, 5, 16, device=device, requires_grad=True)
    loss_labels = torch.randint(0, 32, (2, 5), device=device)

    materialized_tp_weight = Gemma4TensorParallelFusedLinearCrossEntropy.materialize_lm_weight(tp_loss_weight)
    F.cross_entropy(F.linear(loss_hidden, materialized_tp_weight).flatten(0, 1), loss_labels.flatten()).backward()
    F.cross_entropy(F.linear(loss_hidden.detach(), full_loss_weight).flatten(0, 1), loss_labels.flatten()).backward()

    gathered_tp_loss_grad = tp_loss_weight.grad.full_tensor()
    trigger_wait = getattr(gathered_tp_loss_grad, "trigger_wait", None)
    if trigger_wait is not None:
        gathered_tp_loss_grad = trigger_wait()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(gathered_tp_loss_grad, full_loss_weight.grad, rtol=1e-5, atol=1e-5)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
