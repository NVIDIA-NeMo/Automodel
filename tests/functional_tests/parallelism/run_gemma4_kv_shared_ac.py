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

"""Activation checkpointing must wrap whole blocks on KV-shared Gemma4 (E2B/E4B).

``apply_submodule_checkpointing`` deliberately leaves ``self_attn`` unwrapped for
KV-shared models. When the parallelizer routes an E-series model there instead of
wrapping whole decoder blocks, training still runs and the loss does not move, so
the only symptom is memory -- PR #3513 did exactly that and the nightly
Gemma4-E4B benchmark gained 16% peak memory before anyone noticed.

Three checks, the first of which that regression fails:

1. every language decoder block is checkpoint-wrapped, so attention is recomputed
   rather than kept alive for the backward;
2. the checkpointed logits match an un-checkpointed run;
3. so do the gradients, over exactly the same parameter set -- which is what the
   KV-sharing guard was worried about, a replayed block writing K/V twice.

Peak-memory magnitude stays with the nightly benchmark; a proxy this small cannot
measure it without a threshold that says more about the proxy than the code.

Run with::

    torchrun --standalone --nproc-per-node=2 \
        tests/functional_tests/parallelism/run_gemma4_kv_shared_ac.py
"""

from __future__ import annotations

import copy
import os

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy

from nemo_automodel.components.distributed.parallelizer import fsdp2_strategy_parallelize
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gemma4_moe.model import (
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
)

SEQUENCE_LENGTH = 128

# The input is text-only, so the vision tower and the vision->text embedder are
# exactly the parameters expected to come back without a gradient.
VISION_PARAMETER_PREFIXES = ("model.vision_tower.", "model.embed_vision.")


def _tiny_e4b_config() -> Gemma4Config:
    """Build an E4B-shaped config: per-layer embeddings, shared KV, eager attention.

    ``num_kv_shared_layers > 0`` is what puts the model on the KV-shared path, and
    eager attention matches the ``gemma4_4b`` recipe the nightly benchmark runs.
    """
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
        max_position_embeddings=SEQUENCE_LENGTH * 2,
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
            "max_position_embeddings": SEQUENCE_LENGTH * 2,
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
    config._attn_implementation = "eager"
    return config


def _backend() -> BackendConfig:
    """Return the plain PyTorch backend, matching the other Gemma4 functional tests."""
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def _fp32_policy() -> MixedPrecisionPolicy:
    """Keep both runs in fp32 so gradients compare without a dtype tolerance."""
    return MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )


def _as_local(tensor: torch.Tensor) -> torch.Tensor:
    """Return a plain tensor for a value that may be a DTensor gradient.

    Args:
        tensor: Gradient tensor of arbitrary shape, sharded or replicated.

    Returns:
        The same gradient with any DTensor sharding gathered away.
    """
    return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor


def main() -> None:
    """Compare whole-block activation checkpointing against no checkpointing on 2 ranks."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    device_mesh = init_device_mesh(
        "cuda",
        (1, dist.get_world_size(), 1),
        mesh_dim_names=("dp_replicate", "dp_shard_cp", "tp"),
    )

    torch.manual_seed(1234)
    model = Gemma4ForConditionalGeneration(_tiny_e4b_config(), backend=_backend()).to(
        device=device, dtype=torch.float32
    )
    for parameter in model.parameters():
        dist.broadcast(parameter.data, src=0)
    for buffer in model.buffers():
        dist.broadcast(buffer.data, src=0)
    torch.cuda.synchronize(device)
    reference = copy.deepcopy(model)

    checkpointed = fsdp2_strategy_parallelize(
        model,
        device_mesh,
        mp_policy=_fp32_policy(),
        activation_checkpointing=True,
        enable_fsdp2_prefetch=False,
    )
    plain = fsdp2_strategy_parallelize(
        reference,
        device_mesh,
        mp_policy=_fp32_policy(),
        activation_checkpointing=False,
        enable_fsdp2_prefetch=False,
    )

    # 1. Whole blocks are wrapped, so attention sits inside the recomputed region.
    # The sub-module fallback wraps mlp/norms and leaves the block itself bare.
    unwrapped = [
        index
        for index, block in enumerate(checkpointed.model.language_model.layers)
        if not isinstance(block, CheckpointWrapper)
    ]
    if unwrapped:
        raise AssertionError(
            f"KV-shared Gemma4 decoder blocks {unwrapped} are not checkpoint-wrapped; activation "
            "checkpointing fell back to sub-module wrapping, which excludes self_attn"
        )

    # 2. The forward is unchanged.
    input_ids = torch.randint(0, 100, (1, SEQUENCE_LENGTH), device=device)
    checkpointed_logits = checkpointed(input_ids=input_ids).logits
    plain_logits = plain(input_ids=input_ids).logits
    torch.testing.assert_close(
        _as_local(checkpointed_logits),
        _as_local(plain_logits),
        rtol=1e-5,
        atol=1e-5,
        msg=lambda message: f"logits diverged under activation checkpointing: {message}",
    )

    # 3. Recomputing a whole KV-shared block does not change the gradients.
    torch.manual_seed(2026)
    logits_grad = torch.randn_like(_as_local(plain_logits))
    _as_local(checkpointed_logits).backward(logits_grad)
    _as_local(plain_logits).backward(logits_grad)
    torch.cuda.synchronize(device)

    # Checkpoint wrappers insert a `_checkpoint_wrapped_module` path component.
    checkpointed_parameters = {
        name.replace("_checkpoint_wrapped_module.", ""): parameter
        for name, parameter in checkpointed.named_parameters()
    }
    plain_parameters = dict(plain.named_parameters())
    if set(checkpointed_parameters) != set(plain_parameters):
        only_checkpointed = sorted(set(checkpointed_parameters) - set(plain_parameters))
        only_plain = sorted(set(plain_parameters) - set(checkpointed_parameters))
        raise AssertionError(
            f"parameter names differ after wrapping: only-checkpointed={only_checkpointed} only-plain={only_plain}"
        )

    # Gradient presence must match exactly, so a dropped gradient cannot be
    # silently skipped instead of compared.
    checkpointed_with_grad = {name for name, p in checkpointed_parameters.items() if p.grad is not None}
    plain_with_grad = {name for name, p in plain_parameters.items() if p.grad is not None}
    if checkpointed_with_grad != plain_with_grad:
        raise AssertionError(
            "gradient presence differs: "
            f"missing-in-checkpointed={sorted(plain_with_grad - checkpointed_with_grad)} "
            f"missing-in-plain={sorted(checkpointed_with_grad - plain_with_grad)}"
        )

    # A text-only input reaches every parameter except the vision ones, so the
    # ungraded set must be exactly the vision parameters -- no more (a language
    # parameter silently dropping out) and no fewer (the check going vacuous).
    ungraded = set(plain_parameters) - plain_with_grad
    vision_names = {name for name in plain_parameters if name.startswith(VISION_PARAMETER_PREFIXES)}
    if ungraded != vision_names:
        raise AssertionError(
            "ungraded parameters are not exactly the vision path: "
            f"language-path-without-grad={sorted(ungraded - vision_names)} "
            f"vision-path-with-grad={sorted(vision_names - ungraded)}"
        )
    if not vision_names:
        raise AssertionError("the proxy has no vision parameters, so the omission check is vacuous")

    for name in sorted(checkpointed_with_grad):
        torch.testing.assert_close(
            _as_local(checkpointed_parameters[name].grad),
            _as_local(plain_parameters[name].grad),
            rtol=1e-5,
            atol=1e-5,
            msg=lambda message, key=name: f"{key} gradient: {message}",
        )

    if dist.get_rank() == 0:
        print(
            "gemma4 kv-shared activation checkpointing OK "
            f"({len(checkpointed_with_grad)} gradients compared, {len(vision_names)} vision-only omissions)"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
