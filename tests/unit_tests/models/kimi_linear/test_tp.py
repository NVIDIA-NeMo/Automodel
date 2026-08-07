# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.kimi_linear.model import KimiLinearForCausalLM
from nemo_automodel.components.models.kimi_linear.tp import parallelize_kimi_linear
from tests.unit_tests.models.kimi_linear.test_model import _tiny_kimi_config


def _model(use_kda: bool = False):
    """Build a tiny model; the KDA variant needs the optional FLA kernels."""
    if use_kda:
        pytest.importorskip("fla")
    return KimiLinearForCausalLM(_tiny_kimi_config(use_kda=use_kda), backend=BackendConfig(attn="eager"))


def test_plan_shards_full_attention_layers():
    plan = parallelize_kimi_linear(_model())

    assert isinstance(plan["model.layers.1.self_attn.q_proj"], ColwiseParallel)
    assert isinstance(plan["model.layers.1.self_attn.kv_b_proj"], ColwiseParallel)
    assert isinstance(plan["model.layers.1.self_attn.o_proj"], RowwiseParallel)
    # The head-shared compressed latent is not sharded.
    assert "model.layers.1.self_attn.kv_a_proj_with_mqa" not in plan


def test_plan_leaves_kda_layers_replicated():
    # kda_layers=[1] is 1-based, so layer 0 is the KDA layer and layer 1 is MLA.
    plan = parallelize_kimi_linear(_model(use_kda=True))

    assert not any(key.startswith("model.layers.0.self_attn") for key in plan)
    assert "model.layers.1.self_attn.q_proj" in plan


def test_plan_leaves_routed_experts_to_expert_parallelism():
    plan = parallelize_kimi_linear(_model())

    assert not any("experts" in key for key in plan)
    # first_k_dense_replace=1, so layer 0 keeps a dense MLP that is sharded, while
    # layer 1's MoE (also named ``mlp``) gets no projection styles at all.
    assert isinstance(plan["model.layers.0.mlp.down_proj"], RowwiseParallel)
    assert "model.layers.1.mlp.down_proj" not in plan


def test_sequence_parallel_shards_block_boundaries():
    plan = parallelize_kimi_linear(_model(use_kda=True), sequence_parallel=True)

    assert isinstance(plan["model.norm"], SequenceParallel)
    assert isinstance(plan["model.layers.0.input_layernorm"], SequenceParallel)
    assert isinstance(plan["model.layers.0.post_attention_layernorm"], SequenceParallel)
    # A replicated KDA layer and the EP-owned MoE both emit full-sequence outputs
    # that must be scattered back onto the sequence-parallel residual stream.
    assert isinstance(plan["model.layers.0.self_attn"], PrepareModuleOutput)
    assert isinstance(plan["model.layers.1.mlp"], PrepareModuleOutput)


def test_sequence_parallel_is_absent_without_the_flag():
    plan = parallelize_kimi_linear(_model())

    assert "model.norm" not in plan
    assert not any(isinstance(style, (SequenceParallel, PrepareModuleOutput)) for style in plan.values())


def test_model_declares_tensor_and_sequence_parallel_support():
    assert KimiLinearForCausalLM.ModelCapabilities().supports_tp
    assert KimiLinearForCausalLM._supports_sequence_parallel is True
