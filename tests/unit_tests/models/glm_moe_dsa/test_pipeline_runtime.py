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

"""CPU/meta regression for runtime capacity derived from real GLM pipeline layouts."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm_moe_dsa.model import GlmMoeDsaForCausalLM


@pytest.mark.parametrize("attn_backend", ["tilelang", "cudnn", "sdpa"])
@pytest.mark.parametrize("is_first,has_lm_head", [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize("indexshare", [False, True])
def test_pipeline_runtime_uses_glm_stage_token_capacity(attn_backend, is_first, has_lm_head, indexshare):
    config = GlmMoeDsaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=8,
        mlp_layer_types=["dense"],
        indexer_types=["shared" if indexshare else "full"],
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )
    backend = BackendConfig(
        attn=attn_backend,
        linear="torch",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
    )
    with torch.device("meta"):
        model = GlmMoeDsaForCausalLM(config, backend=backend)
    if not has_lm_head:
        model.lm_head = None
    stage = SimpleNamespace(is_first=is_first, submod=model, inputs_meta=None, _configure_outputs_meta=Mock())
    # Packed THD folds batch into T. Dense BSHD must still multiply by batch.
    microbatch_size = 2 if attn_backend == "sdpa" else 1
    ap = AutoPipeline(
        world_mesh={"pp": object()},
        pp_schedule="1f1b",
        pp_microbatch_size=microbatch_size,
        pp_batch_size=2 * microbatch_size,
        device=torch.device("cpu"),
    )
    ap._model_config = config
    ap._info.stages = [stage]
    ap._info.schedule = SimpleNamespace()
    initializer = Mock()
    ap._runtime_initializers = [initializer]

    # Cover initial setup, cached metadata, growth, and shrink via the real hook.
    for seq_len in [4096, 4096, 8192, 2048]:
        ap.update_seq_len(seq_len)
        initializer.prepare.assert_called_with(num_tokens=microbatch_size * seq_len, device=torch.device("cpu"))
        outputs = stage._configure_outputs_meta.call_args.args[0]
        features = config.vocab_size if has_lm_head else config.hidden_size
        expected_shape = (
            (microbatch_size, seq_len, features) if attn_backend == "sdpa" or has_lm_head else (seq_len, features)
        )
        assert outputs[0].shape == expected_shape
        assert len(outputs) == (2 if indexshare and not has_lm_head else 1)
