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

"""Unit tests for model-owned tensor-parallel sidecars."""

from torch import nn
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, SequenceParallel

from nemo_automodel.components.models.baichuan.parallelizer import get_tp_plan as get_baichuan_tp_plan
from nemo_automodel.components.models.falcon_h1.parallelizer import get_tp_plan as get_falcon_h1_tp_plan
from nemo_automodel.components.models.gemma3.parallelizer import get_tp_plan as get_gemma3_tp_plan
from nemo_automodel.components.models.llama.parallelizer import get_tp_plan as get_llama_tp_plan
from nemo_automodel.components.models.mistral3_vlm.parallelizer import get_tp_plan as get_mistral3_vlm_tp_plan
from nemo_automodel.components.models.nemotron_labs_diffusion.parallelizer import get_tp_plan as get_diffusion_tp_plan
from nemo_automodel.components.models.nemotron_nas.parallelizer import get_tp_plan as get_nemotron_nas_tp_plan
from nemo_automodel.components.models.phi.parallelizer import get_tp_plan as get_phi_tp_plan
from nemo_automodel.components.models.phi3.parallelizer import get_tp_plan as get_phi3_tp_plan
from nemo_automodel.components.models.qwen2.parallelizer import get_tp_plan as get_qwen2_tp_plan
from nemo_automodel.components.models.qwen3.parallelizer import get_tp_plan as get_qwen3_tp_plan


class _ModelWithLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()


class _Qwen3Classification(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.score = nn.Linear(2, 2)


def test_llama_sidecar_owns_fused_decoder_paths() -> None:
    plan = get_llama_tp_plan(None, sequence_parallel=True)

    assert "model.layers.*.self_attn.qkv_proj" in plan
    assert "model.layers.*.mlp.gate_up_proj" in plan
    assert isinstance(plan["model.layers.*.input_layernorm"], SequenceParallel)
    assert isinstance(plan["model.layers.*.self_attn.o_proj"], RowwiseParallel)


def test_qwen_sidecars_share_decoder_primitive_but_keep_model_ownership() -> None:
    qwen3_causal = nn.Module()
    qwen3_causal.lm_head = nn.Linear(2, 2)
    for get_plan, model in ((get_qwen2_tp_plan, nn.Module()), (get_qwen3_tp_plan, qwen3_causal)):
        plan = get_plan(model, sequence_parallel=True)
        assert isinstance(plan["model.layers.*.self_attn.q_proj"], ColwiseParallel)
        assert isinstance(plan["model.layers.*.mlp.down_proj"], RowwiseParallel)
        assert "model.layers.*.self_attn.q_norm" not in plan


def test_qwen3_classification_sidecar_replaces_lm_head_with_score() -> None:
    plan = get_qwen3_tp_plan(_Qwen3Classification())

    assert "lm_head" not in plan
    assert isinstance(plan["score"], ColwiseParallel)


def test_gemma3_sidecar_selects_its_local_text_or_vlm_paths() -> None:
    text_plan = get_gemma3_tp_plan(nn.Module(), sequence_parallel=True)
    vlm_plan = get_gemma3_tp_plan(_ModelWithLanguageModel(), sequence_parallel=True)

    assert "model.rotary_emb" in text_plan
    assert "model.language_model.rotary_emb" in vlm_plan
    assert "model.language_model.layers.*.mlp.down_proj" in vlm_plan


def test_nemotron_nas_sidecar_uses_separate_qkv_projections() -> None:
    plan = get_nemotron_nas_tp_plan(None, sequence_parallel=True)

    assert "model.layers.*.self_attn.q_proj" in plan
    assert "model.layers.*.self_attn.qkv_proj" not in plan
    assert isinstance(plan["model.norm"], SequenceParallel)


def test_mistral_vlm_sidecar_scopes_only_the_text_decoder() -> None:
    plan = get_mistral3_vlm_tp_plan(None)

    assert all(key == "lm_head" or key.startswith("model.language_model.") for key in plan)
    assert isinstance(plan["model.language_model.layers.*.mlp.gate_proj"], ColwiseParallel)


def test_special_model_sidecars_preserve_their_model_specific_paths() -> None:
    baichuan = get_baichuan_tp_plan(None)
    falcon = get_falcon_h1_tp_plan(None)
    diffusion = get_diffusion_tp_plan(None, sequence_parallel=True)
    phi = get_phi_tp_plan(None, sequence_parallel=True)
    phi3 = get_phi3_tp_plan(None)

    assert set(baichuan) == {
        "model.layers.*.mlp.gate_proj",
        "model.layers.*.mlp.up_proj",
        "model.layers.*.mlp.down_proj",
    }
    assert any("feed_forward" in key for key in falcon)
    assert not any(".mamba" in key for key in falcon)
    assert "encoder.layers.*.self_attn.q_proj" in diffusion
    assert "diffusion_head" in diffusion
    assert "model.layers.*.self_attn.dense" in phi
    assert "model.layers.*.self_attn.qkv_proj" in phi3
