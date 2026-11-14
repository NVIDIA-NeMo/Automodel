# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""CPU unit tests for the Qwen3-MoE compile_attn (fullgraph attention compile) wiring.

These mock the submodule initializers so Qwen3MoeAttention constructs without a GPU or
TransformerEngine. They cover the compile_attn gating in __init__ (default off, and the
warn/no-op path when attn != 'sdpa'); the actual torch.compile call (attn='sdpa' + GPU)
is pragma-excluded as it only runs on the GPU benchmark.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_moe.layers import Qwen3MoeAttention


def _config():
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

    cfg = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        moe_intermediate_size=64,
        num_experts=0,
        num_experts_per_tok=1,
    )
    cfg.head_dim = 16
    return cfg


@patch("nemo_automodel.components.models.qwen3_moe.layers.initialize_attn_module_and_func")
@patch("nemo_automodel.components.models.qwen3_moe.layers.initialize_rms_norm_module")
@patch("nemo_automodel.components.models.qwen3_moe.layers.initialize_linear_module")
class TestQwen3MoeCompileAttn:
    """compile_attn gating in Qwen3MoeAttention.__init__."""

    def _build(self, mock_lin, mock_rms, mock_attn, backend):
        mock_lin.return_value = Mock()
        mock_rms.return_value = Mock()
        mock_attn.return_value = (Mock(), Mock())
        return Qwen3MoeAttention(_config(), backend)

    def test_compile_attn_off_leaves_uncompiled(self, mock_lin, mock_rms, mock_attn):
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False, compile_attn=False)
        attn = self._build(mock_lin, mock_rms, mock_attn, backend)
        assert attn._compiled_forward is None

    def test_compile_attn_on_non_sdpa_warns_and_noops(self, mock_lin, mock_rms, mock_attn):
        # compile_attn=True but attn != 'sdpa' → warn and leave the forward uncompiled.
        backend = BackendConfig(attn="te", linear="torch", rms_norm="torch", rope_fusion=False, compile_attn=True)
        attn = self._build(mock_lin, mock_rms, mock_attn, backend)
        assert attn._compiled_forward is None

    def test_fa4_compiles_only_fixed_shape_projections(self, mock_lin: Mock, mock_rms: Mock, mock_attn: Mock) -> None:
        backend = BackendConfig(attn="fa4", linear="torch", rms_norm="torch", rope_fusion=False, compile_attn=True)
        with patch("torch.compile", side_effect=lambda fn, **kwargs: fn) as compile_fn:
            attn = self._build(mock_lin, mock_rms, mock_attn, backend)
        assert attn._compiled_forward is None
        assert [call.args[0].__name__ for call in compile_fn.call_args_list] == ["_project_qkv", "_project_output"]
        assert all(call.kwargs == {"fullgraph": True, "dynamic": False} for call in compile_fn.call_args_list)

    @pytest.mark.parametrize("option", [{"linear": "te"}, {"rms_norm": "te"}])
    def test_fa4_rejects_uncompilable_projection_settings(
        self, mock_lin: Mock, mock_rms: Mock, mock_attn: Mock, option: dict[str, str]
    ) -> None:
        settings = {"attn": "fa4", "linear": "torch", "rms_norm": "torch", "rope_fusion": False, "compile_attn": True}
        settings.update(option)
        with pytest.raises(ValueError, match="compiled FA4 requires"):
            self._build(mock_lin, mock_rms, mock_attn, BackendConfig(**settings))

    def test_sdpa_keeps_full_attention_compile(self, mock_lin: Mock, mock_rms: Mock, mock_attn: Mock) -> None:
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False, compile_attn=True)
        with patch("torch.compile", side_effect=lambda fn, **kwargs: fn) as compile_fn:
            attn = self._build(mock_lin, mock_rms, mock_attn, backend)
        assert compile_fn.call_count == 1
        assert compile_fn.call_args.args[0].__name__ == "_forward_impl"
        assert attn._compiled_qkv is None


@pytest.mark.parametrize("thd", [False, True])
def test_te_flattened_attention_output_preserves_token_axes(thd: bool) -> None:
    """TE's already flattened head axis must retain batch/sequence or token axes."""

    def te_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Return TE layout from Q [batch, sequence, heads, dim] or [tokens, heads, dim].

        Args:
            q: Query tensor with batched or THD token axes followed by [heads, dim].
            k: Key tensor with the same token axes and [kv_heads, dim].
            v: Value tensor with the same token axes and [kv_heads, dim].
            **kwargs: Unused TE metadata.

        Returns:
            Tensor preserving token axes with a final flattened [heads * dim] axis.
        """
        return q.flatten(-2)

    config = _config()
    config.dtype = torch.float32
    backend = BackendConfig(attn="te", linear="torch", rms_norm="torch", rope_fusion=False)
    with patch(
        "nemo_automodel.components.models.qwen3_moe.layers.initialize_attn_module_and_func",
        return_value=(None, te_attention),
    ):
        module = Qwen3MoeAttention(config, backend)
    shape = (7, 64) if thd else (2, 7, 64)
    inputs = torch.randn(shape, requires_grad=True)
    frequencies = torch.cat([torch.ones(*shape[:-1], 8), torch.zeros(*shape[:-1], 8)], dim=-1)
    kwargs = {"cu_seqlens": torch.tensor([0, 3, 7], dtype=torch.int32), "max_seqlen": 4} if thd else {}
    output = module(inputs, freqs_cis=frequencies, **kwargs)
    assert output.shape == inputs.shape
    output.square().sum().backward()
    assert inputs.grad.shape == inputs.shape


def test_native_fa4_model_declares_neat_metadata_contract() -> None:
    """The production model opts into document IDs and model-entry offset flattening."""
    from nemo_automodel.components.datasets.utils import neat_packed_collater
    from nemo_automodel.components.models.common.packing import configure_packing
    from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding
    from nemo_automodel.components.models.qwen3_moe.model import Qwen3MoeForCausalLM

    config = _config()
    config.num_hidden_layers = 0
    config.num_experts = 2
    config.dtype = torch.float32
    backend = BackendConfig(
        attn="fa4",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    with (
        patch("torch.cuda.current_device", return_value=0),
        patch(
            "nemo_automodel.components.models.qwen3_moe.model.RotaryEmbedding",
            side_effect=lambda **kwargs: RotaryEmbedding(**{**kwargs, "device": torch.device("cpu")}),
        ),
    ):
        model = Qwen3MoeForCausalLM(config, backend=backend)
    capabilities = configure_packing("fa4", model=model)
    assert capabilities.uses_native_fa4
    assert capabilities.packed_mask_type == "document_ids"
    assert capabilities.return_packed_sequence_metadata
    batch = neat_packed_collater(
        [
            {
                "input_ids": [1, 2, 3, 0],
                "labels": [1, 2, 3, -100],
                "attention_mask": [1, 1, 2, 0],
                "position_ids": [0, 1, 0, 0],
            }
        ],
        packing=capabilities,
    )
    assert batch["attention_mask"].shape == (1, 4)
    assert batch["cu_seqlens"].ndim == 2
    seen = {}

    def observe_metadata(module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        """Observe the metadata after the production model-entry normalizer."""
        seen.update(kwargs)

    handle = model.register_forward_pre_hook(observe_metadata, with_kwargs=True)
    model(**{key: value for key, value in batch.items() if key not in {"labels", "loss_mask"}})
    handle.remove()
    assert seen["cu_seqlens"].tolist() == [0, 2, 3]
    assert seen["packed_token_indices"].tolist() == [0, 1, 2]
