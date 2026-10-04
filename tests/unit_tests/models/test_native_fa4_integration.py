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

"""CPU regressions for model-owned FA4 packing and dispatch."""

from dataclasses import replace
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.attention.utils import preprocess_args_and_kwargs_for_attn
from nemo_automodel.components.datasets.utils import neat_packed_collater
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.packing import configure_packing, get_model_attn_implementation
from tests.unit_tests.attention.test_attention_utils import _reference_varlen_sdpa
from tests.unit_tests.models.step3p5.test_step3p5_model import MockStep3p5Config


def _packed_sample():
    return dict(
        input_ids=[1, 2, 3, 4], labels=[2, -100, 4, -100], attention_mask=[1, 1, 2, 2], position_ids=[0, 1, 0, 1]
    )


@pytest.mark.parametrize("model_family", ["llama", "qwen2", "qwen3", "laguna"])
@pytest.mark.parametrize("backend", ["sdpa", "te", "fa4"])
def test_hf_dispatch_is_model_owned(model_family, backend):
    from nemo_automodel.components.models.laguna.model import LagunaForCausalLM
    from nemo_automodel.components.models.llama.model import LlamaForCausalLM
    from nemo_automodel.components.models.qwen2.model import Qwen2ForCausalLM
    from nemo_automodel.components.models.qwen3.model import Qwen3ForCausalLM

    cls = {
        "llama": LlamaForCausalLM,
        "qwen2": Qwen2ForCausalLM,
        "qwen3": Qwen3ForCausalLM,
        "laguna": LagunaForCausalLM,
    }[model_family]
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.backend = BackendConfig(attn=backend)
    model.config = SimpleNamespace(_attn_implementation="flash_attention_4")
    assert get_model_attn_implementation(model) == "flash_attention_4"


def test_backend_dispatched_model_preserves_document_isolation():
    from nemo_automodel.components.models.qwen3_moe.model import Qwen3MoeForCausalLM

    model = Qwen3MoeForCausalLM.__new__(Qwen3MoeForCausalLM)
    torch.nn.Module.__init__(model)
    model.backend = BackendConfig(attn="sdpa")
    model.config = SimpleNamespace(_attn_implementation="flash_attention_4")
    contract = configure_packing(get_model_attn_implementation(model), model=model)
    batch = neat_packed_collater([_packed_sample()], packing=contract)
    q = k = torch.zeros(1, 4, 1, 1)
    v = torch.tensor([10.0, 10.0, 0.0, 0.0]).reshape(1, 4, 1, 1).requires_grad_()
    q_sdpa, k_sdpa, v_sdpa, kwargs = preprocess_args_and_kwargs_for_attn(
        q, k, v, batch["attention_mask"], model.backend.attn
    )
    output = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, **kwargs)
    torch.testing.assert_close(output.flatten(), torch.tensor([10.0, 10.0, 0.0, 0.0]))
    upstream = torch.tensor([[[[0.0], [0.0], [2.0], [-3.0]]]])
    output.backward(upstream)
    torch.testing.assert_close(v.grad.flatten()[:2], torch.zeros(2))


@pytest.mark.parametrize("layer_type,window_size", [("full_attention", None), ("sliding_attention", 1)])
def test_step3p5_model_derived_packing_matches_sdpa_forward_backward(layer_type, window_size):
    from nemo_automodel.components.models.step3p5.model import Step3p5ForCausalLM

    torch.manual_seed(7)
    backend = BackendConfig(
        attn="fa4",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        dispatcher="torch",
        experts="torch",
        enable_hf_state_dict_adapter=False,
    )
    config = MockStep3p5Config(
        torch_dtype="float32", num_hidden_layers=1, layer_types=[layer_type], sliding_window=window_size
    )
    fa4 = ModuleType("flash_attn.cute")
    fa4.flash_attn_func = None
    fa4.flash_attn_varlen_func = _reference_varlen_sdpa
    with patch.dict("sys.modules", {"flash_attn.cute": fa4}):
        model = Step3p5ForCausalLM(config, backend=backend)
    reference = Step3p5ForCausalLM(config, backend=replace(backend, attn="sdpa"))
    reference.load_state_dict(model.state_dict())
    outputs = []
    for current in (model, reference):
        contract = configure_packing(get_model_attn_implementation(current), model=current)
        batch = neat_packed_collater([_packed_sample()], packing=contract)
        batch.pop("labels")
        outputs.append(current(**batch).logits)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-5)
    upstream = torch.randn_like(outputs[0])
    for output in outputs:
        output.backward(upstream)
    for (name, param), (ref_name, ref_param) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == ref_name
        if param.grad is not None or ref_param.grad is not None:
            torch.testing.assert_close(param.grad, ref_param.grad, rtol=1e-4, atol=1e-5, msg=name)


@pytest.mark.parametrize("with_metadata", [False, True])
def test_qwen3_next_rejects_packed_fa4_before_recurrent_layers(with_metadata):
    from nemo_automodel.components.models.qwen3_next.model import Qwen3NextForCausalLM

    model = Qwen3NextForCausalLM.__new__(Qwen3NextForCausalLM)
    torch.nn.Module.__init__(model)
    model.backend = BackendConfig(attn="fa4")
    metadata = {"cu_seqlens": torch.tensor([0, 2, 4], dtype=torch.int32)} if with_metadata else {}
    with pytest.raises(ValueError, match="Qwen3Next does not support packed FA4"):
        model(torch.ones(2, 4, dtype=torch.long), attention_mask=torch.tensor([[1, 1, 2, 2]]).expand(2, -1), **metadata)


def test_deepseek_v32_rejects_fa4_before_sparse_layers():
    from nemo_automodel.components.models.deepseek_v32.config import DeepseekV32Config
    from nemo_automodel.components.models.deepseek_v32.model import DeepseekV32ForCausalLM

    assert DeepseekV32ForCausalLM._uses_native_fa4 is False
    with pytest.raises(ValueError, match="FA4 is unavailable for DeepSeek V3.2 sparse attention"):
        DeepseekV32ForCausalLM(DeepseekV32Config(), backend=BackendConfig(attn="fa4"))


@pytest.mark.parametrize("microbatch_size", [1, 2])
def test_qwen35_hybrid_packed_fa4_matches_sdpa(microbatch_size):
    """Exercise collation, entry hook, recurrent/full attention, logits and gradients."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

    from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForCausalLM
    from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet
    from tests.unit_tests.models.qwen3_5.test_qwen3_5_dense_backbone import _backend, _tiny_config

    def conv(x, weight, bias, activation, seq_idx):
        """Run separate document convolutions.

        Args:
            x: Activations of shape [1, channels, tokens].
            weight: Convolution weights of shape [channels, kernel].
            bias: Optional bias of shape [channels].
            activation: Activation name, silu.
            seq_idx: Document IDs of shape [1, tokens].

        Returns:
            Convolved activations of shape [1, channels, tokens].
        """
        cuts = [0] + (torch.nonzero(seq_idx[0, 1:] != seq_idx[0, :-1]).flatten() + 1).tolist() + [x.shape[-1]]
        return torch.cat(
            [
                F.silu(
                    F.conv1d(x[:, :, a:b], weight[:, None], bias, padding=weight.shape[-1] - 1, groups=x.shape[1])[
                        :, :, : b - a
                    ]
                )
                for a, b in zip(cuts, cuts[1:])
            ],
            dim=-1,
        )

    def recurrence(q, k, v, *, g, beta, cu_seqlens, cu_seqlens_cpu=None, **kwargs):
        """Evaluate HF's recurrent reference independently for each document.

        Args:
            q: Queries of shape [1, tokens, heads, key_dim].
            k: Keys of shape [1, tokens, heads, key_dim].
            v: Values of shape [1, tokens, heads, value_dim].
            g: Log-decay gates of shape [1, tokens, heads].
            beta: Update gates of shape [1, tokens, heads].
            cu_seqlens: Document boundaries of shape [documents + 1].
            cu_seqlens_cpu: Optional CPU mirror with the same shape.
            **kwargs: Scalar reference-kernel options and optional initial state.

        Returns:
            Outputs of shape [1, tokens, heads, value_dim] and None for state.
        """
        cuts = cu_seqlens.tolist()
        return torch.cat(
            [
                torch_recurrent_gated_delta_rule(q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b], **kwargs)[0]
                for a, b in zip(cuts, cuts[1:])
            ],
            dim=1,
        ), None

    torch.manual_seed(31)
    config = _tiny_config(
        layer_types=("linear_attention", "full_attention"),
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=4,
    )
    fa4 = ModuleType("flash_attn.cute")
    fa4.flash_attn_func = None
    fa4.flash_attn_varlen_func = _reference_varlen_sdpa
    with patch.dict("sys.modules", {"flash_attn.cute": fa4}):
        model = Qwen3_5ForCausalLM(config, backend=replace(_backend(), attn="fa4"))
    reference = Qwen3_5ForCausalLM(config, backend=_backend())
    for current in (model, reference):
        for module in current.modules():
            if isinstance(module, CPAwareGatedDeltaNet):
                module.causal_conv1d_fn = conv
                module.chunk_gated_delta_rule = recurrence
                with torch.no_grad():
                    module.A_log.zero_()
                    module.dt_bias.zero_()
    reference.load_state_dict(model.state_dict())
    samples = [
        dict(
            input_ids=[1, 2, 3, 4, 0, 0],
            labels=[2, -100, 4, -100, -100, -100],
            attention_mask=[1, 1, 2, 2, 0, 0],
            position_ids=[0, 1, 0, 1, 0, 0],
        ),
        dict(
            input_ids=[5, 6, 7, 8, 9, 0],
            labels=[6, 7, -100, 9, -100, -100],
            attention_mask=[1, 1, 1, 2, 2, 0],
            position_ids=[0, 1, 2, 0, 1, 0],
        ),
    ]
    from torch.distributed.pipelining.microbatch import split_args_kwargs_into_chunks

    outputs = []
    for current in (model, reference):
        contract = configure_packing(get_model_attn_implementation(current), model=current)
        batch = neat_packed_collater(samples, packing=contract)
        batch.pop("labels")
        _, chunks = split_args_kwargs_into_chunks((), batch, chunks=2 // microbatch_size)
        outputs.append(torch.cat([current(**chunk).logits for chunk in chunks]))
    valid = torch.tensor([sample["attention_mask"] for sample in samples]) > 0
    torch.testing.assert_close(outputs[0][valid], outputs[1][valid], rtol=1e-4, atol=1e-5)
    upstream = torch.randn_like(outputs[0]) * valid.unsqueeze(-1)
    for output in outputs:
        output.backward(upstream)
    for (name, param), (_, ref_param) in zip(model.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(param.grad, ref_param.grad, rtol=1e-4, atol=1e-5, msg=name)


def test_default_te_qwen3_neat_packing_uses_its_hf_flash_dispatch():
    from transformers import Qwen3Config

    from nemo_automodel.components.models.qwen3.model import Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
    )
    config._attn_implementation = "sdpa"
    # TE is used only by the THD branch; constructing it must not determine NEAT's mask.
    with patch(
        "nemo_automodel.components.models.qwen3.model.initialize_attn_module_and_func",
        return_value=(torch.nn.Identity(), None),
    ):
        model = Qwen3ForCausalLM(config, backend=BackendConfig(attn="te", rope_fusion=False))
    model.config._attn_implementation = "flash_attention_2"
    contract = configure_packing(get_model_attn_implementation(model), model=model)
    assert contract.packed_mask_type == "flash_varlen"
    batch = neat_packed_collater([_packed_sample()], packing=contract)
    assert "attention_mask" not in batch
    assert batch["cu_seq_lens_q"].tolist() == [0, 2, 4]
    assert batch["_packed_seq_ids"].tolist() == [[1, 1, 2, 2]]


@pytest.mark.parametrize("vlm", [False, True])
def test_hf_and_native_packing_contracts_keep_distinct_offset_layouts(vlm):
    from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater
    from nemo_automodel.components.models.common.packing import get_packing_capabilities

    samples = [
        dict(
            input_ids=[1, 2, 3, 0], labels=[2, -100, -100, -100], position_ids=[0, 1, 0, 0], attention_mask=[1, 1, 2, 0]
        ),
        dict(
            input_ids=[4, 5, 0, 0], labels=[5, -100, -100, -100], position_ids=[0, 1, 0, 0], attention_mask=[1, 1, 0, 0]
        ),
    ]
    collate = neat_packed_vlm_collater if vlm else neat_packed_collater
    native = torch.nn.Module()
    native._uses_native_fa4 = True
    native.backend = BackendConfig(attn="fa4")
    hf_contract = configure_packing("flash_attention_2")
    native_contract = get_packing_capabilities(get_model_attn_implementation(native), model=native)
    hf_batch = collate(samples, packing=hf_contract)
    native_batch = collate(samples, packing=native_contract)

    assert "attention_mask" not in hf_batch
    assert "packed_token_indices" not in hf_batch
    assert hf_batch["cu_seq_lens_q"].tolist() == [0, 2, 3, 4, 6, 8]
    assert hf_batch["max_length_q"] == hf_batch["max_length_k"] == 2
    assert native_batch["attention_mask"].tolist() == [[1, 1, 2, 0], [1, 1, 0, 0]]
    assert native_batch["packed_token_indices"].tolist() == [[0, 1, 2, -1], [0, 1, -1, -1]]
    assert native_batch["cu_seqlens"].tolist() == [[0, 2, 3], [0, 2, -1]]
    assert "cu_seq_lens_q" not in native_batch
    torch.testing.assert_close(hf_batch["_packed_seq_ids"], native_batch["_packed_seq_ids"])
