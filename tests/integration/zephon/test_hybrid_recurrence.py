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

"""CUDA coverage for real Zephon batches and native GDN/KDA kernels."""

from contextlib import closing
from dataclasses import replace

import pytest
import torch

from nemo_automodel.components.models.common.packing import configure_packing_for_models
from nemo_automodel.shared.batch import move_batch_to_device
from tests.utils.zephon_text_fixtures import config as config
from tests.utils.zephon_text_fixtures import tokenizer as tokenizer

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GDN/KDA parity requires real CUDA kernels"),
    # Cold FLA compilation is substantially slower than a CPU unit test.
    pytest.mark.timeout(600),
]


def _hybrid_model(family: str) -> torch.nn.Module:
    """Reuse tiny model configs with one recurrent and one full-attention layer."""
    if family == "qwen3_5":
        from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForCausalLM
        from tests.unit_tests.models.qwen3_5.test_qwen3_5_dense_backbone import _backend, _tiny_config

        model_config = _tiny_config(
            layer_types=("linear_attention", "full_attention"),
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
        )
        return Qwen3_5ForCausalLM(model_config, backend=_backend())
    if family == "qwen3_5_moe":
        from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForCausalLM
        from tests.unit_tests.models.qwen3_5.test_qwen3_5_dense_backbone import _backend
        from tests.unit_tests.models.qwen3_5_moe.test_qwen3_5_moe_fp32_gdn import _text_config

        model_config = _text_config()
        # Router balancing is batch-wide and couples documents independently of attention.
        model_config.router_aux_loss_coef = 0.0
        model_config.num_hidden_layers = 2
        model_config.layer_types = ["linear_attention", "full_attention"]
        model_config.linear_key_head_dim = model_config.linear_value_head_dim = 32
        model_config.linear_num_key_heads = model_config.linear_num_value_heads = 2
        return Qwen3_5MoeForCausalLM(model_config, backend=_backend())
    if family == "kimi_linear":
        from nemo_automodel.components.models.kimi_linear.model import KimiLinear48BForCausalLM
        from tests.unit_tests.models.kimi_linear.test_model import _backend_config, _tiny_kimi_config

        model_config = _tiny_kimi_config(use_kda=True)
        model_config.linear_attn_config["head_dim"] = 32
        return KimiLinear48BForCausalLM(model_config, backend=_backend_config())
    from nemo_automodel.components.models.kimi_k3.model import KimiK3ForCausalLM
    from tests.unit_tests.models.kimi_k3.test_pipeline_parallel import _tiny_config, _torch_backend

    model_config = _tiny_config(num_hidden_layers=2, attn_res_block_size=None)
    model_config.linear_attn_config.update(kda_layers=[1], full_attn_layers=[2], head_dim=32)
    return KimiK3ForCausalLM(model_config, backend=_torch_backend())


def _assert_gradient_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare gradient norms while allowing bf16 reduction rounding near zero.

    Args:
        actual: Packed-execution gradient with arbitrary parameter or embedding shape.
        expected: Separate-document gradient with the same shape.
    """
    error = (actual.float() - expected.float()).norm()
    reference = expected.float().norm()
    # Packed and separate FLA kernels use different reduction layouts. A
    # normwise bound avoids relative-error spikes in near-zero components.
    assert error <= 2e-2 * reference + 2e-3, (error.item(), reference.item())


@pytest.mark.parametrize("family", ["qwen3_5", "qwen3_5_moe", "kimi_linear", "kimi_k3"])
@pytest.mark.parametrize("xdoc", [False, True])
def test_real_recurrence_matches_selected_document_layout(config, tokenizer, family, xdoc):
    """Compare logits and gradients against separate or joined document forwards."""
    pytest.importorskip("fla", reason="real GDN/KDA kernel validation requires the FLA extra")
    torch.manual_seed(13)
    model = _hybrid_model(family).cuda()
    model.initialize_weights(buffer_device=torch.device("cuda"), dtype=torch.bfloat16)
    model.train()
    contract = configure_packing_for_models([model])
    config = replace(
        config,
        batch_size=1,
        zephon=replace(
            config.zephon,
            packing=replace(config.zephon.packing, seq_len=10, pack_algorithm="first_fit"),
            training=replace(config.zephon.training, allow_cross_document_attention=xdoc),
        ),
    )
    with closing(config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1, packing_contract=contract)) as loader:
        batch = move_batch_to_device(next(loader), torch.device("cuda"))
    assert batch["_packed_seq_ids"].tolist() == [[1] * 8 + [0] * 2 if xdoc else [1] * 4 + [2] * 4 + [0] * 2]
    batch.pop("labels")
    tokens = batch.pop("input_ids")
    # Different documents can share token IDs, so inspect gradients with respect
    # to token embeddings, whose sequence positions distinguish those occurrences.
    embeddings = model.model.embed_tokens(tokens).detach().requires_grad_()
    packed = model(inputs_embeds=embeddings, **batch).logits[:, 4:8]
    assert torch.isfinite(packed).all()
    loss = packed.float().square().mean()
    loss.backward()
    packed_gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    embedding_gradients = embeddings.grad.clone()
    if xdoc:
        assert embedding_gradients[:, :4].abs().max() > 0
    else:
        assert embedding_gradients[:, :4].abs().max() == 0
    assert embedding_gradients[:, 8:].abs().max() == 0
    model.zero_grad(set_to_none=True)

    selected = slice(0, 8) if xdoc else slice(4, 8)
    reference_embeddings = embeddings[:, selected].detach().requires_grad_()
    reference = model(
        inputs_embeds=reference_embeddings,
        position_ids=batch["position_ids"][:, selected],
    ).logits
    if xdoc:
        reference = reference[:, 4:]
    reference_loss = reference.float().square().mean()
    reference_loss.backward()
    # bf16 projections and packed FLA reductions may use different tiling; these
    # tolerances match the existing CUDA packed-document parity checks.
    torch.testing.assert_close(packed, reference, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(loss, reference_loss, rtol=2e-2, atol=2e-3)
    _assert_gradient_close(embedding_gradients[:, selected], reference_embeddings.grad)
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            _assert_gradient_close(packed_gradients[name], parameter.grad)
