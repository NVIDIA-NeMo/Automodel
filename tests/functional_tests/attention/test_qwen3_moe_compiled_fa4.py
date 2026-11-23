# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Production Qwen packed-FA4 parity with fixed-shape compiled projections."""

from unittest.mock import patch

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from nemo_automodel.components.datasets.packing import build_packed_sequence_metadata
from nemo_automodel.components.distributed.activation_checkpointing import apply_full_layer_checkpointing_to_layers
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.packing import configure_packing
from nemo_automodel.components.models.qwen3_moe.model import Qwen3MoeForCausalLM


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and FlashAttention-4")
@pytest.mark.parametrize("mask_form", ["document_ids", "padding_mask", "packed_offsets"])
@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_compiled_qwen_fa4_packed_forward_backward(mask_form: str, activation_checkpointing: bool) -> None:
    """Real compiled projections preserve document isolation and parameter gradients."""
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FlashAttention-4 requires Hopper or newer hardware")
    pytest.importorskip("flash_attn.cute")
    torch.manual_seed(1234)
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=384,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=2,
        num_experts_per_tok=1,
        mlp_only_layers=[0],
        max_position_embeddings=128,
        torch_dtype=torch.bfloat16,
    )
    backend_options = dict(
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    counter = CompileCounterWithBackend("inductor")
    compile_function = torch.compile
    with patch(
        "nemo_automodel.components.models.qwen3_moe.layers.torch.compile",
        side_effect=lambda function, **options: compile_function(function, backend=counter, **options),
    ):
        candidate = Qwen3MoeForCausalLM(
            config, backend=BackendConfig(attn="fa4", compile_attn=True, **backend_options)
        ).cuda()
    candidate.initialize_weights(dtype=torch.bfloat16)
    capabilities = configure_packing("fa4", model=candidate)
    assert capabilities.packed_mask_type == "document_ids"
    reference = Qwen3MoeForCausalLM(config, backend=BackendConfig(attn="sdpa", **backend_options)).cuda().float()
    reference.load_state_dict(candidate.state_dict())
    candidate_attention = candidate.model.layers["0"].self_attn
    if activation_checkpointing:
        apply_full_layer_checkpointing_to_layers(candidate, [candidate.model.layers["0"]])
    compiled_frames = None

    for layout in (0, 1):
        candidate.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        # Same padded shape, different document counts, lengths and valid-token counts.
        documents = torch.tensor(
            (
                [[1] * 32 + [2] * 48 + [0] * 16, [1] * 24 + [2] * 24 + [3] * 48]
                if layout == 0
                else [[1] * 16 + [2] * 16 + [3] * 48 + [0] * 16, [1] * 32 + [2] * 32 + [0] * 32]
            ),
            device="cuda",
        )
        if mask_form == "padding_mask":
            documents = (documents > 0).long()
        valid = documents > 0
        positions = torch.zeros_like(documents)
        for row in range(2):
            for document in range(1, int(documents[row].max()) + 1):
                selected = documents[row] == document
                positions[row, selected] = torch.arange(int(selected.sum()), device="cuda")
        input_ids = torch.randint(0, config.vocab_size, documents.shape, device="cuda")
        metadata = build_packed_sequence_metadata(documents.cpu())
        arguments = {"position_ids": positions}
        if mask_form != "padding_mask":
            arguments.update(
                cu_seqlens=metadata["cu_seqlens"].cuda(),
                packed_token_indices=metadata["packed_token_indices"].cuda(),
                max_seqlen=metadata["max_seqlen"],
            )
        if mask_form == "document_ids":
            arguments["attention_mask"] = documents
        elif mask_form == "padding_mask":
            arguments["padding_mask"] = ~valid

        actual = candidate(input_ids, **arguments).logits
        causal = torch.arange(96, device="cuda")[:, None] >= torch.arange(96, device="cuda")[None, :]
        allowed = (documents[:, :, None] == documents[:, None, :]) & valid[:, :, None] & valid[:, None, :] & causal
        with sdpa_kernel(SDPBackend.MATH):
            expected = reference(
                input_ids, position_ids=positions, attention_mask=allowed[:, None], padding_mask=~valid
            ).logits
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual[valid].float(), expected[valid], atol=3e-2, rtol=3e-2)
        upstream = torch.randn_like(actual).float() * valid[:, :, None]
        (actual.float() * upstream).sum().backward()
        (expected * upstream).sum().backward()
        for name, parameter in candidate_attention.named_parameters():
            gradient = parameter.grad.float()
            reference_gradient = reference.model.layers["0"].self_attn.get_parameter(name).grad
            assert torch.isfinite(gradient).all()
            torch.testing.assert_close(gradient, reference_gradient, atol=5e-2, rtol=5e-2)
            cosine = torch.nn.functional.cosine_similarity(gradient.flatten(), reference_gradient.flatten(), dim=0)
            assert cosine > 0.995, (name, cosine.item())

        if compiled_frames is None:
            compiled_frames = counter.frame_count
            assert compiled_frames == 2
        else:
            assert counter.frame_count == compiled_frames, "document metadata specialized a projection graph"

        # Changing every other document and padding must leave document 1 unchanged.
        perturbed = input_ids.clone()
        perturbed[documents != 1] = (perturbed[documents != 1] + 17) % config.vocab_size
        unchanged = candidate(perturbed, **arguments).logits.detach()
        torch.testing.assert_close(unchanged[documents == 1], actual.detach()[documents == 1], atol=0, rtol=0)
