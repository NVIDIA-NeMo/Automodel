# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Production Qwen3-MoE THD FlashAttention-4 parity against document-masked SDPA."""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from nemo_automodel._transformers.capabilities import attach_capabilities_and_validate
from nemo_automodel.components.distributed.activation_checkpointing import apply_full_layer_checkpointing_to_layers
from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.packing import validate_thd_packing_support
from nemo_automodel.components.models.qwen3_moe.model import Qwen3MoeForCausalLM

SEQUENCE = 96
# Three documents per raw loader row. Padded spans fill each row and add padding inside
# documents, so FA4's compact offsets differ from the physical THD boundaries.
SEQ_LENS = [[30, 40, 20], [60, 25, 3]]
SEQ_LENS_PADDED = [[32, 40, 24], [64, 28, 4]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and FlashAttention-4")
@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_qwen_fa4_thd_forward_backward(activation_checkpointing: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Raw THD rows reach real FA4 through the shared conversion and match masked SDPA."""
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FlashAttention-4 requires Hopper or newer hardware")
    pytest.importorskip("flash_attn.cute")
    # Packed tokens and the BSHD reference run GEMMs of different shapes. For some shapes
    # cuBLAS splits the reduction and rounds partial sums to BF16; keep them in FP32 instead.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False)
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
    candidate = Qwen3MoeForCausalLM(config, backend=BackendConfig(attn="fa4", **backend_options)).cuda()
    candidate.initialize_weights(dtype=torch.bfloat16)
    reference = Qwen3MoeForCausalLM(config, backend=BackendConfig(attn="sdpa", **backend_options)).cuda().float()
    reference.load_state_dict(candidate.state_dict())
    attach_capabilities_and_validate(candidate, None)
    validate_thd_packing_support([candidate])
    candidate_attention = candidate.model.layers["0"].self_attn
    if activation_checkpointing:
        apply_full_layer_checkpointing_to_layers(candidate, [candidate.model.layers["0"]])

    documents = torch.zeros(len(SEQ_LENS), SEQUENCE, dtype=torch.long)
    positions = torch.zeros(len(SEQ_LENS), SEQUENCE, dtype=torch.long)
    for row, (lengths, padded_lengths) in enumerate(zip(SEQ_LENS, SEQ_LENS_PADDED)):
        offset = 0
        for document, (length, padded) in enumerate(zip(lengths, padded_lengths), start=1):
            documents[row, offset : offset + length] = document
            positions[row, offset : offset + padded] = torch.arange(padded)
            offset += padded
    documents, positions = documents.cuda(), positions.cuda()
    valid = documents > 0
    input_ids = torch.randint(1, config.vocab_size, documents.shape, device="cuda")

    def thd_logits(tokens: torch.Tensor) -> torch.Tensor:
        """Run raw THD rows through the recipe's shared conversion and the FA4 model.

        Args:
            tokens: Token IDs of shape [rows, sequence].

        Returns:
            Logits of shape [rows, sequence, vocab]; physical THD tokens keep the
            row-major loader layout, including padding.
        """
        raw = {
            "input_ids": tokens,
            "labels": tokens.clone(),
            "position_ids": positions,
            "seq_lens": torch.tensor(SEQ_LENS, device="cuda"),
            "seq_lens_padded": torch.tensor(SEQ_LENS_PADDED, device="cuda"),
            "qkv_format": "thd",
        }
        _, batch = ContextParallelSharder(candidate, None, raw, padding_token_id=0, num_chunks=1).shard(raw)
        batch.pop("labels")
        return candidate(**batch).logits.view(len(SEQ_LENS), SEQUENCE, -1)

    actual = thd_logits(input_ids)
    causal = torch.arange(SEQUENCE, device="cuda")[:, None] >= torch.arange(SEQUENCE, device="cuda")[None, :]
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
    # BF16 parameters against an FP32 reference leave near-zero gradient elements without a
    # stable elementwise tolerance; bound the whole-tensor error and direction instead.
    # A CPU emulation of this test with BF16 reference-SDPA kernels stays below 1% relative
    # L2 error with cosine above 0.9999, and matches to 4e-7 in FP32.
    for name, parameter in candidate_attention.named_parameters():
        gradient = parameter.grad.float()
        reference_gradient = reference.model.layers["0"].self_attn.get_parameter(name).grad
        assert torch.isfinite(gradient).all(), name
        relative_error = (gradient - reference_gradient).norm() / reference_gradient.norm()
        assert relative_error < 2e-2, (name, relative_error.item())
        cosine = F.cosine_similarity(gradient.flatten(), reference_gradient.flatten(), dim=0)
        assert cosine > 0.999, (name, cosine.item())

    # Changing every other document and all padding must leave document 1 unchanged.
    perturbed = input_ids.clone()
    perturbed[documents != 1] = perturbed[documents != 1] % (config.vocab_size - 1) + 1
    with torch.no_grad():
        unchanged = thd_logits(perturbed)
    torch.testing.assert_close(unchanged[documents == 1], actual.detach()[documents == 1], atol=0, rtol=0)
