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

"""CPU tests for Qwen3-MoE FA4 input layouts: THD packing and unpacked batches.

FA4 reuses the shared CP1 THD batch conversion that TE uses and prepares private compact
offsets once at decoder entry, as GPT-OSS does. Unpacked batches unpad only when a token is
padding. FA4 kernels are replaced by independent SDPA evaluations, so these tests need no
GPU. Real-kernel THD parity is covered by
tests/functional_tests/attention/test_qwen3_moe_fa4_thd.py.
"""

from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F

import nemo_automodel.components.models.qwen3_moe.model as model_module
from nemo_automodel._transformers.capabilities import attach_capabilities_and_validate
from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.packing import validate_thd_packing_support
from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding
from nemo_automodel.components.models.qwen3_moe.model import Qwen3MoeForCausalLM, Qwen3MoeModel
from tests.unit_tests.attention.test_attention_utils import _reference_varlen_sdpa

pytestmark = pytest.mark.cpu_only

SEQUENCE = 16
# Two raw loader rows of two documents each. Padded lengths fill every row and include
# padding inside documents, so physical and valid THD offsets differ.
SEQ_LENS = [[3, 9], [5, 7]]
SEQ_LENS_PADDED = [[4, 12], [8, 8]]


def _stub(attn: str) -> Qwen3MoeForCausalLM:
    """Return an unbuilt Qwen3-MoE model whose class declarations are real."""
    model = Qwen3MoeForCausalLM.__new__(Qwen3MoeForCausalLM)
    torch.nn.Module.__init__(model)
    model.backend = SimpleNamespace(attn=attn)
    model.config = SimpleNamespace()
    return model


def _raw_thd_batch() -> dict[str, torch.Tensor | str]:
    """Build a raw THD loader batch.

    Returns:
        Mapping with input_ids, labels and document-local position_ids of shape
        [rows, sequence], int64 seq_lens and seq_lens_padded of shape
        [rows, documents], and qkv_format "thd".
    """
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(1, 64, (len(SEQ_LENS), SEQUENCE), generator=generator)
    positions = [[p for padded in row for p in range(padded)] for row in SEQ_LENS_PADDED]
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "position_ids": torch.tensor(positions),
        "seq_lens": torch.tensor(SEQ_LENS),
        "seq_lens_padded": torch.tensor(SEQ_LENS_PADDED),
        "qkv_format": "thd",
    }


def _clone(batch: dict[str, torch.Tensor | str]) -> dict[str, torch.Tensor | str]:
    return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _tiny_model(attn: str, fa4_module: ModuleType) -> Qwen3MoeForCausalLM:
    """Build a two-layer FP32 Qwen3-MoE with routed experts on CPU."""
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        # Batch-level load-balancing gradients differ between one packed batch and
        # separate documents; routing itself stays per token.
        router_aux_loss_coef=0.0,
    )
    config.dtype = torch.float32
    backend = BackendConfig(
        attn=attn,
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    with (
        patch.dict("sys.modules", {"flash_attn.cute": fa4_module}),
        patch("torch.cuda.current_device", return_value=0),
        patch.object(
            model_module,
            "RotaryEmbedding",
            side_effect=lambda **kwargs: RotaryEmbedding(**{**kwargs, "device": torch.device("cpu")}),
        ),
    ):
        model = Qwen3MoeForCausalLM(config, backend=backend)
    return model


@pytest.mark.parametrize("attn,accepted", [("fa4", True), ("te", True), ("sdpa", False)])
def test_fa4_thd_capability_is_backend_gated(attn: str, accepted: bool) -> None:
    """THD validation accepts FA4 only through its declared model path, without side effects."""
    model = _stub(attn)
    if accepted:
        validate_thd_packing_support([model])
    else:
        with pytest.raises(ValueError, match="cannot consume THD"):
            validate_thd_packing_support([model])
    assert model._owns_packed_attention is (attn == "fa4")
    assert model.requires_thd_batch_preparation is (attn == "fa4")
    assert not model._forward_pre_hooks


@pytest.mark.parametrize("attn,expected", [("te", True), ("magi", True), ("fa4", False)])
def test_fa4_thd_keeps_existing_packed_context_parallel_backends(attn: str, expected: bool) -> None:
    """Declaring FA4 THD must not disable TE/Magi packed CP or enable FA4 CP."""
    model = _stub(attn)
    model._mesh = SimpleNamespace(cp_size=2, tp_size=1, pp_size=1, ep_size=1)
    attach_capabilities_and_validate(model, None)
    assert model.supports.cp_size == 2
    assert model.supports.supports_cp_with_sequence_packing is expected


@pytest.mark.parametrize("attn", ["fa4", "te"])
def test_fa4_thd_decoder_prepares_metadata_once(monkeypatch: pytest.MonkeyPatch, attn: str) -> None:
    """One FA4 preparation feeds every layer; other backends receive the original metadata."""
    inner = Qwen3MoeModel.__new__(Qwen3MoeModel)
    torch.nn.Module.__init__(inner)
    inner.backend = SimpleNamespace(attn=attn, rope_fusion=False)
    inner.rotary_emb = None
    inner.embed_tokens = torch.nn.Embedding(8, 4)
    inner.norm = torch.nn.Identity()
    inner.layers = torch.nn.ModuleDict({"0": torch.nn.Identity(), "1": torch.nn.Identity()})
    captures = []
    for layer in inner.layers.values():
        capture = Mock(side_effect=lambda x, **kwargs: x)
        monkeypatch.setattr(layer, "forward", capture)
        captures.append(capture)
    monkeypatch.setattr(model_module, "position_ids_to_freqs_cis", lambda *args, **kwargs: torch.ones(6, 4))
    prepare = Mock(wraps=model_module.prepare_packed_metadata)
    monkeypatch.setattr(model_module, "prepare_packed_metadata", prepare)
    # Two physical documents of three tokens; the last token of each is padding.
    padding = torch.tensor([False, False, True, False, False, True])
    kwargs = {
        "qkv_format": "thd",
        "cu_seqlens": torch.tensor([0, 2, 4], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([0, 3, 6], dtype=torch.int32),
    }

    inner(torch.arange(6), position_ids=torch.tensor([0, 1, 2, 0, 1, 2]), padding_mask=padding, **kwargs)

    first, second = [capture.call_args.kwargs for capture in captures]
    assert first["padding_mask"] is padding
    assert first["cu_seqlens_padded"] is kwargs["cu_seqlens_padded"]
    if attn != "fa4":
        prepare.assert_not_called()
        assert "_fa4_cu_seqlens" not in first and "packed_token_indices" not in first
        return
    assert prepare.call_count == 1
    assert first["_fa4_cu_seqlens"] is second["_fa4_cu_seqlens"]
    assert first["_fa4_cu_seqlens"].tolist() == [0, 2, 4]
    assert first["packed_token_indices"].tolist() == [0, 1, 3, 4]
    assert first["_fa4_max_seqlen"] == 2
    with pytest.raises(ValueError, match="padding_mask length"):
        inner(torch.arange(5), position_ids=torch.arange(5), padding_mask=padding, **kwargs)


def test_fa4_thd_rejects_pipeline_chunks() -> None:
    """Shared FA4 THD preparation is CP1 and single-chunk only; PP chunking fails loudly."""
    with pytest.raises(ValueError, match="single non-pipeline chunk"):
        ContextParallelSharder(_stub("fa4"), None, _raw_thd_batch(), padding_token_id=0, num_chunks=2)


def test_fa4_thd_matches_per_document_sdpa_forward_backward() -> None:
    """Raw THD loader rows follow TE's conversion and match independent per-document SDPA."""
    torch.manual_seed(0)
    fa4 = ModuleType("flash_attn.cute")
    fa4.flash_attn_func = None
    fa4.flash_attn_varlen_func = _reference_varlen_sdpa
    model = _tiny_model("fa4", fa4)
    reference = _tiny_model("sdpa", fa4)
    for module in (model, reference):
        module.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    reference.load_state_dict(model.state_dict())
    attach_capabilities_and_validate(model, None)
    validate_thd_packing_support([model])

    raw = _raw_thd_batch()
    converted = {}
    for name, consumer in (("te", _stub("te")), ("fa4", model)):
        batch = _clone(raw)
        _, converted[name] = ContextParallelSharder(consumer, None, batch, padding_token_id=0, num_chunks=1).shard(
            batch
        )
    assert converted["te"].keys() == converted["fa4"].keys()
    for key, value in converted["te"].items():
        actual = converted["fa4"][key]
        assert torch.equal(value, actual) if isinstance(value, torch.Tensor) else value == actual, key

    inputs = _clone(converted["fa4"])
    inputs.pop("labels")
    logits = model(**inputs).logits
    assert logits.shape == (1, len(SEQ_LENS) * SEQUENCE, 64)
    assert torch.isfinite(logits).all()
    # Physical THD tokens keep the row-major loader layout, including padding.
    documents = []
    valid = torch.zeros(logits.shape[1], dtype=torch.bool)
    for row, (lengths, padded_lengths) in enumerate(zip(SEQ_LENS, SEQ_LENS_PADDED)):
        offset = 0
        for length, padded in zip(lengths, padded_lengths):
            physical = slice(row * SEQUENCE + offset, row * SEQUENCE + offset + length)
            documents.append((raw["input_ids"][row, offset : offset + length], physical))
            valid[physical] = True
            offset += padded
    upstream = torch.randn_like(logits) * valid[None, :, None]
    expected_loss = torch.zeros(())
    for tokens, physical in documents:
        expected = reference(tokens[None], position_ids=torch.arange(tokens.numel())[None]).logits[0]
        torch.testing.assert_close(logits[0, physical], expected, rtol=1e-5, atol=1e-5)
        expected_loss = expected_loss + (expected * upstream[0, physical]).sum()

    (logits * upstream).sum().backward()
    expected_loss.backward()
    for (name, parameter), (reference_name, reference_parameter) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        assert name == reference_name
        gradient = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
        reference_gradient = (
            torch.zeros_like(reference_parameter) if reference_parameter.grad is None else reference_parameter.grad
        )
        torch.testing.assert_close(gradient, reference_gradient, rtol=1e-4, atol=1e-5, msg=name)


@pytest.mark.parametrize("padded", [False, True])
def test_fa4_unpacked_batches_unpad_only_with_padding(padded: bool) -> None:
    """All-valid collated batches keep dense FA4; padded batches unpad and still match SDPA."""
    torch.manual_seed(0)
    calls = {"dense": 0, "varlen": 0}

    def dense_sdpa(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        softmax_scale: float,
        causal: bool,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        """Evaluate dense SDPA for a fake FA4 dense kernel.

        Args:
            q: Query tensor of shape [batch, sequence, heads, head_dim].
            k: Key tensor of shape [batch, sequence, kv_heads, head_dim].
            v: Value tensor of shape [batch, sequence, kv_heads, head_dim].
            softmax_scale: Attention score scale.
            causal: Whether to apply causal attention.
            **kwargs: Unused FA4 options; the window is unbounded here.

        Returns:
            Pair of the output of shape [batch, sequence, heads, head_dim] and None for
            the unused LSE, matching FA4.
        """
        calls["dense"] += 1
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=causal,
            scale=softmax_scale,
            enable_gqa=True,
        )
        return output.transpose(1, 2), None

    def varlen_sdpa(*args: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, None]:
        """Count and evaluate the independent per-document varlen reference."""
        calls["varlen"] += 1
        return _reference_varlen_sdpa(*args, **kwargs)

    fa4 = ModuleType("flash_attn.cute")
    fa4.flash_attn_func = dense_sdpa
    fa4.flash_attn_varlen_func = varlen_sdpa
    model = _tiny_model("fa4", fa4)
    reference = _tiny_model("sdpa", fa4)
    for module in (model, reference):
        module.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    reference.load_state_dict(model.state_dict())
    lengths = [8, 5] if padded else [8, 8]
    input_ids = torch.randint(1, 64, (len(lengths), 8))
    # default_collater always emits padding_mask, which is all False for full-length rows.
    padding_mask = torch.arange(8)[None, :] >= torch.tensor(lengths)[:, None]

    logits = model(input_ids, padding_mask=padding_mask).logits

    layers = len(model.model.layers)
    assert calls == ({"dense": 0, "varlen": layers} if padded else {"dense": layers, "varlen": 0})
    for row, length in enumerate(lengths):
        expected = reference(input_ids[row : row + 1, :length]).logits[0]
        torch.testing.assert_close(logits[row, :length], expected, rtol=1e-5, atol=1e-5)
