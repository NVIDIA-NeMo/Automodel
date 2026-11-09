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

"""CUDA coverage of physical THD metadata and TE padding dispatch."""

from unittest.mock import patch

import pytest
import torch

from nemo_automodel.components.attention.utils import preprocess_args_and_kwargs_for_attn
from nemo_automodel.components.distributed.thd_utils import process_input_for_thd
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd


@pytest.mark.parametrize("real_lengths", [(32, 48), (23, 35), (32, 35)])
@pytest.mark.parametrize("native_preparation", [False, True])
def test_te_physical_boundaries_forward_backward(real_lengths: tuple[int, int], native_preparation: bool) -> None:
    """TE matches independent per-document FP64 attention, including gradients.

    Native preparation absorbs final-only padding. Direct callers retain the
    shorter attention boundary, so these cases exercise both TE dispatch paths.
    Physical offsets and valid-token outputs must agree in both cases.
    """
    if not torch.cuda.is_available():
        pytest.skip("TE THD forward/backward requires CUDA")
    te = pytest.importorskip("transformer_engine.pytorch")
    device = torch.device("cuda")
    physical_lengths = (32, 48)
    input_ids = torch.arange(80, device=device).view(1, 80)
    if native_preparation:
        batch = process_input_for_thd(
            {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
                "position_ids": input_ids.clone(),
                "seq_lens": torch.tensor([real_lengths], device=device),
                "seq_lens_padded": torch.tensor([physical_lengths], device=device),
            }
        )
        metadata = {key: batch[key] for key in ("cu_seqlens", "cu_seqlens_padded", "max_seqlen", "pad_between_seqs")}
        _, _, _, metadata = squeeze_input_for_thd(input_ids, input_ids, None, metadata)
    else:
        metadata = {
            "cu_seqlens": torch.tensor([0, real_lengths[0], sum(real_lengths)], device=device, dtype=torch.int32),
            "cu_seqlens_padded": torch.tensor([0, 32, 80], device=device, dtype=torch.int32),
            "max_seqlen": max(real_lengths),
        }

    torch.manual_seed(17)
    inputs = [torch.randn(80, 4, 64, device=device, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    reference_inputs = [value.detach().double().requires_grad_() for value in inputs]
    q, k, v, kwargs = preprocess_args_and_kwargs_for_attn(*inputs, None, "te", **metadata)
    attention = te.DotProductAttention(4, 64, attention_dropout=0.0, qkv_format="thd")
    output = attention(q, k, v, **kwargs).reshape(80, 4, 64)
    # No production packing/attention helper is used to construct the oracle.
    reference_documents = []
    valid = torch.zeros(80, device=device, dtype=torch.bool)
    offset = 0
    for real, physical in zip(real_lengths, physical_lengths):
        rq, rk, rv = [value[offset : offset + real].transpose(0, 1) for value in reference_inputs]
        causal = torch.ones(real, real, device=device, dtype=torch.bool).tril()
        scores = (rq @ rk.transpose(-1, -2)) / 8.0
        document = (scores.masked_fill(~causal, -torch.inf).softmax(-1) @ rv).transpose(0, 1)
        reference_documents.extend((document, document.new_zeros(physical - real, 4, 64)))
        valid[offset : offset + real] = True
        offset += physical
    reference = torch.cat(reference_documents)

    # BF16 fused reductions have a different order from the FP64 oracle.
    torch.testing.assert_close(output[valid].double(), reference[valid], atol=2e-2, rtol=2e-2)
    upstream = torch.randn_like(output)
    upstream[~valid] = 0
    output.backward(upstream)
    reference.backward(upstream.double())
    for actual, expected in zip(inputs, reference_inputs):
        torch.testing.assert_close(actual.grad.double(), expected.grad, atol=2e-2, rtol=2e-2)
        assert torch.count_nonzero(actual.grad[~valid]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("stage", ["full", "first", "middle", "last"])
@pytest.mark.parametrize("supplied_flag", [False, True])
def test_te_stage_resolves_padding_once_per_microbatch(stage: str, supplied_flag: bool) -> None:
    """Share each pack's padding flag across real backbone and MTP attention."""
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM
    from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_mtp import MockNemotronV3Config

    pytest.importorskip("transformer_engine.pytorch")
    config = MockNemotronV3Config(
        hidden_size=128,
        head_dim=32,
        layers_block_type=["attention", "attention"],
        num_nextn_predict_layers=1,
        mtp_hybrid_override_pattern="*",
    )
    backend = BackendConfig(
        linear="torch", attn="te", rms_norm="torch", dispatcher="torch", enable_hf_state_dict_adapter=False
    )
    model = NemotronHForCausalLM(config, backend=backend).to(device="cuda", dtype=torch.bfloat16).train()
    if stage in ("first", "middle"):
        model.lm_head = None
        model.model.norm = None
        model.mtp = None
    if stage in ("middle", "last"):
        model.model.embed_tokens = None
        inputs = (
            torch.randn(1, 128, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(1, 128, 128, device="cuda", dtype=torch.bfloat16),
        )
    else:
        inputs = (torch.randint(1, 64, (1, 128), device="cuda"),)
    positions = torch.arange(128, device="cuda").view(1, 128) % 64
    physical = torch.tensor([[0, 64, 128, -1000]], device="cuda", dtype=torch.int32)
    # Reuse a stage with alternating layouts to catch flags cached across packs.
    for padded in (True, False, True):
        real = torch.tensor([[0, 63, 126, -1000]], device="cuda", dtype=torch.int32) if padded else physical.clone()
        metadata = dict(qkv_format="thd", cu_seqlens=real, cu_seqlens_padded=physical, max_seqlen=64)
        if supplied_flag:
            metadata["pad_between_seqs"] = padded
        with (
            patch("torch.equal", wraps=torch.equal) as equality,
            patch(
                "nemo_automodel.components.models.nemotron_v3.layers.preprocess_args_and_kwargs_for_attn",
                wraps=preprocess_args_and_kwargs_for_attn,
            ) as preparation,
        ):
            model(*inputs, position_ids=positions, **metadata)
        # Both backbone layers and, on the final/full model, the MTP attention
        # must receive the resolved boolean, avoiding the direct-call fallback.
        assert preparation.call_count == (3 if stage in ("full", "last") else 2)
        assert [call.kwargs.get("pad_between_seqs") for call in preparation.call_args_list] == [
            padded
        ] * preparation.call_count
        assert equality.call_count == (0 if supplied_flag else 1)
