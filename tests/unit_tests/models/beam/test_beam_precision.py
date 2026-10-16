# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""HF forward/backward parity for RoPE and dense SwiGLU, plus shared rotary-table storage."""

import pytest
import torch

from nemo_automodel.components.models.beam.layers import BeamMLP, apply_rope_hf, hf_rope_tables
from nemo_automodel.components.models.beam.model import BeamForCausalLM
from nemo_automodel.components.models.gpt_oss.rope_utils import position_ids_to_freqs_cis
from nemo_automodel.components.moe.layers import MLP

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DTYPES = [torch.bfloat16, torch.float32]


def _tables(batch, seq, head_dim, device, generator):
    """Random fp32 angles -> Automodel unfused ``freqs_cis`` (``cat([cos, sin])`` over half dims) and HF's
    duplicated ``cos``/``sin`` (``cat((freqs, freqs))``), shapes ``[B, S, D]``."""
    angles = (torch.rand(batch, seq, head_dim // 2, generator=generator) * 6.283).to(device)
    cos_h, sin_h = angles.cos(), angles.sin()
    return torch.cat([cos_h, sin_h], -1), torch.cat([cos_h, cos_h], -1), torch.cat([sin_h, sin_h], -1)


def _qk(batch, seq, heads, kv_heads, head_dim, device, dtype, generator):
    q = torch.randn(batch, seq, heads, head_dim, generator=generator).to(device=device, dtype=dtype)
    k = torch.randn(batch, seq, kv_heads, head_dim, generator=generator).to(device=device, dtype=dtype)
    gq = torch.randn(batch, seq, heads, head_dim, generator=generator).to(device=device, dtype=dtype)
    gk = torch.randn(batch, seq, kv_heads, head_dim, generator=generator).to(device=device, dtype=dtype)
    return q, k, gq, gk


class TestRoPE:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
    @pytest.mark.parametrize("layout", ["bshd", "thd"])
    def test_rotation_matches_hf_forward_and_backward(self, device, dtype, layout):
        """Identical q/k, fp32 tables and upstream gradients: outputs and q/k gradients equal HF's bit for bit."""
        from transformers.models.beam.modeling_beam import apply_rotary_pos_emb

        g = torch.Generator().manual_seed(0)
        B, S, H, HKV, D = 2, 16, 4, 2, 16
        freqs_cis, hf_cos, hf_sin = _tables(B, S, D, device, g)
        q, k, gq, gk = _qk(B, S, H, HKV, D, device, dtype, g)
        if layout == "thd":
            q, k, gq, gk = (t.reshape(-1, *t.shape[2:]) for t in (q, k, gq, gk))
            freqs_cis, hf_cos, hf_sin = (t.reshape(-1, D) for t in (freqs_cis, hf_cos, hf_sin))
        head_axis = 2 if layout == "bshd" else 1
        q_hf, k_hf = q.clone().requires_grad_(), k.clone().requires_grad_()
        q_nat, k_nat = q.clone().requires_grad_(), k.clone().requires_grad_()

        oq_hf, ok_hf = apply_rotary_pos_emb(q_hf, k_hf, hf_cos, hf_sin, unsqueeze_dim=head_axis)
        cos, sin = hf_rope_tables(freqs_cis)
        assert cos.shape[-2] == 1 and cos.dtype == torch.float32
        oq, ok = apply_rope_hf(q_nat, cos, sin), apply_rope_hf(k_nat, cos, sin)

        assert oq.dtype == dtype and ok.dtype == dtype
        assert torch.equal(oq, oq_hf) and torch.equal(ok, ok_hf)
        torch.autograd.backward((oq_hf, ok_hf), (gq, gk))
        torch.autograd.backward((oq, ok), (gq, gk))
        assert q_nat.grad.dtype == dtype
        assert torch.equal(q_nat.grad, q_hf.grad) and torch.equal(k_nat.grad, k_hf.grad)

    @pytest.mark.parametrize("positions", ["arange", "offset", "packed"])
    def test_tables_match_hf_rotary_embedding(self, beam_config, backend_config, positions):
        """Automodel's unfused tables (RotaryEmbedding -> position_ids_to_freqs_cis -> hf_rope_tables) equal HF's
        BeamRotaryEmbedding output for plain, offset and packed positions."""
        from transformers.models.beam.modeling_beam import BeamRotaryEmbedding

        S = 24
        if positions == "arange":
            pos = torch.arange(S).unsqueeze(0)
        elif positions == "offset":
            pos = torch.arange(100, 100 + S).unsqueeze(0)
        else:  # two packed sequences of 8 and 16 tokens
            pos = torch.cat([torch.arange(8), torch.arange(16)]).unsqueeze(0)
        hf_rot = BeamRotaryEmbedding(beam_config)
        hf_cos, hf_sin = hf_rot(torch.zeros(1, S, beam_config.head_dim), pos, "sliding_attention")
        model = BeamForCausalLM(beam_config, backend=backend_config)
        rotary = model.model.rotary_emb["sliding_attention"]
        freqs = position_ids_to_freqs_cis(rotary, pos, qkv_format="bshd", for_fused_rope=False)
        cos, sin = hf_rope_tables(freqs)
        assert cos.shape == (1, S, 1, beam_config.head_dim)
        torch.testing.assert_close(cos.squeeze(-2), hf_cos, atol=0, rtol=0)
        torch.testing.assert_close(sin.squeeze(-2), hf_sin, atol=0, rtol=0)


class TestDenseMLP:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
    def test_matches_hf_forward_and_backward(self, beam_config, device, dtype):
        """Identical weights, inputs and upstream gradients: output, input gradient and the three weight gradients
        equal HF BeamMLP bit for bit (same GEMM kernels, same fp32 SwiGLU boundary)."""
        from transformers.models.beam.modeling_beam import BeamMLP as HFBeamMLP

        torch.manual_seed(0)
        hf = HFBeamMLP(beam_config).to(device=device, dtype=dtype)
        nat = BeamMLP(beam_config.hidden_size, beam_config.intermediate_size, "torch", dtype=dtype).to(device)
        assert isinstance(nat, MLP)
        missing, unexpected = nat.load_state_dict(hf.state_dict())
        assert not missing and not unexpected
        x = torch.randn(3, 5, beam_config.hidden_size, device=device).to(dtype)
        g = torch.randn_like(x)
        x_hf, x_nat = x.clone().requires_grad_(), x.clone().requires_grad_()
        y_hf, y_nat = hf(x_hf), nat(x_nat)
        assert y_nat.dtype == dtype and torch.equal(y_nat, y_hf)
        y_hf.backward(g)
        y_nat.backward(g)
        assert torch.equal(x_nat.grad, x_hf.grad)
        hf_params = dict(hf.named_parameters())
        for name, p in nat.named_parameters():
            assert torch.equal(p.grad, hf_params[name].grad), name

    def test_block_builds_beam_mlp_and_rejects_other_activations(self, beam_config, backend_config):
        model = BeamForCausalLM(beam_config, backend=backend_config)
        dense = model.model.layers["0"].mlp
        assert isinstance(dense, BeamMLP) and isinstance(dense, MLP)
        assert not any(isinstance(layer.mlp, BeamMLP) for name, layer in model.model.layers.items() if name != "0")
        beam_config.hidden_act = "gelu"
        with pytest.raises(NotImplementedError, match="hidden_act"):
            BeamForCausalLM(beam_config, backend=backend_config)


@pytest.mark.parametrize("ac", [False, True], ids=["no_ac", "ac"])
@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
def test_rope_tables_shared_through_backward(beam_config, backend_config, ac, packed):
    """One exact FP32 table pair serves every RoPE layer, including recomputation."""
    from nemo_automodel.components.distributed.activation_checkpointing import apply_full_layer_checkpointing_to_layers

    from .utils import capture_rope_calls, hf_model_and_checkpoint, native_from_hf

    _, sd = hf_model_and_checkpoint(beam_config, torch.device("cpu"))
    model = native_from_hf(beam_config, backend_config, sd, torch.device("cpu"))
    if ac:
        apply_full_layer_checkpointing_to_layers(model, list(model.model.layers.values()))
    calls = capture_rope_calls(model)
    saved = set()

    def pack(t):
        """Observe saved rotary-table storage.

        Args:
            t: Tensor of arbitrary shape/dtype; rotary tables have [..., 1, head_dim].

        Returns:
            The input tensor by alias, without modification.
        """
        if t.dtype == torch.float32 and t.dim() >= 3 and t.shape[-2:] == (1, beam_config.head_dim):
            saved.add(t.untyped_storage().data_ptr())
        return t

    ids = torch.randint(0, beam_config.vocab_size, (1, 16), generator=torch.Generator().manual_seed(0))
    kwargs = {}
    if packed:
        kwargs = dict(
            position_ids=torch.arange(8).repeat(2).unsqueeze(0),
            qkv_format="thd",
            cu_seqlens=torch.tensor([0, 8, 16], dtype=torch.int32),
            max_seqlen=8,
        )
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = model(ids, **kwargs).logits
    n_forward = len(calls)
    out.backward(torch.randn(out.shape, generator=torch.Generator().manual_seed(1)))
    n_rope = sum(layer.self_attn.use_rope for layer in model.model.layers.values())
    assert n_forward == n_rope and len(calls) == n_rope * (2 if ac else 1)
    assert all(rec.exact and rec.dtypes == (torch.float32, torch.float32) for rec in calls)
    pairs = {rec.storages for rec in calls}
    assert len(pairs) == 1
    if not ac:
        assert saved == set(next(iter(pairs)))  # backward retained only this shared pair
