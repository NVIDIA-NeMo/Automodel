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

"""Attention dropout (training and recomputation): training-only, HF semantics, on SDPA (unpacked and packed) and TE;
flex rejects a non-zero value; activation-checkpoint recomputation preserves the dropout realization."""

import copy

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.beam.model import BeamForCausalLM

from .utils import hf_model_and_checkpoint, native_from_hf

CUDA = torch.cuda.is_available()


def _models(cfg, backend, device, dropout):
    """Two natives with identical weights: dropout 0 and ``dropout``."""
    hf, sd = hf_model_and_checkpoint(cfg, device)
    m0 = native_from_hf(cfg, backend, sd, device)
    cfg_d = copy.deepcopy(cfg)
    cfg_d.attention_dropout = dropout
    md = BeamForCausalLM(cfg_d, backend=backend).to(device)
    md.load_state_dict(m0.state_dict())
    return hf, m0.eval(), md.eval()


def _ids(cfg, device, n=24):
    return torch.randint(0, cfg.vocab_size, (2, n), generator=torch.Generator().manual_seed(3)).to(device)


def _thd_call(model, ids):
    """Run two packed sequences.

    Args:
        model: Native Beam model.
        ids: Integer tensor of shape [batch, sequence], with sequence >= 24.

    Returns:
        Logits of shape [1, 24, vocab] for documents of lengths 8 and 16.
    """
    lens = (8, 16)
    pos = torch.cat([torch.arange(n, device=ids.device) for n in lens]).unsqueeze(0)
    cu = torch.tensor([0, lens[0], sum(lens)], dtype=torch.int32, device=ids.device)
    return model(ids[:1, : sum(lens)], position_ids=pos, qkv_format="thd", cu_seqlens=cu, max_seqlen=max(lens)).logits


ATTN_DEVICES = [
    ("sdpa", "cpu"),
    pytest.param(("sdpa", "cuda"), marks=pytest.mark.skipif(not CUDA, reason="needs CUDA")),
    pytest.param(("te", "cuda"), marks=pytest.mark.skipif(not CUDA, reason="needs CUDA")),
]


@pytest.fixture
def backend_for(request):
    return lambda attn: request.getfixturevalue("te_backend_config" if attn == "te" else "backend_config")


class TestAttentionDropout:
    @pytest.mark.parametrize("attn_device", ATTN_DEVICES, ids=str)
    def test_eval_equals_no_dropout(self, beam_config, backend_for, attn_device):
        attn, device = attn_device
        _, m0, md = _models(beam_config, backend_for(attn), torch.device(device), 0.5)
        ids = _ids(beam_config, device)
        with torch.no_grad():
            assert torch.equal(md(ids).logits, m0(ids).logits)
            if attn != "flex":
                assert torch.equal(_thd_call(md, ids), _thd_call(m0, ids))

    @pytest.mark.parametrize("attn_device", ATTN_DEVICES, ids=str)
    def test_training_applies_dropout_unpacked_and_packed(self, beam_config, backend_for, attn_device):
        attn, device = attn_device
        _, m0, md = _models(beam_config, backend_for(attn), torch.device(device), 0.5)
        ids = _ids(beam_config, device)
        md.train()
        m0.train()
        a, b = md(ids).logits, md(ids).logits
        assert not torch.equal(a, b) and not torch.equal(a, m0(ids).logits)
        ta, tb = _thd_call(md, ids), _thd_call(md, ids)
        assert not torch.equal(ta, tb)
        # Exercise packed backward independently; the unpacked loss cannot mask missing packed gradients.
        ta.backward(torch.randn_like(ta))
        for layer in md.model.layers.values():
            grad = layer.self_attn.q_proj.weight.grad
            assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
        md.zero_grad(set_to_none=True)
        F.cross_entropy(a[:, :-1].float().reshape(-1, a.shape[-1]), ids[:, 1:].reshape(-1)).backward()
        assert all(torch.isfinite(p.grad).all() for p in md.parameters() if p.grad is not None)

    def test_flex_rejects_nonzero_dropout(self, beam_config, flex_backend_config):
        cfg = copy.deepcopy(beam_config)
        cfg.attention_dropout = 0.1
        with pytest.raises(NotImplementedError, match="attention_dropout"):
            BeamForCausalLM(cfg, backend=flex_backend_config)
        cfg.attention_dropout = 0.0
        assert BeamForCausalLM(cfg, backend=flex_backend_config) is not None

    @pytest.mark.parametrize("attn_device", ATTN_DEVICES, ids=str)
    @pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
    @pytest.mark.parametrize("ac_path", ["submodule", "moe"])
    def test_activation_checkpointing_preserves_the_dropout_realization(
        self, beam_config, backend_for, attn_device, packed, ac_path
    ):
        """Same weights, inputs and RNG state: the AC-wrapped model reproduces the plain model's outputs and input /
        parameter gradients (non-reentrant checkpoint with preserve_rng_state replays the dropout masks)."""
        attn, device = attn_device
        _, _, md = _models(beam_config, backend_for(attn), torch.device(device), 0.5)
        plain = md.train()
        from nemo_automodel.components.distributed.activation_checkpointing import apply_submodule_checkpointing
        from nemo_automodel.components.moe.parallelizer import apply_ac

        acm = copy.deepcopy(plain)
        if ac_path == "submodule":
            apply_submodule_checkpointing(list(acm.model.layers.values()), has_kv_sharing=False)
        else:
            apply_ac(acm)
        ids = _ids(beam_config, device)
        results = []
        for m in (plain, acm):
            m.zero_grad(set_to_none=True)
            torch.manual_seed(1234)
            out = (_thd_call(m, ids) if packed else m(ids).logits).float()
            # Identical nonzero upstream gradients isolate recomputation/RNG correctness on both layouts.
            upstream = torch.randn(out.shape, generator=torch.Generator().manual_seed(17)).to(out.device)
            out.backward(upstream)
            grads = {
                n.replace("_checkpoint_wrapped_module.", ""): p.grad.detach().clone()
                for n, p in m.named_parameters()
                if p.grad is not None
            }
            results.append((out.detach(), grads))
        tol = dict(rtol=1e-5, atol=1e-5)  # fp32 declared tolerance (kernel accumulation order under recompute)
        torch.testing.assert_close(results[1], results[0], **tol)
        # and the realization is dropout-dependent: a different seed changes the output
        torch.manual_seed(99)
        different_seed = _thd_call(plain, ids) if packed else plain(ids).logits
        assert not torch.equal(different_seed.float(), results[0][0])
