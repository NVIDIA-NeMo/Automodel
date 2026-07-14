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

"""Parity tests for the fused LoRA SwiGLU MLP autograd function."""

# ruff: noqa: E741

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components._peft.lora import patch_linear_module
from nemo_automodel.components._peft.lora_mlp import LoRASwiGLUMLPFunction, fused_lora_swiglu_mlp


def _ref_mlp(x, gW, gA, gB, gS, uW, uA, uB, uS, dW, dA, dB, dS):
    e = F.linear(x, gW) + F.linear(F.linear(x, gA) * gS, gB)
    g = F.linear(x, uW) + F.linear(F.linear(x, uA) * uS, uB)
    h = F.silu(e) * g
    return F.linear(h, dW) + F.linear(F.linear(h, dA) * dS, dB)


def _mk(*shape, seed=0):
    g = torch.Generator().manual_seed(sum(shape) + seed)
    return torch.randn(*shape, generator=g) * 0.1


def test_fused_lora_mlp_matches_reference_fp32():
    """Forward output and every gradient match a plain-autograd reference (CPU/float32, strict)."""
    H, I, R, T = 64, 96, 8, 16
    gW, uW, dW = _mk(I, H, seed=1), _mk(I, H, seed=2), _mk(H, I, seed=3)
    gA, uA, dA = _mk(R, H, seed=4), _mk(R, H, seed=5), _mk(R, I, seed=6)
    gB, uB, dB = _mk(I, R, seed=7), _mk(I, R, seed=8), _mk(H, R, seed=9)
    gS, uS, dS = 2.0, 1.5, 0.5
    x0, gout = _mk(2, T, H, seed=10), _mk(2, T, H, seed=11)

    # reference
    xr = x0.clone().requires_grad_(True)
    ref_p = {k: v.clone().requires_grad_(True) for k, v in dict(gA=gA, gB=gB, uA=uA, uB=uB, dA=dA, dB=dB).items()}
    (
        _ref_mlp(
            xr, gW, ref_p["gA"], ref_p["gB"], gS, uW, ref_p["uA"], ref_p["uB"], uS, dW, ref_p["dA"], ref_p["dB"], dS
        )
        * gout
    ).sum().backward()

    # fused
    xf = x0.clone().requires_grad_(True)
    fp = {k: v.clone().requires_grad_(True) for k, v in dict(gA=gA, gB=gB, uA=uA, uB=uB, dA=dA, dB=dB).items()}
    out = LoRASwiGLUMLPFunction.apply(
        xf, gW, fp["gA"], fp["gB"], gS, uW, fp["uA"], fp["uB"], uS, dW, fp["dA"], fp["dB"], dS
    )
    (out * gout).sum().backward()

    def rel(a, b):
        return (a - b).abs().max().item() / (b.abs().max().item() + 1e-9)

    assert rel(xf.grad, xr.grad) < 1e-5
    for k in fp:
        assert rel(fp[k].grad, ref_p[k].grad) < 1e-5, f"grad mismatch for {k}"


def _ref_relu2_mlp(x, uW, uA, uB, uS, dW, dA, dB, dS):
    e = F.linear(x, uW) + F.linear(F.linear(x, uA) * uS, uB)
    f = F.relu(e).pow(2)
    return F.linear(f, dW) + F.linear(F.linear(f, dA) * dS, dB)


def test_fused_relu2_mlp_matches_reference_fp32():
    """ReLU² fused output and all grads match a plain-autograd reference (CPU/float32, strict)."""
    from nemo_automodel.components._peft.lora_mlp import LoRAReLU2MLPFunction

    H, I, R, T = 64, 96, 8, 16
    uW, dW = _mk(I, H, seed=1), _mk(H, I, seed=2)
    uA, dA = _mk(R, H, seed=3), _mk(R, I, seed=4)
    uB, dB = _mk(I, R, seed=5), _mk(H, R, seed=6)
    uS, dS = 1.5, 0.5
    x0, gout = _mk(2, T, H, seed=7), _mk(2, T, H, seed=8)

    xr = x0.clone().requires_grad_(True)
    rp = {k: v.clone().requires_grad_(True) for k, v in dict(uA=uA, uB=uB, dA=dA, dB=dB).items()}
    (_ref_relu2_mlp(xr, uW, rp["uA"], rp["uB"], uS, dW, rp["dA"], rp["dB"], dS) * gout).sum().backward()

    xf = x0.clone().requires_grad_(True)
    fp = {k: v.clone().requires_grad_(True) for k, v in dict(uA=uA, uB=uB, dA=dA, dB=dB).items()}
    out = LoRAReLU2MLPFunction.apply(xf, uW, fp["uA"], fp["uB"], uS, dW, fp["dA"], fp["dB"], dS)
    (out * gout).sum().backward()

    def rel(a, b):
        return (a - b).abs().max().item() / (b.abs().max().item() + 1e-9)

    assert rel(xf.grad, xr.grad) < 1e-5
    for k in fp:
        assert rel(fp[k].grad, rp[k].grad) < 1e-5, k


def test_install_fuses_relu2_mlp():
    """moe.layers.MLP (relu2) routes to the fused ReLU² path once up/down are LoRA-patched."""
    from nemo_automodel.components._peft.lora_mlp import install_fused_lora_mlp
    from nemo_automodel.components.moe.layers import MLP

    torch.manual_seed(0)
    H, I, R = 64, 96, 8
    mlp = MLP(dim=H, inter_dim=I, backend="torch", dtype=torch.float32, activation="relu2")
    mlp.up_proj = patch_linear_module(mlp.up_proj, dim=R, alpha=R, use_triton=False)
    mlp.down_proj = patch_linear_module(mlp.down_proj, dim=R, alpha=R, use_triton=False)
    for m in (mlp.up_proj, mlp.down_proj):
        nn.init.normal_(m.lora_B.weight, std=0.02)

    assert install_fused_lora_mlp(mlp) == 1
    x = torch.randn(2, 16, H)
    out = mlp(x)
    assert type(out.grad_fn).__name__ == "LoRAReLU2MLPFunctionBackward"
    unfused = mlp.down_proj(F.relu(mlp.up_proj(x)).pow(2))
    assert (out - unfused).abs().max().item() < 1e-5


def _make_lora(in_f, out_f, rank, *, dropout=0.0, use_triton=False):
    base = nn.Linear(in_f, out_f, bias=False)
    return patch_linear_module(base, dim=rank, alpha=rank, dropout=dropout, use_triton=use_triton)


def test_fused_helper_matches_unfused_linear_lora():
    """fused_lora_swiglu_mlp(gate, up, down, x) == down(silu(gate(x)) * up(x)) via separate LinearLoRA."""
    torch.manual_seed(0)
    H, I, R, T = 64, 96, 8, 16
    gate, up = _make_lora(H, I, R), _make_lora(H, I, R)
    down = _make_lora(I, H, R)
    # randomize lora_B (init is zeros) so the LoRA path is exercised
    for m in (gate, up, down):
        nn.init.normal_(m.lora_B.weight, std=0.02)

    x = torch.randn(2, T, H)
    unfused = down(F.silu(gate(x)) * up(x))
    fused = fused_lora_swiglu_mlp(gate, up, down, x)
    assert fused is not None
    assert (fused - unfused).abs().max().item() < 1e-5


def test_fused_helper_declines_on_dropout_and_dora():
    """Helper returns None (caller falls back) when fusion preconditions aren't met."""
    H, I, R = 64, 96, 8
    drop_gate = _make_lora(H, I, R, dropout=0.5)
    up, down = _make_lora(H, I, R), _make_lora(I, H, R)
    drop_gate.train()
    x = torch.randn(2, 16, H)
    assert fused_lora_swiglu_mlp(drop_gate, up, down, x) is None

    # DoRA flag on one projection also declines
    gate = _make_lora(H, I, R)
    gate.use_dora = True
    assert fused_lora_swiglu_mlp(gate, up, down, x) is None


def test_fused_helper_declines_on_quantized_base():
    """QLoRA/quantized base weights are packed buffers, not a 2D (out, in) matrix; the fused path
    must decline so the per-linear (dequantizing) path handles them.

    Regression for AM-435: with a 4-bit-packed base, ``F.linear(x, base_weight)`` in the fused
    forward failed with "mat1 and mat2 shapes cannot be multiplied (Nx4096 and 1x14680064)".
    """
    H, I, R = 64, 96, 8
    up, down = _make_lora(H, I, R), _make_lora(I, H, R)
    x = torch.randn(2, 16, H)

    # (a) packed/flattened base weight (shape != (out_features, in_features))
    gate_packed = _make_lora(H, I, R)
    gate_packed.weight = nn.Parameter(torch.zeros(1, I * H // 2), requires_grad=False)
    assert fused_lora_swiglu_mlp(gate_packed, up, down, x) is None

    # (b) bitsandbytes-style quant_state marker on the module
    gate_qs = _make_lora(H, I, R)
    gate_qs.quant_state = object()
    assert fused_lora_swiglu_mlp(gate_qs, up, down, x) is None


def test_fused_helper_declines_on_meta_weights():
    """FSDP/PP graph construction can leave projection params on meta; direct F.linear must not run."""
    H, I, R = 64, 96, 8
    gate, up, down = _make_lora(H, I, R), _make_lora(H, I, R), _make_lora(I, H, R)
    x = torch.randn(2, 16, H)

    gate.weight = nn.Parameter(torch.empty_like(gate.weight, device="meta"), requires_grad=False)
    assert fused_lora_swiglu_mlp(gate, up, down, x) is None


def test_install_skips_meta_weight_mlp():
    """Do not install a fused MLP wrapper when any projection weight is still on meta."""
    from nemo_automodel.components._peft.lora_mlp import install_fused_lora_mlp

    H, I, R = 64, 96, 8
    mlp = _lora_swiglu_mlp(H, I, R)
    mlp.gate_proj.weight = nn.Parameter(torch.empty_like(mlp.gate_proj.weight, device="meta"), requires_grad=False)

    assert install_fused_lora_mlp(mlp) == 0
    assert not getattr(mlp, "_lora_mlp_fused", False)


def _lora_swiglu_mlp(H, I, R):
    """A moe.layers.MLP (swiglu) with all three projections LoRA-patched (torch path, CPU-safe)."""
    from nemo_automodel.components.moe.layers import MLP

    mlp = MLP(dim=H, inter_dim=I, backend="torch", dtype=torch.float32, activation="swiglu")
    for proj in ("gate_proj", "up_proj", "down_proj"):
        setattr(mlp, proj, patch_linear_module(getattr(mlp, proj), dim=R, alpha=R, use_triton=False))
        nn.init.normal_(getattr(mlp, proj).lora_B.weight, std=0.02)
    return mlp


def test_install_fuses_swiglu_and_relu2_together():
    """install_fused_lora_mlp fuses both a SwiGLU MLP and a ReLU² MLP via their respective paths."""
    from nemo_automodel.components._peft.lora_mlp import install_fused_lora_mlp
    from nemo_automodel.components.moe.layers import MLP

    torch.manual_seed(0)
    H, I, R = 64, 96, 8
    swiglu = _lora_swiglu_mlp(H, I, R)
    relu2 = MLP(dim=H, inter_dim=I, backend="torch", dtype=torch.float32, activation="relu2")
    relu2.up_proj = patch_linear_module(relu2.up_proj, dim=R, alpha=R, use_triton=False)
    relu2.down_proj = patch_linear_module(relu2.down_proj, dim=R, alpha=R, use_triton=False)
    for m in (relu2.up_proj, relu2.down_proj):
        nn.init.normal_(m.lora_B.weight, std=0.02)

    assert install_fused_lora_mlp(nn.ModuleList([swiglu, relu2])) == 2
    assert getattr(swiglu, "_lora_mlp_fused", False) and getattr(relu2, "_lora_mlp_fused", False)

    x = torch.randn(2, 16, H)
    assert type(swiglu(x).grad_fn).__name__ == "LoRASwiGLUMLPFunctionBackward"
    assert type(relu2(x).grad_fn).__name__ == "LoRAReLU2MLPFunctionBackward"
    sw_unfused = swiglu.down_proj(F.silu(swiglu.gate_proj(x)) * swiglu.up_proj(x))
    assert (swiglu(x) - sw_unfused).abs().max().item() < 1e-5


def test_is_silu_swiglu_mlp_detects_hf_style_activation():
    """SiLU detection works even when act_fn is not nn.SiLU (HF SiLUActivation / functional silu)."""
    from nemo_automodel.components._peft.lora_mlp import _is_silu_swiglu_mlp

    class _MLP(nn.Module):
        def __init__(self, act):
            super().__init__()
            self.gate_proj = nn.Linear(8, 16, bias=False)
            self.up_proj = nn.Linear(8, 16, bias=False)
            self.down_proj = nn.Linear(16, 8, bias=False)
            self.act_fn = act

    class _HFSiLUActivation(nn.Module):  # mimics transformers SiLUActivation (not nn.SiLU)
        def forward(self, x):
            return F.silu(x)

    assert _is_silu_swiglu_mlp(_MLP(nn.SiLU()))
    assert _is_silu_swiglu_mlp(_MLP(_HFSiLUActivation()))
    assert _is_silu_swiglu_mlp(_MLP(F.silu))
    assert not _is_silu_swiglu_mlp(_MLP(nn.GELU()))  # GeGLU must be excluded


def test_install_fused_lora_mlp_skips_plain_and_is_idempotent():
    """Plain (non-LoRA) MLPs are left alone, and re-running install does not double-swap."""
    from nemo_automodel.components._peft.lora_mlp import install_fused_lora_mlp
    from nemo_automodel.components.moe.layers import MLP

    plain = MLP(dim=64, inter_dim=96, backend="torch", dtype=torch.float32, activation="swiglu")
    assert install_fused_lora_mlp(plain) == 0  # projections are not LoRA-patched

    lora = _lora_swiglu_mlp(64, 96, 8)
    assert install_fused_lora_mlp(lora) == 1
    assert install_fused_lora_mlp(lora) == 0  # idempotent
