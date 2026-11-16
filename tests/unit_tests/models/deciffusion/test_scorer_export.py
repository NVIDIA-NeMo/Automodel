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

"""Scorer equivalence, checkpoint-contract guard and export round trips on tiny CPU models."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from nemo_automodel.components.models.deciffusion.export import (
    checkpoint_fingerprint,
    export_full_checkpoint,
    merge_lora_checkpoint,
    strip_shared_context,
)
from nemo_automodel.components.models.deciffusion.scorer import (
    full_finetune_trainable,
    gather_head_rows,
    option_logits,
    require_bidirectional_text_config,
)

HIDDEN, VOCAB = 8, 32


class TinyDiffusion(nn.Module):
    """Interface stand-in: bidirectional encoder over all positions and a bias-free diffusion head."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.diffusion_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def get_output_embeddings(self) -> nn.Module:
        return self.diffusion_head

    def forward(self, input_ids: torch.Tensor, output_last_hidden_states_only: bool = False, use_cache: bool = True):
        """Encode token ids.

        Args:
            input_ids: Token ids, shape [batch, sequence].

        Returns:
            Namespace with ``last_hidden_state`` [batch, sequence, hidden] (each position sees the whole sequence).
        """
        h = self.embed(input_ids)
        h = h + self.q_proj(h.mean(dim=1, keepdim=True))
        return SimpleNamespace(last_hidden_state=h)


def _config(**overrides):
    base = {
        "architectures": ["NemotronLabsDiffusionModel"],
        "dlm_paradigm": "bidirectional",
        "mask_token_id": 3,
        "deciffusion_shared_context": True,
    }
    return SimpleNamespace(**{**base, **overrides})


def test_bidirectional_text_checkpoint_is_accepted():
    require_bidirectional_text_config(_config())


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"deciffusion_block_causal": True}, "block-causal"),
        ({"deciffusion_bc_blocks": [4096, 8192]}, "block-causal"),
        ({"dlm_paradigm": "block_diff"}, "bidirectional"),
        ({"dlm_paradigm": "autoregressive"}, "bidirectional"),
        ({"vision_config": {"hidden_size": 8}}, "vision"),
        ({"quantization_config": {"quant_method": "modelopt"}}, "quantized"),
        ({"architectures": ["LlamaForCausalLM"]}, "architectures"),
        ({"mask_token_id": None}, "mask_token_id"),
    ],
)
def test_unsupported_variants_fail_clearly(overrides, error):
    with pytest.raises(ValueError, match=error):
        require_bidirectional_text_config(_config(**overrides))


def test_restricted_rows_equal_the_full_head_at_the_mask_and_train_only_the_backbone():
    torch.manual_seed(0)
    model = TinyDiffusion()
    model.diffusion_head.weight.requires_grad_(False)
    codes = [7, 9, 11]
    rows = gather_head_rows(model, codes)
    lookup = {token: rows[i] for i, token in enumerate(codes)}
    prompt = [5, 6, 8, 10]
    logits = option_logits(model, prompt, codes, mask_token_id=3, rows=lookup, device=torch.device("cpu"))
    full = model.diffusion_head(model(torch.tensor([[*prompt, 3]])).last_hidden_state[0, -1])[codes]
    torch.testing.assert_close(logits, full)
    assert logits.dtype == torch.float32
    logits.logsumexp(-1).backward()
    assert model.q_proj.weight.grad is not None and model.diffusion_head.weight.grad is None


def test_head_with_bias_is_rejected():
    model = TinyDiffusion()
    model.diffusion_head = nn.Linear(HIDDEN, VOCAB, bias=True)
    with pytest.raises(ValueError, match="bias"):
        gather_head_rows(model, [1])


def _write_base(root, seed=0, shared=True):
    torch.manual_seed(seed)
    root.mkdir(parents=True)
    weights = {
        "encoder.layers.0.self_attn.q_proj.weight": torch.randn(HIDDEN, HIDDEN).to(torch.bfloat16),
        "diffusion_head.weight": torch.randn(VOCAB, HIDDEN).to(torch.bfloat16),
    }
    save_file(weights, str(root / "model.safetensors"))
    config = {
        "architectures": ["NemotronLabsDiffusionModel"],
        **({"deciffusion_shared_context": True} if shared else {}),
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "tokenizer.json").write_text("{}")
    return weights


def _write_adapter(root, rank=2, alpha=4, **extra):
    torch.manual_seed(1)
    root.mkdir(parents=True)
    a, b = torch.randn(rank, HIDDEN), torch.randn(HIDDEN, rank)
    prefix = "base_model.model.encoder.layers.0.self_attn.q_proj"
    save_file({f"{prefix}.lora_A.weight": a, f"{prefix}.lora_B.weight": b}, str(root / "adapter_model.safetensors"))
    (root / "adapter_config.json").write_text(json.dumps({"r": rank, "lora_alpha": alpha, **extra}))
    return a, b


def test_lora_merge_reproduces_the_adapted_linear_and_strips_shared_context(tmp_path):
    base = _write_base(tmp_path / "base")
    a, b = _write_adapter(tmp_path / "adapter")
    receipt = merge_lora_checkpoint(tmp_path / "base", tmp_path / "adapter", tmp_path / "out")
    merged = load_file(str(tmp_path / "out/model.safetensors"))
    name = "encoder.layers.0.self_attn.q_proj.weight"
    expected = (base[name].float() + 2.0 * (b @ a)).to(torch.bfloat16)
    torch.testing.assert_close(merged[name], expected, atol=0, rtol=0)
    torch.testing.assert_close(merged["diffusion_head.weight"], base["diffusion_head.weight"], atol=0, rtol=0)
    assert receipt["lora_modules"] == 1 and receipt["scale"] == 2.0
    assert "deciffusion_shared_context" not in json.loads((tmp_path / "out/config.json").read_text())
    assert (tmp_path / "out/tokenizer.json").exists()
    # The merged linear computes what the adapted linear computed: x W^T + scale * x A^T B^T.
    x = torch.randn(5, HIDDEN)
    adapted = x @ base[name].float().T + 2.0 * (x @ a.T @ b.T)
    torch.testing.assert_close(x @ merged[name].float().T, adapted, atol=0.15, rtol=0.02)  # bf16 storage


@pytest.mark.parametrize(
    "extra,error",
    [({"use_dora": True}, "DoRA"), ({"rank_pattern": {"q": 4}}, "rank_pattern")],
)
def test_unsupported_adapters_are_rejected(tmp_path, extra, error):
    _write_base(tmp_path / "base")
    _write_adapter(tmp_path / "adapter", **extra)
    with pytest.raises(ValueError, match=error):
        merge_lora_checkpoint(tmp_path / "base", tmp_path / "adapter", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_adapter_for_a_different_base_is_rejected(tmp_path):
    root = tmp_path / "base"
    root.mkdir()
    save_file(
        {"encoder.layers.0.self_attn.q_proj.weight": torch.zeros(HIDDEN + 1, HIDDEN)}, str(root / "model.safetensors")
    )
    (root / "config.json").write_text("{}")
    _write_adapter(tmp_path / "adapter")
    with pytest.raises(ValueError, match="delta"):
        merge_lora_checkpoint(root, tmp_path / "adapter", tmp_path / "out")
    other = tmp_path / "other"
    other.mkdir()
    save_file({"diffusion_head.weight": torch.zeros(VOCAB, HIDDEN)}, str(other / "model.safetensors"))
    with pytest.raises(ValueError, match="different base"):
        merge_lora_checkpoint(other, tmp_path / "adapter", tmp_path / "out2")


def test_existing_output_is_never_overwritten(tmp_path):
    _write_base(tmp_path / "base")
    _write_adapter(tmp_path / "adapter")
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        merge_lora_checkpoint(tmp_path / "base", tmp_path / "adapter", tmp_path / "out")


def test_fingerprint_separates_same_shape_bases(tmp_path):
    _write_base(tmp_path / "a", seed=0)
    _write_base(tmp_path / "b", seed=1)
    assert checkpoint_fingerprint(tmp_path / "a") == checkpoint_fingerprint(tmp_path / "a")
    assert checkpoint_fingerprint(tmp_path / "a") != checkpoint_fingerprint(tmp_path / "b")
    with pytest.raises(FileNotFoundError):
        checkpoint_fingerprint(tmp_path)


def test_full_export_casts_to_base_dtypes_and_rejects_mismatched_tensors(tmp_path):
    base = _write_base(tmp_path / "base")
    trained_dir = tmp_path / "consolidated"
    trained_dir.mkdir()
    trained = {k: v.float() + 0.5 for k, v in base.items()}
    save_file(trained, str(trained_dir / "model.safetensors"))
    export_full_checkpoint(tmp_path / "base", trained_dir, tmp_path / "out")
    out = load_file(str(tmp_path / "out/model.safetensors"))
    for name, tensor in trained.items():
        assert out[name].dtype == torch.bfloat16
        torch.testing.assert_close(out[name], tensor.to(torch.bfloat16), atol=0, rtol=0)
    assert not strip_shared_context(tmp_path / "out")  # already removed by the export
    save_file({"diffusion_head.weight": torch.zeros(VOCAB, HIDDEN)}, str(trained_dir / "model.safetensors"))
    with pytest.raises(ValueError, match="do not match"):
        export_full_checkpoint(tmp_path / "base", trained_dir, tmp_path / "out2")


@pytest.mark.parametrize(
    "name,trainable",
    [
        ("encoder.layers.3.mlp.up_proj.weight", True),
        ("encoder.layers.0._checkpoint_wrapped_module.self_attn.q_proj.weight", True),
        ("encoder.norm.weight", True),
        ("encoder.embed_tokens.weight", False),
        ("diffusion_head.weight", False),
    ],
)
def test_full_finetuning_scope(name, trainable):
    assert full_finetune_trainable(name) is trainable
