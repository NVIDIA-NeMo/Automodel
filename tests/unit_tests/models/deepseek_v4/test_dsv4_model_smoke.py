# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Smoke tests for DeepSeek V4: forward + backward pass with a tiny config.

Run on the cluster (CPU-only, no checkpoint needed):
    PYTHONPATH=/path/to/Automodel_lao python -m pytest \
        tests/unit_tests/models/deepseek_v4/test_dsv4_model_smoke.py -v -s

Or as a standalone script:
    PYTHONPATH=/path/to/Automodel_lao python \
        tests/unit_tests/models/deepseek_v4/test_dsv4_model_smoke.py
"""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4ForCausalLM

# ``MoE.forward`` (in ``nemo_automodel/components/moe/layers.py``)
# unconditionally creates a ``torch.cuda.Stream()`` when the model has a
# shared expert.  On the CPU-only CI image (``L0_Unit_Tests_CPU``) that
# raises ``AcceleratorError: CUDA driver version is insufficient`` even
# though the model parameters live on the CPU.  Gate the forward-running
# smoke tests on CUDA availability — the rest of the file (HC params,
# weight shapes, attn-sink dtype) is pure metadata and runs everywhere.
_REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MoE.forward unconditionally allocates torch.cuda.Stream() for shared experts",
)


def _tiny_config(**overrides) -> DeepseekV4Config:
    """Tiny V4 config: fits in ~1 GB CPU RAM, exercises all code paths."""
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        max_position_embeddings=128,
        rope_theta=10000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 16,
            "original_max_position_embeddings": 64,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        hc_mult=4,
        num_hash_layers=2,
        compress_ratios=[0, 0, 4, 0],  # 4 layers: full, full, csa(fallback), full
        sliding_window=16,
        num_nextn_predict_layers=0,
        rms_norm_eps=1e-6,
        torch_dtype="float32",
    )
    defaults.update(overrides)
    return DeepseekV4Config(**defaults)


def _make_model(config: DeepseekV4Config) -> DeepseekV4ForCausalLM:
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
        dispatcher="torch",
        experts="torch_mm",
    )
    model = DeepseekV4ForCausalLM(config, backend=backend)
    # The model mixes per-module dtype defaults (``embed_tokens`` is built bf16
    # via the now-deprecated ``config.torch_dtype`` path; bare ``nn.Linear``
    # sub-modules pick up the global fp32 default).  Force everything to a
    # single fp32 dtype for the smoke test so matmuls don't hit dtype mismatch.
    # ``DeepseekV4HyperConnection`` keeps its parameters in fp32 explicitly —
    # ``model.float()`` is a no-op for those.
    model = model.float()
    # Zero all float params: Gate/GroupedExperts use torch.empty (uninitialized memory
    # that may contain NaN bit patterns). initialize_weights() is not called in smoke tests.
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point():
                p.zero_()
    model.eval()
    return model


class TestDeepseekV4ModelSmoke:
    @_REQUIRES_CUDA
    def test_forward_shape(self):
        """Forward pass produces logits of the right shape."""
        cfg = _tiny_config()
        model = _make_model(cfg)

        bsz, seq = 2, 8
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq))
        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (bsz, seq, cfg.vocab_size), f"unexpected shape {logits.shape}"
        assert not logits.isnan().any(), "logits contain NaN"
        assert not logits.isinf().any(), "logits contain Inf"

    @_REQUIRES_CUDA
    def test_backward(self):
        """Backward pass produces gradients on all trainable parameters."""
        cfg = _tiny_config()
        model = _make_model(cfg)
        model.train()

        bsz, seq = 2, 8
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq))
        labels = torch.randint(0, cfg.vocab_size, (bsz, seq))

        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), labels.view(-1))
        loss.backward()

        # The Indexer's parameters (``compressor.indexer.*``) feed the top-k
        # selection of compressed positions; the indices are integer-valued
        # so they are non-differentiable by design (this matches the official
        # DSV4 inference reference, where those weights are frozen at load
        # time).  Exclude them from the gradient-presence check.
        missing_grad = [
            n
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None and ".compressor.indexer." not in n
        ]
        assert not missing_grad, f"parameters missing gradients: {missing_grad}"
        assert loss.item() > 0, "loss is zero or negative"

    def test_hc_params_registered(self):
        """HC params are registered on every block via the
        ``DeepseekV4HyperConnection`` submodules ``attn_hc`` / ``ffn_hc``
        (the flat ``hc_attn_*`` / ``hc_ffn_*`` keys exist only in the HF
        checkpoint and are renamed on load).
        """
        cfg = _tiny_config()
        model = _make_model(cfg)

        hc_param_names = {"fn", "base", "scale"}
        for layer_id in range(cfg.num_hidden_layers):
            block = model.model.layers[str(layer_id)]
            for sub_name in ("attn_hc", "ffn_hc"):
                hc = getattr(block, sub_name)
                for pname in hc_param_names:
                    assert hasattr(hc, pname), f"layer {layer_id} missing {sub_name}.{pname}"
                    p = getattr(hc, pname)
                    assert p.dtype == torch.float32, f"{sub_name}.{pname} dtype {p.dtype} != float32"

    def test_hc_attn_fn_shape(self):
        """HC ``fn`` matrix has shape ``(mix_hc, hc_mult * hidden_size)``."""
        cfg = _tiny_config()
        model = _make_model(cfg)

        mix_hc = (2 + cfg.hc_mult) * cfg.hc_mult  # 24
        hc_dim = cfg.hc_mult * cfg.hidden_size  # 256 for tiny

        block = model.model.layers["0"]
        fn = block.attn_hc.fn
        assert fn.shape == (mix_hc, hc_dim), f"attn_hc.fn shape {fn.shape} != ({mix_hc}, {hc_dim})"

    def test_wo_a_weight_shape(self):
        """GroupedOutputProjection has the correct weight shape."""
        cfg = _tiny_config()
        model = _make_model(cfg)

        block = model.model.layers["0"]
        attn = block.self_attn
        n_hpg = cfg.num_attention_heads // cfg.o_groups
        expected = (cfg.o_groups * cfg.o_lora_rank, n_hpg * cfg.head_dim)
        actual = attn.wo_a.weight.shape
        assert actual == expected, f"wo_a weight shape {actual} != {expected}"

    def test_attn_sink_shape_and_dtype(self):
        """``sinks`` is a per-head float32 scalar vector consumed by
        ``eager_attention_with_sink`` (HF PR 45616 named it ``sinks``;
        the HF checkpoint key ``attn_sink`` is renamed to it on load).
        """
        cfg = _tiny_config()
        model = _make_model(cfg)

        block = model.model.layers["0"]
        sink = block.self_attn.sinks
        assert sink.shape == (cfg.num_attention_heads,), f"unexpected shape {sink.shape}"
        assert sink.dtype == torch.float32, f"unexpected dtype {sink.dtype}"

    @_REQUIRES_CUDA
    def test_different_seq_lengths(self):
        """Model runs without error for several sequence lengths."""
        cfg = _tiny_config()
        model = _make_model(cfg)

        # ``Float32RMSNorm.forward`` is ``@torch.compile``'d.  Iterating four
        # different sequence lengths through ``compress_ratio>0`` layers (which
        # build masks of shape ``[..., n_pooled]`` that varies per call)
        # explodes Dynamo's per-shape recompile cache.  Disable the compile
        # for this test — we only care about shape correctness here.
        import torch._dynamo as _dynamo

        with _dynamo.config.patch(recompile_limit=64), _dynamo.config.patch(fail_on_recompile_limit_hit=False):
            for seq in [1, 4, 16, 32]:
                input_ids = torch.randint(0, cfg.vocab_size, (1, seq))
                with torch.no_grad():
                    logits = model(input_ids)
                assert logits.shape == (1, seq, cfg.vocab_size)


if __name__ == "__main__":
    import sys

    tests = TestDeepseekV4ModelSmoke()
    cases = [
        ("forward shape", tests.test_forward_shape),
        ("backward + gradients", tests.test_backward),
        ("HC params registered", tests.test_hc_params_registered),
        ("HC attn_fn shape", tests.test_hc_attn_fn_shape),
        ("wo_a weight shape", tests.test_wo_a_weight_shape),
        ("attn_sink shape/dtype", tests.test_attn_sink_shape_and_dtype),
        ("different seq lengths", tests.test_different_seq_lengths),
    ]

    failed = []
    for name, fn in cases:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed.append(name)

    print()
    if failed:
        print(f"FAILED: {len(failed)}/{len(cases)} tests")
        sys.exit(1)
    else:
        print(f"All {len(cases)} tests passed.")
