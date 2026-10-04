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

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from nemo_automodel.components.models.common.utils import (
    _get_fp32_module_keywords,
    _get_strict_fp32_module_keywords,
    _has_dtensor_params,
    cast_frozen_modules_to_compute_dtype,
    cast_model_to_dtype,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleModel(nn.Module):
    """A small model for dtype testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.linear2 = nn.Linear(4, 2)


class ModelWithFp32Modules(nn.Module):
    """Model that declares modules to keep in fp32 (HF-style)."""

    _keep_in_fp32_modules = ["norm"]

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 2)


class ModelWithStrictFp32(nn.Module):
    """Model that declares strict fp32 modules."""

    _keep_in_fp32_modules_strict = ["head"]

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)


class ModelWithStrictFp32Parameter(nn.Module):
    """Model that declares one strict fp32 parameter by qualified parameter name."""

    _keep_in_fp32_modules_strict = ["mixer.scale"]

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.mixer = nn.Module()
        self.mixer.scale = nn.Parameter(torch.ones(4))


class ModelWithStrictFp32Buffer(nn.Module):
    """Model that declares one strict fp32 buffer by qualified buffer name."""

    _keep_in_fp32_modules_strict = ["router.e_score_correction_bias"]

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.router = nn.Module()
        self.router.register_buffer(
            "e_score_correction_bias",
            torch.tensor([1.001, -2.003, 0.3333, 17.125], dtype=torch.float32),
        )


class _CheckpointWrappedLeaf(nn.Module):
    """Small wrapper that exposes the same child name as activation checkpointing."""

    def __init__(self, wrapped_module):
        super().__init__()
        self._checkpoint_wrapped_module = wrapped_module


class ModelWithCheckpointWrappedDirectFp32Buffer(nn.Module):
    """Model with an fp32 buffer registered directly on a checkpoint-wrapped leaf."""

    _keep_in_fp32_modules = ["attention_sink_bias"]

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        attention = nn.Module()
        attention.register_buffer(
            "attention_sink_bias",
            torch.tensor([1.001, -2.003, 0.3333, 17.125], dtype=torch.float32),
        )
        self.self_attn = _CheckpointWrappedLeaf(attention)


class ModelWithCheckpointWrapperAliasFp32Buffer(nn.Module):
    """Model whose named_buffers emits a forwarded checkpoint-wrapper alias."""

    _keep_in_fp32_modules = ["e_score_correction_bias"]

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.mlp = nn.Module()
        self.mlp.gate = nn.Module()
        self.mlp.gate.register_buffer(
            "e_score_correction_bias",
            torch.tensor([1.001, -2.003, 0.3333, 17.125], dtype=torch.float32),
        )

    def named_buffers(self, prefix="", recurse=True, remove_duplicate=True):
        for name, buffer in super().named_buffers(prefix, recurse, remove_duplicate):
            if name == "mlp.gate.e_score_correction_bias":
                yield "mlp._checkpoint_wrapped_module.gate.e_score_correction_bias", buffer
            else:
                yield name, buffer


class ModelWithBothFp32Attrs(nn.Module):
    """Model with both _keep_in_fp32_modules and _keep_in_fp32_modules_strict."""

    _keep_in_fp32_modules = ["norm"]
    _keep_in_fp32_modules_strict = ["head"]

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 2)


# ---------------------------------------------------------------------------
# Tests for _get_fp32_module_keywords()
# ---------------------------------------------------------------------------


class TestGetFp32ModuleKeywords:
    def test_no_attributes(self):
        model = SimpleModel()
        assert _get_fp32_module_keywords(model) == []

    def test_keep_in_fp32_modules(self):
        model = ModelWithFp32Modules()
        assert _get_fp32_module_keywords(model) == ["norm"]

    def test_keep_in_fp32_modules_strict(self):
        model = ModelWithStrictFp32()
        assert _get_fp32_module_keywords(model) == ["head"]

    def test_keep_in_fp32_modules_strict_tuple(self):
        class Model(nn.Module):
            _keep_in_fp32_modules_strict = ("head",)

            def __init__(self):
                super().__init__()

        model = Model()
        assert _get_fp32_module_keywords(model) == ["head"]
        assert _get_strict_fp32_module_keywords(model) == ["head"]

    def test_keep_in_fp32_modules_strict_set(self):
        class Model(nn.Module):
            _keep_in_fp32_modules_strict = {"head"}

            def __init__(self):
                super().__init__()

        model = Model()
        assert _get_fp32_module_keywords(model) == ["head"]
        assert _get_strict_fp32_module_keywords(model) == ["head"]

    def test_both_attributes_deduped(self):
        model = ModelWithBothFp32Attrs()
        keywords = _get_fp32_module_keywords(model)
        assert "head" in keywords
        assert "norm" in keywords
        assert len(keywords) == 2

    def test_duplicates_removed(self):
        class Model(nn.Module):
            _keep_in_fp32_modules = ["norm", "head"]
            _keep_in_fp32_modules_strict = ["head"]

            def __init__(self):
                super().__init__()

        model = Model()
        keywords = _get_fp32_module_keywords(model)
        assert keywords.count("head") == 1

    def test_none_attributes_ignored(self):
        class Model(nn.Module):
            _keep_in_fp32_modules = None

            def __init__(self):
                super().__init__()

        model = Model()
        assert _get_fp32_module_keywords(model) == []

    def test_set_and_tuple_accepted(self):
        # HuggingFace's PreTrainedModel.__init__ converts a class-level list into an
        # instance-level set; accept set (and tuple) so keep-fp32 keywords are not
        # silently dropped (regression: this no-op'd the gemma4_moe/diffusion_gemma fix).
        class Model(nn.Module):
            _keep_in_fp32_modules = {"norm"}
            _keep_in_fp32_modules_strict = ("head",)

            def __init__(self):
                super().__init__()

        assert set(_get_fp32_module_keywords(Model())) == {"norm", "head"}


# ---------------------------------------------------------------------------
# Tests for cast_model_to_dtype()
# ---------------------------------------------------------------------------


class TestCastModelToDtype:
    def test_simple_model_cast_to_bf16(self):
        model = SimpleModel()
        assert model.linear1.weight.dtype == torch.float32

        cast_model_to_dtype(model, torch.bfloat16)
        assert model.linear1.weight.dtype == torch.bfloat16
        assert model.norm.weight.dtype == torch.bfloat16
        assert model.linear2.weight.dtype == torch.bfloat16

    def test_fp32_modules_preserved(self):
        model = ModelWithFp32Modules()
        cast_model_to_dtype(model, torch.bfloat16)

        assert model.norm.weight.dtype == torch.float32
        assert model.linear.weight.dtype == torch.bfloat16
        assert model.head.weight.dtype == torch.bfloat16

    def test_strict_fp32_modules_preserved(self):
        model = ModelWithStrictFp32()
        cast_model_to_dtype(model, torch.bfloat16)

        assert model.head.weight.dtype == torch.float32
        assert model.linear.weight.dtype == torch.bfloat16

    def test_strict_fp32_modules_preserved_from_tuple(self):
        class Model(nn.Module):
            _keep_in_fp32_modules_strict = ("head",)

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)
                self.head = nn.Linear(4, 2)

        model = Model()
        cast_model_to_dtype(model, torch.bfloat16)

        assert model.head.weight.dtype == torch.float32
        assert model.linear.weight.dtype == torch.bfloat16

    def test_strict_fp32_parameters_preserved(self):
        model = ModelWithStrictFp32Parameter()
        original_scale = torch.tensor([1.001, -2.003, 0.3333, 17.125], dtype=torch.float32)
        with torch.no_grad():
            model.mixer.scale.copy_(original_scale)

        cast_model_to_dtype(model, torch.bfloat16)

        assert model.mixer.scale.dtype == torch.float32
        assert torch.equal(model.mixer.scale, original_scale)
        assert not torch.equal(model.mixer.scale, original_scale.to(torch.bfloat16).float())
        assert model.linear.weight.dtype == torch.bfloat16

    def test_strict_fp32_buffers_preserve_values(self):
        model = ModelWithStrictFp32Buffer()
        original_bias = model.router.e_score_correction_bias.clone()

        cast_model_to_dtype(model, torch.bfloat16)

        assert model.router.e_score_correction_bias.dtype == torch.float32
        assert torch.equal(model.router.e_score_correction_bias, original_bias)
        assert not torch.equal(model.router.e_score_correction_bias, original_bias.to(torch.bfloat16).float())
        assert model.linear.weight.dtype == torch.bfloat16

    def test_checkpoint_wrapper_alias_fp32_buffer_preserves_values(self):
        model = ModelWithCheckpointWrapperAliasFp32Buffer()
        original_bias = model.mlp.gate.e_score_correction_bias.clone()

        cast_model_to_dtype(model, torch.bfloat16)

        assert model.mlp.gate.e_score_correction_bias.dtype == torch.float32
        assert torch.equal(model.mlp.gate.e_score_correction_bias, original_bias)
        assert not torch.equal(model.mlp.gate.e_score_correction_bias, original_bias.to(torch.bfloat16).float())
        assert model.linear.weight.dtype == torch.bfloat16

    def test_checkpoint_wrapped_direct_fp32_buffer_preserves_values(self):
        model = ModelWithCheckpointWrappedDirectFp32Buffer()
        wrapped_attention = model.self_attn._checkpoint_wrapped_module
        original_bias = wrapped_attention.attention_sink_bias.clone()

        cast_model_to_dtype(model, torch.bfloat16)

        assert "attention_sink_bias" not in model.self_attn._buffers
        assert wrapped_attention.attention_sink_bias.dtype == torch.float32
        assert torch.equal(wrapped_attention.attention_sink_bias, original_bias)
        assert not torch.equal(wrapped_attention.attention_sink_bias, original_bias.to(torch.bfloat16).float())
        assert model.linear.weight.dtype == torch.bfloat16

    def test_both_fp32_attrs_preserved(self):
        model = ModelWithBothFp32Attrs()
        cast_model_to_dtype(model, torch.bfloat16)

        assert model.norm.weight.dtype == torch.float32
        assert model.head.weight.dtype == torch.float32
        assert model.linear.weight.dtype == torch.bfloat16

    def test_no_fp32_modules_all_cast(self):
        model = SimpleModel()
        cast_model_to_dtype(model, torch.bfloat16)

        for p in model.parameters():
            assert p.dtype == torch.bfloat16

    def test_fp16_dtype(self):
        model = SimpleModel()
        cast_model_to_dtype(model, torch.float16)

        for p in model.parameters():
            assert p.dtype == torch.float16

    def test_set_valued_keep_in_fp32_preserved(self):
        # Mirrors HF converting _keep_in_fp32_modules (list) to a set on the instance —
        # the gemma4_moe/diffusion_gemma case. cast_model_to_dtype must still restore it.
        class M(nn.Module):
            _keep_in_fp32_modules = {"norm"}

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)
                self.norm = nn.LayerNorm(4)

        model = M()
        cast_model_to_dtype(model, torch.bfloat16)
        assert model.norm.weight.dtype == torch.float32
        assert model.linear.weight.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Tests for sharded (DTensor) models
# ---------------------------------------------------------------------------


@pytest.fixture
def cpu_mesh():
    """Single-rank gloo mesh so parameters can be turned into real DTensors on CPU."""
    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def _shard_parameters(model: nn.Module, mesh) -> None:
    """Replace every parameter with a Shard(0) DTensor of the same dtype and value."""
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            sharded = distribute_tensor(param.detach(), mesh, [Shard(0)])
            module.register_parameter(name, nn.Parameter(sharded, requires_grad=param.requires_grad))


class TestShardedCasting:
    def test_has_dtensor_params_false_for_plain_model(self):
        model = SimpleModel()
        assert not _has_dtensor_params(model)

    def test_has_dtensor_params_true_after_sharding(self, cpu_mesh):
        model = SimpleModel()
        _shard_parameters(model, cpu_mesh)
        assert _has_dtensor_params(model)

    def test_fp32_storage_cast_to_fp32_leaves_sharded_params_untouched(self, cpu_mesh):
        """The post-shard ``initialize_weights(dtype=<storage dtype>)`` call is a no-op on params."""
        model = ModelWithStrictFp32Parameter()
        model.mixer.scale.data = torch.tensor([1.001, -2.003, 0.3333, 17.125])
        _shard_parameters(model, cpu_mesh)
        before = {name: param for name, param in model.named_parameters()}

        cast_model_to_dtype(model, torch.float32)

        for name, param in model.named_parameters():
            assert param is before[name]
            assert param.dtype == torch.float32
        torch.testing.assert_close(model.mixer.scale.full_tensor(), torch.tensor([1.001, -2.003, 0.3333, 17.125]))

    def test_bf16_storage_cast_restores_fp32_buffers_of_sharded_model(self, cpu_mesh):
        """Buffers are never sharded, so fp32 buffers are still restored after FSDP wrapping."""
        model = ModelWithStrictFp32Buffer().to(torch.bfloat16)
        model.router.e_score_correction_bias = torch.tensor([1.001, -2.003, 0.3333, 17.125], dtype=torch.float32)
        _shard_parameters(model, cpu_mesh)

        cast_model_to_dtype(model, torch.bfloat16)

        assert model.linear.weight.dtype == torch.bfloat16
        assert model.router.e_score_correction_bias.dtype == torch.float32
        torch.testing.assert_close(model.router.e_score_correction_bias, torch.tensor([1.001, -2.003, 0.3333, 17.125]))

    def test_mixed_storage_sharded_fp32_param_raises(self, cpu_mesh):
        """A sharded fp32-contract param inside a bf16 model is the unsupported mixed-storage layout."""
        model = ModelWithStrictFp32Parameter()
        model.linear.to(torch.bfloat16)
        _shard_parameters(model, cpu_mesh)

        with pytest.raises(RuntimeError, match="mixer.scale is an fp32-contract parameter"):
            cast_model_to_dtype(model, torch.bfloat16)


class _VLMLike(nn.Module):
    """A frozen tower + trainable backbone, mirroring the VLM finetune layout."""

    def __init__(self, keep_fp32=None):
        super().__init__()
        if keep_fp32 is not None:
            self._keep_in_fp32_modules = keep_fp32
        # Frozen vision tower (excluded from FSDP) with a norm submodule.
        self.vision = nn.Module()
        self.vision.proj = nn.Linear(4, 4)
        self.vision.norm = nn.LayerNorm(4)
        self.vision.register_buffer("pos", torch.zeros(4))
        for p in self.vision.parameters():
            p.requires_grad_(False)
        # Trainable language backbone.
        self.language = nn.Linear(4, 2)


class TestCastFrozenModulesToComputeDtype:
    def test_frozen_tower_cast_trainable_untouched(self):
        model = _VLMLike()
        # Whole model starts fp32.
        assert model.vision.proj.weight.dtype == torch.float32
        assert model.language.weight.dtype == torch.float32

        cast_frozen_modules_to_compute_dtype(model, torch.bfloat16)

        # Frozen tower (params + buffers) cast to the compute dtype.
        assert model.vision.proj.weight.dtype == torch.bfloat16
        assert model.vision.norm.weight.dtype == torch.bfloat16
        assert model.vision.pos.dtype == torch.bfloat16
        # Trainable backbone is left for FSDP to cast.
        assert model.language.weight.dtype == torch.float32

    def test_none_compute_dtype_is_noop(self):
        model = _VLMLike()
        cast_frozen_modules_to_compute_dtype(model, None)
        assert model.vision.proj.weight.dtype == torch.float32

    def test_respects_keep_in_fp32_modules(self):
        model = _VLMLike(keep_fp32=["norm"])
        cast_frozen_modules_to_compute_dtype(model, torch.bfloat16)

        assert model.vision.proj.weight.dtype == torch.bfloat16
        # Frozen norm is pinned to fp32 even though the rest of the tower is bf16.
        assert model.vision.norm.weight.dtype == torch.float32

    def test_already_compute_dtype_noop(self):
        model = _VLMLike()
        model.vision.to(torch.bfloat16)
        cast_frozen_modules_to_compute_dtype(model, torch.bfloat16)
        assert model.vision.proj.weight.dtype == torch.bfloat16

    def test_fully_trainable_model_untouched(self):
        model = SimpleModel()  # all params require grad
        cast_frozen_modules_to_compute_dtype(model, torch.bfloat16)
        for p in model.parameters():
            assert p.dtype == torch.float32

    def test_frozen_weight_cast_with_trainable_adapter_subtree(self):
        model = nn.Linear(4, 4)
        model.weight.requires_grad_(False)
        model.bias.requires_grad_(False)
        model.adapter = nn.Linear(4, 2, bias=False)

        cast_frozen_modules_to_compute_dtype(model, torch.bfloat16)

        assert model.weight.dtype == torch.bfloat16
        assert model.bias.dtype == torch.bfloat16
        assert model.adapter.weight.dtype == torch.bfloat16

    def test_sharded_frozen_param_skipped_but_buffer_cast(self, monkeypatch):
        """A sharded (DTensor) frozen param is left to FSDP, but its fp32 buffers are still cast.

        This mirrors the sharded frozen-tower case (e.g. gemma4 vision tower in the root
        FSDP unit): FSDP all-gathers the params to the compute dtype itself, but never casts
        buffers, so an fp32 buffer would promote bf16 activations back to fp32. We simulate a
        DTensor param with a ``nn.Parameter`` subclass and patch the ``DTensor`` symbol the
        function imports at call time.
        """
        import torch.distributed.tensor as dt_mod

        class _FakeDTensor(nn.Parameter):
            pass

        monkeypatch.setattr(dt_mod, "DTensor", _FakeDTensor, raising=False)

        model = _VLMLike()
        # Mark the frozen proj weight as a "sharded" param (DTensor-like instance).
        model.vision.proj.weight = _FakeDTensor(model.vision.proj.weight.data, requires_grad=False)

        cast_frozen_modules_to_compute_dtype(model, torch.bfloat16)

        # Sharded param is left untouched (FSDP would down-cast it at all-gather).
        assert model.vision.proj.weight.dtype == torch.float32
        # Plain frozen param in the same subtree is still cast.
        assert model.vision.norm.weight.dtype == torch.bfloat16
        # Frozen buffer is always cast (never FSDP-managed) -- the actual fix.
        assert model.vision.pos.dtype == torch.bfloat16


class TestRopeBufferPreserved:
    """Regression: a model-wide bf16 cast must not round rotary frequency buffers.

    ``nn.Module.to(bf16)`` rounds floating-point buffers, including a rotary
    embedding's (non-persistent) ``inv_freq`` / ``freqs_cis``. Building cos/sin from a
    bf16-rounded buffer degrades RoPE precision vs HF (which keeps it fp32) and shows
    up as a large logit/KL divergence when a checkpoint is reloaded in vanilla HF.
    Models guard against this by listing the rotary module (or buffer) in
    ``_keep_in_fp32_modules``; these tests pin that ``cast_model_to_dtype`` honors it
    for non-persistent rope buffers (the mechanism the per-model fixes rely on).
    """

    @staticmethod
    def _rope_model(keep, *, buffer_name="inv_freq", module_name="rotary_emb"):
        class Rope(nn.Module):
            def __init__(self):
                super().__init__()
                # Non-persistent, exactly like the real rotary frequency buffers.
                self.register_buffer(buffer_name, torch.arange(0, 8, 2, dtype=torch.float32), persistent=False)

        class M(nn.Module):
            _keep_in_fp32_modules = keep

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)
                setattr(self, module_name, Rope())

        return M()

    def test_rotary_emb_keyword_keeps_inv_freq_fp32(self):
        # deepseek_v4 / mimo_v2_flash / gemma4_moe / diffusion_gemma case.
        model = self._rope_model(["rotary_emb"])
        cast_model_to_dtype(model, torch.bfloat16)
        assert model.rotary_emb.inv_freq.dtype == torch.float32  # protected
        assert model.linear.weight.dtype == torch.bfloat16  # everything else cast

    def test_unprotected_inv_freq_is_rounded(self):
        # Without the keep-list entry the bug reproduces: inv_freq is rounded to bf16.
        model = self._rope_model([])
        cast_model_to_dtype(model, torch.bfloat16)
        assert model.rotary_emb.inv_freq.dtype == torch.bfloat16

    def test_inv_freq_keyword_protects_non_rotary_named_module(self):
        # minimax_m3_vl vision tower: inv_freq lives on a module not named "rotary_emb",
        # so the "inv_freq" buffer-name keyword is what pins it fp32.
        model = self._rope_model(["inv_freq"], module_name="vision_tower")
        cast_model_to_dtype(model, torch.bfloat16)
        assert model.vision_tower.inv_freq.dtype == torch.float32

    def test_freqs_cis_keyword_protects_buffer(self):
        # deepseek_v3 / kimi_k25_vl: the real-valued frequency buffer is named "freqs_cis".
        model = self._rope_model(["freqs_cis"], buffer_name="freqs_cis", module_name="model")
        cast_model_to_dtype(model, torch.bfloat16)
        assert model.model.freqs_cis.dtype == torch.float32
