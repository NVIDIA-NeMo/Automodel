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

"""Unit tests for the NeMo AutoModel Inkling MoE VLM implementation."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.nn.functional as F
from transformers.models.inkling.modeling_inkling import (  # noqa: E402
    InklingForConditionalGeneration as HFInklingForConditionalGeneration,
)
from transformers.models.inkling.modeling_inkling import InklingMoE as HFInklingMoE  # noqa: E402

from nemo_automodel.components.datasets.vlm.collate_fns import (  # noqa: E402
    build_labels_from_template,
    default_collate_fn,
)
from nemo_automodel.components.models.common import BackendConfig  # noqa: E402
from nemo_automodel.components.models.inkling.layers import InklingDenseMLP, InklingMoE  # noqa: E402
from nemo_automodel.components.models.inkling.model import InklingForConditionalGeneration  # noqa: E402
from nemo_automodel.components.models.inkling.state_dict_adapter import _interleave  # noqa: E402

from .parity_check_inkling import build_tiny_config  # noqa: E402


def _build_models():
    cfg = build_tiny_config()
    hf = HFInklingForConditionalGeneration(cfg).to(dtype=torch.float32).eval()
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", experts="torch")
    nemo = InklingForConditionalGeneration.from_config(cfg, backend=backend).to(dtype=torch.float32).eval()
    return cfg, hf, nemo


def test_sparse_layers_use_inkling_moe():
    cfg, _, nemo = _build_models()
    layers = nemo.model.language_model.layers
    mlp_types = cfg.text_config.mlp_layer_types
    for i, layer in enumerate(layers):
        if mlp_types[i] == "sparse":
            assert isinstance(layer.mlp, InklingMoE)
            assert not isinstance(layer.mlp, HFInklingMoE)
        else:
            assert isinstance(layer.mlp, InklingDenseMLP)


def test_pretrained_load_skips_redundant_full_model_initialization():
    assert InklingForConditionalGeneration._skip_init_weights_on_load is True


def test_pipeline_metadata_uses_unpadded_vocabulary_size():
    cfg = build_tiny_config()
    cfg.text_config.unpadded_vocab_size = cfg.text_config.vocab_size - 8
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
    model = InklingForConditionalGeneration.from_config(cfg, backend=backend)

    inputs_meta, outputs_meta = model.get_pipeline_stage_metas(
        is_first=False,
        microbatch_size=2,
        seq_len=16,
        dtype=torch.float32,
    )

    assert inputs_meta[0].shape == (2, 16, cfg.text_config.hidden_size)
    assert outputs_meta[0].shape == (2, 16, cfg.text_config.unpadded_vocab_size)


def test_requested_dtype_preserves_strict_fp32_modules():
    cfg = build_tiny_config()
    cfg.torch_dtype = torch.bfloat16
    cfg.text_config.torch_dtype = torch.bfloat16
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
    model = InklingForConditionalGeneration.from_config(cfg, backend=backend)

    assert model.model.language_model.layers[0].mlp.gate_up_proj.dtype == torch.bfloat16
    assert model.model.language_model.layers[0].attn_sconv._fp32_params.weight.dtype == torch.float32
    assert model.model.language_model.layers[2].mlp.gate.e_score_correction_bias.dtype == torch.float32
    assert "_fp32_params.e_score_correction_bias" in dict(
        model.model.language_model.layers[2].mlp.gate.named_parameters()
    )


def test_processor_builder_configures_padding(monkeypatch):
    from nemo_automodel.components.models.inkling import processing

    tokenizer = SimpleNamespace(eos_token_id=None, pad_token_id=None, eos_token=None, pad_token=None)
    expected = SimpleNamespace(tokenizer=tokenizer)
    monkeypatch.setattr(processing.AutoProcessor, "from_pretrained", lambda *_args, **_kwargs: expected)

    actual = processing.build_inkling_processor("thinkingmachines/Inkling")

    assert actual is expected
    assert tokenizer.eos_token == "<|content_model_end_sampling|>"
    assert tokenizer.pad_token == tokenizer.eos_token


def test_process_group_warmup_uses_sharded_and_pipeline_groups(monkeypatch):
    import nemo_automodel.components.models.inkling.model as inkling_model

    class FakeDTensor:
        def __init__(self, mesh, placements):
            self.device_mesh = mesh
            self.placements = placements
            self.requires_grad = True

    shard_group = object()
    replicated_group = object()
    pp_group = object()
    mesh = Mock()
    mesh.get_group.side_effect = lambda mesh_dim: (replicated_group, shard_group)[mesh_dim]
    pp_mesh = Mock()
    pp_mesh.get_group.return_value = pp_group
    model_part = Mock()
    model_part.parameters.return_value = [
        FakeDTensor(mesh, (torch.distributed.tensor.Replicate(), Mock())),
    ]

    all_reduce = Mock()
    monkeypatch.setattr(inkling_model, "DTensor", FakeDTensor)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    inkling_model.InklingForConditionalGeneration.warmup_process_groups([model_part], pp_mesh=pp_mesh)

    assert [call.kwargs["group"] for call in all_reduce.call_args_list] == [shard_group, pp_group]


def test_inkling_collator_does_not_require_qwen_vl_utils(monkeypatch):
    import nemo_automodel.components.datasets.vlm.collate_fns as collate_fns

    class InklingTokenizer:
        unk_token_id = 0
        _tokens = {
            "<|message_model|>": 10,
            "<|content_text|>": 11,
            "<|end_message|>": 12,
            "<|content_model_end_sampling|>": 13,
        }

        def convert_tokens_to_ids(self, token):
            return self._tokens.get(token, self.unk_token_id)

    class InklingProcessor:
        tokenizer = InklingTokenizer()
        image_token_id = 99

        def apply_chat_template(self, *_args, **_kwargs):
            input_ids = torch.tensor([[1, 10, 11, 20, 12, 13, 13]])
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "pixel_values": torch.empty(0, 1, 1, 1, 1),
            }

    monkeypatch.setattr(collate_fns, "HAVE_QWEN_VL_UTILS", False)
    examples = [{"conversation": [{"role": "user", "content": [{"type": "text", "text": "Question"}]}]}]

    batch = default_collate_fn(examples, processor=InklingProcessor(), max_length=32)

    assert batch["input_ids"].shape == (1, 6)
    assert batch["num_patches"].tolist() == [0]


def test_inkling_labels_include_sampling_terminator_but_not_padding():
    class InklingTokenizer:
        unk_token_id = 0
        _tokens = {
            "<|message_model|>": 10,
            "<|content_text|>": 11,
            "<|end_message|>": 12,
            "<|content_model_end_sampling|>": 13,
        }

        def convert_tokens_to_ids(self, token):
            return self._tokens.get(token, self.unk_token_id)

    class InklingProcessor:
        tokenizer = InklingTokenizer()

    input_ids = torch.tensor([[1, 2, 10, 11, 20, 21, 12, 13, 13, 13]])
    labels = build_labels_from_template(input_ids, [[]], InklingProcessor())
    expected = torch.tensor([[-100, -100, -100, -100, 20, 21, 12, 13, -100, -100]])
    torch.testing.assert_close(labels, expected)


def test_state_dict_adapter_roundtrip_exact():
    _, hf, nemo = _build_models()
    adapter = nemo.state_dict_adapter
    hf_sd = hf.state_dict()
    native_sd = adapter.from_hf(hf_sd)
    raw_sd = adapter.to_hf(native_sd)
    roundtrip = adapter.from_hf(raw_sd)
    assert set(roundtrip) == set(native_sd)
    assert "model.llm.layers.2.mlp.experts.w13_weight" in raw_sd
    assert "model.llm.layers.2.mlp.gate.bias" in raw_sd
    assert "model.llm.layers.0.mlp.w13_dn.weight" in raw_sd
    assert "model.language_model.layers.2.mlp.gate._fp32_params.e_score_correction_bias" in native_sd
    for key in native_sd:
        assert torch.equal(native_sd[key], roundtrip[key]), f"round-trip mismatch for {key}"


def test_from_hf_load_has_no_missing_or_unexpected_keys():
    _, hf, nemo = _build_models()
    native_sd = nemo.state_dict_adapter.from_hf(hf.state_dict())
    missing, unexpected = nemo.load_state_dict(native_sd, strict=False)
    ignore = lambda ks: [k for k in ks if "rotary" not in k and "inv_freq" not in k]
    assert ignore(missing) == []
    assert ignore(unexpected) == []


def test_logit_parity_kl_below_1e_3():
    cfg, hf, nemo = _build_models()
    native_sd = nemo.state_dict_adapter.from_hf(hf.state_dict())
    nemo.load_state_dict(native_sd, strict=False)

    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.text_config.vocab_size, (1, 24))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        hf_logits = hf(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        nemo_logits = nemo(input_ids=input_ids, attention_mask=attention_mask).logits.float()

    assert hf_logits.shape == nemo_logits.shape
    kl = F.kl_div(
        F.log_softmax(nemo_logits, dim=-1),
        F.log_softmax(hf_logits, dim=-1),
        log_target=True,
        reduction="batchmean",
    ).item()
    assert kl < 1e-3, f"KL divergence too high: {kl}"
    assert (hf_logits - nemo_logits).abs().max().item() < 1e-2


def test_backward_parity():
    cfg, hf, nemo = _build_models()
    native_sd = nemo.state_dict_adapter.from_hf(hf.state_dict())
    nemo.load_state_dict(native_sd, strict=False)
    hf.train()
    nemo.train()

    torch.manual_seed(1)
    input_ids = torch.randint(0, cfg.text_config.vocab_size, (1, 16))
    attention_mask = torch.ones_like(input_ids)
    hf(input_ids=input_ids, attention_mask=attention_mask).logits.float().square().mean().backward()
    nemo(input_ids=input_ids, attention_mask=attention_mask).logits.float().square().mean().backward()

    sparse_layer = next(i for i, layer_type in enumerate(cfg.text_config.mlp_layer_types) if layer_type == "sparse")
    hf_moe = hf.model.language_model.layers[sparse_layer].mlp
    nemo_moe = nemo.model.language_model.layers[sparse_layer].mlp

    torch.testing.assert_close(
        nemo.model.language_model.embed_tokens.weight.grad,
        hf.model.language_model.embed_tokens.weight.grad,
        rtol=1e-4,
        atol=1e-6,
    )
    torch.testing.assert_close(nemo_moe.gate.weight.grad, hf_moe.gate.weight.grad, rtol=1e-4, atol=1e-6)

    raw_gate_up_grad = nemo_moe.experts.gate_and_up_projs.grad.transpose(-1, -2)
    torch.testing.assert_close(
        _interleave(raw_gate_up_grad, 1),
        hf_moe.experts.gate_up_proj.grad,
        rtol=1e-4,
        atol=1e-6,
    )
    torch.testing.assert_close(
        nemo_moe.experts.down_projs.grad.transpose(-1, -2),
        hf_moe.experts.down_proj.grad,
        rtol=1e-4,
        atol=1e-6,
    )


def test_multimodal_tower_parity():
    cfg, hf, nemo = _build_models()
    native_sd = nemo.state_dict_adapter.from_hf(hf.state_dict())
    nemo.load_state_dict(native_sd, strict=False)

    torch.manual_seed(2)
    vision = cfg.vision_config
    pixel_values = torch.randn(
        2,
        vision.temporal_patch_size,
        vision.patch_size,
        vision.patch_size,
        vision.num_channels,
    )
    audio_input_ids = torch.randint(
        0,
        cfg.audio_config.mel_vocab_size,
        (2, 3, cfg.audio_config.n_mel_bins),
    )
    audio_mask = torch.tensor([[True, True, False], [True, False, False]])

    with torch.no_grad():
        hf_image = hf.get_image_features(pixel_values).pooler_output
        nemo_image = nemo.get_image_features(pixel_values).pooler_output
        hf_audio = hf.model.get_audio_features(audio_input_ids, audio_mask).pooler_output
        nemo_audio = nemo.model.get_audio_features(audio_input_ids, audio_mask).pooler_output

    torch.testing.assert_close(nemo_image, hf_image)
    torch.testing.assert_close(nemo_audio, hf_audio)


def test_two_stage_pipeline_forward_parity(monkeypatch):
    import nemo_automodel.components.distributed.pipelining.functional as pipeline

    class FakePPMesh:
        def size(self):
            return 1

        def get_local_rank(self):
            return 0

        def get_group(self, *_args, **_kwargs):
            return None

    class FakePipelineStage:
        def __init__(self, submod, stage_idx, num_stages, device, group=None):
            self.submod = submod
            self.stage_index = stage_idx
            self.num_stages = num_stages
            self.device = device
            self.group = group
            self.is_first = stage_idx == 0
            self.is_last = stage_idx == num_stages - 1

    monkeypatch.setattr(pipeline, "PipelineStage", FakePipelineStage)

    cfg, hf, nemo = _build_models()
    nemo.load_state_dict(nemo.state_dict_adapter.from_hf(hf.state_dict()), strict=False)
    input_ids = torch.randint(0, cfg.text_config.vocab_size - 2, (1, 16))
    input_ids[0, 0] = cfg.image_token_id
    attention_mask = torch.ones_like(input_ids)
    vision = cfg.vision_config
    pixel_values = torch.randn(
        1,
        vision.temporal_patch_size,
        vision.patch_size,
        vision.patch_size,
        vision.num_channels,
    )

    with torch.no_grad():
        expected = nemo(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

    _, model_parts = pipeline.split_model_into_stages(
        model=nemo,
        pp_mesh=FakePPMesh(),
        pp_axis_name="pp",
        pp_schedule="interleaved1f1b",
        device=torch.device("cpu"),
        layers_per_stage=2,
        patch_inner_model=False,
        patch_causal_lm_model=False,
        round_to_pp_multiple="down",
    )
    assert len(model_parts) == 2
    assert model_parts[0].model.language_model.embed_norm is not None

    model_parts[0]._vlm_pixel_values_chunks = [pixel_values]
    model_parts[0]._vlm_chunk_idx = 0
    with torch.no_grad():
        hidden_states = model_parts[0](input_ids, attention_mask=attention_mask, use_cache=False)
        actual = model_parts[1](hidden_states, attention_mask=attention_mask, use_cache=False)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
