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

"""Real CPU VLM loader, mapped Qwen consumption and finite checkpoint replay."""

import json
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
import yaml
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2_5_VLForConditionalGeneration,
)

from nemo_automodel._transformers.registry import resolve_image_position_state
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.zephon.config import (
    ZephonOptions,
    ZephonPackingOptions,
    ZephonRuntimeOptions,
    ZephonSource,
    ZephonSourceOptions,
    ZephonWorkerOptions,
)
from nemo_automodel.components.datasets.zephon.vlm_config import (
    ZephonVlmDataloaderConfig,
    ZephonVlmSourceConfig,
)
from nemo_automodel.components.models.common.packing import (
    configure_packing,
)
from nemo_automodel.recipes._typed_config import RecipeConfig
from tests.utils.vlm_fixtures import inputs as inputs
from tests.utils.vlm_fixtures import processor as processor
from tests.utils.vlm_fixtures import tiny_vlm_config
from tests.utils.zephon_positions import assert_joined_position_ranges


@pytest.fixture
def config(tmp_path, inputs):
    processor_path, documents = inputs
    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / "images.jsonl").write_text("".join(json.dumps(document) + "\n" for document in documents * 3))
    return ZephonVlmDataloaderConfig(
        zephon=ZephonOptions(
            sources=(ZephonSource(name="images", path=str(directory), fmt="jsonl"),),
            packing=ZephonPackingOptions(
                seq_len=63, pack_algorithm="first_fit", pack_num_bins=2, shuffle_after_pack=False
            ),
            source=ZephonSourceOptions(
                chunk_size=3, shuffle_shards=False, shuffle_within_shard=False, shuffle_block_size=None
            ),
            workers=ZephonWorkerOptions(tokenize_parallelism=1),
            runtime=ZephonRuntimeOptions(runner="inline", mtp_mode=False),
        ),
    )


def _build(config, inputs, *, pp_n_microbatches=None):
    model_config = tiny_vlm_config()
    model = Qwen2_5_VLForConditionalGeneration(model_config)
    return config.build(
        pretrained_model_name_or_path=inputs[0],
        dp_rank=0,
        dp_world_size=1,
        batch_size=2,
        position_state_factory=lambda: resolve_image_position_state(model),
        pp_n_microbatches=pp_n_microbatches,
    ).dataloader


def test_example_yaml_route_reads_actual_local_image_source(config, inputs):
    path = Path(__file__).resolve().parents[3] / "examples/vlm_finetune/zephon/train.yaml"
    values = yaml.safe_load(path.read_text())
    values["model"]["pretrained_model_name_or_path"] = inputs[0]
    values["processor"] = {}
    settings = values["dataloader"]["zephon"]
    settings["sources"] = [asdict(source) for source in config.zephon.sources]
    settings["packing"].update(seq_len=40, pack_num_bins=2)
    settings.setdefault("runtime", {}).update(runner="inline", mtp_mode=False)
    resolved = RecipeConfig(ConfigNode(values)).vlm_dataloader
    assert isinstance(resolved, ZephonVlmDataloaderConfig)
    with closing(_build(resolved, inputs)) as loader:
        batch = next(loader)
        assert batch["input_ids"].shape == (2, 40)
        assert batch["labels"].ne(-100).any()
        assert batch["image_grid_thw"].shape[0] > 0


def test_real_loader_checkpoint_and_expanded_media(config, inputs):
    config = replace(config, zephon=replace(config.zephon, runtime=replace(config.zephon.runtime, runner="process")))
    original, restored = _build(config, inputs), _build(config, inputs)
    try:
        first = next(original)
        state = original.state_dict()
        expected = list(original)
        assert expected
        restored.load_state_dict(state)
        actual = list(restored)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for batch in [first, *expected]:
            assert batch["input_ids"].shape == (2, 63)
            assert batch["position_ids"].shape == (3, 2, 63)
            assert "attention_mask" not in batch
            assert batch["_packed_seq_ids"].shape == (2, 63)
            assert batch["input_ids"].eq(9).sum() == (batch["image_grid_thw"].prod(-1) // 4).sum()
            assert batch["pixel_values"].shape[0] == batch["image_grid_thw"].prod(-1).sum()
            assert batch["_packed_seq_ids"][batch["padding_mask"]].eq(0).all()
    finally:
        original.close()
        restored.close()


def _forward(model, batch):
    """Consume IDs [B,L], mRoPE [3,B,L], compact boundaries [B,L] and media."""
    configure_packing(model.config._attn_implementation, model=model)
    model_inputs = {name: value for name, value in batch.items() if name != "labels"}
    return model(**model_inputs, use_cache=False).logits


@pytest.mark.parametrize("attention", ["sdpa", "eager"])
def test_actual_adapter_xdoc_policy_and_backward(config, inputs, attention):
    torch.manual_seed(123)
    model_config = tiny_vlm_config()
    model_config._attn_implementation = attention
    model = Qwen2_5_VLForConditionalGeneration(model_config).train()
    off_loader = _build(config, inputs)
    on_loader = _build(
        replace(
            config,
            zephon=replace(
                config.zephon, training=replace(config.zephon.training, allow_cross_document_attention=True)
            ),
        ),
        inputs,
    )
    try:
        off, on = next(off_loader), next(on_loader)
        assert_joined_position_ranges(off["position_ids"], on["position_ids"], off["_packed_seq_ids"])
        for key in off.keys() - {"_packed_seq_ids", "position_ids"}:
            torch.testing.assert_close(off[key], on[key], atol=0, rtol=0)
        later = off["_packed_seq_ids"].gt(1)
        assert later.any()
        before = _forward(model, off)
        after = _forward(model, on)
        loss = torch.nn.functional.cross_entropy(after.flatten(0, 1), on["labels"].flatten(), ignore_index=-100)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(model.lm_head.weight.grad).all()
        grad = model.model.visual.patch_embed.proj.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        changed_off, changed_on = dict(off), dict(on)
        tokens = off["input_ids"].clone()
        tokens[off["_packed_seq_ids"].eq(1) & tokens.eq(5)] = 7
        changed_off["input_ids"] = changed_on["input_ids"] = tokens
        torch.testing.assert_close(_forward(model, changed_off)[later], before[later], atol=0, rtol=0)
        assert not torch.allclose(_forward(model, changed_on)[later], after[later], atol=1e-6, rtol=1e-6)
        excluded = ~off["_packed_seq_ids"].bool()
        if "_packed_seq_ids" in on:
            assert on["_packed_seq_ids"][excluded].eq(0).all()
        else:
            assert torch.equal(excluded, off["padding_mask"])
    finally:
        off_loader.close()
        on_loader.close()


def test_unsupported_modes_fail_before_iteration(config, inputs):
    with pytest.raises(ValueError, match="first_fit or best_fit"):
        replace(
            config,
            zephon=replace(
                config.zephon,
                packing=replace(config.zephon.packing, pack_algorithm="wrap", pack_num_bins=None, drop_oversized=False),
            ),
        )
    with pytest.raises(ValueError, match="media-aware"):
        replace(config, zephon=replace(config.zephon, source=replace(config.zephon.source, token_estimation=True)))
    with pytest.raises(ValueError, match="context parallelism"):
        config.validate_parallelism(cp_size=2)
    config.validate_parallelism(cp_size=1)
    with pytest.raises(ValueError, match="evenly divide"):
        _build(config, inputs, pp_n_microbatches=3)
    with pytest.raises(ValueError, match="no certified"):
        resolve_image_position_state(
            LlamaForCausalLM(
                LlamaConfig(
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    vocab_size=32,
                )
            )
        )


def test_text_only_without_fake_images_omits_media(config, inputs, tmp_path):
    directory = tmp_path / "text-only"
    directory.mkdir()
    document = inputs[1][2]
    (directory / "text.jsonl").write_text("".join(json.dumps(document) + "\n" for _ in range(12)))
    config = replace(
        config,
        dataset_config=ZephonVlmSourceConfig(inject_fake_images=False),
        zephon=replace(config.zephon, sources=(ZephonSource(name="text", path=str(directory), fmt="jsonl"),)),
    )
    with closing(_build(config, inputs)) as loader:
        batch = next(loader)
        assert "pixel_values" not in batch and "image_grid_thw" not in batch
        model_config = tiny_vlm_config()
        model_config._attn_implementation = "sdpa"
        assert torch.isfinite(_forward(Qwen2_5_VLForConditionalGeneration(model_config), batch)).all()


def test_pipeline_media_ownership_preserves_packed_documents(config, inputs):
    ordinary = _build(config, inputs)
    pipeline = _build(config, inputs, pp_n_microbatches=2)
    try:
        expected, actual = next(ordinary), next(pipeline)
        chunks = actual.pop("_vlm_pp_media_chunks")
        for key in ("input_ids", "labels", "position_ids", "_packed_seq_ids", "mm_token_type_ids"):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        torch.testing.assert_close(torch.cat(chunks["pixel_values"]), expected["pixel_values"])
        torch.testing.assert_close(torch.cat(chunks["image_grid_hws"]), expected["image_grid_thw"])
        for index, (pixels, grids) in enumerate(zip(chunks["pixel_values"], chunks["image_grid_hws"], strict=True)):
            assert pixels.shape[0] == grids.prod(-1).sum()
            assert expected["input_ids"][index].eq(9).sum() == (grids.prod(-1) // 4).sum()
    finally:
        ordinary.close()
        pipeline.close()


def test_parent_position_factory_and_processor_context_do_not_enter_workers(config, inputs, monkeypatch):
    import os
    import threading
    from contextlib import contextmanager

    from nemo_automodel.components.datasets.zephon import pipeline as pipeline_module

    config = replace(config, zephon=replace(config.zephon, runtime=replace(config.zephon.runtime, runner="process")))
    parent_pid = os.getpid()
    unpicklable_parent_state = threading.Lock()
    events = []

    def position_factory():
        assert os.getpid() == parent_pid
        with unpicklable_parent_state:
            events.append("positions")
            model = Qwen2_5_VLForConditionalGeneration(tiny_vlm_config())
            return resolve_image_position_state(model)

    def native_rope(*args, **kwargs):
        raise AssertionError("Bound model RoPE callbacks must not enter Zephon workers")

    @contextmanager
    def processor_context():
        events.append("processor-enter")
        yield
        events.append("processor-exit")

    create_source = pipeline_module.create_source_pipeline

    def create_source_after_context(*args, **kwargs):
        assert events == ["positions", "processor-enter", "processor-exit"]
        events.append("source")
        return create_source(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "create_source_pipeline", create_source_after_context)
    result = config.build(
        pretrained_model_name_or_path=inputs[0],
        dp_rank=0,
        dp_world_size=1,
        batch_size=2,
        position_state_factory=position_factory,
        get_rope_index=native_rope,
        dataset_build_context=processor_context(),
        cp_size=1,
    )
    try:
        batch = next(result.dataloader)
        assert batch["input_ids"].shape == (2, 63)
        assert batch["position_ids"].shape == (3, 2, 63)
        assert events == ["positions", "processor-enter", "processor-exit", "source"]
    finally:
        result.dataloader.close()

    with pytest.raises(ValueError, match="context parallel"):
        config.build(pretrained_model_name_or_path=inputs[0], dp_rank=0, dp_world_size=1, batch_size=2, cp_size=2)
