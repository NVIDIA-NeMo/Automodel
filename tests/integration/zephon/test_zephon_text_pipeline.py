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

"""Real CPU source, packing, mixture, and checkpoint-replay pipelines."""

import json
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.packing import BatchOutputFormat
from nemo_automodel.components.datasets.zephon import ZephonDataloaderConfig
from nemo_automodel.components.datasets.zephon.config import ZephonPretokenizedTextSourceConfig
from nemo_automodel.components.models.common.packing import PackingCapabilities
from tests.utils.zephon_text_fixtures import config as config
from tests.utils.zephon_text_fixtures import tokenizer as tokenizer


def test_real_finite_stream_and_continuing_iter(config, tokenizer):
    with closing(config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)) as loader:
        first = next(loader)
        assert iter(loader) is loader
        second = next(iter(loader))
        assert not torch.equal(first["input_ids"], second["input_ids"])
        assert first["input_ids"].shape == (2, 7)
        assert first["labels"].tolist()[0] == [3, 4, 1, 3, 4, 5, 1]
        assert first["position_ids"].tolist()[0] == [0, 1, 2, 3, 0, 1, 2]
        assert first["padding_mask"].dtype == torch.bool
        assert not first["padding_mask"].any()  # EOS/PAD is still a real token.
        assert first["labels"].eq(1).any()
        rest = list(loader)
        assert len(rest) == 7  # 36*4 tokens / (8 tokens * 2 docs per batch).
        assert list(loader) == []


def test_text_explicit_ids_override_insertion_without_changing_tokenizer(
    config: ZephonDataloaderConfig, tokenizer: PreTrainedTokenizerFast
) -> None:
    config = replace(
        config,
        dataset_config=replace(
            config.dataset_config, tokenize_special_tokens="bos_eos", bos_token_id=0, eos_token_id=40
        ),
        zephon=replace(config.zephon, packing=replace(config.zephon.packing, seq_len=4)),
    )
    with closing(config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)) as loader:
        batch = next(loader)
        assert batch["input_ids"][0].tolist() == [0, 2, 3, 4]
        assert batch["labels"][0].tolist() == [2, 3, 4, 40]
    assert tokenizer.bos_token_id is None
    assert tokenizer.eos_token_id == 1


@pytest.mark.parametrize("backend", ["sdpa", "flash_attention_2"])
def test_configured_backend_reaches_packed_loader(config, tokenizer, backend, monkeypatch):
    from types import SimpleNamespace

    from nemo_automodel._transformers import NeMoAutoModelForCausalLM
    from nemo_automodel._transformers.kernel_patches import _apply_preload_overrides
    from nemo_automodel.components.models.common.packing import configure_packing_for_models
    from nemo_automodel.recipes.llm.train_ft import build_model

    class Consumer(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            return kwargs

    def construct(*, attn_implementation, has_packed_sequence, **kwargs):
        assert has_packed_sequence is True
        selected, _ = _apply_preload_overrides(1, 1, has_packed_sequence, attn_implementation, False)
        model = Consumer()
        model.config = SimpleNamespace(_attn_implementation=selected)
        return model

    monkeypatch.setattr(NeMoAutoModelForCausalLM, "from_pretrained", construct)
    model = build_model(
        cfg_model=ConfigNode({"_target_": construct, "attn_implementation": backend}),
        cfg_peft=None,
        seed=42,
        has_packed_sequence=config.is_packed,
    )
    assert model.config._attn_implementation == backend
    contract = configure_packing_for_models([model])
    with closing(config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1, packing_contract=contract)) as loader:
        batch = next(loader)
        assert batch["_packed_seq_ids"].shape == batch["input_ids"].shape == (2, 7)
        assert batch["_packed_seq_ids"].tolist()[0] == [1, 1, 1, 1, 2, 2, 2]
        assert "document_ids" not in batch and "attention_document_ids" not in batch
        assert "cu_seq_lens_q" not in batch and "attention_mask" not in batch
        received = model(**batch)
        if backend == "sdpa":
            assert received["attention_mask"].shape == (2, 1, 7, 7)
            assert received["attention_mask"][0, 0, 3, 0]
            assert not received["attention_mask"][0, 0, 4, 0]
        else:
            assert received["cu_seq_lens_q"].tolist() == [0, 4, 7, 11, 14]
            assert "attention_mask" not in received


def test_fresh_loader_checkpoint_replays_inflight_batches(config, tokenizer):
    loader = config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)
    restored = config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)
    try:
        next(loader)
        state = loader.state_dict()
        assert isinstance(state["zephon"], bytes)
        expected = [next(loader) for _ in range(3)]
        restored.load_state_dict(state)
        for wanted in expected:
            torch.testing.assert_close(next(restored), wanted)
    finally:
        loader.close()
        restored.close()


def test_xdoc_and_boundary_label_mask_are_independent(config, tokenizer):
    configs = [
        replace(
            config,
            zephon=replace(
                config.zephon,
                training=replace(config.zephon.training, allow_cross_document_attention=xdoc, mask_segment_starts=mask),
            ),
        )
        for xdoc, mask in ((False, False), (True, False), (False, True), (True, True))
    ]
    loaders = [cfg.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1) for cfg in configs]
    try:
        batches = [next(loader) for loader in loaders]
        torch.testing.assert_close(batches[0]["labels"], batches[1]["labels"])
        torch.testing.assert_close(batches[2]["labels"], batches[3]["labels"])
        assert batches[2]["labels"][0, 3].item() == -100
        torch.testing.assert_close(batches[2]["position_ids"], batches[0]["position_ids"])
        for batch in (batches[1], batches[3]):
            expected_positions = torch.arange(batch["input_ids"].shape[-1]).expand_as(batch["input_ids"])
            valid = ~batch["padding_mask"]
            torch.testing.assert_close(batch["position_ids"][valid], expected_positions[valid])
            torch.testing.assert_close(batch["_packed_seq_ids"], valid.long())
            assert "document_ids" not in batch
    finally:
        for loader in loaders:
            loader.close()


def test_common_text_build_does_not_enter_native_context_or_wrap_collation(config, tokenizer):
    class NativeOnlyContext:
        def __enter__(self):
            raise AssertionError("Zephon initialization must stay on every rank")

        def __exit__(self, *_):
            pass

    def native_collate_wrapper(collator):
        raise AssertionError("to_training must remain the final collator")

    with closing(
        config.build(
            tokenizer=tokenizer,
            dp_rank=0,
            dp_world_size=1,
            dataset_build_context=NativeOnlyContext(),
            collate_wrapper=native_collate_wrapper,
            pp_enabled=True,
            supports_seq_lens=False,
            cp_size=1,
            ignore_index=-321,
        )
    ) as loader:
        batch = next(loader)
        assert batch["input_ids"].shape == (2, 7)
        assert loader.synchronized_stream
        assert loader.use_distributed_checkpointing

    with pytest.raises(ValueError, match="context parallel"):
        config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1, cp_size=2)


def test_document_id_model_consumes_real_loader_and_respects_xdoc(config, tokenizer):
    from nemo_automodel.components.models.common.packing import configure_packing
    from nemo_automodel.components.models.flint import FlintConfig, FlintForCausalLM

    torch.manual_seed(19)
    model_config = FlintConfig(
        vocab_size=48,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["full_attention"],
        mlp_only_layers=[0],
        no_rope_layer_indices=[0],
        max_position_embeddings=32,
        dtype=torch.float32,
    )
    model = FlintForCausalLM(model_config).eval()
    contract = configure_packing("eager", model=model)
    assert contract.packed_mask_type == "document_ids" and not contract.return_packed_sequence_metadata
    for xdoc in (False, True):
        selected = replace(
            config,
            zephon=replace(
                config.zephon, training=replace(config.zephon.training, allow_cross_document_attention=xdoc)
            ),
        )
        with closing(
            selected.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1, packing_contract=contract)
        ) as loader:
            batch = next(loader)
            assert "document_ids" not in batch
            assert batch["attention_mask"].tolist()[0] == ([1] * 7 if xdoc else [1] * 4 + [2] * 3)
            inputs = {key: value for key, value in batch.items() if key != "labels"}
            logits = model(**inputs).logits
            changed_ids = batch["input_ids"].clone()
            changed_ids[:, :4] = 30
            changed = model(**{**inputs, "input_ids": changed_ids}).logits
            if xdoc:
                assert not torch.allclose(logits[:, 4:], changed[:, 4:])
            else:
                torch.testing.assert_close(logits[:, 4:], changed[:, 4:], atol=0, rtol=0)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), batch["labels"].flatten())
            loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            model.zero_grad()


def test_raw_text_without_packing_keeps_documents_separate_and_pads(
    config: ZephonDataloaderConfig, tokenizer: PreTrainedTokenizerFast
) -> None:
    selected = replace(
        config,
        zephon=replace(
            config.zephon,
            packing=replace(config.zephon.packing, pack_algorithm=None, seq_len=None),
            source=replace(config.zephon.source, chunk_size=1),
        ),
    )
    contract = PackingCapabilities("block_causal", max_sequence_length=7)
    with closing(selected.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1, packing_contract=contract)) as loader:
        batch = next(loader)
        assert batch["input_ids"].tolist() == [[2, 3, 4, 1, 1, 1, 1], [3, 4, 5, 1, 1, 1, 1]]
        assert batch["labels"].tolist() == [[3, 4, 1, -100, -100, -100, -100], [4, 5, 1, -100, -100, -100, -100]]
        assert batch["_packed_seq_ids"].tolist() == [[1, 1, 1, 1, 0, 0, 0]] * 2
    assert selected.zephon.packing.seq_len is None
    Path(config.zephon.sources[0].path, "text.jsonl").write_text('{"text":"w0 w1 w2 w3 w4 w5 w6 w7 w8"}\n' * 2)
    with closing(selected.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1, packing_contract=contract)) as loader:
        with pytest.raises((ValueError, RuntimeError), match=r"exceeding seq_len\+1"):
            next(loader)


def test_pretokenized_measurement_allocates_tokens_and_finite_repeat_cap(config, tokenizer, tmp_path):
    from nemo_automodel.components.datasets.zephon.config import ZephonSource

    sources = []
    for name, tokens in (("short", [2, 1]), ("long", [3] * 7 + [1])):
        path = tmp_path / name
        path.mkdir()
        (path / "data.jsonl").write_text((json.dumps({"tokens": tokens}) + "\n") * 64)
        sources.append(ZephonSource(name=name, path=str(path), fmt="jsonl"))
    pretokenized = replace(
        config,
        batch_size=1,
        dataset_config=ZephonPretokenizedTextSourceConfig(tokens_field="tokens"),
        zephon=replace(
            config.zephon,
            sources=tuple(sources),
            packing=replace(config.zephon.packing, seq_len=31),
            source=replace(
                config.zephon.source,
                chunk_size=8,
                exhausted_policy="repeat",
                stop_after_passes=4,
                token_estimation={"calibration_samples": 8, "calibration_shards_min": 1, "calibration_shards_max": 1},
            ),
        ),
    )
    with closing(pretokenized.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)) as loader:
        short = total = 0
        for batch in loader:
            tokens = torch.cat((batch["input_ids"], batch["labels"][:, -1:]), dim=1)
            short += 2 * int(tokens.eq(2).sum())
            total += tokens.numel()
        assert 0.42 < short / total < 0.58
        assert 0 < total < 5000
        assert list(loader) == []


def test_pretokenized_mixed_supervision_preserves_all_targets(config, tokenizer):
    path = Path(config.zephon.sources[0].path) / "text.jsonl"
    path.write_text(
        json.dumps({"input_ids": [2], "loss_mask": [0]})
        + "\n"
        + json.dumps({"input_ids": [3, 4], "loss_mask": [0, 0]})
        + "\n"
        + json.dumps({"input_ids": [5, 6, 1]})
        + "\n"
    )
    config = replace(
        config,
        batch_size=1,
        dataset_config=ZephonPretokenizedTextSourceConfig(),
        zephon=replace(
            config.zephon,
            packing=replace(config.zephon.packing, seq_len=5),
            source=replace(config.zephon.source, chunk_size=1),
        ),
    )
    with closing(config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)) as loader:
        batch = next(loader)
        assert batch["input_ids"].tolist() == [[2, 3, 4, 5, 6]]
        assert batch["labels"].tolist() == [[-100, -100, 5, 6, 1]]
        assert batch["position_ids"].tolist() == [[0, 0, 1, 0, 1]]


@pytest.mark.parametrize("output_format", ["batched", "thd"])
def test_prepacked_boundaries_and_single_document_fallback(
    config: ZephonDataloaderConfig, tokenizer: PreTrainedTokenizerFast, output_format: BatchOutputFormat
) -> None:
    path = Path(config.zephon.sources[0].path, "text.jsonl")
    path.write_text(
        json.dumps({"input_ids": [2, 3, 4, 5, 6, 1], "positions": [0, 1, 2, 0, 1, 2]})
        + "\n"
        + json.dumps({"input_ids": [7, 1, 8, 1]})
        + "\n"
    )
    selected = replace(
        config,
        dataset_config=ZephonPretokenizedTextSourceConfig(),
        output_format=output_format,
        zephon=replace(
            config.zephon,
            packing=replace(config.zephon.packing, pack_algorithm=None),
            runtime=replace(config.zephon.runtime, runner="process" if output_format == "batched" else "threads"),
            source=replace(config.zephon.source, chunk_size=1),
            training=replace(config.zephon.training, mask_segment_starts=True),
        ),
    )
    for xdoc in (False, True):
        selected = replace(
            selected,
            zephon=replace(
                selected.zephon, training=replace(selected.zephon.training, allow_cross_document_attention=xdoc)
            ),
        )
        with closing(selected.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)) as loader:
            batch = next(loader)
            assert batch["input_ids"].tolist() == [[2, 3, 4, 5, 6, 1, 1], [7, 1, 8, 1, 1, 1, 1]]
            assert batch["labels"].tolist() == [[3, 4, -100, 6, 1, -100, -100], [1, 8, 1, -100, -100, -100, -100]]
            assert batch["_packed_seq_ids"].tolist() == [
                [1, 1, 1, 1 if xdoc else 2, 1 if xdoc else 2, 1 if xdoc else 2, 0],
                [1, 1, 1, 1, 0, 0, 0],
            ]
            assert batch["position_ids"][0, :6].tolist() == ([0, 1, 2, 3, 4, 5] if xdoc else [0, 1, 2, 0, 1, 2])
            # EOS inside the second supplied sequence never implies a boundary.
            assert batch["position_ids"][1, :4].tolist() == [0, 1, 2, 3]
            if output_format == "thd":
                assert batch["seq_lens"][0].tolist() == ([6] if xdoc else [3, 3])
                assert batch["seq_lens_padded"][0].tolist() == ([7] if xdoc else [3, 4])
