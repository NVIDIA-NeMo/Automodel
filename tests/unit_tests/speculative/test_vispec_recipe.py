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

"""Unit tests for the ViSpec recipe glue and the VLM answer-regeneration script."""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import LlamaConfig

from nemo_automodel.cli.utils import resolve_recipe_name
from nemo_automodel.components.speculative import regenerate_vlm
from nemo_automodel.components.speculative.eagle.draft_llama_v12 import LlamaEagleDraftModel
from nemo_automodel.components.speculative.eagle.vispec_core import VispecTrainerModule
from nemo_automodel.components.speculative.eagle.vispec_draft import VispecDraftModel
from nemo_automodel.components.speculative.eagle.vispec_target import HFVispecTargetModel
from nemo_automodel.recipes.llm.train_vispec import (
    TrainVispecRecipe,
    TrainVispecStage1Recipe,
    _apply_image_token_budget,
    _resolve_image_token_id,
    _seed_draft_initialization,
)

HIDDEN = 32
VOCAB = 64
IMAGE_TOKEN_ID = 7


def _config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        draft_num_hidden_layers=1,
        vispec_num_query_tokens=2,
    )


class _StubBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)

    def forward(self, input_ids, attention_mask, output_hidden_states=False, **kwargs):
        """Return HF-shaped hidden states for a stub target.

        Args:
            input_ids: Tensor of shape [batch, sequence].
            attention_mask: Tensor of shape [batch, sequence].
            output_hidden_states: Whether to return the per-layer states.

        Returns:
            An object whose ``hidden_states`` is a 2-tuple of [batch, sequence, hidden].
        """
        embeds = self.embed(input_ids)

        class _Output(SimpleNamespace):
            def __getitem__(self, index):
                if index == 0:
                    return self.hidden_states[-1]
                raise IndexError(index)

        return _Output(hidden_states=(embeds, embeds * 2.0))


class _StubTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _StubBaseModel()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def get_input_embeddings(self):
        return self.model.embed


def _recipe() -> TrainVispecRecipe:
    """Wire a recipe with tiny models, skipping distributed setup entirely."""
    torch.manual_seed(0)
    recipe = TrainVispecRecipe.__new__(TrainVispecRecipe)
    recipe.device = torch.device("cpu")
    recipe.draft_model = VispecDraftModel(_config())
    target = _StubTarget()
    recipe.target_wrapper = HFVispecTargetModel(target, image_token_id=IMAGE_TOKEN_ID)
    recipe.trainer_module = VispecTrainerModule(recipe.draft_model, target_lm_head=target.lm_head, mtp_steps=1)
    return recipe


class TestImageTokenResolution:
    def test_prefers_config_field(self):
        config = SimpleNamespace(image_token_id=151655)
        assert _resolve_image_token_id(config, processor=None) == 151655

    def test_falls_back_to_processor(self):
        config = SimpleNamespace(image_token_id=None)
        processor = SimpleNamespace(
            image_token="<|image_pad|>",
            tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda token: 42),
        )
        assert _resolve_image_token_id(config, processor) == 42

    def test_raises_when_unresolvable(self):
        config = SimpleNamespace(image_token_id=None)
        with pytest.raises(ValueError, match="image token"):
            _resolve_image_token_id(config, SimpleNamespace(image_token=None))


class TestImageTokenBudget:
    def test_caps_both_flat_attributes_and_the_size_dict(self):
        processor = SimpleNamespace(
            image_processor=SimpleNamespace(max_pixels=99, min_pixels=1, size={"max_pixels": 99, "min_pixels": 1})
        )
        _apply_image_token_budget(processor, {"image_max_pixels": 802816, "image_min_pixels": 3136})
        assert processor.image_processor.max_pixels == 802816
        assert processor.image_processor.min_pixels == 3136
        assert processor.image_processor.size == {"max_pixels": 802816, "min_pixels": 3136}

    def test_unset_keys_leave_the_processor_default_alone(self):
        processor = SimpleNamespace(image_processor=SimpleNamespace(max_pixels=99, min_pixels=1))
        _apply_image_token_budget(processor, {})
        assert processor.image_processor.max_pixels == 99
        assert processor.image_processor.min_pixels == 1

    def test_processor_without_an_image_processor_is_a_no_op(self):
        _apply_image_token_budget(SimpleNamespace(), {"image_max_pixels": 802816})


class TestDraftInitializationSeed:
    def test_prefers_explicit_seed(self, monkeypatch):
        seeds = []
        monkeypatch.setattr(torch, "manual_seed", lambda seed: seeds.append(seed))

        assert _seed_draft_initialization({"seed": 17, "shuffle_seed": 42}) == 17
        assert seeds == [17]

    def test_falls_back_to_shuffle_seed_then_default(self, monkeypatch):
        seeds = []
        monkeypatch.setattr(torch, "manual_seed", lambda seed: seeds.append(seed))

        assert _seed_draft_initialization({"shuffle_seed": 13}) == 13
        assert _seed_draft_initialization({}) == 42
        assert seeds == [13, 42]


class TestComputeMetrics:
    def test_forwards_vision_inputs_and_returns_finite_loss(self):
        recipe = _recipe()
        batch = {
            "input_ids": torch.tensor([[1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 5, 6, 8, 9]]),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
            "loss_mask": torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1]]),
            "pixel_values": torch.randn(4, 8),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }
        metrics = recipe._compute_metrics(batch)
        assert torch.isfinite(metrics.loss)
        assert metrics.valid_tokens.item() == 3

    def test_loss_components_are_logged_under_vispec_names(self):
        recipe = _recipe()
        metrics = SimpleNamespace(
            prob_loss=torch.tensor(1.5), rank_loss=torch.tensor(0.25), valid_tokens=torch.tensor(37)
        )
        assert recipe._loss_components(metrics) == {"prob_loss": 1.5, "rank_loss": 0.25, "valid_tokens": 37.0}

    def test_valid_tokens_is_logged_so_an_empty_loss_mask_is_visible(self):
        recipe = _recipe()
        metrics = SimpleNamespace(
            prob_loss=torch.tensor(0.0), rank_loss=torch.tensor(0.0), valid_tokens=torch.tensor(0)
        )
        assert recipe._loss_components(metrics)["valid_tokens"] == 0.0


class TestStage1Init:
    def test_none_is_a_no_op(self):
        recipe = _recipe()
        before = recipe.draft_model.fc.weight.clone()
        recipe._load_stage1_draft(None)
        torch.testing.assert_close(recipe.draft_model.fc.weight, before)

    def test_loads_shared_weights_and_keeps_vision_modules(self, tmp_path):
        recipe = _recipe()
        stage1 = LlamaEagleDraftModel(_config())
        path = tmp_path / "model.safetensors"
        save_file({k: v.contiguous() for k, v in stage1.state_dict().items()}, str(path))

        img_fc_before = recipe.draft_model.img_fc.weight.clone()
        recipe._load_stage1_draft(str(path))
        torch.testing.assert_close(recipe.draft_model.fc.weight, stage1.fc.weight)
        torch.testing.assert_close(recipe.draft_model.img_fc.weight, img_fc_before)

    def test_rejects_checkpoint_missing_draft_weights(self, tmp_path):
        recipe = _recipe()
        path = tmp_path / "partial.safetensors"
        save_file({"fc.weight": recipe.draft_model.fc.weight.clone().contiguous()}, str(path))
        with pytest.raises(ValueError, match="missing draft weights"):
            recipe._load_stage1_draft(str(path))

    def test_warns_on_unexpected_keys(self, tmp_path, caplog):
        recipe = _recipe()
        stage1 = LlamaEagleDraftModel(_config())
        state = {k: v.contiguous() for k, v in stage1.state_dict().items()}
        state["not_a_draft_param"] = torch.zeros(2)
        path = tmp_path / "extra.safetensors"
        save_file(state, str(path))
        with caplog.at_level("WARNING"):
            recipe._load_stage1_draft(str(path))
        assert "unexpected key" in caplog.text


class TestRegenerateVlm:
    def test_load_source_dataset_uses_json_builder_for_local_file(self, monkeypatch, tmp_path):
        source = tmp_path / "source.jsonl"
        source.write_text('{"image": "a.png"}\n')
        calls = []
        expected = object()
        monkeypatch.setattr(
            regenerate_vlm,
            "load_dataset",
            lambda *args, **kwargs: calls.append((args, kwargs)) or expected,
        )

        assert regenerate_vlm._load_source_dataset(str(source), "train") is expected
        assert calls == [(("json",), {"data_files": str(source), "split": "train"})]

    def test_load_target_model_uses_explicit_cuda_placement(self, monkeypatch):
        class _FakeModel:
            def __init__(self):
                self.device = None

            def eval(self):
                return self

            def to(self, device):
                self.device = device
                return self

        fake_model = _FakeModel()
        calls = []
        monkeypatch.setattr(
            regenerate_vlm.AutoModelForImageTextToText,
            "from_pretrained",
            classmethod(lambda cls, *args, **kwargs: calls.append((args, kwargs)) or fake_model),
        )
        monkeypatch.setattr(regenerate_vlm.torch.cuda, "is_available", lambda: True)

        assert regenerate_vlm._load_target_model("local/model") is fake_model
        assert calls == [(("local/model",), {"torch_dtype": "auto"})]
        assert fake_model.device.type == "cuda"

    def test_extract_prompt_strips_image_placeholder(self):
        example = {
            "image": "a/b.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe this."},
                {"from": "gpt", "value": "A short caption."},
            ],
        }
        assert regenerate_vlm._extract_prompt(example) == ("Describe this.", "a/b.jpg")

    def test_extract_prompt_returns_none_without_image(self):
        assert regenerate_vlm._extract_prompt({"conversations": [{"from": "human", "value": "hi"}]}) is None

    def test_extract_prompt_returns_none_without_human_turn(self):
        example = {"image": "a.jpg", "conversations": [{"from": "gpt", "value": "hi"}]}
        assert regenerate_vlm._extract_prompt(example) is None

    def test_build_messages_appends_length_instruction_after_the_image(self):
        messages = regenerate_vlm._build_messages("Describe.", "/data/a.jpg", regenerate_vlm.LENGTH_INSTRUCTION)
        content = messages[0]["content"]
        assert [item["type"] for item in content] == ["text", "image", "text"]
        assert content[2]["text"] == regenerate_vlm.LENGTH_INSTRUCTION

    def test_build_messages_without_instruction_or_prompt(self):
        content = regenerate_vlm._build_messages("", "/data/a.jpg", "")[0]["content"]
        assert [item["type"] for item in content] == ["image"]

    def test_write_meta_is_consumable_by_make_meta_dataset(self, tmp_path):
        meta_path = regenerate_vlm._write_meta(tmp_path, "/images")
        meta = json.loads(meta_path.read_text())
        entry = meta["vispec_stage2"]
        assert entry["file_name"] == "data.jsonl"
        assert entry["media_dir"] == "/images"
        assert entry["columns"] == {"messages": "conversations", "images": "images"}
        assert entry["tags"] == {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        }

    def test_main_rejects_empty_slice(self):
        argv = [
            "--model",
            "m",
            "--dataset",
            "d",
            "--image-root",
            "/i",
            "--output-dir",
            "/o",
            "--start",
            "10",
            "--end",
            "10",
        ]
        with pytest.raises(ValueError, match="must be greater than"):
            regenerate_vlm.main(argv)

    def test_regenerate_writes_target_answers_without_the_length_instruction(self, tmp_path, monkeypatch):
        """The stored prompt must be the original one: the length instruction is a
        generation-time device, and training on it would teach the draft a prompt
        shape that never occurs at deployment."""
        image_root = tmp_path / "images"
        image_root.mkdir()
        (image_root / "a.jpg").write_bytes(b"")
        (image_root / "b.jpg").write_bytes(b"")

        rows = [
            {"image": "a.jpg", "conversations": [{"from": "human", "value": "<image>\nDescribe."}]},
            {"image": "missing.jpg", "conversations": [{"from": "human", "value": "<image>\nSkip me."}]},
            {"image": "b.jpg", "conversations": [{"from": "gpt", "value": "no human turn"}]},
        ]
        prompts_seen = []

        class _FakeDataset(list):
            def shuffle(self, seed):
                return self

            def select(self, indices):
                return _FakeDataset(self[i] for i in indices)

        monkeypatch.setattr(regenerate_vlm, "load_dataset", lambda *a, **k: _FakeDataset(rows))
        monkeypatch.setattr(regenerate_vlm.AutoProcessor, "from_pretrained", classmethod(lambda cls, *a, **k: object()))
        monkeypatch.setattr(regenerate_vlm, "_load_target_model", lambda model: object())

        def _fake_generate(model, processor, messages, config):
            prompts_seen.append(messages)
            return "a long generated answer"

        monkeypatch.setattr(regenerate_vlm, "_generate_answer", _fake_generate)

        config = regenerate_vlm.RegenerationConfig(
            model="m",
            dataset="d",
            split="train",
            image_root=str(image_root),
            output_dir=str(tmp_path / "out"),
            start=0,
            end=3,
            max_new_tokens=8,
            temperature=0.0,
            shuffle_seed=42,
            length_instruction=regenerate_vlm.LENGTH_INSTRUCTION,
        )
        data_path = regenerate_vlm.regenerate(config)

        written = [json.loads(line) for line in data_path.read_text().splitlines()]
        assert len(written) == 1  # the missing image and the human-turn-less row are skipped
        assert written[0]["images"] == ["a.jpg"]
        assert written[0]["conversations"][0]["value"] == "<image>\nDescribe."
        assert regenerate_vlm.LENGTH_INSTRUCTION not in written[0]["conversations"][0]["value"]
        assert written[0]["conversations"][1] == {"from": "gpt", "value": "a long generated answer"}
        # ...but the instruction WAS sent to the model.
        assert prompts_seen[0][0]["content"][-1]["text"] == regenerate_vlm.LENGTH_INSTRUCTION
        assert (tmp_path / "out" / "meta.json").exists()


class _StubConfigNode(dict):
    """Minimal stand-in for the recipe ConfigNode (attribute + ``get`` access)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _FakeDistributedDataParallel(nn.Module):
    """Minimal DDP stand-in that preserves parameter discovery in setup tests."""

    def __init__(self, module, **kwargs):
        super().__init__()
        self.module = module
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        """Forward to the wrapped trainer module."""
        return self.module(*args, **kwargs)


def _patch_setup_dependencies(monkeypatch, tmp_path, dataloader_batches=2):
    """Stub out everything ``setup()`` would download or distribute."""
    module = "nemo_automodel.recipes.llm.train_vispec"
    monkeypatch.setattr(
        f"{module}.initialize_distributed",
        lambda **kwargs: SimpleNamespace(device=torch.device("cpu"), world_size=1, is_main=True),
    )
    monkeypatch.setattr(f"{module}.setup_logging", lambda: None)

    target_config = SimpleNamespace(
        architectures=["Qwen2_5_VLForConditionalGeneration"],
        image_token_id=IMAGE_TOKEN_ID,
        get_text_config=_config,
    )
    monkeypatch.setattr(f"{module}.AutoConfig", SimpleNamespace(from_pretrained=lambda *a, **k: target_config))
    monkeypatch.setattr(f"{module}.AutoProcessor", SimpleNamespace(from_pretrained=lambda *a, **k: object()))
    monkeypatch.setattr(f"{module}.NeMoAutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: object()))
    monkeypatch.setattr(
        f"{module}.NeMoAutoModelForImageTextToText",
        SimpleNamespace(from_pretrained=lambda *a, **k: _StubTarget()),
    )
    monkeypatch.setattr(f"{module}.build_dspark_vlm_dataloader", lambda **kwargs: [None] * dataloader_batches)
    monkeypatch.setattr(TrainVispecRecipe, "_build_checkpointer", lambda self, target_path: None)
    monkeypatch.setattr(TrainVispecRecipe, "load_checkpoint", lambda self, restore_from=None: None)

    return _StubConfigNode(
        recipe_args=_StubConfigNode(
            target_model_name_or_path="stub/qwen2.5-vl",
            output_dir=str(tmp_path / "out"),
            seq_length=64,
            micro_batch_size=1,
            grad_accumulation_steps=2,
            num_epochs=1,
            num_query_tokens=2,
            mtp_steps=2,
            prob_loss_weight=5.0,
            rank_loss_weight=0.25,
        ),
        optimizer=_StubConfigNode(lr=3.0e-6),
        dataset=_StubConfigNode(),
    )


def _patch_stage1_setup_dependencies(monkeypatch, tmp_path, dataloader_batches=2):
    """Stub every external dependency used by the text-only stage-1 setup."""
    module = "nemo_automodel.recipes.llm.train_vispec"
    monkeypatch.setattr(
        f"{module}.initialize_distributed",
        lambda **kwargs: SimpleNamespace(device=torch.device("cpu"), world_size=1, is_main=True),
    )
    monkeypatch.setattr(f"{module}.setup_logging", lambda: None)

    target_config = SimpleNamespace(
        architectures=["Qwen2_5_VLForConditionalGeneration"],
        get_text_config=_config,
    )
    monkeypatch.setattr(f"{module}.AutoConfig", SimpleNamespace(from_pretrained=lambda *a, **k: target_config))
    monkeypatch.setattr(f"{module}.NeMoAutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: object()))
    monkeypatch.setattr(
        f"{module}.NeMoAutoModelForImageTextToText",
        SimpleNamespace(from_pretrained=lambda *a, **k: _StubTarget()),
    )
    monkeypatch.setattr(f"{module}.build_eagle3_dataloader", lambda **kwargs: [None] * dataloader_batches)
    monkeypatch.setattr(TrainVispecStage1Recipe, "_build_checkpointer", lambda self, target_path: None)
    monkeypatch.setattr(TrainVispecStage1Recipe, "load_checkpoint", lambda self, restore_from=None: None)

    return _StubConfigNode(
        recipe_args=_StubConfigNode(
            target_model_name_or_path="stub/qwen2.5-vl",
            train_data_path="stub/text-data",
            output_dir=str(tmp_path / "out"),
            seq_length=64,
            micro_batch_size=2,
            grad_accumulation_steps=2,
            num_epochs=1,
        ),
        optimizer=_StubConfigNode(lr=3.0e-5),
    )


class TestSetup:
    def test_rejects_micro_batch_size_above_one(self, monkeypatch, tmp_path):
        cfg = _patch_setup_dependencies(monkeypatch, tmp_path)
        cfg["recipe_args"]["micro_batch_size"] = 4
        with pytest.raises(ValueError, match="micro_batch_size=1"):
            TrainVispecRecipe(cfg).setup()

    def test_builds_a_vispec_draft_and_trainer_from_config(self, monkeypatch, tmp_path):
        recipe = TrainVispecRecipe(_patch_setup_dependencies(monkeypatch, tmp_path))
        recipe.setup()

        assert isinstance(recipe.draft_model, VispecDraftModel)
        assert recipe.draft_model.num_query_tokens == 2
        assert isinstance(recipe.trainer_module, VispecTrainerModule)
        assert recipe.trainer_module.mtp_steps == 2
        assert recipe.trainer_module.prob_loss_weight == 5.0
        assert recipe.trainer_module.rank_loss_weight == 0.25
        # Embeddings are copied from the target and frozen by default, so they
        # must not reach the optimizer.
        torch.testing.assert_close(
            recipe.draft_model.embed_tokens.weight, recipe.target_wrapper.get_input_embeddings().weight
        )
        assert not recipe.draft_model.embed_tokens.weight.requires_grad
        optimizer_params = {id(p) for group in recipe.optimizer.param_groups for p in group["params"]}
        assert id(recipe.draft_model.embed_tokens.weight) not in optimizer_params
        assert id(recipe.draft_model.img_fc.weight) in optimizer_params

    def test_lr_schedule_covers_every_optimizer_step(self, monkeypatch, tmp_path):
        recipe = TrainVispecRecipe(_patch_setup_dependencies(monkeypatch, tmp_path, dataloader_batches=5))
        recipe.setup()
        # 5 batches / grad_accumulation_steps=2 -> 3 steps (the trailing partial
        # window still runs an optimizer step).
        assert recipe.total_optim_steps == 3


class TestStage1Setup:
    def test_builds_a_plain_eagle_draft_with_a_vlm_teacher(self, monkeypatch, tmp_path):
        recipe = TrainVispecStage1Recipe(_patch_stage1_setup_dependencies(monkeypatch, tmp_path))
        recipe.setup()

        assert isinstance(recipe.draft_model, LlamaEagleDraftModel)
        assert not isinstance(recipe.trainer_module, VispecTrainerModule)
        torch.testing.assert_close(
            recipe.draft_model.embed_tokens.weight, recipe.target_wrapper.get_input_embeddings().weight
        )
        assert not recipe.draft_model.embed_tokens.weight.requires_grad

    def test_runs_a_text_only_vlm_teacher_step(self, monkeypatch, tmp_path):
        recipe = TrainVispecStage1Recipe(_patch_stage1_setup_dependencies(monkeypatch, tmp_path))
        recipe.setup()

        metrics = recipe._compute_metrics(
            {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 8, 9]]),
                "attention_mask": torch.ones(1, 8, dtype=torch.long),
                "loss_mask": torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1]]),
            }
        )

        assert torch.isfinite(metrics.loss)
        assert metrics.valid_tokens.item() == 3

    def test_builds_a_validation_dataloader_when_configured(self, monkeypatch, tmp_path):
        calls = []
        cfg = _patch_stage1_setup_dependencies(monkeypatch, tmp_path)
        cfg["recipe_args"]["val_data_path"] = "stub/validation-data"
        monkeypatch.setattr(
            "nemo_automodel.recipes.llm.train_vispec.build_eagle3_dataloader",
            lambda **kwargs: calls.append(kwargs) or [None],
        )

        recipe = TrainVispecStage1Recipe(cfg)
        recipe.setup()

        assert len(calls) == 2
        assert calls[1]["data_path"] == "stub/validation-data"
        assert calls[1]["shuffle"] is False

    def test_wraps_the_trainer_with_ddp_for_multiple_ranks(self, monkeypatch, tmp_path):
        cfg = _patch_stage1_setup_dependencies(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "nemo_automodel.recipes.llm.train_vispec.initialize_distributed",
            lambda **kwargs: SimpleNamespace(device=torch.device("cpu"), world_size=2, is_main=True),
        )
        monkeypatch.setattr(
            "nemo_automodel.recipes.llm.train_vispec.DistributedDataParallel", _FakeDistributedDataParallel
        )

        recipe = TrainVispecStage1Recipe(cfg)
        recipe.setup()

        assert isinstance(recipe.trainer_module, _FakeDistributedDataParallel)
        assert recipe.trainer_module.kwargs["find_unused_parameters"] is False

    def test_rejects_target_sharding(self, monkeypatch, tmp_path):
        cfg = _patch_stage1_setup_dependencies(monkeypatch, tmp_path)
        cfg["distributed"] = _StubConfigNode(tp_size=2)
        with pytest.raises(NotImplementedError, match="target-model sharding"):
            TrainVispecStage1Recipe(cfg).setup()

    def test_rejects_packing(self, monkeypatch, tmp_path):
        cfg = _patch_stage1_setup_dependencies(monkeypatch, tmp_path)
        cfg["recipe_args"]["packed_sequence_size"] = 128
        with pytest.raises(NotImplementedError, match="packed_sequence_size"):
            TrainVispecStage1Recipe(cfg).setup()


def test_stage1_recipe_name_resolves():
    assert (
        resolve_recipe_name("TrainVispecStage1Recipe")
        == "nemo_automodel.recipes.llm.train_vispec.TrainVispecStage1Recipe"
    )


def test_main_requires_a_config(monkeypatch):
    from nemo_automodel.recipes.llm import train_vispec

    monkeypatch.setattr("sys.argv", ["train_vispec"])
    with pytest.raises(ValueError, match="--config"):
        train_vispec.main()


@pytest.mark.parametrize(
    ("recipe_name", "expected_class_name"),
    [
        ("TrainVispecStage1Recipe", "TrainVispecStage1Recipe"),
        ("TrainVispecRecipe", "TrainVispecRecipe"),
    ],
)
def test_main_selects_the_recipe_from_config(monkeypatch, recipe_name, expected_class_name):
    """The stage-1 and stage-2 example YAMLs share one CLI entrypoint."""
    from nemo_automodel.recipes.llm import train_vispec

    events = []

    class _FakeRecipe:
        def __init__(self, cfg):
            self.cfg = cfg
            events.append(("init", type(self).__name__))

        def setup(self):
            events.append(("setup", type(self).__name__))

        def run_train_validation_loop(self):
            events.append(("run", type(self).__name__))

    stage1_cls = type("TrainVispecStage1Recipe", (_FakeRecipe,), {})
    stage2_cls = type("TrainVispecRecipe", (_FakeRecipe,), {})
    monkeypatch.setattr(train_vispec, "parse_args_and_load_config", lambda _: {"recipe": recipe_name})
    monkeypatch.setattr(train_vispec, "TrainVispecStage1Recipe", stage1_cls)
    monkeypatch.setattr(train_vispec, "TrainVispecRecipe", stage2_cls)

    train_vispec.main("test.yaml")

    assert events == [
        ("init", expected_class_name),
        ("setup", expected_class_name),
        ("run", expected_class_name),
    ]


def test_main_rejects_an_unknown_recipe(monkeypatch):
    from nemo_automodel.recipes.llm import train_vispec

    monkeypatch.setattr(train_vispec, "parse_args_and_load_config", lambda _: {"recipe": "TrainEagle1Recipe"})

    with pytest.raises(ValueError, match="Unsupported ViSpec recipe"):
        train_vispec.main("test.yaml")
