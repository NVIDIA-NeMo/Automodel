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

"""Merged CLEF release logits match a trained tiny Qwen3.5/LoRA decision model."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.clef.export import export_release
from nemo_automodel.components.models.clef.joint_schema_head import (
    ClefHeadConfig,
    EncodedQuestion,
    EncodedRecord,
    JointSchemaHead,
    LexicalTable,
    load_joint_head,
    save_joint_head,
)
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.qwen3_5.model import (
    Qwen3_5ForConditionalGeneration as NativeQwen3_5ForConditionalGeneration,
)
from nemo_automodel.recipes.decision.train_clef import TrainClefDecisionRecipe


def _config() -> Qwen3_5Config:
    return Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config={
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 32,
            "layer_types": ["full_attention"],
            "attn_implementation": "eager",
            "tie_word_embeddings": False,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "out_hidden_size": 16,
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
        tie_word_embeddings=False,
    )


def _logits(model: nn.Module, head: JointSchemaHead) -> torch.Tensor:
    """Score the same local record using the trained or exported model.

    Args:
        model: Qwen3.5 model containing weights [output_features, input_features]
            and optional trainable LoRA A [rank, input_features]/B [output_features, rank].
        head: Joint schema head with its native parameter layouts and fp32 arithmetic.

    Returns:
        Tensor of shape [questions, options], here [2, 2], in fp32.
    """
    ids = torch.tensor([[1, 2, 3, 4]])
    valid = torch.ones_like(ids)
    encoded = EncodedRecord(
        (1, 2, 3, 4),
        (
            EncodedQuestion("first", 1, (0, 1), ((1, 2), (2, 3)), ("a", "b")),
            EncodedQuestion("second", 1, (0, 1), ((1, 2), (2, 3)), ("a", "b")),
        ),
        "example",
    )
    output = model(input_ids=ids, attention_mask=valid, use_cache=False, logits_to_keep=1, output_hidden_states=True)
    lexical = LexicalTable(list(range(64)), model.get_output_embeddings().weight.detach())
    return torch.stack(head(output.hidden_states[-1], ids, valid, [encoded], lexical)[0])


@pytest.mark.parametrize("backbone", ["hf", "native"])
@pytest.mark.parametrize("stage_input", ["directory", "symlink", "text_pointer"])
def test_actual_merged_release_reload_preserves_trained_decision_logits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backbone: str, stage_input: str
) -> None:
    torch.manual_seed(17)
    released = Qwen3_5ForConditionalGeneration(_config())
    base = tmp_path / "base"
    released.save_pretrained(base)
    model = released
    if backbone == "native":
        model = NativeQwen3_5ForConditionalGeneration(
            _config(),
            backend=BackendConfig(
                linear="torch",
                attn="sdpa",
                rms_norm="torch",
                rope_fusion=False,
                dispatcher="torch",
                enable_hf_state_dict_adapter=True,
            ),
        )
        model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
        model.load_state_dict(model.state_dict_adapter.from_hf(released.state_dict()))
        roundtrip = model.state_dict_adapter.to_hf(model.state_dict())
        assert roundtrip.keys() == released.state_dict().keys()
        for name, value in released.state_dict().items():
            torch.testing.assert_close(roundtrip[name], value, rtol=0, atol=0)
    peft = PeftConfig(
        target_modules=["*q_proj", "*v_proj"],
        dim=2,
        alpha=6,
        lora_dtype=torch.float32,
        use_memory_efficient_lora=False,
    )
    assert apply_lora_to_linear_modules(model, peft) == 2
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("lora_B.weight"):
                nn.init.normal_(parameter, std=0.1)
    head = ClefHeadConfig(hidden_size=16, width=16, routing_layers=1, layers=1, heads=2, feedforward=32).build()
    save_joint_head(head, base, dtype=torch.float32)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad] + list(head.parameters()), lr=0.05
    )
    model.train()
    head.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(_logits(model, head), torch.tensor([0, 1])).backward()
        optimizer.step()
    stage = tmp_path / "stage"
    run = TrainClefDecisionRecipe.__new__(TrainClefDecisionRecipe)
    run.model, run.head, run.peft_config = model, head, peft
    run.tag = "sft"
    run.cfg = {}
    run.dist_env = SimpleNamespace(is_main=True)
    run.checkpointer = CheckpointingConfig(checkpoint_dir=tmp_path, is_peft=True, save_consolidated=False).build(
        dp_rank=0, tp_rank=0, pp_rank=0
    )
    destination = tmp_path / "merged"
    try:
        run._save_stage(stage, selection_score=0.25)
        if stage_input == "directory":
            export_release(base, revision="local-fixture", stage=stage, out=destination)
        else:
            pointer = tmp_path / "LOWEST_VAL"
            assert pointer.is_symlink()
            if stage_input == "text_pointer":
                pointer.unlink()
                pointer.with_suffix(".txt").write_text(stage.name)
            script = Path(__file__).resolve().parents[4] / "examples/decision/export_clef_release.py"
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    str(script),
                    "--base",
                    str(base),
                    "--revision",
                    "local-fixture",
                    "--stage",
                    str(pointer),
                    "--out",
                    str(destination),
                ],
            )
            runpy.run_path(str(script), run_name="__main__")
    finally:
        run.checkpointer.close()
    merged = Qwen3_5ForConditionalGeneration.from_pretrained(destination, dtype=torch.bfloat16).eval()
    merged_head = load_joint_head(destination, dtype=torch.float32).eval()
    cast_model_to_dtype(model, torch.bfloat16)
    model.eval()
    head.eval()
    with torch.no_grad():
        expected = _logits(model, head)
        actual = _logits(merged, merged_head)
        base_logits = _logits(Qwen3_5ForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16).eval(), head)
    # The release exporter deliberately writes a bf16 backbone. Merging before
    # that rounding differs from the live bf16 low-rank matmuls by a few ulps.
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-2)
    assert float((expected - base_logits).abs().max()) > 1e-5, "fixture must distinguish a dropped adapter"
    assert not any("lora_" in key for key in merged.state_dict())
    for name, parameter in head.state_dict().items():
        torch.testing.assert_close(merged_head.state_dict()[name], parameter, rtol=0, atol=0)


def test_native_stage_initialization_restores_weights_and_keeps_optimizer_state_fresh(tmp_path: Path) -> None:
    from copy import deepcopy

    from nemo_automodel.components.checkpoint.utils import read_checkpoint_pointer

    torch.manual_seed(23)
    peft = PeftConfig(
        target_modules=["*q_proj", "*v_proj"],
        dim=2,
        alpha=6,
        lora_dtype=torch.float32,
        use_memory_efficient_lora=False,
    )
    base = Qwen3_5ForConditionalGeneration(_config())
    target_model = deepcopy(base)
    assert apply_lora_to_linear_modules(base, peft) == 2
    assert apply_lora_to_linear_modules(target_model, peft) == 2
    source = TrainClefDecisionRecipe.__new__(TrainClefDecisionRecipe)
    source.model = base
    source.head = ClefHeadConfig(hidden_size=16, width=16, routing_layers=1, layers=1, heads=2, feedforward=32).build()
    source.peft_config = peft
    source.tag = "sft"
    source.cfg = {}
    source.dist_env = SimpleNamespace(is_main=True)
    source.checkpointer = CheckpointingConfig(checkpoint_dir=tmp_path, is_peft=True, save_consolidated=False).build(
        dp_rank=0, tp_rank=0, pp_rank=0
    )
    source_optimizer = torch.optim.AdamW(
        [parameter for parameter in source.model.parameters() if parameter.requires_grad]
        + list(source.head.parameters()),
        lr=0.05,
    )
    target = TrainClefDecisionRecipe.__new__(TrainClefDecisionRecipe)
    target.model = target_model
    target.head = ClefHeadConfig(hidden_size=16, width=16, routing_layers=1, layers=1, heads=2, feedforward=32).build()
    target.peft_config = peft
    target.tag = "rl"
    target.cfg = {}
    target.dist_env = SimpleNamespace(is_main=True)
    target.checkpointer = CheckpointingConfig(checkpoint_dir=tmp_path, is_peft=True, save_consolidated=False).build(
        dp_rank=0, tp_rank=0, pp_rank=0
    )
    target_optimizer = torch.optim.AdamW(
        [parameter for parameter in target.model.parameters() if parameter.requires_grad]
        + list(target.head.parameters()),
        lr=0.05,
    )
    target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=1, gamma=0.1)
    fresh_schedule = deepcopy(target_scheduler.state_dict())
    parameter_ids = [id(parameter) for group in target_optimizer.param_groups for parameter in group["params"]]
    try:
        for step in range(2):
            source_optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(_logits(source.model, source.head), torch.tensor([0, 1])).backward()
            source_optimizer.step()
            source._save_stage(tmp_path / f"step_{step:08d}", selection_score=0.2 + 0.1 * step)
            if step == 0:
                first_adapter = {
                    name: parameter.detach().clone()
                    for name, parameter in source.model.named_parameters()
                    if parameter.requires_grad
                }
                first_head = {name: value.detach().clone() for name, value in source.head.state_dict().items()}
        assert source_optimizer.state
        latest = tmp_path / "step_00000001"
        assert read_checkpoint_pointer(tmp_path, "LOWEST_VAL").resolve() == latest.resolve()
        assert read_checkpoint_pointer(tmp_path, "LATEST").resolve() == latest.resolve()
        assert (tmp_path / "step_00000000" / "stage.json").exists()
        target._load_stage(tmp_path / "step_00000000")
        for name, parameter in target.model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter, first_adapter[name], atol=0, rtol=0)
        for name, value in target.head.state_dict().items():
            torch.testing.assert_close(value, first_head[name], atol=0, rtol=0)
        target._load_stage(tmp_path / "LOWEST_VAL")
        with torch.no_grad():
            torch.testing.assert_close(
                _logits(target.model.eval(), target.head.eval()),
                _logits(source.model.eval(), source.head.eval()),
                atol=2e-6,
                rtol=2e-5,
            )
        assert not target_optimizer.state
        assert target_scheduler.state_dict() == fresh_schedule
        assert [
            id(parameter) for group in target_optimizer.param_groups for parameter in group["params"]
        ] == parameter_ids
    finally:
        source.checkpointer.close()
        target.checkpointer.close()
