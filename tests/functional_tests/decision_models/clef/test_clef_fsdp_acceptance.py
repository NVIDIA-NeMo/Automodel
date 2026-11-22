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

"""Real two-GPU acceptance of CLEF's LoRA-backbone/replicated-head optimizer step.

A tiny Qwen3.5 conditional-generation model uses actual AutoModel LoRA modules,
FSDP2, CUDA/NCCL collectives and the production CLEF training loop. The reference
is the same unsharded model in CPU fp32, with a single global question mean.
No pretrained weights, remote model code or dataset download is needed.

Run with CUDA_VISIBLE_DEVICES set to the two devices assigned to the job::

    uv run pytest tests/functional_tests/decision_models/clef/test_clef_fsdp_acceptance.py -v

Optimizer and sharded-stage reload checks are separate parametrized cases.
Merged-export parity runs independently on CPU.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from transformers import Qwen3_5Config

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.decision.records import question_targets
from nemo_automodel.components.loss.decision import DecisionLossConfig
from nemo_automodel.components.loss.decision_batch import DecisionTargets, QuestionLogits, assemble_decision_batch
from nemo_automodel.components.models.clef.joint_schema_head import (
    ClefHeadConfig,
    EncodedQuestion,
    EncodedRecord,
    LexicalTable,
    clef_option_orders,
)
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
from nemo_automodel.recipes._typed_config import RecipeConfig
from nemo_automodel.recipes.decision.train_clef import TrainClefDecisionRecipe

_RESULT_PREFIX = "CLEF_FSDP_ACCEPTANCE "


def _model_config() -> Qwen3_5Config:
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


def _items(accumulation: int, *, empty_rank: bool) -> list:
    items = []
    for index in range(2 * accumulation):
        count = 1 if index % 2 == 0 else 3
        questions = tuple(
            EncodedQuestion(f"q{question}", 1, (0, 1), ((1, 2), (2, 3)), ("a", "b")) for question in range(count)
        )
        record = {
            "id": str(index),
            "state": "example",
            "questions": {
                question.question_id: {"type": "choice", "criteria": {"a": "A", "b": "B"}} for question in questions
            },
            "answers": {}
            if empty_rank and index % 2 == 0
            else {question.question_id: "a" if index % 2 == 0 else "b" for question in questions},
        }
        items.append((record, EncodedRecord((1, 2 + index, 3, 4), questions, str(index))))
    return items


def _recipe(
    output: Path, *, device: torch.device, accumulation: int, empty_rank: bool, max_norm: float
) -> TrainClefDecisionRecipe:
    torch.manual_seed(7)
    recipe = TrainClefDecisionRecipe.__new__(TrainClefDecisionRecipe)
    recipe.model = Qwen3_5ForConditionalGeneration(
        _model_config(),
        backend=BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            rope_fusion=False,
            dispatcher="torch",
            enable_hf_state_dict_adapter=True,
        ),
    )
    recipe.model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    recipe.peft_config = PeftConfig(
        target_modules=["*q_proj", "*v_proj"],
        dim=2,
        alpha=6,
        lora_dtype=torch.float32,
        use_memory_efficient_lora=False,
    )
    matched = apply_lora_to_linear_modules(recipe.model, recipe.peft_config)
    assert matched == 2
    with torch.no_grad():
        for name, parameter in recipe.model.named_parameters():
            if name.endswith("lora_B.weight"):
                nn.init.normal_(parameter, std=0.02)
    recipe.head = ClefHeadConfig(hidden_size=16, width=16, routing_layers=1, layers=1, heads=2, feedforward=32).build()
    recipe.lexical = LexicalTable(list(range(64)), recipe.model.get_output_embeddings().weight.detach().clone())
    recipe.pad_token_id = 0
    recipe.model.to(device)
    recipe.head.to(device)
    recipe.lexical.to(device)
    recipe.device = device
    recipe.device_mesh = None
    # Seed 2 preserves the different question populations across the two DP shards.
    recipe.seed = 2
    recipe.tag = "sft"
    recipe.dist_env = SimpleNamespace(is_main=device.type == "cpu" or dist.get_rank() == 0)
    recipe.output_dir = output
    recipe.loss_config = DecisionLossConfig(label_smoothing=0.0, brier_weight=0.0, ordinal_weight=0.0)
    recipe.cfg = RecipeConfig(
        ConfigNode(
            {
                "lr_scheduler": None,
                "train": {
                    "epochs": 1,
                    "local_batch_size": 1,
                    "global_batch_size": 2 * accumulation,
                    "warmup_ratio": 0.0,
                    "max_grad_norm": max_norm,
                    "val_every_steps": 100,
                },
            }
        )
    )
    recipe.splits = {"train": _items(accumulation, empty_rank=empty_rank)}
    recipe._get_dp_group_size = lambda: 1 if device.type == "cpu" else dist.get_world_size()
    recipe._get_dp_rank = lambda: 0 if device.type == "cpu" else dist.get_rank()
    recipe._dp_group = lambda: None if device.type == "cpu" else dist.group.WORLD
    # This test isolates the distributed optimizer step. Separate tests exercise actual stage/export I/O.
    recipe._save_stage = lambda path, **kwargs: None
    return recipe


def _trainable_parameters(recipe: TrainClefDecisionRecipe) -> dict[str, nn.Parameter]:
    return {
        f"{part_name}.{name}": parameter
        for part_name, part in (("backbone", recipe.model), ("head", recipe.head))
        for name, parameter in part.named_parameters()
        if parameter.requires_grad
    }


def _full_gradients(recipe: TrainClefDecisionRecipe) -> dict[str, torch.Tensor]:
    """Read real parameter gradients, gathering FSDP shards on every rank.

    Args:
        recipe: CLEF recipe whose trainable backbone gradients have DTensor Shard(0)
            placement and whose head gradients are replicated ordinary tensors.

    Returns:
        CPU fp32 tensors in each parameter's full global shape; LoRA A is
        [rank, input_features], LoRA B is [output_features, rank], and head
        tensors retain their corresponding unsharded parameter layouts.
    """
    result = {}
    for name, parameter in _trainable_parameters(recipe).items():
        assert parameter.grad is not None, f"missing gradient for {name}"
        gradient = parameter.grad
        if isinstance(gradient, DTensor):
            gradient = gradient.full_tensor()
        result[name] = gradient.detach().float().cpu().clone()
    return result


def _global_loss(recipe: TrainClefDecisionRecipe) -> torch.Tensor:
    """Compute the independent, single-process global question mean.

    Args:
        recipe: Unsharded CPU recipe whose model/head parameters are fp32.

    Returns:
        Scalar tensor connected to every trainable adapter and head parameter.
    """
    items = recipe.splits["train"]
    questions = [question_targets(record, clef_option_orders(encoded)) for record, encoded in items]
    targets = DecisionTargets.from_questions(record_ids=[str(record["id"]) for record, _ in items], questions=questions)
    scores = [
        [
            QuestionLogits(str(record["id"]), target.question_id, target.option_ids, row)
            for target, row in zip(record_targets, record_scores, strict=True)
        ]
        for (record, _), record_targets, record_scores in zip(items, questions, recipe._forward(items), strict=True)
    ]
    batch = assemble_decision_batch(logits=scores, targets=targets)
    loss, _ = recipe.loss_config.build()(batch.logits, batch.targets)
    return loss


def _assert_parameters(recipe: TrainClefDecisionRecipe, reference: TrainClefDecisionRecipe) -> None:
    expected = _trainable_parameters(reference)
    for name, parameter in _trainable_parameters(recipe).items():
        full = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
        torch.testing.assert_close(full.detach().float().cpu(), expected[name], atol=5e-6, rtol=3e-4, msg=name)


def _run_case(output: Path, *, accumulation: int, empty_rank: bool, max_norm: float) -> TrainClefDecisionRecipe:
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    reference = _recipe(
        output, device=torch.device("cpu"), accumulation=accumulation, empty_rank=empty_rank, max_norm=max_norm
    )
    _global_loss(reference).backward()
    reference_gradients = _full_gradients(reference)
    reference_norm = nn.utils.clip_grad_norm_(list(_trainable_parameters(reference).values()), max_norm)
    clipped_reference_gradients = _full_gradients(reference)
    if max_norm < 1:
        assert float(reference_norm) > max_norm, "the clipping case must actually exceed the joint budget"
    reference_optimizers = [
        torch.optim.SGD((parameter for parameter in reference.model.parameters() if parameter.requires_grad), lr=0.05),
        torch.optim.SGD(reference.head.parameters(), lr=0.05),
    ]
    for optimizer in reference_optimizers:
        optimizer.step()

    actual = _recipe(output, device=device, accumulation=accumulation, empty_rank=empty_rank, max_norm=max_norm)
    local_targets = [
        target
        for record, encoded in actual._shard(actual.splits["train"], shuffle=True, epoch=0)
        for target in question_targets(record, clef_option_orders(encoded))
    ]
    local_counts = torch.tensor(
        [len(local_targets), sum(target.label != -100 for target in local_targets)], device=device
    )
    rank_counts = [torch.empty_like(local_counts) for _ in range(2)]
    dist.all_gather(rank_counts, local_counts)
    assert int(rank_counts[0][0]) != int(rank_counts[1][0]), "fixture must have unequal question counts"
    if empty_rank:
        assert sum(int(count[1]) == 0 for count in rank_counts) == 1, (
            "fixture must have one entirely feedback-free rank"
        )
    else:
        assert all(int(count[1]) > 0 for count in rank_counts)
    mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("dp_shard",))
    fp32_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32, reduce_dtype=torch.float32, output_dtype=torch.float32
    )
    fully_shard(actual.model, mesh=mesh, mp_policy=fp32_policy)
    actual.device_mesh = mesh
    actual.optimizer = [
        torch.optim.SGD((parameter for parameter in actual.model.parameters() if parameter.requires_grad), lr=0.05)
    ]
    actual.head_optimizer = torch.optim.SGD(actual.head.parameters(), lr=0.05)
    assert all(isinstance(parameter, DTensor) for parameter in actual.model.parameters() if parameter.requires_grad)
    assert all(not isinstance(parameter, DTensor) for parameter in actual.head.parameters())

    raw_gradients = {}
    clipped_gradients = {}
    synchronize_head = actual._sync_head_grads

    def synchronize_and_record() -> None:
        synchronize_head()
        raw_gradients.update(_full_gradients(actual))

    def record_clipped(optimizer, args, kwargs) -> None:
        clipped_gradients.update(_full_gradients(actual))

    actual._sync_head_grads = synchronize_and_record
    handle = actual.optimizer[0].register_step_pre_hook(record_clipped)
    try:
        actual._train()
    finally:
        handle.remove()
    assert raw_gradients.keys() == reference_gradients.keys()
    assert all(parameter.grad is None for parameter in actual.model.parameters() if not parameter.requires_grad)
    assert any("lora_A" in name and float(value.abs().max()) > 0 for name, value in raw_gradients.items())
    assert any("lora_B" in name and float(value.abs().max()) > 0 for name, value in raw_gradients.items())
    for name in raw_gradients:
        torch.testing.assert_close(raw_gradients[name], reference_gradients[name], atol=5e-6, rtol=3e-4, msg=name)
        torch.testing.assert_close(
            clipped_gradients[name], clipped_reference_gradients[name], atol=5e-6, rtol=3e-4, msg=name
        )
    actual_norm = torch.linalg.vector_norm(torch.cat([gradient.flatten() for gradient in clipped_gradients.values()]))
    assert float(actual_norm) <= max_norm + 2e-6
    _assert_parameters(actual, reference)
    if dist.get_rank() == 0:
        rows = [json.loads(line) for line in (output / "training_log.jsonl").read_text().splitlines()]
        reported_norm = next(row["grad_norm"] for row in rows if "grad_norm" in row)
        assert reported_norm == pytest.approx(float(reference_norm), rel=3e-4, abs=5e-6)
    dist.barrier()
    return actual


def _run_stage_reload(output: Path) -> None:
    actual = _run_case(output, accumulation=2, empty_rank=True, max_norm=0.1)
    actual.model.eval()
    actual.head.eval()
    checkpointer = CheckpointingConfig(
        checkpoint_dir=output, is_peft=True, save_consolidated=False, is_async=False
    ).build(dp_rank=dist.get_rank(), tp_rank=0, pp_rank=0, process_group=dist.group.WORLD)
    actual.checkpointer = checkpointer
    del actual._save_stage
    batch = actual.splits["train"][:1]
    with torch.no_grad():
        expected_logits = [[row.detach().clone() for row in record] for record in actual._forward(batch)]
    expected_parameters = {
        name: (parameter.full_tensor() if isinstance(parameter, DTensor) else parameter).detach().clone()
        for name, parameter in _trainable_parameters(actual).items()
    }
    parameter_ids = {name: id(parameter) for name, parameter in _trainable_parameters(actual).items()}
    stage = output / "stage"
    try:
        actual._save_stage(stage)
        with torch.no_grad():
            for parameter in _trainable_parameters(actual).values():
                parameter.add_(0.125)
        actual._load_stage(stage)
        with torch.no_grad():
            restored_logits = actual._forward(batch)
        for expected_record, restored_record in zip(expected_logits, restored_logits, strict=True):
            for expected_row, restored_row in zip(expected_record, restored_record, strict=True):
                torch.testing.assert_close(restored_row, expected_row, atol=2e-6, rtol=2e-5)
        for name, parameter in _trainable_parameters(actual).items():
            assert id(parameter) == parameter_ids[name], f"stage initialization replaced optimizer parameter {name}"
            full = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
            torch.testing.assert_close(full, expected_parameters[name], atol=0, rtol=0, msg=name)
        if dist.get_rank() == 0:
            assert (stage / "model" / "adapter_model.safetensors").exists()
            assert (stage / "head" / ".metadata").exists()
        dist.barrier()
    finally:
        checkpointer.close()


def _run_worker(output: Path, *, case: str) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    try:
        if case == "optimizer":
            for accumulation, empty_rank, max_norm in ((1, False, 1e6), (2, False, 0.1), (2, True, 0.1)):
                case_output = output / f"accum-{accumulation}-empty-{empty_rank}-norm-{max_norm}"
                case_output.mkdir(parents=True, exist_ok=True)
                _run_case(case_output, accumulation=accumulation, empty_rank=empty_rank, max_norm=max_norm)
        else:
            _run_stage_reload(output)
        if dist.get_rank() == 0:
            print(_RESULT_PREFIX + case + " PASS", flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two assigned CUDA GPUs for actual FSDP2 acceptance")
@pytest.mark.parametrize("case", ["optimizer", "stage_reload"])
def test_clef_lora_fsdp_training_acceptance(tmp_path, case: str) -> None:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--worker-output",
        str(tmp_path),
        "--case",
        case,
    ]
    # Respect the scheduler's assigned devices; do not expose GPUs outside CUDA_VISIBLE_DEVICES.
    environment = os.environ.copy()
    repository = str(Path(__file__).resolve().parents[4])
    environment["PYTHONPATH"] = repository + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        command, cwd=repository, env=environment, text=True, capture_output=True, timeout=600, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _RESULT_PREFIX + case + " PASS" in completed.stdout, completed.stdout + completed.stderr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-output", type=Path, required=True)
    parser.add_argument("--case", choices=("optimizer", "stage_reload"), required=True)
    arguments = parser.parse_args()
    _run_worker(arguments.worker_output, case=arguments.case)
