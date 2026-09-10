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

"""Validate actual EP training and exact checkpoint continuation on a tiny V4.1.

Run with torchrun on one GB200 node. The default 50-step model preserves all
40 attention layers, both Engram layers, candidate pruning, and activation
checkpointing. A fresh model resumes halfway through the uninterrupted run.
Model, Adam, scheduler, RNG, loader position, batches, losses and final weights
must match exactly. An explicit TileLang diagnostic mode measures nondeterministic
backward continuation without declaring numerical differences a pass. Checkpoint
restoration itself remains exact; this script does not establish pretrained
logit parity.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from run_reference_parity import _native_source_manifest
from tokenizers import Tokenizer, models
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerFast

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.distributed import DistributedSetup, FSDP2Config, ParallelismSizes
from nemo_automodel.components.distributed.config import MoEParallelizerConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import (
    DeepseekV41Config,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
)
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm


@dataclass(frozen=True)
class TrainingConfig:
    """Dimensions and runtime controls for the complete tiny layer schedule."""

    output_dir: Path
    steps: int = 50
    checkpoint_step: int | None = None
    sequence_length: int = 32
    seed: int = 42
    experts: str = "torch_linear"
    attention_backend: str = "tilelang"
    activation_checkpointing: bool = True
    diagnostic_continuation: bool = False
    repeat_uninterrupted: bool = False

    def __post_init__(self) -> None:
        if self.steps < 2 or self.sequence_length < 16:
            raise ValueError("Use at least two steps and 16 tokens to exercise a resumed candidate-pruning path")
        split = self.steps // 2 if self.checkpoint_step is None else self.checkpoint_step
        if not 0 < split < self.steps:
            raise ValueError("checkpoint_step must be strictly between zero and steps")
        if self.diagnostic_continuation and self.attention_backend != "tilelang":
            raise ValueError("SDPA continuation must remain exact; diagnostics are only for TileLang backward")
        object.__setattr__(self, "checkpoint_step", split)

    def build_model(self, setup: DistributedSetup) -> torch.nn.Module:
        """Construct through NeMo's real FSDP2 and HybridEP onboarding path."""
        vocab = {"[UNK]": 0, " The": 1, "the": 2, "THE": 3, "é": 4, "E": 5, " ": 6, "x": 7, "\ufffd": 8}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="[UNK]")), unk_token="[UNK]"
        )
        text = DeepseekV41TextConfig(
            vocab_size=64,
            hidden_size=64,
            moe_intermediate_size=128,
            num_hidden_layers=40,
            num_attention_heads=4,
            # The TileLang MMA schedule needs at least 64 channels. Keep the
            # smaller SDPA fixture for isolated checkpoint debugging.
            head_dim=64 if self.attention_backend == "tilelang" else 32,
            qk_rope_head_dim=8,
            q_lora_rank=32,
            o_lora_rank=32,
            o_groups=1,
            n_routed_experts=max(4, dist.get_world_size()),
            num_experts_per_tok=2,
            index_n_heads=2,
            index_head_dim=32,
            index_topk=8,
            candidate_topk_blocks=2,
            candidate_block_size=4,
            engram_num_embeddings=[113, 253],
            engram_vocab_size=11,
            engram_n_heads=2,
            engram_head_dim=4,
            engram_pad_token_id=2,
            engram_compressed_vocab_size=6,
            num_nextn_predict_layers=0,
            dspark_block_size=0,
            dspark_noise_token_id=0,
            dtype="bfloat16",
        )
        return NeMoAutoModelForCausalLM.from_config(
            DeepseekV41Config(text_config=text, vision_config=DeepseekV41VisionConfig(num_hidden_layers=0)),
            backend=BackendConfig(
                attn=self.attention_backend,
                linear="torch",
                rms_norm="torch_fp32",
                experts=self.experts,
                dispatcher="hybridep",
                gate_precision="float32",
                enable_hf_state_dict_adapter=True,
            ),
            tokenizer=tokenizer,
            distributed_setup=setup,
            use_liger_kernel=False,
            use_sdpa_patching=False,
            torch_dtype=torch.bfloat16,
        )

    def build_dataloader(self) -> StatefulDataLoader:
        """Use a real stateful shuffled loader with distinct per-rank examples."""
        return StatefulDataLoader(
            TokenDataset(self.steps * 2, self.sequence_length, self.seed + 1000 * dist.get_rank()),
            batch_size=1,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(self.seed + dist.get_rank()),
        )

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.AdamW:
        """Keep frozen inference-only indexer weights outside Adam state."""
        return torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-3)


@dataclass(frozen=True)
class TokenDataset(Dataset):
    """Index-deterministic token examples, leaving iterator state to torchdata."""

    size: int
    sequence_length: int
    seed: int

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self.seed + index)
        return torch.randint(0, 9, (self.sequence_length,), generator=generator, device="cpu")


@dataclass(frozen=True)
class StepResult:
    """Observed batch, loss and norm for one optimizer update."""

    step: int
    loss: float
    grad_norm: float
    tokens: tuple[int, ...]


@dataclass
class TrainingProgress:
    """Checkpointed next optimizer-step index."""

    step: int = 0

    def state_dict(self) -> dict[str, int]:
        return {"step": self.step}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.step = state["step"]


def _snapshot(value: Any) -> Any:
    """Copy checkpoint-state leaves to local CPU storage without gathering."""
    if isinstance(value, torch.Tensor):
        local = value.to_local() if isinstance(value, DTensor) else value
        return local.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot(item) for item in value)
    return value


def _assert_exact(actual: Any, expected: Any, path: str) -> None:
    """Compare nested checkpoint states exactly, including every tensor leaf."""
    if isinstance(expected, torch.Tensor):
        local = actual.to_local() if isinstance(actual, DTensor) else actual
        if local.dtype != expected.dtype or local.shape != expected.shape or not torch.equal(local.cpu(), expected):
            raise AssertionError(
                f"Checkpoint continuation differs at {path}: "
                f"actual={local.shape}/{local.dtype}, expected={expected.shape}/{expected.dtype}"
            )
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(
                f"Checkpoint continuation keys differ at {path}: "
                f"missing={list(expected.keys() - actual.keys())[:10]}, "
                f"unexpected={list(actual.keys() - expected.keys())[:10]}"
            )
        for key in expected:
            _assert_exact(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError(f"Checkpoint continuation sequence differs at {path}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_exact(left, right, f"{path}.{index}")
    elif actual != expected:
        raise AssertionError(f"Checkpoint continuation differs at {path}: {actual!r} != {expected!r}")


def _training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rng: StatefulRNG,
    loader: StatefulDataLoader,
    progress: TrainingProgress,
) -> dict[str, Any]:
    """Capture framework state with stable optimizer parameter names.

    Raw optimizer integer IDs may change when DCP restores FSDP parameters.
    Canonical FQNs compare every moment and parameter-group field against the
    same actual parameter, independent of that serialization detail.
    Before the first update, inspect the empty optimizer directly so DCP does
    not initialize its moments merely to capture the random initial state.
    """
    return {
        "model": model.state_dict(),
        "optimizer": get_optimizer_state_dict(
            model, optimizer, options=StateDictOptions(flatten_optimizer_state_dict=True)
        )
        if optimizer.state
        else optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": rng.state_dict(),
        "dataloader": loader.state_dict(),
        "progress": progress.state_dict(),
    }


def _tensor_differences(actual: Any, expected: Any, path: str, changes: dict[str, Any]) -> int:
    """Audit every final tensor and record differences without imposing a tolerance.

    Shapes, dtypes, and non-tensor state must still match exactly. Each changed
    floating tensor records its error and reference norms, including references
    that are identically zero, so no relative-error floor hides small moments.
    """
    if isinstance(expected, torch.Tensor):
        local = actual.to_local() if isinstance(actual, DTensor) else actual
        if local.shape != expected.shape or local.dtype != expected.dtype:
            _assert_exact(actual, expected, path)
        local = local.detach().cpu()
        if torch.equal(local, expected):
            return 1
        if not local.is_floating_point() or not torch.isfinite(local).all():
            _assert_exact(actual, expected, path)
        difference = local.double() - expected.double()
        error_norm = torch.linalg.vector_norm(difference).item()
        reference_norm = torch.linalg.vector_norm(expected.double()).item()
        changes[path] = {
            "numel": local.numel(),
            "changed_elements": torch.count_nonzero(difference).item(),
            "max_abs_error": difference.abs().max().item(),
            "error_l2": error_norm,
            "reference_l2": reference_norm,
            "relative_rmse": error_norm / reference_norm if reference_norm else None,
        }
        return 1
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            _assert_exact(actual, expected, path)
        return sum(_tensor_differences(actual[key], value, f"{path}.{key}", changes) for key, value in expected.items())
    if isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            _assert_exact(actual, expected, path)
        return sum(
            _tensor_differences(left, right, f"{path}.{index}", changes)
            for index, (left, right) in enumerate(zip(actual, expected))
        )
    _assert_exact(actual, expected, path)
    return 0


def _step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
    setup: DistributedSetup,
    step: int,
) -> StepResult:
    """Run one real forward/backward, verifying all trainable gradients."""
    # Make every RNG family affect the observed batch, so restoring an unused
    # RNG state cannot produce a false-positive continuation comparison.
    offset = random.randrange(9) + int(np.random.randint(9)) + int(torch.randint(9, (), device="cpu"))
    inputs = (batch.cuda() + offset + torch.randint(9, (), device="cuda")) % 9
    optimizer.zero_grad(set_to_none=True)
    print(json.dumps({"rank": dist.get_rank(), "step": step, "phase": "forward"}), flush=True)
    output = model(inputs, labels=inputs)
    if not torch.isfinite(output.loss):
        raise AssertionError(f"Non-finite loss at step {step}")
    print(json.dumps({"rank": dist.get_rank(), "step": step, "phase": "backward"}), flush=True)
    output.loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is None:
            raise AssertionError(f"Missing gradient at step {step}: {name}")
        if parameter.grad is not None:
            gradient = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
            if not torch.isfinite(gradient).all():
                raise AssertionError(f"Non-finite gradient at step {step}: {name}")
    norm = scale_grads_and_clip_grad_norm(
        max_grad_norm=1.0,
        model_parts=[model],
        moe_mesh=setup.mesh_context.moe_mesh,
        dp_group_size=dist.get_world_size(),
    )
    optimizer.step()
    result = StepResult(step, output.loss.item(), float(norm), tuple(inputs.cpu().flatten().tolist()))
    print(
        json.dumps({"rank": dist.get_rank(), "step": step, "loss": result.loss, "grad_norm": result.grad_norm}),
        flush=True,
    )
    return result


def _final_differences(actual: dict[str, Any], expected: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Compare numerical model/Adam state and require all other final state exactly."""
    changes: dict[str, Any] = {}
    tensor_count = 0
    for field in actual:
        if field in ("model", "optimizer"):
            tensor_count += _tensor_differences(actual[field], expected[field], field, changes)
        else:
            _assert_exact(actual[field], expected[field], f"final.{field}")
    return tensor_count, changes


def _training_source_manifest() -> dict[str, str]:
    """Record shared numerical, initialization, and checkpoint implementation."""
    root = Path(__file__).resolve().parents[4]
    manifest = _native_source_manifest()
    paths = {Path(__file__).resolve()}
    for directory in ("checkpoint", "distributed", "training"):
        paths.update((root / "nemo_automodel/components" / directory).rglob("*.py"))
    for path in paths:
        manifest[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(manifest.items()))


def run(config: TrainingConfig) -> None:
    """Compare an uninterrupted run against restoration into a fresh model."""
    source_manifest = _training_source_manifest()
    rank = dist.get_rank()
    rng = StatefulRNG(config.seed)
    setup = DistributedSetup.build(
        strategy=FSDP2Config(),
        parallelism_sizes=ParallelismSizes(ep_size=dist.get_world_size()),
        moe_parallel_config=MoEParallelizerConfig(
            reshard_after_forward=True,
            lm_head_precision="float32",
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=None, cast_forward_inputs=False
            ),
        ),
        activation_checkpointing=config.activation_checkpointing,
        world_size=dist.get_world_size(),
    )
    construction_rng = _snapshot(rng.state_dict()) if config.repeat_uninterrupted else None
    model = config.build_model(setup)
    optimizer = config.build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)
    loader = config.build_dataloader()
    iterator = iter(loader)
    rng = StatefulRNG(config.seed + 500, ranked=True)
    checkpointer = Checkpointer(
        CheckpointingConfig(
            checkpoint_dir=str(config.output_dir),
            model_save_format="torch_save",
            save_consolidated=False,
            model_cache_dir=str(config.output_dir / "cache"),
            model_repo_id="test/deepseek-v41",
            dequantize_base_checkpoint=True,
        ),
        dp_rank=rank,
        tp_rank=0,
        pp_rank=0,
        process_group=dist.group.WORLD,
        moe_mesh=setup.mesh_context.moe_mesh,
    )
    checkpoint_path = str(config.output_dir / "checkpoint")
    baseline: list[StepResult] = []
    progress = TrainingProgress()
    initial_state = (
        _snapshot(_training_state(model, optimizer, scheduler, rng, loader, progress))
        if config.repeat_uninterrupted
        else None
    )
    split_state = None
    try:
        for index in range(config.steps):
            baseline.append(_step(model, optimizer, next(iterator), setup, index))
            scheduler.step()
            progress.step = index + 1
            if index + 1 == config.checkpoint_step:
                checkpointer.save_model(model, checkpoint_path)
                checkpointer.save_optimizer(optimizer, model, checkpoint_path, scheduler=scheduler)
                split_state = _snapshot(_training_state(model, optimizer, scheduler, rng, loader, progress))
                checkpointer.save_on_global_ranks(rng, "rng", checkpoint_path)
                checkpointer.save_on_dp_ranks(loader, "dataloader", checkpoint_path)
                checkpointer.save_on_global_ranks(progress, "progress", checkpoint_path)
        expected = _snapshot(_training_state(model, optimizer, scheduler, rng, loader, progress))
        del optimizer, scheduler, iterator, loader, model
        gc.collect()
        torch.cuda.empty_cache()
        model = config.build_model(setup)
        optimizer = config.build_optimizer(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)
        loader = config.build_dataloader()
        checkpointer.load_model(model, str(Path(checkpoint_path) / "model"))
        checkpointer.load_optimizer(optimizer, model, checkpoint_path, scheduler=scheduler)
        checkpointer.load_on_dp_ranks(loader, "dataloader", checkpoint_path)
        iterator = iter(loader)
        checkpointer.load_on_global_ranks(rng, "rng", checkpoint_path)
        progress = TrainingProgress()
        checkpointer.load_on_global_ranks(progress, "progress", checkpoint_path)
        restored = _training_state(model, optimizer, scheduler, rng, loader, progress)
        _assert_exact(restored, split_state, "restored")
        continuation: list[StepResult] = []
        for index in range(progress.step, config.steps):
            result = _step(model, optimizer, next(iterator), setup, index)
            continuation.append(result)
            scheduler.step()
            progress.step = index + 1
            if result.tokens != baseline[index].tokens or result.step != baseline[index].step:
                raise AssertionError(f"Resumed batch at step {index} differs from the uninterrupted run")
            if result != baseline[index] and not config.diagnostic_continuation:
                raise AssertionError(f"Resumed step {index} differs: {result} vs {baseline[index]}")
        final_state = _training_state(model, optimizer, scheduler, rng, loader, progress)
        tensor_count, changes = _final_differences(final_state, expected)
        continuation_exact = not changes and all(item == baseline[item.step] for item in continuation)
        if not config.diagnostic_continuation:
            _assert_exact(final_state, expected, "final")
        result = {
            "passed": continuation_exact,
            "checkpoint_restore_exact": True,
            "continuation_exact": continuation_exact,
            "diagnostic_continuation": config.diagnostic_continuation,
            "numerical_tolerance_applied": False,
            "rank": rank,
            "world_size": dist.get_world_size(),
            "steps": config.steps,
            "checkpoint_step": config.checkpoint_step,
            "num_hidden_layers": 40,
            "engram_layers": [1, 14],
            "dispatcher": "hybridep",
            "experts": config.experts,
            "attention_backend": config.attention_backend,
            "head_dim": model.config.text_config.head_dim,
            "source_hashes_before_training": source_manifest,
            "activation_checkpointing": config.activation_checkpointing,
            "restore_exact_fields": list(restored),
            "continuation_exact_fields": ["scheduler", "rng", "dataloader", "progress", "batches"],
            "compared_final_tensors": tensor_count,
            "changed_final_tensors": changes,
            "baseline": [asdict(item) for item in baseline],
            "continuation": [asdict(item) for item in continuation],
            "max_loss_abs_error": max(abs(item.loss - baseline[item.step].loss) for item in continuation),
            "max_grad_norm_abs_error": max(
                abs(item.grad_norm - baseline[item.step].grad_norm) for item in continuation
            ),
        }
        if config.repeat_uninterrupted:
            # Reconstruct the same random initial model without loading any
            # training checkpoint, then measure the backend's own noise floor.
            del optimizer, scheduler, iterator, loader, model, restored, final_state
            gc.collect()
            torch.cuda.empty_cache()
            rng.load_state_dict(construction_rng)
            model = config.build_model(setup)
            optimizer = config.build_optimizer(model)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)
            loader = config.build_dataloader()
            iterator = iter(loader)
            rng = StatefulRNG(config.seed + 500, ranked=True)
            progress = TrainingProgress()
            _assert_exact(
                _training_state(model, optimizer, scheduler, rng, loader, progress), initial_state, "repeated_initial"
            )
            repeated: list[StepResult] = []
            for index in range(config.steps):
                step_result = _step(model, optimizer, next(iterator), setup, index)
                repeated.append(step_result)
                if step_result.tokens != baseline[index].tokens:
                    raise AssertionError(f"Independent uninterrupted batch {index} differs from the first baseline")
                scheduler.step()
                progress.step = index + 1
            repeat_count, repeat_changes = _final_differences(
                _training_state(model, optimizer, scheduler, rng, loader, progress), expected
            )
            result["uninterrupted_repeat"] = {
                "initial_state_exact": True,
                "numerical_tolerance_applied": False,
                "trajectory_exact": not repeat_changes and repeated == baseline,
                "compared_final_tensors": repeat_count,
                "changed_final_tensors": repeat_changes,
                "steps": [asdict(item) for item in repeated],
                "max_loss_abs_error": max(abs(item.loss - baseline[item.step].loss) for item in repeated),
                "max_grad_norm_abs_error": max(
                    abs(item.grad_norm - baseline[item.step].grad_norm) for item in repeated
                ),
            }
            result["passed"] = result["passed"] and result["uninterrupted_repeat"]["trajectory_exact"]
            if not config.diagnostic_continuation:
                _assert_exact(repeated, baseline, "repeated_steps")
                if repeat_changes:
                    raise AssertionError("Independent uninterrupted final state differs under strict validation")
        if _training_source_manifest() != source_manifest:
            raise AssertionError("Implementation source changed during training; discard this validation run")
        result["source_manifest_unchanged"] = True
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / f"rank_{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
        dist.barrier()
    finally:
        checkpointer.close()


def main() -> None:
    """Parse standalone functional-run controls and initialize the GPU ranks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experts", choices=("torch_linear", "torch_mm", "gmm"), default="torch_linear")
    parser.add_argument("--attention-backend", choices=("sdpa", "tilelang"), default="tilelang")
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--diagnostic-continuation",
        action="store_true",
        help="Measure TileLang backward differences without treating them as a numerical continuation pass",
    )
    parser.add_argument(
        "--repeat-uninterrupted",
        action="store_true",
        help="Measure backend noise with an independent uninterrupted run after checking its initial state exactly",
    )
    config = TrainingConfig(**vars(parser.parse_args()))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        run(config)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
