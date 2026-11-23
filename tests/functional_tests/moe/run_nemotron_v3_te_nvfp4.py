#!/usr/bin/env python
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

r"""NVFP4 precision map of a small Nemotron 3 model, checked against Transformer Engine's own state.

The model, the expert-parallel dispatcher, FSDP2, activation checkpointing and the ``te_nvfp4`` block are built
by the training recipe from ``nemotron_v3_te_nvfp4_proxy.yaml``. Three passes run on identical tokens:

1. Forward without a Transformer Engine autocast (twice): the high-precision reference and its repeatability.
2. Forward and backward under the recipe's autocast, as the train step runs them.
3. The recipe's own training loop for the configured steps.

A forward hook on every Transformer Engine ``Linear`` / ``GroupedLinear`` reads, right after the module ran,
whether Transformer Engine had quantization enabled for it (``module.fp8``) and the recipe it stored for it
(``module.fp8_meta["recipe"]``). The expected precision of each module is derived here from the layer pattern
and from the constants below, not from the code under test. Passes 2 and 3 must match it for every module, and
pass 1 must show every module in high precision, so a map that quantizes the wrong modules, or none, fails.

Before the passes, the two ranks compare the weights they drew at random initialization: the shards of the
FSDP-managed projections and each rank's own routed experts must differ between the ranks. Ranks hold one
random state once a sharded tensor has been drawn, so a weight drawn on a rank's local tensor comes out equal
on both and fails here.

Scope: a small randomly initialized model on mock tokens. This checks which kernels run, that loss and
gradients are finite, and that the quantized forward stays close to the high-precision one. It is no evidence
of convergence.

Usage::

    torchrun --nproc_per_node=2 tests/functional_tests/moe/run_nemotron_v3_te_nvfp4.py \
        --config tests/functional_tests/moe/nemotron_v3_te_nvfp4_proxy.yaml \
        --checkpoint.checkpoint_dir <dir>
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

DEFAULT_CONFIG = "tests/functional_tests/moe/nemotron_v3_te_nvfp4_proxy.yaml"

# The recipe under test, restated: the test fails if the YAML or the code drift from it.
LAYER_PATTERN = ["mamba", "moe", "attention", "mamba", "moe", "mamba", "moe", "mamba"]
HIGH_PRECISION_TAIL_LAYERS = 3
NVFP4_EXPERT_ALIGNMENT = 128
# Relative error of the quantized logits against the high-precision logits of the same weights and tokens.
# One NVFP4 GEMM of random operands is off by about 0.15 and the Mamba blocks dominate the residual stream
# of this randomly initialized model, so the logits differ by about 0.3 (GB300: 0.32). A forward that
# quantizes nothing gives exactly 0; a GEMM that returns wrong rows gives errors near or above 1.
MAX_LOGITS_ERROR = 0.6
MAX_LOSS_ERROR = 0.02


def expected_precision(name: str) -> str:
    """Precision the recipe gives the GEMM module ``name``: ``nvfp4``, ``mxfp8`` or ``high_precision``."""
    parts = [part for part in name.split(".") if part != "_checkpoint_wrapped_module"]
    if parts[0] == "mtp":
        return "high_precision"
    assert parts[:2] == ["model", "layers"] and parts[3] == "mixer", name
    layer = int(parts[2])
    if layer >= len(LAYER_PATTERN) - HIGH_PRECISION_TAIL_LAYERS:
        return "high_precision"
    kind, module = LAYER_PATTERN[layer], ".".join(parts[4:])
    if kind == "mamba":
        return {"in_proj": "nvfp4", "out_proj": "mxfp8"}[module]
    if kind == "attention":
        assert module in ("q_proj", "k_proj", "v_proj", "o_proj"), name
        return "high_precision"
    return {
        "experts.gate_up_linear": "nvfp4",
        "experts.down_linear": "nvfp4",
        "shared_experts.up_proj": "nvfp4",
        "shared_experts.down_proj": "nvfp4",
        "fc1_latent_proj": "high_precision",
        "fc2_latent_proj": "high_precision",
    }[module]


def expected_module_names() -> set[str]:
    """Names of the GEMM modules the layer pattern and the MTP head ([attention, moe]) must produce.

    ``lm_head`` is not among them: it is a ``torch.nn.Linear`` (``LMHead``) and runs in the model dtype.
    """
    attention = ["q_proj", "k_proj", "v_proj", "o_proj"]
    moe = [
        "experts.gate_up_linear",
        "experts.down_linear",
        "shared_experts.up_proj",
        "shared_experts.down_proj",
        "fc1_latent_proj",
        "fc2_latent_proj",
    ]
    by_kind = {"mamba": ["in_proj", "out_proj"], "attention": attention, "moe": moe}
    names = {f"model.layers.{i}.mixer.{module}" for i, kind in enumerate(LAYER_PATTERN) for module in by_kind[kind]}
    names |= {f"mtp.layers.0.mixer.{module}" for module in attention} | {"mtp.layers.0.eh_proj"}
    names |= {f"mtp.layers.1.mixer.{module}" for module in moe}
    return names


@dataclass
class ExecutionRecord:
    """What Transformer Engine reported for its modules while they ran."""

    precisions: dict[str, set[str]] = field(default_factory=dict)
    """GEMM module name -> every precision one of its forwards ran in."""
    quantizers: dict[str, set[str]] = field(default_factory=dict)
    """GEMM module name -> class names of the forward quantizers of its quantized forwards."""
    attention_quantized: set[bool] = field(default_factory=set)
    padded_splits: dict[str, list[list[int]]] = field(default_factory=dict)
    """Grouped projection name -> the per-expert row counts of each of its forwards."""
    raw_splits: dict[str, list[list[int]]] = field(default_factory=dict)
    """Padding module name -> the per-expert row counts it received."""

    def clear(self) -> None:
        self.precisions.clear()
        self.quantizers.clear()
        self.attention_quantized.clear()
        self.padded_splits.clear()
        self.raw_splits.clear()


def install_hooks(model: torch.nn.Module, record: ExecutionRecord) -> list[str]:
    """Record Transformer Engine's state after every forward of its GEMM and attention modules."""
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.attention import DotProductAttention

    def canonical(name: str) -> str:
        return ".".join(part for part in name.split(".") if part != "_checkpoint_wrapped_module")

    def gemm_hook(name: str):
        def hook(module, args, output) -> None:
            if not module.fp8:
                precision = "high_precision"
            else:
                recipe = module.fp8_meta["recipe"]
                precision = "nvfp4" if recipe.nvfp4() else "mxfp8" if recipe.mxfp8() else type(recipe).__name__
                record.quantizers.setdefault(name, set()).update(
                    type(quantizer).__name__ for quantizer in module.quantizers["scaling_fwd"]
                )
            record.precisions.setdefault(name, set()).add(precision)

        return hook

    def splits_hook(name: str, target: dict[str, list[list[int]]]):
        def hook(module, args) -> None:
            target.setdefault(name, []).append([int(count) for count in args[1]])

        return hook

    names = []
    for raw_name, module in model.named_modules():
        name = canonical(raw_name)
        if isinstance(module, (te.Linear, te.GroupedLinear)):
            names.append(name)
            module.register_forward_hook(gemm_hook(name))
            if isinstance(module, te.GroupedLinear):
                module.register_forward_pre_hook(splits_hook(name, record.padded_splits))
        elif isinstance(module, DotProductAttention):
            module.register_forward_hook(lambda module, args, output: record.attention_quantized.add(bool(module.fp8)))
        elif isinstance(module, te.Fp8Padding):
            module.register_forward_pre_hook(splits_hook(name, record.raw_splits))
    return names


def check_executed(record: ExecutionRecord, names: list[str], *, quantized: bool, what: str) -> dict[str, int]:
    """Compare what ran with the expectation; return the number of modules per precision."""
    assert set(names) == expected_module_names(), sorted(set(names) ^ expected_module_names())
    counts: dict[str, int] = {}
    quantizer_of = {"nvfp4": "NVFP4Quantizer", "mxfp8": "MXFP8Quantizer"}
    for name in names:
        assert name in record.precisions, f"{what}: {name} did not run"
        expected = expected_precision(name) if quantized else "high_precision"
        assert record.precisions[name] == {expected}, f"{what}: {name} ran {record.precisions[name]}, not {expected}"
        if expected != "high_precision":
            assert record.quantizers[name] == {quantizer_of[expected]}, (name, record.quantizers[name])
        counts[expected] = counts.get(expected, 0) + 1
    assert record.attention_quantized == {False}, f"{what}: attention kernels ran quantized"
    return counts


def check_expert_padding(record: ExecutionRecord) -> int:
    """Every NVFP4 grouped GEMM got token groups in multiples of the alignment; return how many were padded."""
    padded_groups = 0
    assert record.padded_splits, "no grouped GEMM ran"
    for name, calls in record.padded_splits.items():
        nvfp4 = expected_precision(name) == "nvfp4"
        for splits in calls:
            assert sum(splits) > 0, name
            if nvfp4:
                assert all(count % NVFP4_EXPERT_ALIGNMENT == 0 for count in splits), (name, splits)
    for name, calls in record.raw_splits.items():
        # Only the experts that ran a quantized recipe ask for padding.
        assert expected_precision(name.replace("fp8_padding", "gate_up_linear")) == "nvfp4", name
        padded_groups += sum(count % NVFP4_EXPERT_ALIGNMENT != 0 for splits in calls for count in splits)
    return padded_groups


def check_random_initialization_differs_between_ranks(model: torch.nn.Module) -> int:
    """Compare the randomly drawn block weights of the two ranks; return how many were compared.

    Every weight with two or more dimensions in the decoder blocks and the MTP head is drawn at random:
    Mamba, attention, latent and shared-expert projections and the router (one FSDP shard per rank), the Mamba
    convolution, and the routed experts (each rank holds its own half of them). The leading values of each
    rank's local tensor are exchanged and must not be equal.

    Args:
        model: The model after the recipe built, sharded and initialized it.

    Returns:
        Number of weights compared.
    """
    from torch.distributed.tensor import DTensor

    names, leading_values = [], []
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2 or not name.startswith(("model.layers.", "mtp.layers.")):
            continue
        local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
        assert local.numel() >= 64, name
        names.append(name)
        leading_values.append(local.detach().flatten()[:64].float())
    assert any(".experts." in name for name in names) and any(".shared_experts." in name for name in names), names
    mine = torch.stack(leading_values)
    gathered = [torch.empty_like(mine) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, mine)
    assert len(gathered) == 2, "the comparison is written for two ranks"
    equal = [name for index, name in enumerate(names) if torch.equal(gathered[0][index], gathered[1][index])]
    assert not equal, f"both ranks hold the same initial values in {len(equal)} of {len(names)} weights: {equal}"
    return len(names)


def main() -> None:
    cfg = parse_args_and_load_config(DEFAULT_CONFIG)
    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    model = recipe.model_parts[0]
    rank = dist.get_rank()
    compared_weights = check_random_initialization_differs_between_ranks(model)
    # The config spells Mamba and attention layers "linear_attention" and "full_attention".
    canonical = {"linear_attention": "mamba", "full_attention": "attention"}
    pattern = [canonical.get(kind, kind) for kind in model.config.layers_block_type]
    assert pattern == LAYER_PATTERN, pattern
    assert type(model.model.layers["1"].mixer.experts).__name__ == "GroupedExpertsTE"

    record = ExecutionRecord()
    names = install_hooks(model, record)
    model.train()
    generator = torch.Generator().manual_seed(20261123 + rank)
    input_ids = torch.randint(1, model.config.vocab_size, (2, 256), generator=generator).to(recipe.dist_env.device)

    def loss_of(logits: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits[:, :-1].float().flatten(0, 1), input_ids[:, 1:].flatten())

    # Pass 1: no Transformer Engine autocast. Nothing may run quantized, MXFP8 modules included.
    with torch.no_grad():
        reference = model(input_ids=input_ids).logits.float()
        counts_reference = check_executed(record, names, quantized=False, what="high-precision pass")
        assert not record.raw_splits, "experts padded their tokens without a quantized recipe"
        repeat = model(input_ids=input_ids).logits.float()
    repeat_error = float((repeat - reference).norm() / reference.norm())
    reference_loss = float(loss_of(reference))

    # Pass 2: forward and backward under the recipe's autocast.
    record.clear()
    with recipe.te_quantization.maybe_te_autocast():
        logits = model(input_ids=input_ids).logits
        loss = loss_of(logits)
        loss.backward()
    counts = check_executed(record, names, quantized=True, what="quantized pass")
    padded_groups = check_expert_padding(record)
    logits_error = float((logits.detach().float() - reference).norm() / reference.norm())
    loss_error = abs(float(loss) - reference_loss) / reference_loss
    print(
        f"NVFP4_PRECISION_METRICS rank={rank} loss={float(loss):.5f} reference_loss={reference_loss:.5f} "
        f"loss_error={loss_error:.5f} logits_error={logits_error:.5f} repeat_error={repeat_error:.2e}",
        flush=True,
    )
    assert math.isfinite(float(loss))
    assert repeat_error < 1e-3 < logits_error < MAX_LOGITS_ERROR, (repeat_error, logits_error)
    assert loss_error < MAX_LOSS_ERROR, (float(loss), reference_loss)
    missing, nonfinite, zero = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name.startswith("mtp."):
            continue  # the MTP head is not part of this pass's loss; pass 3 trains it
        if parameter.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        elif float(parameter.grad.float().abs().max()) == 0.0:
            zero.append(name)
    assert not missing and not nonfinite and not zero, (missing, nonfinite, zero)
    model.zero_grad(set_to_none=True)

    # Pass 3: the recipe's own steps (MTP loss, activation-checkpoint recompute, FusedAdam master weights).
    record.clear()
    recipe.run_train_validation_loop()
    check_executed(record, names, quantized=True, what="training loop")
    check_expert_padding(record)

    result = {
        "rank": rank,
        "modules_by_precision": counts,
        "modules_high_precision_pass": counts_reference,
        "padded_expert_groups": padded_groups,
        "reference_loss": reference_loss,
        "quantized_loss": float(loss),
        "loss_relative_error": loss_error,
        "logits_relative_error": logits_error,
        "repeat_relative_error": repeat_error,
        "weights_that_differ_between_ranks": compared_weights,
    }
    if rank == 0:
        log = os.path.join(cfg.checkpoint.checkpoint_dir, "training.jsonl")
        steps = [json.loads(line) for line in open(log) if line.strip()]
        assert len(steps) == cfg.step_scheduler.max_steps, len(steps)
        for step in steps:
            assert math.isfinite(step["loss"]) and math.isfinite(step["grad_norm"]) and step["grad_norm"] > 0, step
        result["training_loop_loss"] = [step["loss"] for step in steps]
        result["training_loop_grad_norm"] = [step["grad_norm"] for step in steps]
    # With a learned gate the token groups are not aligned by themselves; a rank that padded none proves nothing.
    assert padded_groups > 0, "no expert token group needed padding"
    print("NVFP4_PRECISION_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
