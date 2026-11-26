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

The training recipe builds the model from ``CONFIG_YAML`` below. On identical tokens it runs
(1) a forward without autocast, twice (high-precision reference and its repeatability), (2) forward and backward
under the recipe's autocast, (3) the recipe's own training loop. A forward hook on every TE ``Linear`` /
``GroupedLinear`` reads whether TE quantized the module and with which recipe; the expected precision is derived
here from the layer pattern, not from the code under test. Before that, the two ranks compare their randomly
initialized weights, which must differ (a weight drawn on a local shard comes out equal on both ranks).

Scope: which kernels run, finite loss and gradients, quantized logits close to high precision. No convergence evidence.

Usage::

    torchrun --nproc_per_node=2 tests/functional_tests/moe/run_nemotron_v3_te_nvfp4.py \
        --checkpoint.checkpoint_dir <dir>
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

# The configuration of the recipe: a small randomly initialized Nemotron 3 model (2 GPUs with NVFP4 kernels).
#
# Same module kinds and the same te_nvfp4 block as
# examples/llm_pretrain/nemotron_3_5_super_vl/nemotron_3_5_super_vl_te_nvfp4_gb300.yaml: Mamba, MoE with a
# latent space and a shared expert, attention, one MTP depth of [attention, moe], Transformer Engine experts
# behind HybridEP, FSDP2 with activation checkpointing, and FusedAdam with FP32 master weights.
# 8 layers with a high-precision tail of 3: layers 0-4 hold every module kind once below the tail, and
# layers 5-7 (Mamba, MoE, Mamba) are the tail.
#
# Every GEMM dimension is a multiple of 128, as in the full model (the Mamba input projection has 1408 = 11 x 128
# outputs; the full model has 18560 = 145 x 128). 2 x 256 tokens per rank reach the 8 routed experts with a
# learned gate, so the per-expert token counts are not multiples of the NVFP4 alignment by themselves.
CONFIG_YAML = """\
recipe: TrainFinetuneRecipeForNextTokenPrediction

step_scheduler:
  global_batch_size: 4
  local_batch_size: 2
  ckpt_every_steps: 1000
  val_every_steps: 1000
  num_epochs: null
  max_steps: 3

dist_env:
  backend: nccl
  timeout_minutes: 10

rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 42
  ranked: true

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_config
  dtype: bfloat16
  num_nextn_predict_layers: 1
  config:
    _target_: transformers.NemotronHConfig
    architectures: [NemotronHForCausalLM]
    vocab_size: 512
    hidden_size: 256
    layers_block_type: [mamba, moe, attention, mamba, moe, mamba, moe, mamba]
    num_attention_heads: 8
    num_key_value_heads: 2
    head_dim: 32
    mamba_num_heads: 128
    mamba_head_dim: 4
    ssm_state_size: 16
    n_groups: 8
    chunk_size: 128
    intermediate_size: 128
    n_routed_experts: 8
    num_experts_per_tok: 2
    moe_intermediate_size: 128
    moe_shared_expert_intermediate_size: 256
    moe_latent_size: 128
    num_nextn_predict_layers: 1
    mtp_layers_block_type: [attention, moe]
    max_position_embeddings: 4096
    pad_token_id: 0
    dtype: bfloat16
  backend:
    _target_: nemo_automodel.components.models.nemotron_v3.layers.NemotronV3BackendConfig
    attn: te
    linear: te
    mamba_linear: te
    rms_norm: te
    experts: te
    # One grouped GEMM per expert projection.
    te_grouped_tensor: true
    dispatcher: hybridep
    gate_precision: float32
    fake_balanced_gate: false
    enable_hf_state_dict_adapter: false
    te_nvfp4:
      _target_: nemo_automodel.components.models.common.TENvfp4Config
      # Empty next to a precision map; lm_head, the default entry, is a torch linear and runs in BF16 anyway.
      exclude_fqns: []
      precision_map:
        _target_: nemo_automodel.components.models.nemotron_v3.quantization.NemotronV3PrecisionMap
        high_precision_tail_layers: 3
        mamba_in_proj: nvfp4
        mamba_out_proj: mxfp8
        routed_experts: nvfp4
        shared_experts: nvfp4
        attention: high_precision
        latent_proj: high_precision
        mtp: high_precision

checkpoint:
  enabled: false

distributed:
  strategy: fsdp2
  tp_size: 1
  cp_size: 1
  pp_size: 1
  ep_size: 2
  sequence_parallel: false
  activation_checkpointing: true

mtp:
  scaling_factor: 0.3

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

dataset:
  _target_: nemo_automodel.components.datasets.llm.mock_iterable_dataset.MockIterableDatasetConfig
  vocab_size: 512
  seq_len: 256
  include_position_ids: false
  exclude_token_ids: [0]

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  num_workers: 0

optimizer:
  _target_: transformer_engine.pytorch.optimizers.FusedAdam
  lr: 1.0e-4
  betas: [0.9, 0.95]
  eps: 1.0e-8
  weight_decay: 0.1
  adam_w_mode: true
  master_weights: true
  master_weight_dtype: float32
  store_param_remainders: false
  exp_avg_dtype: float32
  exp_avg_sq_dtype: float32

clip_grad_norm:
  max_norm: 1.0
"""

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


def load_config() -> ConfigNode:
    """Load ``CONFIG_YAML`` the way the recipe loads a YAML file, with the command-line overrides applied."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8") as config_file:
        config_file.write(CONFIG_YAML)
        config_file.flush()
        return parse_args_and_load_config(config_file.name)


def main() -> None:
    cfg = load_config()
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
        check_executed(record, names, quantized=False, what="high-precision pass")
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
        "padded_expert_groups": padded_groups,
        "weights_that_differ_between_ranks": compared_weights,
    }
    if rank == 0:
        log = os.path.join(cfg.checkpoint.checkpoint_dir, "training.jsonl")
        steps = [json.loads(line) for line in open(log) if line.strip()]
        assert len(steps) == cfg.step_scheduler.max_steps, len(steps)
        for step in steps:
            assert math.isfinite(step["loss"]) and math.isfinite(step["grad_norm"]) and step["grad_norm"] > 0, step
    # With a learned gate the token groups are not aligned by themselves; a rank that padded none proves nothing.
    assert padded_groups > 0, "no expert token group needed padding"
    print("NVFP4_PRECISION_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
