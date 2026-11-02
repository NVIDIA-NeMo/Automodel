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

"""Two-rank parity of GLM DSA context parallelism against the single-process model (cuDNN DSA kernels).

Each rank runs the same small GLM DSA model on the same packed batch three ways: the full sequence
without CP (the reference), and its CP2 shard under the contiguous and the head/tail query layouts.
A layout is correct only if every full-indexer layer selects the same keys as the reference for the
local queries, the local logits match the reference rows, and the parameter gradients summed over the
two ranks match the reference gradients. The batch packs three documents whose real lengths differ
from their padded storage, so chunk boundaries fall inside documents and inside padding. The head/tail
case also runs under activation checkpointing, which takes the gathered latent through the replay path.

Both ranks agree on every verdict through a collective before raising, so a failing layout ends the test
with the assertion message instead of leaving the peer rank waiting in a collective.
"""

from __future__ import annotations

import datetime
import itertools
import os
import socket
import sys
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm_moe_dsa import cp as glm_cp
from nemo_automodel.components.models.glm_moe_dsa import layers as glm_layers
from nemo_automodel.components.models.glm_moe_dsa.model import GlmMoeDsaForCausalLM
from nemo_automodel.components.models.glm_moe_dsa.optimized_kernels import is_cudnn_dsa_available

# Two worker processes each import torch, build the model four times and run several cuDNN graphs.
pytestmark = pytest.mark.timeout(600)

_WORLD_SIZE = 2
# Real and padded-storage lengths of the three packed documents: 1024 padded tokens = four head/tail
# chunks of 256 at CP2, with chunk boundaries inside document 0 (at 256), document 1 (512, 768) and
# padding rows at 300-319, 770-799 and 1000-1023.
_REAL_LENGTHS = (300, 450, 200)
_PADDED_LENGTHS = (320, 480, 224)
# Relative error budget for bf16 logits and for summed bf16 gradients (max abs error / max abs reference).
_LOGITS_TOLERANCE = 1e-2
_GRAD_TOLERANCE = 2e-2


class _Mesh:
    """Minimal CP mesh for ``_slice_thd_chunk_for_cp``: size, group and this rank."""

    def __init__(self, size: int, rank: int):
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def get_group(self):
        return dist.group.WORLD

    def get_local_rank(self) -> int:
        return self._rank


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _config() -> GlmMoeDsaConfig:
    """GLM-5.x head geometry (the kernels fix the latent and indexer dims) with few heads and layers."""
    return GlmMoeDsaConfig(
        vocab_size=1024,
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=3,
        intermediate_size=1024,
        moe_intermediate_size=256,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        attention_bias=False,
        q_lora_rank=256,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=64,
        rope_interleave=True,
        indexer_rope_interleave=True,
        # Dense MLPs: the gather order only touches attention, and per-token expert routing could flip on
        # bf16 noise between the sharded and the full-sequence run, which would hide behind the comparison.
        mlp_layer_types=["dense", "dense", "dense"],
        indexer_types=["full", "shared", "full"],
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )


def _backend(layout: str) -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="cudnn",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
        cp_query_layout=layout,
    )


def _model(layout: str, device: torch.device, reference_state: dict[str, torch.Tensor] | None):
    torch.manual_seed(1234)
    model = GlmMoeDsaForCausalLM(_config(), backend=_backend(layout))
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    if reference_state is not None:
        model.load_state_dict(reference_state)
    for param in model.parameters():
        param.data = param.data.to(torch.bfloat16)
    return model.to(device).train()


def _global_batch(device: torch.device, vocab_size: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    total = sum(_PADDED_LENGTHS)
    position_ids = torch.cat([torch.arange(padded) for padded in _PADDED_LENGTHS])
    padding_mask = torch.cat(
        [torch.arange(padded) >= real for real, padded in zip(_REAL_LENGTHS, _PADDED_LENGTHS, strict=True)]
    )
    batch = {
        "input_ids": torch.randint(0, vocab_size, (total,), generator=generator),
        "labels": torch.full((total,), -100, dtype=torch.int64),
        "position_ids": position_ids,
        "padding_mask": padding_mask,
        "cu_seqlens": torch.tensor([0, *itertools.accumulate(_REAL_LENGTHS)], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([0, *itertools.accumulate(_PADDED_LENGTHS)], dtype=torch.int32),
        "max_seqlen": torch.tensor(max(_REAL_LENGTHS), dtype=torch.int32),
    }
    return {key: value.to(device) for key, value in batch.items()}


def _expected_local_indices(layout: str, rank: int, total: int, device: torch.device) -> torch.Tensor:
    """The global query positions ``rank`` must own: one run (contiguous) or chunks r and 2cp-1-r (head/tail)."""
    if layout == "contiguous":
        local = total // _WORLD_SIZE
        return torch.arange(rank * local, (rank + 1) * local, device=device)
    chunk = total // (2 * _WORLD_SIZE)
    head = torch.arange(rank * chunk, (rank + 1) * chunk, device=device)
    tail_start = (2 * _WORLD_SIZE - 1 - rank) * chunk
    return torch.cat([head, torch.arange(tail_start, tail_start + chunk, device=device)])


def _capture_topk(calls: list[torch.Tensor]):
    """Wrap the layers' cuDNN indexer entry point so every full layer's selection is recorded."""
    original = glm_layers.cudnn_indexer_topk

    def recording(*args, **kwargs):
        out = original(*args, **kwargs)
        calls.append(out.detach().clone())
        return out

    glm_layers.cudnn_indexer_topk = recording
    return original


def _forward_backward(model, inputs: dict, loss_weights: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one step on ``inputs`` with a fixed linear loss; return the logits and every parameter's gradient.

    Parameters without a gradient (the indexer: its selection is discrete) report zeros, so both ranks
    issue the same collectives when the gradients are summed.
    """
    model.zero_grad(set_to_none=True)
    logits = model(**inputs).logits.squeeze(0)
    (logits.float() * loss_weights).sum().backward()
    # checkpoint_wrapper inserts ``_checkpoint_wrapped_module`` into parameter names; key by the plain name.
    grads = {
        name.replace("._checkpoint_wrapped_module", ""): (
            param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
        )
        for name, param in model.named_parameters()
    }
    return logits.detach(), grads


def _relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    scale = reference.float().abs().max().clamp_min(1e-6)
    return float((actual.float() - reference.float()).abs().max() / scale)


def _log(rank: int, started: float, message: str) -> None:
    print(f"[parity rank {rank} +{time.monotonic() - started:6.1f}s] {message}", file=sys.stderr, flush=True)


def _agree_or_raise(problems: list[str], tag: str, device: torch.device) -> None:
    """Share each rank's verdict; raise on every rank if any rank found a problem."""
    flag = torch.tensor([1 if problems else 0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if int(flag.item()):
        raise AssertionError(f"{tag}: " + ("; ".join(problems) if problems else "the peer rank reported a failure"))


def _check_layout(
    rank: int,
    layout: str,
    device: torch.device,
    reference_state: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_weights: torch.Tensor,
    ref_logits: torch.Tensor,
    ref_grads: dict[str, torch.Tensor],
    ref_topk: list[torch.Tensor],
    *,
    activation_checkpointing: bool,
    started: float,
) -> dict[str, float]:
    tag = f"rank {rank} {layout}{' +AC' if activation_checkpointing else ''}"
    _log(rank, started, f"{tag}: building the model")
    model = _model(layout, device, reference_state)
    if activation_checkpointing:
        for key, layer in list(model.model.layers.items()):
            model.model.layers[key] = checkpoint_wrapper(
                layer,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                context_fn=layer.nemo_checkpoint_context_fn(None),
            )
    sliced = glm_cp._slice_thd_chunk_for_cp(
        batch,
        cp_mesh=_Mesh(_WORLD_SIZE, rank),
        cp_group=dist.group.WORLD,
        cp_size=_WORLD_SIZE,
        cp_rank=rank,
        padding_token_id=0,
        query_layout=layout,
    )
    local = sliced["glm_dsa_cp_query_indices"].long()
    problems: list[str] = []
    expected_local = _expected_local_indices(layout, rank, ref_logits.shape[0], device)
    if not torch.equal(local, expected_local):
        problems.append(f"the batch slicer handed this rank {local.tolist()[:4]}... instead of the {layout} shard")
    _agree_or_raise(problems, tag, device)
    inputs = {key: value for key, value in sliced.items() if key != "labels"}

    captured: list[torch.Tensor] = []
    original = _capture_topk(captured)
    try:
        _log(rank, started, f"{tag}: forward/backward on {local.numel()} local tokens")
        logits, grads = _forward_backward(model, inputs, loss_weights.index_select(0, local))
    finally:
        glm_layers.cudnn_indexer_topk = original
    _log(rank, started, f"{tag}: comparing with the reference")

    metrics: dict[str, float] = {}
    if not activation_checkpointing:
        if len(captured) != len(ref_topk):
            problems.append(f"{len(captured)} indexer selections captured, reference has {len(ref_topk)}")
        for layer_idx, (local_topk, full_topk) in enumerate(zip(captured, ref_topk)):
            expected = full_topk.index_select(0, local)
            differing = int((local_topk != expected).any(dim=-1).any(dim=-1).sum())
            metrics[f"topk_rows_differing_layer{layer_idx}"] = differing
            if differing:
                problems.append(
                    f"full layer {layer_idx} selected other keys for {differing} of {local.numel()} local queries"
                )

    metrics["logits_rel_err"] = _relative_error(logits, ref_logits.index_select(0, local))
    if metrics["logits_rel_err"] > _LOGITS_TOLERANCE:
        problems.append(f"logits relative error {metrics['logits_rel_err']:.4f} > {_LOGITS_TOLERANCE}")

    worst = 0.0
    worst_name = ""
    for name in sorted(ref_grads):
        summed = grads[name].float().clone()
        dist.all_reduce(summed, op=dist.ReduceOp.SUM)
        err = _relative_error(summed, ref_grads[name])
        if err > worst:
            worst, worst_name = err, name
    metrics["grad_rel_err_max"] = worst
    if worst > _GRAD_TOLERANCE:
        problems.append(f"gradient of {worst_name} relative error {worst:.4f} > {_GRAD_TOLERANCE}")
    _log(rank, started, f"{tag}: {metrics}")
    _agree_or_raise(problems, tag, device)
    return metrics


def _parity_worker(rank: int, port: int) -> None:
    started = time.monotonic()
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(_WORLD_SIZE)
        # A rank that dies must not leave its peer waiting on a collective until the pytest timeout.
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=datetime.timedelta(seconds=120),
            device_id=torch.device("cuda", rank),
        )
        device = torch.device("cuda", rank)
        _log(rank, started, "process group ready; building the reference model")

        reference = _model("contiguous", device, None)
        reference_state = {key: value.detach().cpu().float() for key, value in reference.state_dict().items()}
        batch = _global_batch(device, reference.config.vocab_size)
        total = batch["input_ids"].shape[0]
        loss_weights = torch.randn(total, reference.config.vocab_size, generator=torch.Generator().manual_seed(2)).to(
            device
        )

        # Reference: the whole packed sequence in one process (no CP keys in the batch).
        reference_inputs = {key: value for key, value in batch.items() if key != "labels"}
        reference_inputs["qkv_format"] = "thd"
        ref_topk: list[torch.Tensor] = []
        original = _capture_topk(ref_topk)
        try:
            _log(rank, started, f"reference forward/backward on {total} tokens")
            ref_logits, ref_grads = _forward_backward(reference, reference_inputs, loss_weights)
        finally:
            glm_layers.cudnn_indexer_topk = original
        assert len(ref_topk) == 2, "two full-indexer layers select keys"
        assert ref_logits.shape[0] == total
        del reference

        report = {}
        for layout, activation_checkpointing in (("contiguous", False), ("head_tail", False), ("head_tail", True)):
            report[(layout, activation_checkpointing)] = _check_layout(
                rank,
                layout,
                device,
                reference_state,
                batch,
                loss_weights,
                ref_logits,
                ref_grads,
                ref_topk,
                activation_checkpointing=activation_checkpointing,
                started=started,
            )
        if rank == 0:
            for key, metrics in report.items():
                print(f"GLM DSA CP parity {key}: {metrics}", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cudnn_dsa_available()
    or not dist.is_nccl_available()
    or torch.cuda.device_count() < _WORLD_SIZE,
    reason="requires the cuDNN DSA kernels, NCCL and at least two CUDA devices",
)
def test_cp2_contiguous_and_head_tail_match_the_single_process_model() -> None:
    mp.spawn(_parity_worker, args=(_free_port(),), nprocs=_WORLD_SIZE, join=True)
