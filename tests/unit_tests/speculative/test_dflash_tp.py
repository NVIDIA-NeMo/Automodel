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

"""CPU unit tests for DFlash target-model tensor parallelism.

Covers the pieces verifiable without multiple GPUs: the trainer keeps the frozen
target lm_head / embed_tokens as non-registered references (so a DDP-wrapped
trainer only sees the plain draft params), it gathers their tensor-parallel
(DTensor) outputs to plain tensors, and the recipe resolves the ``dp`` submesh it
keys the draft DDP group / sampler / checkpointer on. Real multi-rank TP is
validated on the server.
"""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

import nemo_automodel.recipes.llm.train_dflash as train_dflash
from nemo_automodel.components.loss.dllm_loss import DFlashDecayLoss
from nemo_automodel.components.speculative.dflash.core import (
    DFlashStepMetrics,
    DFlashTrainerModule,
    NoValidAnchorsError,
    _to_full_tensor,
)
from nemo_automodel.components.speculative.dflash.domino_core import DominoStepMetrics, DominoTrainerModule
from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel
from nemo_automodel.components.speculative.dflash.registry import resolve_dflash_draft_spec
from nemo_automodel.recipes.llm.train_dflash import TrainDFlashRecipe

VOCAB = 64
HIDDEN = 32
NUM_TARGET_LAYERS = 8
TARGET_LAYER_IDS = [1, 3, 5]
BLOCK_SIZE = 4
MASK_ID = VOCAB - 1


@pytest.fixture
def single_rank_pg():
    import torch.distributed as dist

    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    already = dist.is_initialized()
    if not already:
        dist.init_process_group(backend="gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        yield
    finally:
        if not already:
            dist.destroy_process_group()


def _draft_model(dflash_config_extra=None):
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    cfg.num_target_layers = NUM_TARGET_LAYERS
    cfg.block_size = BLOCK_SIZE
    cfg.dflash_config = {"mask_token_id": MASK_ID, "target_layer_ids": TARGET_LAYER_IDS}
    if dflash_config_extra:
        cfg.dflash_config.update(dflash_config_extra)
    cfg._attn_implementation = "sdpa"
    return Qwen3DFlashDraftModel(cfg)


def _tp_target_modules():
    """A column-parallel lm_head + vocab-parallel embed_tokens, both returning DTensors."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

    lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
    embed = nn.Embedding(VOCAB, HIDDEN)
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
    parallelize_module(lm_head, mesh, ColwiseParallel(output_layouts=Shard(-1), use_local_output=False))
    parallelize_module(
        embed, mesh, RowwiseParallel(input_layouts=Replicate(), output_layouts=Replicate(), use_local_output=False)
    )
    lm_head.requires_grad_(False)
    embed.requires_grad_(False)
    return lm_head, embed


def _run_two_rank_materialized_head(rank: int, init_file: str) -> None:
    """Exercise real TP weight gathering and fused draft-gradient parity."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard
    from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        torch.manual_seed(23)
        head = nn.Linear(HIDDEN, VOCAB, bias=False)
        reference_weight = head.weight.detach().clone()
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        parallelize_module(head, mesh, ColwiseParallel(output_layouts=Shard(-1), use_local_output=False))
        head.requires_grad_(False)
        embed = nn.Embedding(VOCAB, HIDDEN).requires_grad_(False)
        trainer = DFlashTrainerModule(
            draft_model=nn.Identity(),
            target_lm_head=head,
            target_embed_tokens=embed,
            mask_token_id=MASK_ID,
            block_size=BLOCK_SIZE,
            attention_backend="sdpa",
            num_anchors=2,
            use_fused_linear_ce=True,
            linear_ce_chunk_size=3,
        )

        gathered_weight, gathered_bias = trainer._materialize_frozen_lm_head(torch.device("cpu"))
        torch.testing.assert_close(gathered_weight, reference_weight)
        assert gathered_bias is None

        B, N = 2, 2
        T = N * (BLOCK_SIZE - 1)
        hidden = torch.randn(B, T, HIDDEN, requires_grad=True)
        hidden_ref = hidden.detach().clone().requires_grad_(True)
        target_ids = torch.randint(0, VOCAB, (B, T))
        block_mask = torch.ones(B, T)
        loss_fn = DFlashDecayLoss(loss_gamma=4.0, chunk_size=3, normalize="mean")
        actual, actual_correct = loss_fn.forward_fused_with_correct(
            hidden,
            gathered_weight,
            target_ids,
            block_mask,
            block_size=BLOCK_SIZE,
        )
        expected_logits = torch.nn.functional.linear(hidden_ref, reference_weight)
        expected = loss_fn(expected_logits, target_ids, block_mask, block_size=BLOCK_SIZE)

        torch.testing.assert_close(actual.total_loss, expected.total_loss)
        assert torch.equal(actual_correct, expected_logits.argmax(dim=-1) == target_ids)
        actual.total_loss.backward()
        expected.total_loss.backward()
        torch.testing.assert_close(hidden.grad, hidden_ref.grad, atol=1e-6, rtol=1e-5)
    finally:
        dist.destroy_process_group()


class _AncestorOwnedFrozenHead(nn.Module):
    """Target root whose child head parameter is owned by root-level FSDP2."""

    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)


def _run_two_rank_fsdp_ancestor_head_forward(rank: int, init_file: str) -> None:
    """Exercise full DFlash forwards against an ancestor-owned FSDP2 head."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FSDPModule, fully_shard
    from torch.distributed.tensor import DTensor

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(31)
        target_root = _AncestorOwnedFrozenHead().requires_grad_(False)
        expected_head_weight = target_root.lm_head.weight.detach().clone()
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))

        # Only the ancestor is an FSDP unit. The child lm_head has no forward
        # hook that may be called independently; its parameter is nevertheless
        # a sharded DTensor owned by the ancestor's FSDP state.
        fully_shard(target_root, mesh=mesh)
        assert isinstance(target_root, FSDPModule)
        assert isinstance(target_root.lm_head.weight, DTensor)
        assert not isinstance(target_root.lm_head, FSDPModule)

        embed = nn.Embedding(VOCAB, HIDDEN).requires_grad_(False)
        trainer = DFlashTrainerModule(
            draft_model=_draft_model(),
            target_lm_head=target_root.lm_head,
            target_embed_tokens=embed,
            mask_token_id=MASK_ID,
            block_size=BLOCK_SIZE,
            attention_backend="sdpa",
            num_anchors=4,
            loss_decay_gamma=4.0,
            use_fused_linear_ce=True,
            linear_ce_chunk_size=2,
        )

        gathered_weight, gathered_bias = trainer._materialize_frozen_lm_head(torch.device("cpu"))
        torch.testing.assert_close(gathered_weight, expected_head_weight)
        assert gathered_bias is None

        input_ids = torch.randint(0, VOCAB - 1, (1, 12))
        hidden = torch.randn(1, 12, len(TARGET_LAYER_IDS) * HIDDEN)
        loss_mask = torch.ones(1, 12) if rank == 0 else torch.zeros(1, 12)
        # Rank 0 constructs four blocks (12 predicted positions, six CE
        # chunks). Rank 1's only legal candidate is the final possible anchor,
        # whose three continuation positions are supervised, so it constructs
        # one block (three predicted positions, two chunks).
        if rank == 1:
            loss_mask[:, 8:] = 1

        metrics = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
        assert torch.isfinite(metrics.loss)
        assert metrics.valid_blocks.item() == (4 if rank == 0 else 1)
        metrics.loss.backward()
        draft_grads = [parameter.grad for parameter in trainer.draft_model.parameters() if parameter.grad is not None]
        assert draft_grads
        assert all(torch.isfinite(grad).all() for grad in draft_grads)
        assert target_root.lm_head.weight.grad is None

        block_counts = torch.tensor([metrics.valid_blocks.item()], dtype=torch.int64)
        gathered_counts = [torch.empty_like(block_counts) for _ in range(2)]
        dist.all_gather(gathered_counts, block_counts)
        assert [int(count.item()) for count in gathered_counts] == [4, 1]

        # A data-dependent empty-anchor rank must still participate in the
        # ancestor-owned head's full_tensor collective before raising. Rank 0
        # completes its unequal local work, then both ranks reach the barrier.
        no_anchor_mask = torch.ones(1, 12) if rank == 0 else torch.zeros(1, 12)
        if rank == 0:
            result = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=no_anchor_mask)
            assert torch.isfinite(result.loss)
        else:
            with pytest.raises(NoValidAnchorsError):
                trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=no_anchor_mask)
        dist.barrier()
    finally:
        dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# _to_full_tensor
# --------------------------------------------------------------------------- #
def test_to_full_tensor_is_noop_on_plain_tensor():
    plain = torch.randn(2, 3)
    assert _to_full_tensor(plain) is plain


def test_to_full_tensor_gathers_vocab_sharded_dtensor(single_rank_pg):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    full = torch.randn(2, 4, 6)
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
    sharded = distribute_tensor(full, mesh, [Shard(-1)])
    assert hasattr(sharded, "full_tensor")

    out = _to_full_tensor(sharded)
    assert not hasattr(out, "full_tensor")
    torch.testing.assert_close(out, full)


def test_materialized_frozen_head_moves_to_compute_device():
    head = nn.Linear(HIDDEN, VOCAB, bias=True).requires_grad_(False)
    trainer = DFlashTrainerModule(
        draft_model=nn.Identity(),
        target_lm_head=head,
        target_embed_tokens=nn.Embedding(VOCAB, HIDDEN).requires_grad_(False),
        mask_token_id=MASK_ID,
        block_size=BLOCK_SIZE,
        use_fused_linear_ce=True,
    )

    weight, bias = trainer._materialize_frozen_lm_head(torch.device("meta"))

    assert weight.device.type == "meta"
    assert bias is not None and bias.device.type == "meta"


def test_materialized_frozen_head_treats_empty_bias_placeholder_as_no_bias():
    head = nn.Linear(HIDDEN, VOCAB, bias=False).requires_grad_(False)
    head.bias = nn.Parameter(torch.empty(0), requires_grad=False)
    trainer = DFlashTrainerModule(
        draft_model=nn.Identity(),
        target_lm_head=head,
        target_embed_tokens=nn.Embedding(VOCAB, HIDDEN).requires_grad_(False),
        mask_token_id=MASK_ID,
        block_size=BLOCK_SIZE,
        use_fused_linear_ce=True,
    )

    _, bias = trainer._materialize_frozen_lm_head(torch.device("cpu"))

    assert bias is None


# --------------------------------------------------------------------------- #
# Trainer with a tensor-parallel target lm_head + embed_tokens
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("use_fused_linear_ce", [False, True])
def test_trainer_runs_with_tensor_parallel_target(single_rank_pg, use_fused_linear_ce):
    """A column-parallel lm_head and vocab-parallel embed_tokens return DTensors;
    the trainer must gather them, keep them non-registered (so DDP sees only the
    draft), run a finite forward, and flow gradients to the draft."""
    torch.manual_seed(0)
    draft = _draft_model()
    lm_head, embed = _tp_target_modules()

    trainer = DFlashTrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=MASK_ID,
        block_size=BLOCK_SIZE,
        attention_backend="sdpa",
        num_anchors=8,
        loss_decay_gamma=7.0,
        use_fused_linear_ce=use_fused_linear_ce,
        linear_ce_chunk_size=3,
    )

    # The frozen target modules are non-registered: a DDP-wrapped trainer must
    # only see the plain draft params (no sharded DTensor params to broadcast).
    trainer_param_ids = {id(p) for p in trainer.parameters()}
    assert all(id(p) not in trainer_param_ids for p in lm_head.parameters())
    assert all(id(p) not in trainer_param_ids for p in embed.parameters())
    assert trainer_param_ids == {id(p) for p in draft.parameters()}

    input_ids = torch.randint(0, VOCAB - 1, (2, 24))
    hidden = torch.randn(2, 24, len(TARGET_LAYER_IDS) * HIDDEN)
    loss_mask = torch.ones(2, 24)
    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)

    assert isinstance(out, DFlashStepMetrics)
    assert torch.isfinite(out.loss) and out.loss.item() > 0

    out.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in draft.parameters() if p.grad is not None)
    grad = sum(p.grad.abs().sum().item() for p in draft.parameters() if p.grad is not None)
    assert grad > 0


def test_materialized_tp_head_two_rank_loss_and_gradient_parity(tmp_path):
    """Two real ranks gather the frozen TP weight once and match the dense reference."""
    mp.spawn(_run_two_rank_materialized_head, args=(str(tmp_path / "dflash_tp_head"),), nprocs=2, join=True)


def test_fsdp2_ancestor_owned_head_full_forward_with_unequal_rank_work(tmp_path):
    """Full forwards gather a root-FSDP2-owned head before rank-local chunking."""
    mp.spawn(
        _run_two_rank_fsdp_ancestor_head_forward,
        args=(str(tmp_path / "dflash_fsdp_ancestor_head"),),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("shift_label", [True, False])
def test_domino_trainer_runs_with_tensor_parallel_target(single_rank_pg, shift_label):
    """Regression: Domino used to consume the TP target's ``lm_head`` /
    ``embed_tokens`` outputs without ``_to_full_tensor``, so the vocab-sharded
    DTensors crashed the reshape / GRU / cross_entropy on the first step. Both
    ``shift_label`` branches embed block tokens, so both are exercised."""
    torch.manual_seed(0)
    draft = _draft_model(
        {
            "projector_type": "domino",
            "emb_dim": 16,
            "gru_hidden_dim": 16,
            "pure_draft_prefix_len": 1,
            "shift_label": shift_label,
        }
    )
    lm_head, embed = _tp_target_modules()

    trainer = DominoTrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=MASK_ID,
        block_size=BLOCK_SIZE,
        attention_backend="sdpa",
        num_anchors=8,
        loss_decay_gamma=7.0,
        shift_label=shift_label,
    )

    input_ids = torch.randint(0, VOCAB - 1, (2, 24))
    hidden = torch.randn(2, 24, len(TARGET_LAYER_IDS) * HIDDEN)
    loss_mask = torch.ones(2, 24)
    # lambda_base=0.5 keeps both the corrected and the base CE in the loss, so
    # gradients flow through the correction head and the backbone alike.
    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, lambda_base=0.5)

    assert isinstance(out, DominoStepMetrics)
    assert torch.isfinite(out.loss) and out.loss.item() > 0

    out.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in draft.parameters() if p.grad is not None)
    head_grad = sum(
        p.grad.abs().sum().item()
        for m in (draft.prefix_gru, draft.embed_proj)
        for p in m.parameters()
        if p.grad is not None
    )
    assert head_grad > 0


# --------------------------------------------------------------------------- #
# _submesh_or_none
# --------------------------------------------------------------------------- #
def test_submesh_or_none_handles_none_present_and_missing(monkeypatch):
    assert train_dflash._submesh_or_none(None, "dp") is None

    monkeypatch.setattr(train_dflash, "get_flat_mesh", lambda mesh, name: ("submesh", name))
    assert train_dflash._submesh_or_none(object(), "dp") == ("submesh", "dp")

    def _raise(mesh, name):
        raise KeyError(name)

    monkeypatch.setattr(train_dflash, "get_flat_mesh", _raise)
    assert train_dflash._submesh_or_none(object(), "dp") is None


# --------------------------------------------------------------------------- #
# TrainDFlashRecipe seams: target build / DDP group / checkpoint dp_rank
# --------------------------------------------------------------------------- #
def _bare_recipe(**attrs):
    """A recipe instance with ``setup()`` bypassed and only the attrs under test set."""
    recipe = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    for name, value in attrs.items():
        setattr(recipe, name, value)
    return recipe


class _TargetStub:
    """Stand-in for a loaded target model; records ``.to`` / ``requires_grad_``."""

    def __init__(self):
        self.to_calls = []
        self.requires_grad_calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self

    def requires_grad_(self, flag):
        self.requires_grad_calls.append(flag)
        return self


def test_build_target_model_single_gpu_path(monkeypatch):
    """No ``distributed:`` section: no mesh is built, ``from_pretrained`` is called
    without ``distributed_setup``, and ``.to(device)`` places the target."""
    captured = {}
    stub = _TargetStub()

    def _fake_from_pretrained(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return stub

    monkeypatch.setattr(
        train_dflash, "NeMoAutoModelForCausalLM", SimpleNamespace(from_pretrained=_fake_from_pretrained)
    )

    recipe = _bare_recipe(
        cfg={},  # no "distributed" key
        dist_env=SimpleNamespace(world_size=1),
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
    )
    out = recipe._build_target_model(
        {"target_attn_implementation": "sdpa"}, "target/path", resolve_dflash_draft_spec(["Qwen3ForCausalLM"])
    )

    assert out is stub
    assert recipe.dist_setup is None and recipe.device_mesh is None and recipe.dp_mesh is None
    assert "distributed_setup" not in captured["kwargs"]
    assert captured["kwargs"]["attn_implementation"] == "sdpa"
    assert captured["path"] == "target/path"
    assert stub.to_calls == [recipe.device]  # placed by .to on the single-GPU path
    assert stub.requires_grad_calls == [False]


def test_build_target_model_tensor_parallel_path(monkeypatch):
    """A ``distributed:`` section resolves the mesh + dp submesh, passes
    ``distributed_setup`` to ``from_pretrained`` (sharded in place), and skips ``.to``."""
    captured = {}
    stub = _TargetStub()

    def _fake_from_pretrained(path, **kwargs):
        captured["kwargs"] = kwargs
        return stub

    sentinel_mesh = object()
    sentinel_dp = object()
    # cp is unset here (tp_size=2), so the "cp" submesh is a size-1 mesh.
    sentinel_cp = SimpleNamespace(size=lambda: 1)
    dist_setup = SimpleNamespace(mesh_context=SimpleNamespace(device_mesh=sentinel_mesh))
    monkeypatch.setattr(
        train_dflash, "NeMoAutoModelForCausalLM", SimpleNamespace(from_pretrained=_fake_from_pretrained)
    )
    monkeypatch.setattr(train_dflash, "create_distributed_setup_from_config", lambda cfg, world_size: dist_setup)
    monkeypatch.setattr(
        train_dflash, "_submesh_or_none", lambda mesh, name: sentinel_cp if name == "cp" else sentinel_dp
    )

    recipe = _bare_recipe(
        cfg={"distributed": {"tp_size": 2}},
        dist_env=SimpleNamespace(world_size=2),
        device=torch.device("cpu"),
        compute_dtype=torch.bfloat16,
    )
    out = recipe._build_target_model({}, "target/path", resolve_dflash_draft_spec(["Qwen3ForCausalLM"]))

    assert out is stub
    assert recipe.dist_setup is dist_setup
    assert recipe.device_mesh is sentinel_mesh
    assert recipe.dp_mesh is sentinel_dp
    assert recipe.cp_mesh is sentinel_cp
    assert captured["kwargs"]["distributed_setup"] is dist_setup
    assert stub.to_calls == []  # sharded in place by from_pretrained, never moved
    assert stub.requires_grad_calls == [False]


def test_draft_ddp_process_group_no_mesh():
    # tp_size=1 -> no mesh -> default full-world group (None).
    recipe = _bare_recipe(dp_mesh=None, dist_env=SimpleNamespace(world_size=2))
    assert recipe._draft_ddp_process_group() is None


def test_draft_ddp_process_group_full_world_when_dp_spans_world():
    # dp spans the whole world (no tp) -> default full-world group (None).
    dp_mesh = SimpleNamespace(size=lambda: 2, get_group=lambda: "DP_GROUP")
    recipe = _bare_recipe(dp_mesh=dp_mesh, dist_env=SimpleNamespace(world_size=2))
    assert recipe._draft_ddp_process_group() is None


def test_draft_ddp_process_group_restricts_to_dp_axis():
    # dp smaller than world (tp>1) -> reduce only over the dp sub-group.
    dp_mesh = SimpleNamespace(size=lambda: 1, get_group=lambda: "DP_GROUP")
    recipe = _bare_recipe(dp_mesh=dp_mesh, dist_env=SimpleNamespace(world_size=2))
    assert recipe._draft_ddp_process_group() == "DP_GROUP"


def test_build_checkpointer_keys_dp_rank_on_dp_mesh(monkeypatch, tmp_path):
    """The checkpoint shard is keyed on the dp coordinate (identical across tp
    ranks of a replica), and tp_rank stays 0 (the draft is never tp-sharded)."""
    captured = {}
    monkeypatch.setattr(train_dflash, "CheckpointingConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(train_dflash, "Checkpointer", lambda **kw: captured.update(kw) or SimpleNamespace(**kw))

    recipe = _bare_recipe(
        cfg={},  # no "checkpoint" section
        output_dir=tmp_path,
        draft_model=torch.nn.Linear(2, 2),
        dp_mesh=SimpleNamespace(get_local_rank=lambda: 3),
    )
    recipe._build_checkpointer("target/path")

    assert captured["dp_rank"] == 3
    assert captured["tp_rank"] == 0


def test_build_checkpointer_falls_back_to_global_rank(monkeypatch, tmp_path):
    # tp_size=1 -> dp_mesh None -> key on the global rank (0 here, no dist init).
    captured = {}
    monkeypatch.setattr(train_dflash, "CheckpointingConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(train_dflash, "Checkpointer", lambda **kw: captured.update(kw) or SimpleNamespace(**kw))
    monkeypatch.setattr(train_dflash.dist, "is_initialized", lambda: False)

    recipe = _bare_recipe(
        cfg={},
        output_dir=tmp_path,
        draft_model=torch.nn.Linear(2, 2),
        dp_mesh=None,
    )
    recipe._build_checkpointer("target/path")

    assert captured["dp_rank"] == 0
