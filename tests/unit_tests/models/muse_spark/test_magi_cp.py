# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed Magi metadata must follow each interleaved pipeline microbatch."""

from types import SimpleNamespace

import pytest
import torch
from torch.distributed.pipelining.microbatch import split_args_kwargs_into_chunks

from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.models.muse_spark import magi_cp
from tests.unit_tests.models.muse_spark.conftest import tiny_muse_spark_config, tiny_muse_spark_model


class _FakeDeviceMesh:
    """Named CP-only mesh used to exercise public sharder resolution."""

    mesh_dim_names = ("cp",)

    def __init__(self, cp_mesh) -> None:
        self._cp_mesh = cp_mesh

    def __getitem__(self, name: str):
        if name != "cp":
            raise KeyError(name)
        return self._cp_mesh


@pytest.mark.parametrize("pipeline,rows", [(False, 1), (True, 1), (True, 3)])
def test_magi_shards_token_masks_and_retains_independent_microbatch_plans(
    monkeypatch, pipeline: bool, rows: int
) -> None:
    """Exercise real THD conversion and PyTorch microbatch splitting with distinct packs."""
    counts = [[2, 6], [3, 5], [4, 4]][:rows]
    ids = torch.arange(rows * 8).reshape(rows, 8)
    positions = torch.stack([torch.cat([torch.arange(n) for n in row]) for row in counts])
    batch = {
        "input_ids": ids,
        "labels": ids + 1,
        "position_ids": positions,
        "seq_lens": torch.tensor([[a, b - 1] for a, b in counts]),
        "loss_mask": (ids % 3 != 0).float(),
        "seq_lens_padded": torch.tensor(counts),
        "qkv_format": "thd",
    }
    partition_by_boundary = {
        2: torch.tensor([0, 1, 6, 7]),
        3: torch.tensor([0, 2, 5, 7]),
        4: torch.tensor([1, 3, 4, 6]),
    }

    def key_for(cu: torch.Tensor, *args, **kwargs):
        """Map CPU document boundaries [documents+1] to a fake CP partition."""
        return SimpleNamespace(indices=partition_by_boundary[int(cu[1])], total_seqlen_q=8, boundary=int(cu[1]))

    def dispatch(tensor: torch.Tensor, key, **kwargs) -> torch.Tensor:
        """Select CP-local [tokens] from the global [sequence] input."""
        return tensor[key.indices]

    api = SimpleNamespace(
        magi_attn_varlen_key=key_for,
        make_varlen_key_for_new_mask_after_dispatch=lambda cuq, cuk, key, **kw: key,
        get_position_ids=lambda key: key.indices,
        dispatch=dispatch,
    )
    configs = SimpleNamespace(DistAttnConfig=lambda **kw: kw, DispatchConfig=lambda **kw: kw)
    monkeypatch.setattr(magi_cp, "safe_import", lambda name: (True, api if name.endswith(".api") else configs))
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    mesh = SimpleNamespace(size=lambda: 2, get_group=lambda: group)
    model = SimpleNamespace(
        config=tiny_muse_spark_config(),
        backend=SimpleNamespace(attn="magi"),
        cp_mesh=mesh,
        _is_pipeline_parallel_stage=lambda: pipeline,
    )
    model.prepare_model_inputs_for_cp = lambda current_batch, num_chunks=1: magi_cp.prepare_magi_inputs(
        model, current_batch, num_chunks=num_chunks
    )
    sharder = ContextParallelSharder(model, _FakeDeviceMesh(mesh), batch, num_chunks=rows)
    _, result = sharder.shard(batch)
    layout = sharder.shard_layout
    assert result["padding_mask"].shape == ((rows, 4) if pipeline else (4,))
    expected_padding = torch.stack([partition_by_boundary[row[0]] == 7 for row in counts])
    assert torch.equal(result["padding_mask"].reshape(rows, -1), expected_padding)
    if not pipeline:
        assert layout.input_row_shape == (1, 8)
        assert torch.equal(result["input_ids"], ids[0, partition_by_boundary[2]])
        return
    assert layout is None
    assert torch.equal(result["input_ids"], ids)
    input_ids = result.pop("input_ids")
    inputs, kwargs = split_args_kwargs_into_chunks((input_ids,), result, rows)
    for index in (*reversed(range(rows)), 0):
        chunk = kwargs[index]
        boundary = counts[index][0]
        local_indices = partition_by_boundary[boundary]
        plan = chunk["_muse_spark_magi_plan"].select(chunk["_muse_spark_magi_microbatch"])
        assert plan.keys[0].boundary == boundary
        assert torch.equal(chunk["position_ids"][0], positions[index, local_indices])
        assert torch.equal(chunk["labels"][0], ids[index, local_indices] + 1)
        assert torch.equal(chunk["loss_mask"][0], batch["loss_mask"][index, local_indices])
        assert torch.equal(inputs[index][0][0, local_indices], ids[index, local_indices])
        assert chunk["_muse_spark_magi_microbatch"].device.type == "cpu"


def test_magi_pp_rejects_multiple_packs_per_microbatch(monkeypatch) -> None:
    """Keep the supported one-pack-per-microbatch contract explicit."""
    model = SimpleNamespace(config=tiny_muse_spark_config(), _is_pipeline_parallel_stage=lambda: True)
    monkeypatch.setattr(magi_cp, "safe_import", lambda name: (False, None))
    batch = {"input_ids": torch.zeros(4, 8, dtype=torch.long), "qkv_format": "thd"}
    sharder = magi_cp.prepare_magi_inputs(model, batch, num_chunks=2)["cp_sharder"]
    with pytest.raises(ValueError, match="one packed row per microbatch"):
        sharder.shard_batch(None, None, batch)


def test_magi_without_pp_rejects_multiple_microbatches() -> None:
    """The flat non-PP token layout represents one pack only."""
    model = SimpleNamespace(config=tiny_muse_spark_config(), _is_pipeline_parallel_stage=lambda: False)
    with pytest.raises(ValueError, match="without PP requires one packed microbatch"):
        magi_cp.prepare_magi_inputs(model, {"qkv_format": "thd"}, num_chunks=2)


@pytest.mark.parametrize("index", [-1, 2])
def test_magi_batch_plans_reject_out_of_range_index(index: int) -> None:
    """Reject invalid schedule identities instead of selecting another plan."""
    plan = magi_cp.MuseMagiPlan({0: SimpleNamespace(total_seqlen_q=8)})
    plans = magi_cp._MuseMagiBatchPlans((plan, plan))
    with pytest.raises(IndexError, match="outside"):
        plans.select(torch.tensor([index]))


def test_muse_forward_selects_the_pipeline_microbatch_plan() -> None:
    """Resolve the immutable plan before passing metadata to the backbone."""

    class CaptureBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.plan = None
            self.padding_mask = None

        def forward(self, *, inputs_embeds: torch.Tensor, **kwargs):
            self.plan = kwargs["_muse_spark_magi_plan"]
            self.padding_mask = kwargs["padding_mask"]
            return SimpleNamespace(last_hidden_state=inputs_embeds)

    model = tiny_muse_spark_model()
    capture = CaptureBackbone()
    model.model = capture
    model.lm_head = None
    first = magi_cp.MuseMagiPlan({0: SimpleNamespace(total_seqlen_q=4)})
    second = magi_cp.MuseMagiPlan({0: SimpleNamespace(total_seqlen_q=4)})
    plans = magi_cp._MuseMagiBatchPlans((first, second))
    padding_mask = torch.tensor([[False, False, False, True]])

    model(
        inputs_embeds=torch.zeros(1, 4, model.config.hidden_size),
        position_ids=torch.arange(4).unsqueeze(0),
        padding_mask=padding_mask,
        qkv_format="thd",
        _muse_spark_magi_plan=plans,
        _muse_spark_magi_microbatch=torch.tensor([1]),
    )

    assert capture.plan is second
    assert torch.equal(capture.padding_mask, padding_mask.squeeze(0))
