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

"""Tests for the export dtype mapping when the ranks that save a model hold different tensors."""

import json
import logging
import os
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file

from nemo_automodel.components.checkpoint._backports.hf_utils import FQN_TO_DTYPE_MAPPING_FILENAME
from nemo_automodel.components.checkpoint.checkpointing import CheckpointingConfig

pytestmark = pytest.mark.cpu_only

_REPO_ID = "test/base-checkpoint"
_WORLD_SIZE = 3
_EXPERT_COUNT = 4
_MODULES = ("shared", *(f"experts.{index}" for index in range(_EXPERT_COUNT)))
# Modules per shard of the base checkpoint. Rank 0 writes output shard 1 and rank 1 writes output shard 2, so each
# of them writes tensors of an expert that only the other one holds.
_BASE_SHARDS = (("shared", "experts.0", "experts.2"), ("experts.1", "experts.3"))


class _SaveCase(NamedTuple):
    """One save of the model from a world of three ranks.

    Attributes:
        default_group: Whether all three ranks save through the default group. Otherwise ranks 0 and 1 save through
            a process group of their own and rank 2 takes no part.
        rank_local_experts: Whether ranks 0 and 1 hold only their own two of the four experts, as under expert
            parallelism, instead of all of them. Rank 2 holds the shared weight only.
        base_dtypes: Dtype of the base checkpoint in the Hugging Face cache of each rank, or ``None`` for a rank
            without a cached base checkpoint.
        pinning_ranks: Ranks whose model carries the adapter that pins the biases to FP32.
        single_rank_consolidation: Whether rank 0 alone writes the consolidated export.
        save_consolidated: Whether the save writes the consolidated export at all.
    """

    default_group: bool = False
    rank_local_experts: bool = True
    base_dtypes: tuple[torch.dtype | None, torch.dtype | None, torch.dtype | None] = (torch.bfloat16,) * 3
    pinning_ranks: tuple[int, ...] = (0, 1, 2)
    single_rank_consolidation: bool = False
    save_consolidated: bool = True


_SAVE_CASES = {
    # A checkpointer of the default group creates a process group of all ranks, which every rank has to do at the
    # same point. This save therefore comes first, before ranks 0 and 1 create process groups of their own.
    "default_group": _SaveCase(default_group=True),
    "every_rank": _SaveCase(),
    "single_rank": _SaveCase(single_rank_consolidation=True),
    "offline": _SaveCase(save_consolidated=False),
    # Rank 0 has no cached base checkpoint and pins nothing, so its own mapping is empty.
    "empty_mapping_on_rank_0": _SaveCase(
        base_dtypes=(None, torch.bfloat16, None), pinning_ranks=(1,), single_rank_consolidation=True
    ),
    "different_base_checkpoints": _SaveCase(base_dtypes=(torch.bfloat16, torch.float16, None)),
    "same_tensors": _SaveCase(rank_local_experts=False),
    "no_base_checkpoint": _SaveCase(base_dtypes=(None, None, None)),
}


class _FP32BiasAdapter:
    """State-dict adapter whose exported names equal the native names and whose biases must stay FP32."""

    def to_hf(self, state_dict: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """Export the tensors under their native names.

        Args:
            state_dict: Native names mapped to tensors of arbitrary shape.
            **kwargs: Unused export options.

        Returns:
            The same names mapped to the same tensors, without copies.
        """
        return dict(state_dict)

    def forced_hf_dtype_mapping(self, state_dict: dict[str, torch.Tensor]) -> dict[str, str]:
        """Pin the biases among the tensors that this rank exports.

        Args:
            state_dict: Exported names mapped to tensors of arbitrary shape; only the names are read.

        Returns:
            The safetensors dtype ``F32`` for every bias in ``state_dict``.
        """
        return {key: "F32" for key in state_dict if key.endswith(".bias")}


def _saved_values() -> dict[str, float]:
    """Return the fill value of every exported tensor.

    The weight values are exact in BF16 and F16. The bias values are not exact in BF16, so a cast shows in the value.
    """
    values = {"shared.weight": -1.0}
    for index in range(_EXPERT_COUNT):
        values[f"experts.{index}.weight"] = float(index)
        values[f"experts.{index}.bias"] = index + 0.3
    return values


def _build_model(expert_indices: range, pins_biases: bool) -> torch.nn.Module:
    """Build an FP32 model with a shared weight and one linear layer, of its own size, per local expert."""
    model = torch.nn.Module()
    model.shared = torch.nn.Linear(4, 4, bias=False)
    model.experts = torch.nn.ModuleDict({str(index): torch.nn.Linear(4, 4 * (index + 1)) for index in expert_indices})
    values = _saved_values()
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            parameter.fill_(values[key])
    if pins_biases:
        model.state_dict_adapter = _FP32BiasAdapter()
    # The complete key list, computed before sharding and stamped on every rank.
    model._pre_shard_hf_state_dict_keys = list(values)
    return model


def _write_base_checkpoint(cache_dir: Path, dtype: torch.dtype) -> None:
    """Cache a base checkpoint that stores every tensor of the model in ``dtype``."""
    snapshot = cache_dir / f"models--{_REPO_ID.replace('/', '--')}" / "snapshots" / "main"
    snapshot.mkdir(parents=True)
    weight_map = {}
    for index, modules in enumerate(_BASE_SHARDS, start=1):
        filename = f"model-{index:05d}-of-{len(_BASE_SHARDS):05d}.safetensors"
        tensors = {key: torch.zeros(1, dtype=dtype) for key in _saved_values() if key.rsplit(".", 1)[0] in modules}
        save_file(tensors, str(snapshot / filename))
        weight_map.update(dict.fromkeys(tensors, filename))
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def _run_save_worker(rank: int, init_file: str, root: str) -> None:
    """Save every case from a world of three ranks.

    The checkpoint of a case is written to ``root/<case>/checkpoint``. A rank reads the base checkpoint from
    ``root/<case>/cache_rank<rank>``, which may be absent.
    """
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    # Rank 2 waits on the default group while the other two ranks save, so that group gets the longer timeout.
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=_WORLD_SIZE, timeout=timedelta(seconds=120)
    )
    try:
        # A rank that enters a collective of this group without its peer fails after the timeout instead of hanging.
        save_group = dist.new_group(ranks=[0, 1], backend="gloo", timeout=timedelta(seconds=30))
        for name, case in _SAVE_CASES.items():
            if rank == 2 and not case.default_group:
                continue
            if rank == 2:
                experts = range(0)
            elif case.rank_local_experts:
                experts = range(2 * rank, 2 * rank + 2)
            else:
                experts = range(_EXPERT_COUNT)
            checkpointer = CheckpointingConfig(
                checkpoint_dir=os.path.join(root, name),
                model_cache_dir=os.path.join(root, name, f"cache_rank{rank}"),
                model_repo_id=_REPO_ID,
                save_consolidated=case.save_consolidated,
                single_rank_consolidation=case.single_rank_consolidation,
            ).build(dp_rank=rank, tp_rank=0, pp_rank=0, process_group=None if case.default_group else save_group)
            try:
                model = _build_model(experts, pins_biases=rank in case.pinning_ranks)
                checkpointer.save_model(model, os.path.join(root, name, "checkpoint"))
            finally:
                checkpointer.close()
        # Rank 2 never enters the collectives of the group of ranks 0 and 1; the world is still usable by all ranks.
        check = torch.ones(1)
        dist.all_reduce(check)
        assert check.item() == _WORLD_SIZE
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def saved_models(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Save every case from one world of three ranks and return the ``model`` directory of each checkpoint."""
    root = tmp_path_factory.mktemp("dtype_mapping")
    for name, case in _SAVE_CASES.items():
        for rank, dtype in enumerate(case.base_dtypes):
            if dtype is not None:
                _write_base_checkpoint(root / name / f"cache_rank{rank}", dtype)
    mp.spawn(_run_save_worker, args=(str(root / "dist_init"), str(root)), nprocs=_WORLD_SIZE, join=True)
    return {name: root / name / "checkpoint" / "model" for name in _SAVE_CASES}


def _read_export(consolidated: Path) -> tuple[dict[str, torch.Tensor], int]:
    """Read a consolidated export through its index.

    Returns:
        The exported names mapped to tensors of arbitrary shape, and the ``total_size`` that the index states.
    """
    index = json.loads((consolidated / "model.safetensors.index.json").read_text())
    exported = {}
    for filename in set(index["weight_map"].values()):
        exported.update(load_file(str(consolidated / filename)))
    assert set(index["weight_map"]) == set(exported)
    return exported, index["metadata"]["total_size"]


def _assert_export(consolidated: Path, weight_dtypes: dict[str, torch.dtype]) -> None:
    """Check that an export holds every tensor with the expected dtype and exactly its saved value.

    ``weight_dtypes`` names the modules whose weight is expected in another dtype than FP32. Every bias is expected
    in FP32.
    """
    exported, _ = _read_export(consolidated)
    expected = {}
    for key, value in _saved_values().items():
        module, kind = key.rsplit(".", 1)
        dtype = weight_dtypes.get(module, torch.float32) if kind == "weight" else torch.float32
        expected[key] = (dtype, [torch.tensor(value).item()])
    assert {key: (tensor.dtype, tensor.float().unique().tolist()) for key, tensor in exported.items()} == expected


def _consolidate_offline(model_dir: Path, output_dir: Path, monkeypatch: pytest.MonkeyPatch, *options: str) -> None:
    """Run the offline consolidation tool in this process, as one worker."""
    from tools import offline_hf_consolidation as tool

    arguments = ["--backend", "gloo", "--model-name", _REPO_ID, "--input-dir", str(model_dir)]
    monkeypatch.setattr("sys.argv", ["offline_hf_consolidation", *arguments, "--output-dir", str(output_dir), *options])
    with (
        patch.object(tool, "initialize_distributed"),
        patch.object(tool, "get_world_size_safe", return_value=1),
        patch.object(tool, "get_rank_safe", return_value=0),
    ):
        tool.main()


@pytest.mark.parametrize("case", ["every_rank", "single_rank", "default_group"])
def test_export_restores_base_dtypes_of_tensors_held_by_another_rank(saved_models, case):
    """Each rank holds two of four experts: every weight returns to BF16 and every pinned bias stays FP32."""
    _assert_export(saved_models[case] / "consolidated", dict.fromkeys(_MODULES, torch.bfloat16))


def test_index_total_size_counts_the_bytes_of_the_exported_tensors(saved_models):
    """Rank 0 sizes the index with the dtypes that every rank applied to the output shards it wrote."""
    exported, total_size = _read_export(saved_models["every_rank"] / "consolidated")

    assert total_size == sum(tensor.numel() * tensor.element_size() for tensor in exported.values())


def test_offline_consolidation_restores_base_dtypes_of_tensors_held_by_another_rank(saved_models, monkeypatch):
    """The mapping that rank 0 writes for the offline tool names the tensors of both ranks."""
    model_dir = saved_models["offline"]
    dtype_mapping = json.loads((model_dir / ".hf_metadata" / FQN_TO_DTYPE_MAPPING_FILENAME).read_text())
    assert dtype_mapping == {key: "F32" if key.endswith(".bias") else "BF16" for key in _saved_values()}

    _consolidate_offline(model_dir, model_dir / "consolidated", monkeypatch)

    _assert_export(model_dir / "consolidated", dict.fromkeys(_MODULES, torch.bfloat16))


def test_offline_cast_keeps_the_pinned_biases_of_another_rank_in_fp32(saved_models, monkeypatch):
    """A requested BF16 cast changes the weights only: the biases that rank 0 does not hold keep their exact values."""
    model_dir = saved_models["offline"]

    _consolidate_offline(model_dir, model_dir / "consolidated_bf16", monkeypatch, "--cast-dtype", "bf16")

    _assert_export(model_dir / "consolidated_bf16", dict.fromkeys(_MODULES, torch.bfloat16))


def test_rank_with_an_empty_mapping_applies_the_dtypes_of_its_peer(saved_models):
    """Rank 0 consolidates alone, has no cached base checkpoint and pins nothing, so its own mapping is empty.

    Both ranks still enter the same collectives. The tensors that rank 1 exports return to BF16. The weights that
    only rank 0 exports keep their saved dtype, because no rank read their base dtype.
    """
    restored = dict.fromkeys(("shared", "experts.2", "experts.3"), torch.bfloat16)

    _assert_export(saved_models["empty_mapping_on_rank_0"] / "consolidated", restored)


def test_ranks_with_different_base_checkpoints_use_the_dtype_of_the_lowest_rank(saved_models):
    """Rank 0 cached a BF16 base checkpoint and rank 1 an F16 one: the weight that both export follows rank 0."""
    expected = {
        "shared": torch.bfloat16,
        "experts.0": torch.bfloat16,
        "experts.1": torch.bfloat16,
        "experts.2": torch.float16,
        "experts.3": torch.float16,
    }

    _assert_export(saved_models["different_base_checkpoints"] / "consolidated", expected)


def test_export_restores_base_dtypes_when_both_ranks_hold_every_tensor(saved_models):
    """Both ranks export all tensors and derive the same mapping."""
    _assert_export(saved_models["same_tensors"] / "consolidated", dict.fromkeys(_MODULES, torch.bfloat16))


def test_export_keeps_saved_dtypes_without_cached_base_checkpoint(saved_models):
    """Without base checkpoint headers there is no dtype to restore, whichever rank writes a tensor."""
    _assert_export(saved_models["no_base_checkpoint"] / "consolidated", {})


def _build_dtype_mapping_as_rank(
    rank: int,
    reference: Path,
    base_dtype: torch.dtype,
    weight_names: list[str],
    all_gather_object: Callable[..., None],
) -> dict[str, str] | None:
    """Build the dtype mapping of one rank of a two-rank group whose collectives are replaced.

    Args:
        rank: Rank of this process in the group.
        reference: Empty directory for the base checkpoint that this rank reads.
        base_dtype: Dtype in which the base checkpoint stores the weights.
        weight_names: Names of the weights that this rank exports.
        all_gather_object: Replacement for ``torch.distributed.all_gather_object``.

    Returns:
        The mapping that the checkpointer derived for this rank.
    """
    save_file({name: torch.zeros(1, dtype=base_dtype) for name in weight_names}, str(reference / "model.safetensors"))
    checkpointer = CheckpointingConfig(
        checkpoint_dir=str(reference), model_repo_id=str(reference), save_consolidated=False
    ).build(dp_rank=rank, tp_rank=0, pp_rank=0)
    model_state = SimpleNamespace(model=[SimpleNamespace()])
    state_dict = dict.fromkeys(weight_names, torch.zeros(1))
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.get_rank", return_value=rank),
        patch("torch.distributed.all_gather_object", side_effect=all_gather_object),
    ):
        return checkpointer._maybe_build_original_dtype_mapping(model_state, state_dict)


def test_ranks_with_equal_dtype_mappings_send_one_digest(tmp_path):
    """Cost check: a rank whose peer reports the same digest keeps its mapping and does not send it."""
    sent = []

    def gather_from_equal_peer(object_list, obj, group=None):
        sent.append(obj)
        object_list[:] = [obj, obj]

    mapping = _build_dtype_mapping_as_rank(0, tmp_path, torch.bfloat16, ["shared.weight"], gather_from_equal_peer)

    assert mapping == {"shared.weight": "BF16"}
    assert len(sent) == 1
    assert mapping not in sent


@pytest.mark.parametrize("rank", [0, 1])
def test_ranks_that_disagree_on_a_dtype_use_the_lowest_rank_and_rank_0_warns(tmp_path, caplog, rank):
    """Both ranks derive the same mapping; rank 0 reports how many names differ and the first of them."""
    weight_names = (["shared.weight", "experts.0.weight"], ["shared.weight", "experts.2.weight"])
    base_dtypes = (torch.bfloat16, torch.float16)
    peer_mapping = dict.fromkeys(weight_names[1 - rank], "F16" if rank == 0 else "BF16")

    def gather_from_peer(object_list, obj, group=None):
        object_list[rank] = obj
        object_list[1 - rank] = "digest of the other rank" if isinstance(obj, str) else peer_mapping

    caplog.set_level(logging.WARNING)
    mapping = _build_dtype_mapping_as_rank(rank, tmp_path, base_dtypes[rank], weight_names[rank], gather_from_peer)

    assert mapping == {"shared.weight": "BF16", "experts.0.weight": "BF16", "experts.2.weight": "F16"}
    warned = "report different export dtypes for 1 tensor(s), the first being shared.weight" in caplog.text
    assert warned == (rank == 0)
