# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for the pipeline-parallel PEFT-adapter gather.

Background
----------
Under ``pp_size > 1`` the local PEFT save path in
:class:`ModelState` only collects adapter tensors from the local PP stages,
so the on-disk adapter is missing every layer owned by another stage (a
grouped-experts MoE could save only ~1/pp of its layers). The fix all-gathers
the per-rank PEFT dicts across the PP group and unions ``target_modules`` the
same way.

Most of these tests fake ``torch.distributed`` so the merge/union logic runs on CPU
with no real process group — the distributed primitive is trivial; the
correctness we care about is the merge, the dedup, the world==1 short-circuit,
and the degenerate-gather warning. Modules trained in full next to LoRA move
between stages with their own collectives, which a two-rank gloo test exercises.
"""

import logging
import os
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
from torch import nn

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.addons import (
    _extract_modules_to_save,
    _extract_target_modules,
    _extract_target_parameters,
)
from nemo_automodel.components.checkpoint.stateful_wrappers import (
    ModelState,
    _gather_peft_state_dict_across_pp,
    _trainable_module_layout,
)
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin


class _FakePPGroup:
    """Marker object standing in for a real ProcessGroup in tests."""

    def __init__(self, ranks_state_dicts):
        # ranks_state_dicts: list of per-rank payloads the fake all_gather returns.
        self.ranks_state_dicts = ranks_state_dicts


class _V5MoEAdapter(MoESplitExpertsStateDictMixin):
    """Minimal adapter that opts into fused PEFT v5 expert parameters."""

    @property
    def _v5_peft_target_parameters(self):
        return ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")


def _install_fake_distributed(monkeypatch, per_rank_payloads):
    """Patch torch.distributed so collectives operate on ``per_rank_payloads``.

    ``get_world_size`` returns len(per_rank_payloads); ``all_gather_object``
    fills the output list with a copy of ``per_rank_payloads`` (i.e. what every
    rank would receive after the gather).
    """
    world = len(per_rank_payloads)

    def fake_get_world_size(group=None):
        return world

    def fake_all_gather_object(out_list, obj, group=None):
        # Simulate the collective: every rank ends up with all ranks' objects.
        for i, payload in enumerate(per_rank_payloads):
            out_list[i] = payload

    monkeypatch.setattr(torch.distributed, "get_world_size", fake_get_world_size)
    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)


def _t(val):
    """Tiny tensor factory so dicts hold real tensors (CPU)."""
    return torch.tensor([float(val)])


# ---------------------------------------------------------------------------
# _gather_peft_state_dict_across_pp
# ---------------------------------------------------------------------------
class TestGatherPeftStateDictAcrossPP:
    def test_merges_disjoint_layers_from_all_ranks(self, monkeypatch):
        """The canonical bug case: each rank holds a disjoint layer subset;
        the gather must produce the union of all of them."""
        # pp=4, 2 layers/rank, keys disjoint by layer index (mirrors real PP).
        rank0 = {
            "base_model.model.model.layers.0.self_attn.wq_a.lora_A.weight": _t(0),
            "base_model.model.model.layers.1.self_attn.wq_a.lora_A.weight": _t(1),
        }
        rank1 = {
            "base_model.model.model.layers.2.self_attn.wq_a.lora_A.weight": _t(2),
            "base_model.model.model.layers.3.self_attn.wq_a.lora_A.weight": _t(3),
        }
        rank2 = {
            "base_model.model.model.layers.4.self_attn.wq_a.lora_A.weight": _t(4),
            "base_model.model.model.layers.5.self_attn.wq_a.lora_A.weight": _t(5),
        }
        rank3 = {
            "base_model.model.model.layers.6.self_attn.wq_a.lora_A.weight": _t(6),
            "base_model.model.model.layers.7.self_attn.wq_a.lora_A.weight": _t(7),
        }
        payloads = [rank0, rank1, rank2, rank3]
        _install_fake_distributed(monkeypatch, payloads)

        # Call from rank 0's perspective (local_state_dict == rank0).
        merged = _gather_peft_state_dict_across_pp(rank0, _FakePPGroup(payloads))

        assert len(merged) == 8
        for layer in range(8):
            key = f"base_model.model.model.layers.{layer}.self_attn.wq_a.lora_A.weight"
            assert key in merged, f"missing layer {layer}"

    def test_world_size_one_returns_local_unchanged(self, monkeypatch):
        """pp=1 (single rank): no gather, return the local dict object as-is."""
        local = {"base_model.model.model.layers.0.self_attn.wq_a.lora_A.weight": _t(0)}
        _install_fake_distributed(monkeypatch, [local])

        merged = _gather_peft_state_dict_across_pp(local, _FakePPGroup([local]))

        assert merged is local  # short-circuit returns the same object

    def test_duplicate_key_resolves_to_lowest_rank(self, monkeypatch):
        """If the same FQN appears on two ranks (e.g. a replicated param),
        the lowest-rank value must win deterministically."""
        dup_key = "base_model.model.model.embed_tokens.lora_A.weight"
        rank0 = {dup_key: _t(100)}
        rank1 = {dup_key: _t(200)}
        payloads = [rank0, rank1]
        _install_fake_distributed(monkeypatch, payloads)

        merged = _gather_peft_state_dict_across_pp(rank0, _FakePPGroup(payloads))

        assert len(merged) == 1
        assert merged[dup_key].item() == 100.0  # rank 0 wins

    def test_handles_empty_rank_payload(self, monkeypatch):
        """A rank that trained no PEFT params (empty dict) must not break the
        merge (defensive: e.g. a stage that holds only the embed/head)."""
        rank0 = {"base_model.model.model.layers.0.self_attn.wq_a.lora_A.weight": _t(0)}
        rank1 = {}  # empty
        rank2 = {"base_model.model.model.layers.4.self_attn.wq_a.lora_A.weight": _t(4)}
        payloads = [rank0, rank1, rank2]
        _install_fake_distributed(monkeypatch, payloads)

        merged = _gather_peft_state_dict_across_pp(rank0, _FakePPGroup(payloads))

        assert len(merged) == 2
        assert "base_model.model.model.layers.0.self_attn.wq_a.lora_A.weight" in merged
        assert "base_model.model.model.layers.4.self_attn.wq_a.lora_A.weight" in merged

    def test_degenerate_gather_emits_warning(self, monkeypatch, caplog):
        """If the gather adds nothing beyond the largest single rank (e.g. a
        wrong/global group was passed), warn loudly — this is the failure this
        fix exists to catch."""
        # All ranks report the SAME single key → merged size == max per-rank size.
        same = {"base_model.model.model.layers.0.self_attn.wq_a.lora_A.weight": _t(0)}
        payloads = [same, same]
        _install_fake_distributed(monkeypatch, payloads)

        with caplog.at_level(logging.WARNING):
            merged = _gather_peft_state_dict_across_pp(same, _FakePPGroup(payloads))

        assert len(merged) == 1
        assert any("may be INCOMPLETE" in rec.message for rec in caplog.records)

    def test_single_nonempty_rank_does_not_warn(self, monkeypatch, caplog):
        """A valid PP layout can place all trainable adapters on one stage, with
        every other stage contributing an empty dict (per_rank=[N, 0, ...]).
        Then merged_n == max(per_rank) is correct, not a collapse, so the
        INCOMPLETE warning must NOT fire."""
        rank0 = {
            "base_model.model.model.layers.0.self_attn.wq_a.lora_A.weight": _t(0),
            "base_model.model.model.layers.1.self_attn.wq_a.lora_A.weight": _t(1),
        }
        payloads = [rank0, {}, {}]  # only rank 0 holds adapters
        _install_fake_distributed(monkeypatch, payloads)

        with caplog.at_level(logging.WARNING):
            merged = _gather_peft_state_dict_across_pp(rank0, _FakePPGroup(payloads))

        assert len(merged) == 2
        assert not any("may be INCOMPLETE" in rec.message for rec in caplog.records)

    def test_healthy_gather_does_not_warn(self, monkeypatch, caplog):
        rank0 = {"base_model.model.model.layers.0.self_attn.wq_a.lora_A.weight": _t(0)}
        rank1 = {"base_model.model.model.layers.1.self_attn.wq_a.lora_A.weight": _t(1)}
        payloads = [rank0, rank1]
        _install_fake_distributed(monkeypatch, payloads)

        with caplog.at_level(logging.WARNING):
            _gather_peft_state_dict_across_pp(rank0, _FakePPGroup(payloads))

        assert not any("may be INCOMPLETE" in rec.message for rec in caplog.records)


def _pipeline_stage(index: int) -> nn.Module:
    """Stage ``index`` of a two-stage pipeline: LoRA in its decoder layer, other modules trained in full.

    The first stage holds a bf16 embedding. The last holds an fp32 norm with a persistent 0-dim int64
    buffer and a bf16 LM head. Every rank builds both stages and gets the same values.
    """
    torch.manual_seed(index)
    stage = nn.Module()
    stage.model = nn.Module()
    if index == 0:
        stage.model.embed_tokens = nn.Embedding(8, 4, dtype=torch.bfloat16)
    stage.model.layers = nn.ModuleDict({str(index): nn.Module()})
    stage.model.layers[str(index)].q_proj = nn.Linear(4, 4)
    if index == 1:
        stage.model.norm = nn.LayerNorm(4)
        stage.model.norm.register_buffer("steps", torch.tensor(3))
        stage.lm_head = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
    apply_lora_to_linear_modules(stage, PeftConfig(target_modules=["*.q_proj"], dim=2, alpha=4, use_triton=False))
    with torch.no_grad():
        for name, parameter in stage.named_parameters():
            parameter.copy_(torch.randn_like(parameter))
            parameter.requires_grad_("lora_" in name or not name.startswith("model.layers."))
    return stage


def _run_pipeline_stage_save_worker(rank: int, init_file: str) -> None:
    """Each rank saves its own stage and must return the whole adapter with exact values and dtypes."""
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        stages = [_pipeline_stage(0), _pipeline_stage(1)]
        pickled_payloads = []
        all_gather_object = torch.distributed.all_gather_object

        def recording_all_gather_object(object_list, obj, group=None):
            pickled_payloads.append(obj)
            all_gather_object(object_list, obj, group=group)

        with patch.object(torch.distributed, "all_gather_object", recording_all_gather_object):
            state = ModelState(stages[rank], is_peft=True, pp_group=torch.distributed.group.WORLD).state_dict()

        expected = {
            f"base_model.model.{name}": parameter.detach()
            for stage in stages
            for name, parameter in stage.named_parameters()
            if parameter.requires_grad
        }
        expected["base_model.model.model.norm.steps"] = stages[1].model.norm.steps
        assert state.keys() == expected.keys(), sorted(state.keys() ^ expected.keys())
        for key, value in expected.items():
            assert state[key].dtype == value.dtype, (key, state[key].dtype, value.dtype)
            torch.testing.assert_close(
                state[key], value, rtol=0, atol=0, msg=lambda formatted, key=key: f"{key}: {formatted}"
            )
        # The tensors of modules stored whole never enter the pickled gather, which stages every rank's
        # payload on every rank's device at once.
        pickled_keys = {key for payload in pickled_payloads if isinstance(payload, dict) for key in payload}
        assert pickled_keys and all(".lora_" in key for key in pickled_keys), sorted(pickled_keys)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.run_only_on("CPU")
def test_pipeline_stages_share_fully_trained_modules_without_pickling_them(tmp_path):
    """Every stage ends up with the other stage's modules trained in full, moved one tensor at a time."""
    torch.multiprocessing.spawn(
        _run_pipeline_stage_save_worker, args=(str(tmp_path / "dist_init"),), nprocs=2, join=True
    )


# ---------------------------------------------------------------------------
# _extract_target_modules PP union
# ---------------------------------------------------------------------------
def _make_model_with_named_modules(module_names):
    """Build a dummy model whose ``named_modules`` yields the given names
    (each target gets a ``.lora_A`` leaf so _extract_target_modules picks it up)."""
    root = nn.Module()
    for name in module_names:
        parts = name.split(".")
        parent = root
        for part in parts[:-1]:
            if not hasattr(parent, part):
                setattr(parent, part, nn.Module())
            parent = getattr(parent, part)
        setattr(parent, parts[-1], nn.Identity())
    return root


class TestExtractTargetModulesPPUnion:
    def test_unions_target_modules_across_pp_ranks(self, monkeypatch):
        """adapter_config.json target_modules must list every stage's layers,
        not just the local stage's (the second half of the truncation bug)."""
        # Local (rank 0) model only has layer 0's modules.
        local_model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.wq_a.lora_A",
                "model.layers.0.self_attn.wkv.lora_A",
            ]
        )
        # Other ranks contribute layers 1 and 2 (what the gather would supply).
        rank1_targets = ["model.layers.1.self_attn.wq_a", "model.layers.1.self_attn.wkv"]
        rank2_targets = ["model.layers.2.self_attn.wq_a", "model.layers.2.self_attn.wkv"]
        local_targets = ["model.layers.0.self_attn.wkv", "model.layers.0.self_attn.wq_a"]
        payloads = [local_targets, rank1_targets, rank2_targets]
        _install_fake_distributed(monkeypatch, payloads)

        result = _extract_target_modules(local_model, pp_group=_FakePPGroup(payloads))

        # Union covers all three layers.
        for layer in (0, 1, 2):
            assert f"model.layers.{layer}.self_attn.wq_a" in result
            assert f"model.layers.{layer}.self_attn.wkv" in result

    def test_no_pp_group_returns_local_only(self):
        """Without a PP group (pp_size==1 or non-PP), behaviour is unchanged:
        only the local model's modules are returned."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.wq_a.lora_A",
            ]
        )
        result = _extract_target_modules(model, pp_group=None)
        assert result == ["model.layers.0.self_attn.wq_a"]


class TestExtractModulesToSavePPUnion:
    def test_stage_layouts_are_combined_before_naming_modules(self, monkeypatch):
        """Every stage names the same modules, and a partial view never widens one.

        The last stage holds model.norm but no decoder layer, so on its own every parameter
        under ``model`` looks trained in full and ``model`` itself would be listed. HF PEFT
        would then wrap the whole backbone and fail to load it from the adapter.
        """
        first = nn.Module()
        first.model = nn.Module()
        first.model.embed_tokens = nn.Embedding(4, 2)
        first.model.layers = nn.ModuleList([nn.Module()])
        first.model.layers[0].q_proj = nn.Linear(2, 2).requires_grad_(False)
        first.model.layers[0].q_proj.lora_A = nn.Linear(2, 1, bias=False)
        last = nn.Module()
        last.model = nn.Module()
        last.model.norm = nn.LayerNorm(2)
        last.lm_head = nn.Linear(2, 4)
        layouts = [_trainable_module_layout([first]), _trainable_module_layout([last])]
        _install_fake_distributed(monkeypatch, layouts)

        for stage in (first, last):
            assert _extract_modules_to_save([stage], pp_group=_FakePPGroup(layouts)) == [
                "lm_head",
                "model.embed_tokens",
                "model.norm",
            ]


class TestExtractTargetParametersPPUnion:
    def test_remote_expert_lora_enables_v5_metadata_on_every_stage(self, monkeypatch):
        """A PP rank without experts must emit metadata when another rank owns them."""
        local_model = nn.Module()
        local_model.state_dict_adapter = _V5MoEAdapter()
        payloads = [False, True]
        _install_fake_distributed(monkeypatch, payloads)

        result = _extract_target_parameters(local_model, pp_group=_FakePPGroup(payloads))

        assert result == ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
