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

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nemo_automodel.components.speculative.serve_sglang import build_sglang_argv, resolve_draft_artifacts


def _build_args(draft: str, algorithm: str = "EAGLE3", *, print_only: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        target="meta-llama/Llama-3.1-8B-Instruct",
        draft=draft,
        algorithm=algorithm,
        num_steps=3,
        topk=1,
        num_draft_tokens=4,
        host="0.0.0.0",
        port=30000,
        mem_fraction_static=0.75,
        dtype="bfloat16",
        tp_size=1,
        trust_remote_code=False,
        print_only=print_only,
        extra=[],
    )


def test_resolve_draft_artifacts_prefers_existing_export_dir(tmp_path: Path):
    checkpoint_dir = tmp_path / "epoch_0_step_1000"
    model_dir = checkpoint_dir / "model"
    model_dir.mkdir(parents=True)
    # Seed with an already-correct architectures field so this test exercises
    # only the path-resolution branch, not the in-place rewrite branch.
    config_path = model_dir / "config.json"
    config_path.write_text(json.dumps({"architectures": ["LlamaForCausalLMEagle3"]}), encoding="utf-8")
    config_mtime_before = config_path.stat().st_mtime_ns
    (model_dir / "model.safetensors").write_bytes(b"weights")
    torch.save(torch.arange(4), model_dir / "speculative_token_map.pt")

    resolved_model, resolved_token_map = resolve_draft_artifacts(str(checkpoint_dir), "EAGLE3")

    assert resolved_model == str(model_dir)
    assert resolved_token_map == str(model_dir / "speculative_token_map.pt")
    assert config_path.stat().st_mtime_ns == config_mtime_before, (
        "config.json should not be rewritten when already correct"
    )


def test_resolve_draft_artifacts_rewrites_stale_exported_config(tmp_path: Path):
    checkpoint_dir = tmp_path / "epoch_0_step_1000"
    model_dir = checkpoint_dir / "model"
    model_dir.mkdir(parents=True)
    stale_config = model_dir / "config.json"
    stale_config.write_text(json.dumps({"architectures": ["LlamaEagle3DraftModel"]}), encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    torch.save(torch.arange(4), model_dir / "speculative_token_map.pt")

    resolved_model, resolved_token_map = resolve_draft_artifacts(str(checkpoint_dir), "EAGLE3")

    assert resolved_model == str(model_dir)
    assert resolved_token_map == str(model_dir / "speculative_token_map.pt")
    rewritten = json.loads(stale_config.read_text(encoding="utf-8"))
    assert rewritten["architectures"] == ["LlamaForCausalLMEagle3"]


def test_resolve_draft_artifacts_dry_run_does_not_modify_stale_config(tmp_path: Path):
    """dry_run=True must return expected paths but never touch files on disk."""
    checkpoint_dir = tmp_path / "epoch_0_step_1000"
    model_dir = checkpoint_dir / "model"
    model_dir.mkdir(parents=True)
    stale_config = model_dir / "config.json"
    stale_content = json.dumps({"architectures": ["LlamaEagle3DraftModel"]})
    stale_config.write_text(stale_content, encoding="utf-8")
    config_mtime_before = stale_config.stat().st_mtime_ns
    (model_dir / "model.safetensors").write_bytes(b"weights")
    torch.save(torch.arange(4), model_dir / "speculative_token_map.pt")

    resolved_model, resolved_token_map = resolve_draft_artifacts(str(checkpoint_dir), "EAGLE3", dry_run=True)

    assert resolved_model == str(model_dir)
    assert resolved_token_map == str(model_dir / "speculative_token_map.pt")
    # The stale config must be left untouched in dry-run mode.
    assert stale_config.stat().st_mtime_ns == config_mtime_before, "config.json must not be rewritten in dry-run mode"
    assert stale_config.read_text(encoding="utf-8") == stale_content


def test_resolve_draft_artifacts_eagle1_skips_token_map_and_architecture_rewrite(
    tmp_path: Path,
    monkeypatch,
):
    """EAGLE-1 / EAGLE-2 do not need a token map and must not have architectures rewritten."""
    checkpoint_dir = tmp_path / "epoch_0_step_1000"
    checkpoint_dir.mkdir()
    original_archs = ["LlamaEagleDraftModel"]
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"architectures": original_archs, "model_type": "llama"}), encoding="utf-8"
    )
    (checkpoint_dir / "draft_model.pt").write_bytes(b"placeholder")
    # No eagle3_meta.pt: this is EAGLE-1 / EAGLE-2.

    def _fake_save_file(state_dict, path: str) -> None:
        Path(path).write_bytes(b"fake-safetensors")

    monkeypatch.setattr(
        "nemo_automodel.components.speculative.serve_sglang._load_safetensors_save_file",
        lambda: _fake_save_file,
    )
    monkeypatch.setattr(
        "nemo_automodel.components.speculative.serve_sglang._torch_load",
        lambda path: {"lm_head.weight": torch.ones(2, 2)},
    )

    resolved_model, resolved_token_map = resolve_draft_artifacts(str(checkpoint_dir), "EAGLE")

    model_dir = checkpoint_dir / "model"
    assert resolved_model == str(model_dir)
    assert resolved_token_map is None, "EAGLE-1 / EAGLE-2 should not emit a token map"
    exported_archs = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["architectures"]
    assert exported_archs == original_archs, (
        "architectures should not be rewritten for algorithms outside the EAGLE3 map"
    )


def test_export_patches_num_hidden_layers_from_state_dict(tmp_path: Path, monkeypatch):
    """Exported config must reflect the drafter's actual layer count, not the target model's."""
    checkpoint_dir = tmp_path / "epoch_0_step_1000"
    checkpoint_dir.mkdir()
    # Simulate a target-model config with num_hidden_layers=32.
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"architectures": ["LlamaEagle3DraftModel"], "num_hidden_layers": 32}), encoding="utf-8"
    )
    (checkpoint_dir / "draft_model.pt").write_bytes(b"placeholder")
    torch.save({"selected_token_ids": torch.tensor([3, 7, 9])}, checkpoint_dir / "eagle3_meta.pt")

    # Drafter state dict has only one transformer layer.
    drafter_state_dict = {
        "fc.weight": torch.ones(2, 2),
        "layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
        "layers.0.mlp.gate_proj.weight": torch.ones(2, 2),
    }

    def _fake_save_file(state_dict, path: str) -> None:
        Path(path).write_bytes(b"fake-safetensors")

    monkeypatch.setattr(
        "nemo_automodel.components.speculative.serve_sglang._load_safetensors_save_file",
        lambda: _fake_save_file,
    )
    monkeypatch.setattr(
        "nemo_automodel.components.speculative.serve_sglang._torch_load",
        lambda path: (
            drafter_state_dict if path.name == "draft_model.pt" else {"selected_token_ids": torch.tensor([3, 7, 9])}
        ),
    )

    resolve_draft_artifacts(str(checkpoint_dir), "EAGLE3")

    exported_config = json.loads((checkpoint_dir / "model" / "config.json").read_text(encoding="utf-8"))
    assert exported_config["num_hidden_layers"] == 1, (
        "exported config must use the drafter's layer count (1), not the target model's (32)"
    )
    assert exported_config["architectures"] == ["LlamaForCausalLMEagle3"]


def test_build_sglang_argv_exports_recipe_checkpoint_and_passes_token_map(
    tmp_path: Path,
    monkeypatch,
):
    checkpoint_dir = tmp_path / "epoch_0_step_1000"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"architectures": ["LlamaEagle3DraftModel"]}), encoding="utf-8"
    )
    (checkpoint_dir / "draft_model.pt").write_bytes(b"placeholder")
    torch.save({"selected_token_ids": torch.tensor([3, 7, 9])}, checkpoint_dir / "eagle3_meta.pt")

    saved = {}

    def _fake_save_file(state_dict, path: str) -> None:
        saved["state_dict"] = state_dict
        Path(path).write_bytes(b"fake-safetensors")

    monkeypatch.setattr(
        "nemo_automodel.components.speculative.serve_sglang._load_safetensors_save_file",
        lambda: _fake_save_file,
    )
    monkeypatch.setattr(
        "nemo_automodel.components.speculative.serve_sglang._torch_load",
        lambda path: (
            {"lm_head.weight": torch.ones(2, 2)}
            if path.name == "draft_model.pt"
            else {"selected_token_ids": torch.tensor([3, 7, 9])}
        ),
    )

    argv = build_sglang_argv(_build_args(str(checkpoint_dir)))

    model_dir = checkpoint_dir / "model"
    token_map_path = model_dir / "speculative_token_map.pt"
    assert (model_dir / "model.safetensors").exists()
    assert (model_dir / "config.json").exists()
    assert token_map_path.exists()
    assert list(saved["state_dict"]) == ["lm_head.weight"]
    torch.testing.assert_close(saved["state_dict"]["lm_head.weight"], torch.ones(2, 2))
    assert "--speculative-draft-model-path" in argv
    assert argv[argv.index("--speculative-draft-model-path") + 1] == str(model_dir)
    assert "--speculative-token-map" in argv
    assert argv[argv.index("--speculative-token-map") + 1] == str(token_map_path)
