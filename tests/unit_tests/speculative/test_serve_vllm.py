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

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from nemo_automodel.components.speculative import serve_vllm
from nemo_automodel.components.speculative.serve_vllm import (
    build_vllm_argv,
    main,
    resolve_draft_artifacts,
)


def _build_args(
    draft: str,
    *,
    method: str = "eagle3",
    num_speculative_tokens: int | None = None,
    print_only: bool = False,
    trust_remote_code: bool = False,
    max_model_len: int | None = None,
    extra: list[str] | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        target="Qwen/Qwen3-8B",
        draft=draft,
        method=method,
        num_speculative_tokens=num_speculative_tokens,
        host="0.0.0.0",
        port=8000,
        tp_size=1,
        draft_tp_size=1,
        gpu_memory_utilization=0.8,
        dtype="bfloat16",
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
        print_only=print_only,
        extra=extra or [],
    )


def _write_draft_checkpoint(
    model_dir: Path,
    *,
    architectures: list[str],
    parallel: bool = True,
    prefixed: bool = True,
) -> Path:
    """Write a tiny real HF-style draft checkpoint and return the config path.

    ``prefixed=True`` mirrors the Automodel ``self.model`` wrapper (weights named
    ``model.*``); ``prefixed=False`` mirrors an already-vLLM-standard draft.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    pre = "model." if prefixed else ""
    tensors = {
        f"{pre}embed_tokens.weight": torch.zeros(4, 2),
        f"{pre}layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
        f"{pre}norm.weight": torch.zeros(2),
        "lm_head.weight": torch.zeros(4, 2),
        "d2t": torch.zeros(4, dtype=torch.long),
        "t2d": torch.zeros(4, dtype=torch.bool),
    }
    if parallel:
        tensors["mask_hidden"] = torch.zeros(1, 1, 6)
    save_file(tensors, str(model_dir / "model.safetensors"), metadata={"format": "pt"})

    config = {"architectures": architectures, "model_type": "qwen3", "draft_vocab_size": 4}
    if parallel:
        config.update({"parallel_drafting": True, "mask_token_id": 151669, "num_depths": 8})
    config_path = model_dir / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    # A tokenizer-ish asset that the export should carry along.
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return config_path


def _safetensors_keys(path: Path) -> set[str]:
    with safe_open(str(path), framework="pt") as f:
        return set(f.keys())


def _speculative_config_from_argv(argv: list[str]) -> dict:
    return json.loads(argv[argv.index("--speculative-config") + 1])


# --------------------------------------------------------------------------- #
# Automodel draft (model.* prefixed weights) -> remapped vllm_export/ dir.
# --------------------------------------------------------------------------- #
def test_resolve_exports_remapped_weights(tmp_path: Path):
    """An Automodel draft is exported with the ``model.`` prefix stripped and config fixed."""
    model_dir = tmp_path / "epoch_0_step_2000" / "model" / "consolidated"
    src_config = _write_draft_checkpoint(model_dir, architectures=["LlamaEagle3DraftModel"])

    draft_path, config = resolve_draft_artifacts(str(tmp_path / "epoch_0_step_2000"))

    export_dir = model_dir / "vllm_export"
    assert draft_path == str(export_dir)
    # Weight keys are de-prefixed; top-level tensors are untouched.
    assert _safetensors_keys(export_dir / "model.safetensors") == {
        "embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight",
        "norm.weight",
        "lm_head.weight",
        "d2t",
        "t2d",
        "mask_hidden",
    }
    # Exported config carries both fixups.
    exported = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
    assert exported["architectures"] == ["LlamaForCausalLMEagle3"]
    assert exported["pard_token"] == 151669
    # Tokenizer asset travels with the export.
    assert (export_dir / "tokenizer_config.json").exists()
    # The source checkpoint is left untouched (non-destructive).
    assert json.loads(src_config.read_text(encoding="utf-8"))["architectures"] == ["LlamaEagle3DraftModel"]
    assert config["parallel_drafting"] is True


def test_resolve_reads_keys_and_remaps_index(tmp_path: Path):
    """A sharded draft: keys are read from the index and the exported index is de-prefixed."""
    model_dir = tmp_path / "model"
    _write_draft_checkpoint(model_dir, architectures=["LlamaEagle3DraftModel"])
    # Add a safetensors index so resolution reads keys from it (the real sharded layout).
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {
            "model.embed_tokens.weight": "model.safetensors",
            "lm_head.weight": "model.safetensors",
            "d2t": "model.safetensors",
        },
    }
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    draft_path, _ = resolve_draft_artifacts(str(model_dir))

    exported_index = json.loads((Path(draft_path) / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert set(exported_index["weight_map"]) == {"embed_tokens.weight", "lm_head.weight", "d2t"}


def test_export_is_cached_until_source_changes(tmp_path: Path):
    """A second resolve reuses the export; touching the source weights rebuilds it."""
    model_dir = tmp_path / "model"
    _write_draft_checkpoint(model_dir, architectures=["LlamaEagle3DraftModel"])
    export_config = model_dir / "vllm_export" / "config.json"

    resolve_draft_artifacts(str(model_dir))
    first_mtime = export_config.stat().st_mtime_ns
    resolve_draft_artifacts(str(model_dir))
    assert export_config.stat().st_mtime_ns == first_mtime, "fresh export must be reused, not rebuilt"


def test_resolve_dry_run_returns_export_path_without_writing(tmp_path: Path):
    """dry_run returns the would-be export path but creates nothing and touches nothing."""
    model_dir = tmp_path / "model"
    src_config = _write_draft_checkpoint(model_dir, architectures=["LlamaEagle3DraftModel"])
    content_before = src_config.read_text(encoding="utf-8")

    draft_path, _ = resolve_draft_artifacts(str(model_dir), dry_run=True)

    assert draft_path == str(model_dir / "vllm_export")
    assert not (model_dir / "vllm_export").exists()
    assert src_config.read_text(encoding="utf-8") == content_before


def test_resolve_peagle_without_mask_token_raises_before_writing(tmp_path: Path):
    """parallel_drafting with no mask_token_id fails, and no partial export is left behind."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)
    save_file(
        {"model.embed_tokens.weight": torch.zeros(4, 2), "lm_head.weight": torch.zeros(4, 2)},
        str(model_dir / "model.safetensors"),
        metadata={"format": "pt"},
    )
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["LlamaEagle3DraftModel"], "parallel_drafting": True}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no mask_token_id"):
        resolve_draft_artifacts(str(model_dir))
    assert not (model_dir / "vllm_export" / "model.safetensors").exists()


# --------------------------------------------------------------------------- #
# vLLM-standard draft (no model. prefix) -> in-place config rewrite, no export.
# --------------------------------------------------------------------------- #
def test_resolve_vllm_standard_draft_rewrites_in_place(tmp_path: Path):
    """A draft whose weights are already vLLM-standard only gets the config fixups."""
    model_dir = tmp_path / "model"
    config_path = _write_draft_checkpoint(model_dir, architectures=["LlamaEagle3DraftModel"], prefixed=False)

    draft_path, _ = resolve_draft_artifacts(str(model_dir))

    assert draft_path == str(model_dir)
    assert not (model_dir / "vllm_export").exists()
    rewritten = json.loads(config_path.read_text(encoding="utf-8"))
    assert rewritten["architectures"] == ["LlamaForCausalLMEagle3"]
    assert rewritten["pard_token"] == 151669


def test_resolve_vllm_standard_already_correct_is_untouched(tmp_path: Path):
    """A vLLM-standard config already carrying arch + pard_token is not rewritten."""
    model_dir = tmp_path / "model"
    config_path = _write_draft_checkpoint(model_dir, architectures=["LlamaForCausalLMEagle3"], prefixed=False)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["pard_token"] = config["mask_token_id"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    mtime_before = config_path.stat().st_mtime_ns

    resolve_draft_artifacts(str(model_dir))

    assert config_path.stat().st_mtime_ns == mtime_before


def test_resolve_passes_through_hf_repo_id(tmp_path: Path):
    """A non-existent path is treated as a Hub repo id and passed through with an empty config."""
    draft_path, config = resolve_draft_artifacts("khazic/peagle-qwen3-8b")

    assert draft_path == "khazic/peagle-qwen3-8b"
    assert config == {}


def test_resolve_existing_dir_without_weights_raises(tmp_path: Path):
    """A real directory with no config/weights is a user error, not a Hub id."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")  # config but no weights

    with pytest.raises(ValueError, match="no config.json \\+ weights"):
        resolve_draft_artifacts(str(tmp_path))


# --------------------------------------------------------------------------- #
# build_vllm_argv / speculative-config.
# --------------------------------------------------------------------------- #
def test_build_argv_peagle_sets_parallel_drafting_and_points_at_export(tmp_path: Path):
    """A P-EAGLE head: speculative-config carries parallel_drafting, K=num_depths, export path."""
    model_dir = tmp_path / "model"
    _write_draft_checkpoint(model_dir, architectures=["LlamaEagle3DraftModel"])

    argv = build_vllm_argv(_build_args(str(model_dir)))

    spec = _speculative_config_from_argv(argv)
    assert spec["method"] == "eagle3"
    assert spec["model"] == str(model_dir / "vllm_export")
    assert spec["num_speculative_tokens"] == 8  # derived from num_depths
    assert spec["parallel_drafting"] is True
    assert spec["draft_tensor_parallel_size"] == 1
    assert argv[argv.index("--model") + 1] == "Qwen/Qwen3-8B"
    assert "vllm.entrypoints.openai.api_server" in argv


def test_build_argv_plain_eagle3_omits_parallel_drafting(tmp_path: Path):
    """A non-parallel EAGLE-3 head: no parallel_drafting key, K must be given explicitly."""
    model_dir = tmp_path / "model"
    _write_draft_checkpoint(model_dir, architectures=["LlamaEagle3DraftModel"], parallel=False)

    argv = build_vllm_argv(_build_args(str(model_dir), num_speculative_tokens=3))

    spec = _speculative_config_from_argv(argv)
    assert spec["num_speculative_tokens"] == 3
    assert "parallel_drafting" not in spec


def test_build_argv_plain_eagle3_without_k_raises(tmp_path: Path):
    """A non-parallel head with no num_depths and no CLI K is a hard error."""
    model_dir = tmp_path / "model"
    _write_draft_checkpoint(model_dir, architectures=["LlamaForCausalLMEagle3"], parallel=False, prefixed=False)

    with pytest.raises(ValueError, match="num-speculative-tokens is required"):
        build_vllm_argv(_build_args(str(model_dir)))


def test_build_argv_passes_through_optional_flags(tmp_path: Path):
    """max-model-len, trust-remote-code and trailing extras are forwarded."""
    model_dir = tmp_path / "model"
    _write_draft_checkpoint(model_dir, architectures=["LlamaEagle3DraftModel"])

    argv = build_vllm_argv(
        _build_args(
            str(model_dir),
            trust_remote_code=True,
            max_model_len=4096,
            extra=["--enable-chunked-prefill"],
        )
    )

    assert argv[argv.index("--max-model-len") + 1] == "4096"
    assert "--trust-remote-code" in argv
    assert "--enable-chunked-prefill" in argv


def test_parse_args_strips_leading_double_dash():
    """A leading ``--`` separator before extras is stripped."""
    args = serve_vllm._parse_args(["--target", "Qwen/Qwen3-8B", "--draft", "/tmp/d", "--", "--enable-chunked-prefill"])
    assert args.extra == ["--enable-chunked-prefill"]


# --------------------------------------------------------------------------- #
# main() dispatch.
# --------------------------------------------------------------------------- #
def test_main_print_only_prints_command(tmp_path: Path, capsys):
    """--print-only prints the resolved command, exits 0, and writes nothing."""
    model_dir = tmp_path / "model"
    _write_draft_checkpoint(model_dir, architectures=["LlamaEagle3DraftModel"])

    rc = main(["--target", "Qwen/Qwen3-8B", "--draft", str(model_dir), "--print-only"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "vllm.entrypoints.openai.api_server" in out
    assert not (model_dir / "vllm_export").exists()


def test_main_exits_when_vllm_missing(tmp_path: Path, monkeypatch):
    """Without --print-only a missing vLLM exits with code 2 before launching."""
    model_dir = tmp_path / "model"
    _write_draft_checkpoint(model_dir, architectures=["LlamaEagle3DraftModel"])
    monkeypatch.setattr(serve_vllm, "safe_import", lambda name: (False, None))

    with pytest.raises(SystemExit) as exc:
        main(["--target", "Qwen/Qwen3-8B", "--draft", str(model_dir)])
    assert exc.value.code == 2


def test_main_launches_via_subprocess(tmp_path: Path, monkeypatch):
    """When vLLM is present and not print-only, the built argv is dispatched."""
    model_dir = tmp_path / "model"
    _write_draft_checkpoint(model_dir, architectures=["LlamaEagle3DraftModel"])
    monkeypatch.setattr(serve_vllm, "safe_import", lambda name: (True, object()))
    monkeypatch.delattr(serve_vllm.os, "execv", raising=False)
    captured = {}

    def _fake_call(cmd):
        captured["cmd"] = cmd
        return 0

    monkeypatch.setattr(serve_vllm.subprocess, "call", _fake_call)

    rc = main(["--target", "Qwen/Qwen3-8B", "--draft", str(model_dir)])

    assert rc == 0
    assert "vllm.entrypoints.openai.api_server" in captured["cmd"]


def test_main_execv_path_replaces_process(tmp_path: Path, monkeypatch):
    """With os.execv available the command is dispatched via execv (process replacement)."""
    model_dir = tmp_path / "model"
    _write_draft_checkpoint(model_dir, architectures=["LlamaEagle3DraftModel"])
    monkeypatch.setattr(serve_vllm, "safe_import", lambda name: (True, object()))
    captured = {}

    class _Execv(Exception):
        pass

    def _fake_execv(path, cmd):
        captured["path"] = path
        captured["cmd"] = cmd
        raise _Execv  # execv never returns; raise to hand control back to the test

    monkeypatch.setattr(serve_vllm.os, "execv", _fake_execv)

    with pytest.raises(_Execv):
        main(["--target", "Qwen/Qwen3-8B", "--draft", str(model_dir)])

    assert captured["path"] == captured["cmd"][0]
    assert "vllm.entrypoints.openai.api_server" in captured["cmd"]
