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

"""Serve an Automodel-trained EAGLE-3 / P-EAGLE drafter with vLLM.

This is the vLLM companion to ``serve_sglang``. It exists because the
parallel-drafting (P-EAGLE) head produced by the EAGLE-3 recipe with
``parallel_drafting: true`` is *not* servable by SGLang today (``serve_sglang``
rejects it); its inference runtime is vLLM's parallel-drafting path
(vLLM >= 0.16, https://github.com/vllm-project/speculators/pull/480). The same
entry point also serves a plain (non-parallel) EAGLE-3 draft under vLLM.

The EAGLE drafter checkpoints produced by the recipes are written by the
consolidated checkpointer as an HF-style ``model/`` directory
(``model.safetensors`` + ``config.json``), optionally nested one level deeper in
a ``consolidated/`` subdir. The draft already stores the ``d2t`` / ``t2d``
vocab-remap buffers inside its weights in the exact offset form vLLM expects
(``target_id = draft_id + d2t[draft_id]``), so -- unlike the SGLang path -- no
separate speculative-token-map file is emitted.

Three fixups bridge the Automodel on-disk format to what vLLM loads:

* ``architectures``: the recipe writes ``["LlamaEagle3DraftModel"]`` (the
  Automodel class name), which is not in vLLM's model registry. vLLM routes
  every EAGLE-3 draft on ``LlamaForCausalLMEagle3`` (``PEagleDraftModel`` /
  ``Eagle3LlamaForCausalLM`` are registered aliases), so the field is rewritten.
* ``pard_token`` (P-EAGLE only): vLLM's parallel-drafting proposer reads the
  masked-slot token id from ``pard_token`` / ``ptd_token_id`` /
  ``dflash_config.mask_token_id``, but the recipe only writes the top-level
  ``mask_token_id``. The value is copied to ``pard_token`` (mirroring vLLM's own
  speculators ``update_peagle`` mapping).
* weight keys: Automodel wraps the draft as ``self.model``, so its weights are
  saved with a leading ``model.`` (``model.embed_tokens.weight``). vLLM's loader
  re-adds ``model.`` to every non-top-level weight, so the prefix is stripped
  during a one-time export into a ``vllm_export/`` subdir (the source checkpoint
  is left untouched). A draft whose weights are already vLLM-standard skips the
  export and only gets the config fixups in place.

The script then shells out to ``python -m vllm.entrypoints.openai.api_server``
with the right ``--speculative-config``.

NOTE -- vLLM is NOT bundled with the NeMo-AutoModel container image and is
intentionally NOT declared in ``pyproject.toml``. To use this entry point,
install it yourself into the same environment:

    uv pip install "vllm>=0.16"

Refer to https://github.com/vllm-project/vllm for the version matching your
CUDA / PyTorch stack. If vLLM is missing this script exits with a clear install
hint rather than crashing on import.

Typical usage (after training produces a P-EAGLE checkpoint at
``./outputs/peagle/checkpoints/epoch_2_step_44326``):

    python -m nemo_automodel.components.speculative.serve_vllm \\
        --target Qwen/Qwen3-8B \\
        --draft ./outputs/peagle/checkpoints/epoch_2_step_44326 \\
        --num-speculative-tokens 8

``--num-speculative-tokens`` defaults to the draft config's ``num_depths`` (K),
so for a P-EAGLE head it can be omitted. Pass ``--print-only`` to inspect the
command without launching it; in that mode the on-disk ``architectures`` rewrite
is skipped and the printed paths reflect what a real launch would produce.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from nemo_automodel.shared.import_utils import safe_import, safe_import_from

logger = logging.getLogger(__name__)

_VLLM_INSTALL_HINT = (
    "vllm is not installed in this environment. Install it manually with "
    '`uv pip install "vllm>=0.16"` (the parallel-drafting / P-EAGLE runtime landed in 0.16; '
    "see https://github.com/vllm-project/vllm for CUDA / PyTorch compatibility) and re-run "
    "this script. vLLM is intentionally not bundled with the NeMo-AutoModel container."
)
_SAFETENSORS_INSTALL_HINT = (
    "safetensors is required to remap an Automodel drafter checkpoint for vLLM. "
    "Install it with `uv pip install safetensors` and re-run this script."
)

# Automodel wraps the draft as ``self.model = LlamaModel(...)`` so its weights are
# saved with a leading ``model.`` (e.g. ``model.embed_tokens.weight``). vLLM's
# ``Eagle3LlamaForCausalLM`` loader re-adds ``model.`` to every non-top-level
# weight, so an Automodel draft becomes ``model.model.embed_tokens.weight`` and
# fails to load. We strip the wrapper prefix during export so vLLM resolves it.
_MODEL_PREFIX = "model."
_VLLM_EXPORT_DIRNAME = "vllm_export"

# The EAGLE recipes write ``architectures=["LlamaEagle3DraftModel"]`` (the
# Automodel class name) into the drafter config, but vLLM's model registry
# routes every EAGLE-3 draft on ``LlamaForCausalLMEagle3`` (``PEagleDraftModel``
# and ``Eagle3LlamaForCausalLM`` are registered aliases). We rewrite during
# resolution so the drafter directory is consumable by vLLM unchanged.
_AUTOMODEL_DRAFT_ARCHITECTURE = "LlamaEagle3DraftModel"
_VLLM_DRAFT_ARCHITECTURE = "LlamaForCausalLMEagle3"


def _check_vllm_available() -> None:
    """Verify the ``vllm`` package can actually be imported, else exit (code 2)."""
    ok, _ = safe_import("vllm")
    if not ok:
        logger.error(_VLLM_INSTALL_HINT)
        raise SystemExit(2)


def _has_hf_weight_file(path: Path) -> bool:
    """Return True if ``path`` contains an HF-style weight artifact."""
    return any(
        (path / name).exists() for name in ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")
    )


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load a drafter ``config.json`` into a dict."""
    with config_path.open("r") as f:
        return json.load(f)


def _vllm_config_updates(config: dict[str, Any]) -> dict[str, Any]:
    """Return the ``config.json`` key changes needed for vLLM to load this draft.

    Two fixups bridge the Automodel on-disk format to what vLLM reads:

    * ``architectures`` -> the vLLM-canonical ``LlamaForCausalLMEagle3`` (the
      recipe writes the Automodel class name ``LlamaEagle3DraftModel``, which
      vLLM's registry does not know).
    * for a P-EAGLE head, ``pard_token`` (= ``mask_token_id``). This mirrors
      vLLM's own speculators ``update_peagle`` mapping; the parallel-drafting
      proposer reads ``pard_token`` / ``ptd_token_id`` /
      ``dflash_config.mask_token_id``, none of which the recipe writes -- it
      only writes the top-level ``mask_token_id``.

    Returns an empty dict when the config already satisfies vLLM.
    """
    updates: dict[str, Any] = {}
    if config.get("architectures") != [_VLLM_DRAFT_ARCHITECTURE]:
        updates["architectures"] = [_VLLM_DRAFT_ARCHITECTURE]
    if config.get("parallel_drafting"):
        has_mask_key = (
            "pard_token" in config or "ptd_token_id" in config or "mask_token_id" in (config.get("dflash_config") or {})
        )
        if not has_mask_key:
            mask_token_id = config.get("mask_token_id")
            if mask_token_id is None:
                raise ValueError(
                    "draft config has parallel_drafting=true but no mask_token_id, so the "
                    "pard_token that vLLM's parallel-drafting proposer requires cannot be derived."
                )
            updates["pard_token"] = mask_token_id
    return updates


def _rewrite_config_for_vllm(config_path: Path, config: dict[str, Any]) -> None:
    """Patch ``config_path`` in place with the keys vLLM needs (architectures, pard_token).

    Takes the already-parsed ``config`` so the caller's read is not repeated.
    No-op when the config already satisfies vLLM. The write is staged through a
    sibling ``.tmp`` file and finalized with ``os.replace`` so an interrupted
    write cannot leave the destination half-truncated.
    """
    updates = _vllm_config_updates(config)
    if not updates:
        return
    logger.info("Patching draft config for vLLM: %s", updates)
    tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump({**config, **updates}, f, indent=2)
    os.replace(tmp_path, config_path)


def _draft_weight_keys(draft_dir: Path) -> list[str]:
    """Return the weight tensor names in ``draft_dir``.

    Reads the names from ``model.safetensors.index.json`` when sharded, otherwise
    from the single ``model.safetensors`` header (no tensor data is loaded).
    Returns an empty list when neither is present / readable.
    """
    index_path = draft_dir / "model.safetensors.index.json"
    if index_path.exists():
        return list(_load_config(index_path).get("weight_map", {}).keys())
    single = draft_dir / "model.safetensors"
    if single.exists():
        ok, safe_open = safe_import_from("safetensors", "safe_open")
        if ok:
            with safe_open(str(single), framework="pt") as f:
                return list(f.keys())
    return []


def _needs_weight_remap(weight_keys: list[str]) -> bool:
    """True when any weight carries the Automodel ``self.model`` wrapper prefix."""
    return any(k.startswith(_MODEL_PREFIX) for k in weight_keys)


def _remap_key_for_vllm(key: str) -> str:
    """Strip the ``self.model`` wrapper prefix so vLLM's loader (which re-adds it) resolves it.

    Top-level tensors (``lm_head.weight``, ``d2t``, ``t2d``, ``mask_hidden``) do
    not carry the prefix and are returned unchanged.
    """
    return key[len(_MODEL_PREFIX) :] if key.startswith(_MODEL_PREFIX) else key


def _load_safetensors_io():
    """Return ``(load_file, save_file)`` from ``safetensors.torch`` or exit with a hint."""
    ok_load, load_file = safe_import_from("safetensors.torch", "load_file")
    ok_save, save_file = safe_import_from("safetensors.torch", "save_file")
    if not (ok_load and ok_save):
        logger.error(_SAFETENSORS_INSTALL_HINT)
        raise SystemExit(2)
    return load_file, save_file


def _export_is_fresh(draft_dir: Path, export_dir: Path) -> bool:
    """True when ``export_dir`` holds a complete config + weights newer than the source weights or config."""
    exported_config = export_dir / "config.json"
    if not exported_config.exists() or not _has_hf_weight_file(export_dir):
        return False
    src_files = [
        *draft_dir.glob("model-*.safetensors"),
        draft_dir / "model.safetensors",
        draft_dir / "config.json",
    ]
    src_mtime = max((f.stat().st_mtime_ns for f in src_files if f.exists()), default=0)
    return exported_config.stat().st_mtime_ns >= src_mtime


def _export_for_vllm(draft_dir: Path, config: dict[str, Any]) -> Path:
    """Materialize a vLLM-loadable copy of ``draft_dir`` in a ``vllm_export/`` subdir.

    The safetensors keys are remapped to strip the Automodel ``self.model``
    wrapper prefix, the config receives the architectures / pard_token fixups, and
    the tokenizer assets are copied alongside. The source checkpoint is left
    untouched. The export is cached and rebuilt only when stale.
    """
    export_dir = draft_dir / _VLLM_EXPORT_DIRNAME
    if _export_is_fresh(draft_dir, export_dir):
        logger.info("Reusing fresh vLLM export at %s", export_dir)
        return export_dir

    # Validate the config fixups up front so a bad draft fails before any writes.
    config_updates = _vllm_config_updates(config)
    load_file, save_file = _load_safetensors_io()
    export_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(draft_dir.glob("model-*.safetensors")) or [draft_dir / "model.safetensors"]
    for shard in shards:
        if not shard.exists():
            continue
        remapped = {_remap_key_for_vllm(k): v for k, v in load_file(str(shard)).items()}
        save_file(remapped, str(export_dir / shard.name), metadata={"format": "pt"})

    index_path = draft_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = _load_config(index_path)
        index["weight_map"] = {_remap_key_for_vllm(k): v for k, v in index["weight_map"].items()}
        with (export_dir / "model.safetensors.index.json").open("w") as f:
            json.dump(index, f, indent=2)

    exported_config = dict(config)
    exported_config.update(config_updates)
    with (export_dir / "config.json").open("w") as f:
        json.dump(exported_config, f, indent=2)

    # Copy tokenizer / chat-template assets so vLLM can load the draft standalone.
    for asset in draft_dir.iterdir():
        if (
            asset.is_file()
            and asset.suffix in (".json", ".jinja", ".txt", ".model")
            and not asset.name.startswith("model-")
            and asset.name not in ("config.json", "model.safetensors.index.json")
        ):
            shutil.copy2(asset, export_dir / asset.name)

    logger.info("Exported vLLM-compatible draft to %s", export_dir)
    return export_dir


def _find_draft_dir(draft_path: Path) -> Path | None:
    """Return the HF-style drafter directory under ``draft_path``, if present.

    Accepts the outer ``epoch_<E>_step_<S>`` directory, its ``model/`` subdir, or
    the consolidated checkpointer's ``model/consolidated/`` (and a bare
    ``consolidated/``) layout, as well as a directory that is already the HF
    model dir. Returns the first candidate holding ``config.json`` + a weight
    file, else ``None``.
    """
    candidates = [
        draft_path,
        draft_path / "model",
        draft_path / "model" / "consolidated",
        draft_path / "consolidated",
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "config.json").exists() and _has_hf_weight_file(candidate):
            return candidate
    return None


def resolve_draft_artifacts(draft: str, *, dry_run: bool = False) -> tuple[str, dict[str, Any]]:
    """Resolve a user-supplied drafter path to the directory and config vLLM expects.

    Args:
        draft: A local path to a recipe checkpoint (the ``epoch_*`` dir, its
            ``model/`` / ``model/consolidated/`` subdir, or the HF model dir
            itself) or a Hugging Face Hub repo id.
        dry_run: When True, no on-disk ``architectures`` rewrite is performed and
            the returned paths reflect what a real launch would produce.

    Returns:
        ``(draft_path, config)`` where ``draft_path`` is what vLLM's
        ``--speculative-config`` ``model`` field should point at and ``config``
        is the parsed drafter ``config.json`` (empty for a passed-through Hub
        repo id, which cannot be introspected locally).
    """
    p = Path(draft)
    if not p.exists():
        # Treat a non-existent path as a Hugging Face Hub repo id and pass it
        # through untouched; vLLM resolves and downloads it at launch.
        return draft, {}

    draft_dir = _find_draft_dir(p)
    if draft_dir is None:
        raise ValueError(
            f"--draft {draft!r} exists but no config.json + weights were found under it "
            "(looked in ./, model/, model/consolidated/, consolidated/). Point --draft at the "
            "checkpoint directory produced by an EAGLE-3 / P-EAGLE recipe."
        )

    config_path = draft_dir / "config.json"
    config = _load_config(config_path)

    # Automodel drafts carry a ``self.model`` wrapper prefix on their weights that
    # vLLM cannot load; those need a remapped export. A draft whose weights are
    # already vLLM-standard only needs the in-place config fixups.
    if _needs_weight_remap(_draft_weight_keys(draft_dir)):
        export_dir = draft_dir / _VLLM_EXPORT_DIRNAME
        if dry_run:
            return str(export_dir), config
        return str(_export_for_vllm(draft_dir, config)), config

    if not dry_run:
        _rewrite_config_for_vllm(config_path, config)
    return str(draft_dir), config


def _resolve_num_speculative_tokens(cli_value: int | None, config: dict[str, Any]) -> int:
    """Resolve K (number of speculative tokens), defaulting to the draft's ``num_depths``."""
    if cli_value is not None:
        return cli_value
    num_depths = config.get("num_depths")
    if num_depths is None:
        raise ValueError(
            "--num-speculative-tokens is required: the draft config has no ``num_depths`` to "
            "derive it from (only P-EAGLE / parallel-drafting heads carry it). Pass the value "
            "explicitly, e.g. --num-speculative-tokens 4."
        )
    return int(num_depths)


def _build_speculative_config(args: argparse.Namespace, draft_path: str, config: dict[str, Any]) -> dict[str, Any]:
    """Build the vLLM ``--speculative-config`` dict for a resolved draft.

    ``parallel_drafting`` is a SpeculativeConfig field that vLLM defaults to
    False; it must be set here for a P-EAGLE head (the draft's own
    ``config.json`` flag is not auto-promoted to the speculative config).
    """
    speculative_config: dict[str, Any] = {
        "method": args.method,
        "model": draft_path,
        "num_speculative_tokens": _resolve_num_speculative_tokens(args.num_speculative_tokens, config),
        "draft_tensor_parallel_size": args.draft_tp_size,
    }
    if config.get("parallel_drafting"):
        speculative_config["parallel_drafting"] = True
    return speculative_config


def build_vllm_argv(args: argparse.Namespace) -> list[str]:
    """Build the ``python -m vllm.entrypoints.openai.api_server`` argv for a config."""
    draft_path, config = resolve_draft_artifacts(args.draft, dry_run=args.print_only)
    speculative_config = _build_speculative_config(args, draft_path, config)

    argv: list[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.target,
        "--speculative-config",
        json.dumps(speculative_config),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tp_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--dtype",
        args.dtype,
    ]
    if args.max_model_len is not None:
        argv += ["--max-model-len", str(args.max_model_len)]
    if args.trust_remote_code:
        argv.append("--trust-remote-code")
    if args.extra:
        argv += list(args.extra)
    return argv


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the vLLM serve helper."""
    parser = argparse.ArgumentParser(
        prog="serve_vllm",
        description=(
            "Launch vLLM with an Automodel-trained EAGLE-3 / P-EAGLE drafter. "
            "Requires `uv pip install vllm` in the current environment; vLLM is not "
            "bundled with the Automodel container."
        ),
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target (base) model path or HuggingFace repo id, e.g. Qwen/Qwen3-8B.",
    )
    parser.add_argument(
        "--draft",
        required=True,
        help=(
            "Path to the drafter checkpoint directory produced by an EAGLE-3 / P-EAGLE recipe "
            "(e.g. outputs/peagle/checkpoints/epoch_2_step_44326). The ``model/`` and "
            "``model/consolidated/`` subdirs are auto-selected."
        ),
    )
    parser.add_argument(
        "--method",
        default="eagle3",
        choices=["eagle", "eagle3"],
        help="Speculative method passed to vLLM (P-EAGLE uses eagle3 + parallel_drafting).",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=None,
        help="K, number of speculative tokens. Defaults to the draft config's num_depths.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor-parallel size for the target model.")
    parser.add_argument(
        "--draft-tp-size",
        type=int,
        default=1,
        help="--speculative-config draft_tensor_parallel_size (draft model TP).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="--gpu-memory-utilization passed through to vLLM.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Inference dtype (must match the dtype used during EAGLE training).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="--max-model-len passed through to vLLM (default: model config).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward --trust-remote-code to vLLM (needed for custom target architectures).",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help=(
            "Print the resolved vLLM command and exit without launching it. "
            "Skips the on-disk architectures rewrite; the printed paths reflect "
            "what would be produced on a real launch."
        ),
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded verbatim to vLLM (prefix with `--`).",
    )
    parsed = parser.parse_args(argv)
    if parsed.extra and parsed.extra[0] == "--":
        parsed.extra = parsed.extra[1:]
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Validate the environment, resolve the drafter ckpt, then exec vLLM.

    Returns the vLLM server's exit code, or ``2`` if vLLM is missing.
    """
    logging.basicConfig(
        level=os.environ.get("NEMO_AUTOMODEL_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args(argv)
    if not args.print_only:
        # Fail fast before any expensive config resolution / rewrite work.
        _check_vllm_available()
    cmd = build_vllm_argv(args)
    logger.info("vLLM command: %s", " ".join(cmd))
    if args.print_only:
        print(" ".join(cmd))
        return 0
    if hasattr(os, "execv") and Path(cmd[0]).is_absolute() and Path(cmd[0]).is_file():
        os.execv(cmd[0], cmd)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
