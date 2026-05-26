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

"""Serve an Automodel-trained EAGLE / EAGLE-3 drafter with SGLang.

The EAGLE drafter checkpoints produced by the EAGLE recipes
(``recipes/llm/train_eagle{1,2,3}.py``) are saved as ``draft_model.pt`` plus
recipe metadata. This script converts that layout into an HF/SGLang-readable
``model/`` directory when needed, then shells out to
``python -m sglang.launch_server`` with the right speculative-decoding flags.

NOTE — SGLang is NOT bundled with the NeMo-AutoModel container image and
is intentionally NOT declared in ``pyproject.toml``. To use this entry
point, install it yourself into the same environment:

    uv pip install "sglang>=0.5.9"

Refer to https://github.com/sgl-project/sglang for the version matching
your CUDA / PyTorch stack. If SGLang is missing this script exits with a
clear install hint rather than crashing on import.

Typical usage (after training produces a checkpoint at
``./checkpoints/epoch_0_step_1000``):

    python -m nemo_automodel.components.speculative.serve_sglang \\
        --target meta-llama/Llama-3.1-8B-Instruct \\
        --draft ./checkpoints/epoch_0_step_1000 \\
        --algorithm EAGLE3 \\
        --num-steps 3 --topk 1 --num-draft-tokens 4

Pass ``--print-only`` to inspect the command without launching it; in that
mode no checkpoint export is performed and the printed paths reflect what
would be produced on a real launch.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import torch

from nemo_automodel.shared.import_utils import safe_import, safe_import_from

logger = logging.getLogger(__name__)

_SGLANG_INSTALL_HINT = (
    "sglang is not installed in this environment. Install it manually with "
    '`uv pip install "sglang>=0.5.9"` (see https://github.com/sgl-project/sglang '
    "for CUDA / PyTorch compatibility) and re-run this script. SGLang is "
    "intentionally not bundled with the NeMo-AutoModel container."
)
_SAFETENSORS_INSTALL_HINT = (
    "safetensors is required to export Automodel EAGLE checkpoints for SGLang. "
    "Install it with `uv pip install safetensors` and re-run this script."
)

# The EAGLE recipes write ``architectures=["LlamaEagle3DraftModel"]`` (the
# Automodel class name) into the drafter config, but SGLang's model registry
# routes on its own canonical class names. We rewrite during export so the
# exported ``model/`` directory is consumable by SGLang (and any other HF
# tooling) without depending on the ``--speculative-algorithm`` flag to
# override architecture dispatch.
_SGLANG_ARCHITECTURE_FOR_ALGORITHM = {
    "EAGLE3": "LlamaForCausalLMEagle3",
}


def _has_hf_weight_file(path: Path) -> bool:
    """Return True if ``path`` already contains a HF-style weight artifact."""
    return any(
        (path / name).exists() for name in ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")
    )


def _check_sglang_available() -> None:
    """Verify the ``sglang`` package can actually be imported, else exit (code 2)."""
    ok, _ = safe_import("sglang")
    if not ok:
        logger.error(_SGLANG_INSTALL_HINT)
        raise SystemExit(2)


def _load_safetensors_save_file() -> Callable[..., None]:
    """Return ``safetensors.torch.save_file`` or exit with an install hint."""
    ok, save_file = safe_import_from("safetensors.torch", "save_file")
    if not ok:
        logger.error(_SAFETENSORS_INSTALL_HINT)
        raise SystemExit(2)
    return save_file


def _torch_load(path: Path) -> Any:
    """Load a torch pickle, preferring ``weights_only=True`` when supported."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _rewrite_config_for_sglang(
    src_config_path: Path,
    dst_config_path: Path,
    algorithm: str,
    *,
    num_hidden_layers: int | None = None,
) -> None:
    """Copy ``src_config_path`` to ``dst_config_path`` and normalize ``architectures``.

    For algorithms in ``_SGLANG_ARCHITECTURE_FOR_ALGORITHM`` the
    ``architectures`` field is rewritten to the SGLang-canonical class name
    (e.g. ``LlamaForCausalLMEagle3``). For other algorithms the original
    field is preserved. When ``num_hidden_layers`` is provided it is written
    into the config so the exported drafter reflects its actual depth rather
    than the target model's depth. The write is staged through a sibling
    ``.tmp`` file and finalized with ``os.replace`` so an interrupted write
    cannot leave the destination half-truncated when rewriting in place.
    """
    with src_config_path.open("r") as f:
        config = json.load(f)
    expected = _SGLANG_ARCHITECTURE_FOR_ALGORITHM.get(algorithm)
    if expected is not None and config.get("architectures") != [expected]:
        logger.info("Rewriting architectures: %s -> %s", config.get("architectures"), [expected])
        config["architectures"] = [expected]
    if num_hidden_layers is not None and config.get("num_hidden_layers") != num_hidden_layers:
        logger.info("Rewriting num_hidden_layers: %s -> %d", config.get("num_hidden_layers"), num_hidden_layers)
        config["num_hidden_layers"] = num_hidden_layers
    tmp_path = dst_config_path.with_suffix(dst_config_path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(config, f, indent=2)
    os.replace(tmp_path, dst_config_path)


def _config_needs_rewrite(config_path: Path, algorithm: str) -> bool:
    """Return True when ``config_path`` does not match the SGLang architecture for ``algorithm``."""
    expected = _SGLANG_ARCHITECTURE_FOR_ALGORITHM.get(algorithm)
    if expected is None or not config_path.exists():
        return False
    with config_path.open("r") as f:
        config = json.load(f)
    return config.get("architectures") != [expected]


def _infer_num_hidden_layers(state_dict: dict[str, Any]) -> int | None:
    """Infer num_hidden_layers from a state dict by counting unique layer indices."""
    indices = {int(m.group(1)) for key in state_dict if (m := re.search(r"\blayers\.(\d+)\b", key))}
    return len(indices) if indices else None


def _regenerate_token_map(meta_path: Path, token_map_path: Path) -> None:
    """Extract ``selected_token_ids`` from a recipe meta file into a SGLang token map."""
    meta = _torch_load(meta_path)
    selected_token_ids = meta.get("selected_token_ids")
    if selected_token_ids is None:
        raise ValueError(f"{meta_path} is missing selected_token_ids required for EAGLE3 SGLang serving.")
    torch.save(selected_token_ids.cpu(), token_map_path)


def _maybe_export_training_checkpoint(
    checkpoint_dir: Path, algorithm: str, *, dry_run: bool = False
) -> tuple[Path, Path | None]:
    """Convert recipe-native EAGLE checkpoints into an HF/SGLang-readable directory.

    Args:
        checkpoint_dir: Recipe checkpoint dir, expected to contain
            ``draft_model.pt`` and ``config.json`` (and ``eagle3_meta.pt`` for
            EAGLE-3).
        algorithm: Speculative algorithm name, used to pick the right
            SGLang architecture and to decide whether a token map is needed.
        dry_run: When True, return the paths that *would* be produced
            without writing anything.

    Returns:
        ``(export_dir, token_map_path_or_None)``.
    """
    draft_model_path = checkpoint_dir / "draft_model.pt"
    config_path = checkpoint_dir / "config.json"
    if not draft_model_path.exists() or not config_path.exists():
        return checkpoint_dir, None

    export_dir = checkpoint_dir / "model"
    exported_weights = export_dir / "model.safetensors"
    exported_config = export_dir / "config.json"
    token_map_path: Path | None = None
    if algorithm == "EAGLE3" and (checkpoint_dir / "eagle3_meta.pt").exists():
        token_map_path = export_dir / "speculative_token_map.pt"

    if dry_run:
        return export_dir, token_map_path

    export_dir.mkdir(parents=True, exist_ok=True)
    state_dict: dict[str, Any] | None = None
    if not exported_weights.exists():
        logger.info("Exporting draft checkpoint %s -> %s", draft_model_path, exported_weights)
        save_file = _load_safetensors_save_file()
        state_dict = _torch_load(draft_model_path)
        save_file(state_dict, str(exported_weights))
    if not exported_config.exists():
        if state_dict is None:
            state_dict = _torch_load(draft_model_path)
        num_hidden_layers = _infer_num_hidden_layers(state_dict)
        _rewrite_config_for_sglang(config_path, exported_config, algorithm, num_hidden_layers=num_hidden_layers)
    if token_map_path is not None and not token_map_path.exists():
        _regenerate_token_map(checkpoint_dir / "eagle3_meta.pt", token_map_path)
    return export_dir, token_map_path


def resolve_draft_artifacts(draft: str, algorithm: str, *, dry_run: bool = False) -> tuple[str, str | None]:
    """Resolve a user-supplied drafter path to the model and token-map paths SGLang expects.

    Accepts either the outer ``epoch_<E>_step_<S>`` directory or the inner
    ``model/`` directory; HF Hub repo ids are passed through untouched.

    Args:
        draft: A local path or HF Hub repo id.
        algorithm: Speculative algorithm name.
        dry_run: When True, no on-disk export is performed and the returned
            paths reflect what *would* be produced on a real launch.

    Returns:
        ``(draft_path, token_map_path_or_None)`` suitable for SGLang flags.
    """
    p = Path(draft)
    if not p.exists():
        return draft, None

    candidate_dirs = [p]
    if p.is_dir() and (p / "model").is_dir():
        candidate_dirs.insert(0, p / "model")

    for candidate in candidate_dirs:
        if not candidate.is_dir():
            continue
        if not ((candidate / "config.json").exists() and _has_hf_weight_file(candidate)):
            continue
        config_path = candidate / "config.json"
        if _config_needs_rewrite(config_path, algorithm):
            if dry_run:
                logger.warning(
                    "Existing config at %s has stale architectures for %s; --print-only "
                    "skips the rewrite. Running the printed command verbatim may fail; "
                    "rerun this script without --print-only first to heal the config.",
                    config_path,
                    algorithm,
                )
            else:
                _rewrite_config_for_sglang(config_path, config_path, algorithm)
        token_map_path = candidate / "speculative_token_map.pt"
        if token_map_path.exists() or algorithm != "EAGLE3":
            return str(candidate), str(token_map_path) if token_map_path.exists() else None
        # EAGLE3 + already-exported model dir but token map missing: try to
        # regenerate it from a sibling ``eagle3_meta.pt`` (one level up if the
        # user pointed at the inner ``model/`` directory).
        parent = candidate.parent if candidate.name == "model" else candidate
        meta_path = parent / "eagle3_meta.pt"
        if meta_path.exists():
            if dry_run:
                return str(candidate), str(token_map_path)
            _regenerate_token_map(meta_path, token_map_path)
            return str(candidate), str(token_map_path)
        logger.warning(
            "EAGLE3 model dir %s is missing speculative_token_map.pt and no sibling "
            "eagle3_meta.pt was found at %s; SGLang will fail to start without a token map.",
            candidate,
            meta_path,
        )
        return str(candidate), None

    if not p.is_dir():
        raise ValueError(f"--draft must be a directory or a Hugging Face repo id, got {draft!r}.")

    resolved_dir, token_map_path = _maybe_export_training_checkpoint(p, algorithm, dry_run=dry_run)
    if resolved_dir != p:
        logger.info("Resolved drafter path %s -> %s", p, resolved_dir)
        return str(resolved_dir), str(token_map_path) if token_map_path is not None else None
    return str(p), None


def build_sglang_argv(args: argparse.Namespace) -> list[str]:
    """Build the ``python -m sglang.launch_server`` argv for a given config."""
    draft_path, token_map_path = resolve_draft_artifacts(args.draft, args.algorithm, dry_run=args.print_only)
    argv: list[str] = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model",
        args.target,
        "--speculative-algorithm",
        args.algorithm,
        "--speculative-draft-model-path",
        draft_path,
        "--speculative-num-steps",
        str(args.num_steps),
        "--speculative-eagle-topk",
        str(args.topk),
        "--speculative-num-draft-tokens",
        str(args.num_draft_tokens),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--dtype",
        args.dtype,
    ]
    if args.tp_size > 1:
        argv += ["--tp-size", str(args.tp_size)]
    if token_map_path is not None:
        argv += ["--speculative-token-map", token_map_path]
    if args.trust_remote_code:
        argv.append("--trust-remote-code")
    if args.extra:
        argv += list(args.extra)
    return argv


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the serve helper."""
    parser = argparse.ArgumentParser(
        prog="serve_sglang",
        description=(
            "Launch SGLang with an Automodel-trained EAGLE/EAGLE3 drafter. "
            "Requires `uv pip install sglang` in the current environment; "
            "SGLang is not bundled with the Automodel container."
        ),
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target (base) model path or HuggingFace repo id, e.g. meta-llama/Llama-3.1-8B-Instruct.",
    )
    parser.add_argument(
        "--draft",
        required=True,
        help=(
            "Path to the drafter checkpoint directory produced by an EAGLE recipe "
            "(e.g. checkpoints/epoch_0_step_1000). If the directory contains a "
            "`model/` subdir, it is auto-selected."
        ),
    )
    parser.add_argument(
        "--algorithm",
        default="EAGLE3",
        choices=["EAGLE", "EAGLE3"],
        help="Speculative decoding algorithm to use in SGLang.",
    )
    parser.add_argument("--num-steps", type=int, default=3, help="--speculative-num-steps.")
    parser.add_argument("--topk", type=int, default=1, help="--speculative-eagle-topk.")
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=4,
        help="--speculative-num-draft-tokens.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server bind host.")
    parser.add_argument("--port", type=int, default=30000, help="Server port.")
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.75,
        help="--mem-fraction-static passed through to SGLang.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Inference dtype (must match the dtype used during EAGLE training).",
    )
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor-parallel size for SGLang.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward --trust-remote-code to SGLang (needed for custom target architectures).",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help=(
            "Print the resolved sglang command and exit without launching it. "
            "Skips checkpoint export entirely; the printed paths reflect what "
            "would be produced on a real launch."
        ),
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded verbatim to sglang.launch_server (prefix with `--`).",
    )
    parsed = parser.parse_args(argv)
    if parsed.extra and parsed.extra[0] == "--":
        parsed.extra = parsed.extra[1:]
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Validate the environment, resolve the drafter ckpt, then exec sglang.

    Returns the SGLang server's exit code, or ``2`` if SGLang or safetensors
    is missing.
    """
    logging.basicConfig(
        level=os.environ.get("NEMO_AUTOMODEL_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args(argv)
    if not args.print_only:
        # Fail fast before any expensive checkpoint export work.
        _check_sglang_available()
    cmd = build_sglang_argv(args)
    logger.info("SGLang command: %s", " ".join(cmd))
    if args.print_only:
        print(" ".join(cmd))
        return 0
    if hasattr(os, "execv") and Path(cmd[0]).is_absolute() and Path(cmd[0]).is_file():
        os.execv(cmd[0], cmd)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
