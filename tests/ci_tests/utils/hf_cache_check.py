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

#!/usr/bin/env python3
"""Check whether every HF repo referenced by a recipe YAML is already present
in the local Hugging Face Hub cache (``$HF_HOME/hub/``).

Pure inspection — never downloads, never modifies the cache. The exit code
IS the decision so the caller shell reduces to::

    python3 hf_cache_check.py --config <recipe> || export HF_HUB_OFFLINE=0

Exit codes:
    0   every referenced HF repo is present in the local cache
    1   at least one repo is missing, ``$HF_HOME`` is unset, or the check
        errored — treat as "cache miss, need online"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

# Keys under which recipe YAMLs declare HF repo IDs. Covers:
#   - model.pretrained_model_name_or_path
#   - processor.pretrained_model_name_or_path (VLM)
#   - ci.checkpoint_robustness.model.pretrained_model_name_or_path
#   - ci.checkpoint_robustness.tokenizer_name
#   - any future nested block using the same keys
REPO_ID_KEYS = ("pretrained_model_name_or_path", "tokenizer_name")


def _walk_repo_ids(node: Any) -> Iterable[str]:
    """Yield every string value in ``node`` reached via a key in ``REPO_ID_KEYS``."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in REPO_ID_KEYS and isinstance(value, str):
                yield value
            else:
                yield from _walk_repo_ids(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_repo_ids(item)


def _is_hf_repo_id(value: str) -> bool:
    """Heuristic: exactly one ``/``, no leading ``/`` or ``.``, no whitespace."""
    if not value or value.startswith("/") or value.startswith("."):
        return False
    if any(ch.isspace() for ch in value):
        return False
    return value.count("/") == 1


def _extract_repo_ids(recipe_path: Path) -> list[str]:
    """Deduplicated, ordered list of HF-form repo IDs referenced by ``recipe_path``."""
    with recipe_path.open("r", encoding="utf-8") as f:
        recipe = yaml.safe_load(f) or {}

    seen: dict[str, None] = {}
    for value in _walk_repo_ids(recipe):
        if _is_hf_repo_id(value) and value not in seen:
            seen[value] = None
    return list(seen)


def _repo_folder(repo_id: str) -> str:
    """Deterministic cache folder name (``models--<org>--<name>``).

    Prefers ``huggingface_hub.file_download.repo_folder_name`` when available so
    we stay locked to upstream's convention; falls back to the documented
    ``models--{org}--{name}`` form otherwise.
    """
    try:
        from huggingface_hub.file_download import repo_folder_name

        return repo_folder_name(repo_id=repo_id, repo_type="model")
    except Exception:
        return "models--" + repo_id.replace("/", "--")


def _has_snapshot(repo_dir: Path) -> bool:
    """A cache entry is usable iff ``snapshots/<rev>/`` exists with at least one file."""
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return False
    for rev in snapshots.iterdir():
        if rev.is_dir() and any(rev.iterdir()):
            return True
    return False


def check_cache(recipe_path: Path, hf_home: Path) -> tuple[list[str], list[str]]:
    """Return ``(cached, missing)`` for the recipe's referenced repo IDs."""
    hub_dir = hf_home / "hub"
    cached: list[str] = []
    missing: list[str] = []
    for repo_id in _extract_repo_ids(recipe_path):
        repo_dir = hub_dir / _repo_folder(repo_id)
        (cached if _has_snapshot(repo_dir) else missing).append(repo_id)
    return cached, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Recipe YAML path")
    args = parser.parse_args()

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        print("[hf_cache_check] HF_HOME unset; treating as cache miss", file=sys.stderr)
        return 1
    if not args.config.is_file():
        print(f"[hf_cache_check] recipe not found: {args.config}; treating as cache miss", file=sys.stderr)
        return 1

    cached, missing = check_cache(args.config, Path(hf_home))
    if missing:
        print(f"[hf_cache_check] miss in {hf_home}: {' '.join(missing)}")
        return 1
    print(f"[hf_cache_check] all {len(cached)} required repos cached in {hf_home}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
