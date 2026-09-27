#!/usr/bin/env python3
"""Validate and resolve the manifest-driven Titans evaluation model registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_registry(path: Path) -> list[dict[str, Any]]:
    """Load and validate one evaluation registry."""
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or not isinstance(payload.get("models"), list):
        raise ValueError(f"{path}: unsupported registry schema")
    models = payload["models"]
    names = [model.get("name") for model in models]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{path}: every model requires a non-empty name")
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: model names must be unique")
    for model in models:
        if not isinstance(model.get("run_root"), str):
            raise ValueError(f"{path}: {model['name']} requires run_root")
        if not isinstance(model.get("memory_model"), bool):
            raise ValueError(f"{path}: {model['name']} requires boolean memory_model")
        if not isinstance(model.get("enabled", True), bool):
            raise ValueError(f"{path}: {model['name']} enabled must be boolean")
    return models


def resolve_checkpoint(model: dict[str, Any], work_root: Path) -> Path:
    """Resolve an enabled registry entry to its consolidated checkpoint."""
    root = Path(model["run_root"])
    if not root.is_absolute():
        root = work_root / "outputs" / root
    latest = root / "LATEST"
    if not latest.is_symlink():
        raise FileNotFoundError(f"{model['name']}: missing checkpoint link {latest}")
    return (root / latest.readlink() / "model" / "consolidated").resolve()


def evaluation_rows(
    registry_path: Path,
    work_root: Path,
    *,
    model_name: str | None = None,
) -> list[tuple[str, Path, bool]]:
    """Expand enabled models into checkpoint and TTT-mode rows."""
    rows = []
    for model in load_registry(registry_path):
        if not model.get("enabled", True):
            continue
        if model_name is not None and model["name"] != model_name:
            continue
        checkpoint = resolve_checkpoint(model, work_root)
        modes = (True, False) if model["memory_model"] else (True,)
        rows.extend((model["name"], checkpoint, enabled) for enabled in modes)
    if model_name is not None and not rows:
        raise KeyError(f"No enabled model named {model_name!r}")
    return rows


def main() -> None:
    """Print tab-separated rows for shell launchers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--model")
    args = parser.parse_args()
    for name, checkpoint, enable_ttt in evaluation_rows(
        args.registry,
        args.work_root,
        model_name=args.model,
    ):
        print(f"{name}\t{checkpoint}\t{int(enable_ttt)}")


if __name__ == "__main__":
    main()
