# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import json, os, platform, shutil, subprocess, sys, tempfile
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None  # noqa

def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except Exception as e:
        return f"<<{cmd} failed: {e}>>"

def _write(path: Path, text: str) -> Optional[Path]:
    try:
        path.write_text(text)
        return path
    except Exception:
        return None

def _git_root(start: Path) -> Path:
    out = _run(["git", "rev-parse", "--show-toplevel"])
    p = Path(out.strip())
    return p if p.exists() else start

def _detect_docker_context() -> Dict[str, Any]:
    """
    Best-effort capture of the image reference and digest.
    Works when:
      - WANDB_DOCKER / AUTOMODEL_DOCKER_IMAGE / DOCKER_IMAGE is set
      - or docker/podman/nerdctl is present and can inspect the ref
    """
    info: Dict[str, Any] = {
        "ref_env": os.getenv("WANDB_DOCKER")
                   or os.getenv("AUTOMODEL_DOCKER_IMAGE")
                   or os.getenv("DOCKER_IMAGE"),
        "engine": None,
        "digest_ref": None,
        "raw_inspect": None,
        "os_release": None,
        "container_runtime_guess": None,
    }

    # quick OS fingerprint (useful if reproducing on a different base image)
    try:
        info["os_release"] = Path("/etc/os-release").read_text()
    except Exception:
        pass

    # try to guess runtime from env files commonly present in containers
    if Path("/run/.containerenv").exists():
        info["container_runtime_guess"] = "podman"
    elif Path("/.dockerenv").exists():
        info["container_runtime_guess"] = "docker"

    # find a client we can call
    for cand in ("docker", "podman", "nerdctl"):
        if shutil.which(cand):
            info["engine"] = cand
            break

    ref = info["ref_env"]
    if ref and info["engine"]:
        if info["engine"] in ("docker", "podman"):
            fmt = "{{index .RepoDigests 0}}"
            info["digest_ref"] = _run([info["engine"], "image", "inspect", ref, "--format", fmt]).strip()
            info["raw_inspect"] = _run([info["engine"], "image", "inspect", ref])
        elif info["engine"] == "nerdctl":
            out = _run(["nerdctl", "image", "inspect", ref])
            info["raw_inspect"] = out
            # RepoDigests appears in JSON; keep it simple and let the replicate tool parse later
            # (we still save the raw inspect for exact provenance)
    return info

SAFE_ENV_PREFIXES = (
    "CUDA_", "CUDNN_", "NCCL_", "TRANSFORMERS_CACHE", "HF_HOME",
    "UV_", "PIP_", "PYTORCH_", "WANDB_", "NEMO_", "WORLD_SIZE", "RANK", "LOCAL_RANK",
)

def log_environment_bundle(run: "wandb.sdk.wandb_run.Run",
                           project_root: Path | None = None,
                           artifact_name: str = "runtime-env") -> None:
    """
    Capture env + docker + resolver state into a W&B artifact tied to the run.
    Safe to call on every entrypoint before heavy work starts.
    """
    assert wandb is not None, "wandb must be importable"
    root = (project_root or _git_root(Path.cwd())).resolve()

    files: list[Path] = []
    tmp = Path(tempfile.mkdtemp(prefix="automodel-env-"))

    # 1) system & python metadata
    meta = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "uname": platform.uname()._asdict(),
        "pip_version": _run([sys.executable, "-m", "pip", "--version"]),
        "pip_config_debug": _run([sys.executable, "-m", "pip", "config", "debug"]),
        "nvidia_smi": _run(["bash", "-lc", "nvidia-smi -x -q || nvidia-smi || true"]),
        "nvcc": _run(["bash", "-lc", "nvcc --version || true"]),
        "git_commit": _run(["git", "rev-parse", "HEAD"]).strip(),
        "git_status": _run(["git", "status", "--porcelain"]),
        # whitelist non-secret env only
        "env": {k: v for k, v in os.environ.items() if any(k.startswith(p) for p in SAFE_ENV_PREFIXES)},
    }
    p = _write(tmp / "env-metadata.json", json.dumps(meta, indent=2))
    if p: files.append(p)

    # 2) docker context
    docker_info = _detect_docker_context()
    p = _write(tmp / "docker.json", json.dumps(docker_info, indent=2))
    if p: files.append(p)

    # 3) resolver snapshots
    # Prefer uv if present (even if we launched with plain python)
    if (root / "pyproject.toml").exists() and shutil.which("uv"):
        for name in ("pyproject.toml", "uv.lock"):
            f = root / name
            if f.exists(): files.append(f)
        p = _write(tmp / "pip-freeze.txt", _run(["uv", "pip", "freeze"]))
        if p: files.append(p)
        p = _write(tmp / "requirements-uv.txt", _run(["uv", "export", "--frozen", "--format", "requirements-txt"]))
        if p: files.append(p)
    else:
        p = _write(tmp / "pip-freeze.txt", _run([sys.executable, "-m", "pip", "freeze"]))
        if p: files.append(p)
        # Conda (optional)
        if os.environ.get("CONDA_PREFIX") or shutil.which("conda"):
            p = _write(tmp / "conda-env.yml", _run(["conda", "env", "export"]))
            if p: files.append(p)

    # 4) include project files if present
    for fname in ("requirements.txt", "requirements-dev.txt", "pyproject.toml", "setup.cfg", "setup.py"):
        f = root / fname
        if f.exists():
            files.append(f)

    # 5) ship artifact
    art = wandb.Artifact(artifact_name, type="environment")
    for f in files:
        try:
            art.add_file(str(f))
        except Exception:
            pass
    run.log_artifact(art)

    # Also pin as a run summary for quick access
    run.summary["environment_artifact"] = art.id
    if docker_info.get("digest_ref"):
        run.summary["docker_image"] = docker_info["digest_ref"]
    elif docker_info.get("ref_env"):
        run.summary["docker_image"] = docker_info["ref_env"]
