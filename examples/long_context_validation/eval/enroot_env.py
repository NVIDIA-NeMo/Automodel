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

"""Enroot execution backend for mini-swe-agent (Docker-less SWE-bench on Slurm/pyxis clusters).

Mirrors ``minisweagent.environments.docker.DockerEnvironment`` but drives per-instance
SWE-bench repo environments with **enroot** instead of Docker, since this cluster has
enroot + pyxis and no Docker daemon.

Lifecycle per SWE-bench instance:
  1. import  ``docker://swebench/sweb.eval.x86_64.<iid>:latest`` -> a node-local ``.sqsh``
     (cached; import uses node-local temp + capped zstd to avoid the lustre-whiteout and
     parallel-zstd-OOM failures seen on this cluster).
  2. create  a named, writable container from the ``.sqsh``.
  3. execute each agent command via ``enroot start --rw`` sharing that container's rootfs,
     so ``/testbed`` edits persist across commands (each command still runs in a fresh
     subshell, matching the docker backend's contract).
  4. cleanup removes the container (the ``.sqsh`` is kept for reuse).

Select it in a mini-swe-agent config with::

    environment:
      environment_class: enroot_env.EnrootEnvironment

(``enroot_env`` must be importable, i.e. this file's dir on ``PYTHONPATH``.)
"""

import logging
import os
import shlex
import subprocess
import uuid
from pathlib import Path
from typing import Any

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge
from pydantic import BaseModel


class EnrootEnvironmentConfig(BaseModel):
    """Configuration for an enroot-backed SWE-bench execution environment."""

    image: str
    """Container image. Either ``docker://...`` (imported to a .sqsh) or a path to an existing .sqsh."""
    cwd: str = "/testbed"
    """Working directory in which to execute commands."""
    env: dict[str, str] = {}
    """Environment variables to set in the container."""
    forward_env: list[str] = []
    """Host environment variables to forward into the container (only if set on the host)."""
    timeout: int = 60
    """Timeout (seconds) for executing a single command."""
    executable: str = os.getenv("MSWEA_ENROOT_EXECUTABLE", "enroot")
    """Path to the enroot executable."""
    interpreter: list[str] = ["bash", "-c"]
    """Interpreter for commands. ``bash -c`` (non-login) so BASH_ENV is honored (conda activate testbed)."""
    sqsh_dir: str = os.getenv("MSWEA_ENROOT_SQSH_DIR", "/tmp/mswea_sqsh")
    """Directory holding imported .sqsh images (reused across instances if present)."""
    import_timeout: int = 900
    """Timeout (seconds) for importing an image. Kept under the cluster's 20-min idle-GPU
    cancellation window so a hung import fails (and frees the instance) rather than letting
    the served GPUs sit idle long enough to be auto-cancelled."""
    max_processors: int = int(os.getenv("ENROOT_MAX_PROCESSORS", "8"))
    """Cap parallel zstd workers during import (avoids OOM seen with nproc-many workers)."""


class EnrootEnvironment:
    """Docker-less per-instance SWE-bench environment backed by an enroot container."""

    def __init__(self, *, config_class: type = EnrootEnvironmentConfig, logger: logging.Logger | None = None, **kwargs):
        """Execute bash commands in a per-instance enroot container. See ``EnrootEnvironmentConfig``."""
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.container_name: str | None = None
        self.config = config_class(**kwargs)
        self._start_container()

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def _sqsh_path(self) -> Path:
        """Resolve the .sqsh to create the container from, importing it if needed."""
        if self.config.image.endswith(".sqsh"):
            return Path(self.config.image)
        # Normalize to enroot's docker URI form: docker://[REGISTRY#]REPO[:TAG].
        # SWE-bench image names arrive as "docker.io/swebench/sweb.eval...:latest"; enroot
        # uses '#' (not '/') as the registry separator, so a literal "docker.io/" in the repo
        # path resolves a nonexistent repo. Strip it (docker.io is enroot's default registry).
        ref = self.config.image.replace("docker://", "").lstrip("/")
        if ref.startswith("docker.io/"):
            ref = ref[len("docker.io/") :]
        safe = ref.replace("/", "+").replace(":", "+")
        sqsh_dir = Path(self.config.sqsh_dir)
        sqsh_dir.mkdir(parents=True, exist_ok=True)
        sqsh = sqsh_dir / f"{safe}.sqsh"
        if sqsh.exists() and sqsh.stat().st_size > 0:
            self.logger.info(f"Reusing cached image {sqsh}")
            return sqsh
        docker_ref = f"docker://{ref}"
        env = dict(os.environ, ENROOT_MAX_PROCESSORS=str(self.config.max_processors))
        # Import to a unique temp path then atomically rename, so a shared (lustre) image cache
        # is safe under concurrent runs (e.g. base + SFT importing the same image at once):
        # duplicate imports are redundant but never corrupt the cached .sqsh.
        tmp = sqsh.with_name(f"{sqsh.name}.tmp.{uuid.uuid4().hex[:8]}")
        self.logger.info(f"Importing {docker_ref} -> {sqsh}")
        proc = subprocess.run(
            [self.config.executable, "import", "-o", str(tmp), docker_ref],
            capture_output=True,
            text=True,
            timeout=self.config.import_timeout,
            env=env,
        )
        if proc.returncode != 0:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            raise RuntimeError(
                f"enroot import failed for {docker_ref} (rc={proc.returncode}).\n"
                f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
            )
        os.replace(tmp, sqsh)
        return sqsh

    def _enroot_env(self) -> dict:
        """Env for enroot subcommands. ENROOT_MOUNT_HOME=n is CRITICAL: by default enroot
        bind-mounts the host's home over the container's /root, so /root/.bashrc becomes the
        HOST bashrc instead of the image's `conda activate testbed` line — commands then run
        in the wrong Python env. Disabling the home mount lets the image's bashrc run."""
        return dict(os.environ, ENROOT_MOUNT_HOME="n")

    def _start_container(self):
        """Import (if needed) and create a writable enroot container for this instance."""
        sqsh = self._sqsh_path()
        name = f"mswea-{uuid.uuid4().hex[:8]}"
        self.logger.debug(f"Creating enroot container {name} from {sqsh}")
        subprocess.run(
            [self.config.executable, "create", "--name", name, str(sqsh)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
            env=self._enroot_env(),
        )
        self.container_name = name
        self.logger.info(f"Created enroot container {name}")

    def _enroot_start_cmd(self, command: str, cwd: str) -> list[str]:
        cmd = [self.config.executable, "start", "--rw"]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["--env", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["--env", f"{key}={value}"])
        # enroot has no working-dir flag; cd into cwd inside the shell.
        full = f"cd {shlex.quote(cwd)} && {command}"
        cmd.extend([self.container_name, *self.config.interpreter, full])
        return cmd

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the enroot container and return the result as a dict."""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd
        assert self.container_name, "Container not created"
        cmd = self._enroot_start_cmd(command, cwd)
        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=self._enroot_env(),
            )
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        """Raise Submitted when the agent emits the submission marker (matches docker backend)."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def cleanup(self):
        """Remove the enroot container (keeps the .sqsh for reuse)."""
        if getattr(self, "container_name", None) is not None:
            subprocess.Popen(f"{self.config.executable} remove -f {self.container_name} >/dev/null 2>&1 &", shell=True)

    def __del__(self):
        self.cleanup()
