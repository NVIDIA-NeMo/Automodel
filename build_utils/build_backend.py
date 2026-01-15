import ast
import logging
import os
import subprocess
from pathlib import Path

from setuptools import build_meta as orig

_root = Path(__file__).parent.parent.resolve()
logger = logging.getLogger(__name__)


def is_git_repo():
    return (_root / ".git").is_dir()


def get_git_commit_hash(cwd: Path | None, length: int = 10) -> str | None:
    try:
        cmd = ["git", "rev-parse", f"--short={length}", "HEAD"]
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL).strip().decode("utf-8")
    except Exception:
        return None


def get_git_tag(cwd: Path | None = None) -> str:
    try:
        cmd = ["git", "describe", "--tags", "--match", "v*", "--exact-match"]
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL).strip().decode("utf-8")
    except Exception:
        return ""


def get_version_info_match(file_path: Path) -> dict[str, str]:
    content = file_path.read_text(encoding="utf-8")
    tree = ast.parse(content)
    results = {}

    for node in tree.body:
        match node:
            case ast.Assign(targets=[ast.Name(id=key)], value=ast.Constant(value=val)) if key in {
                "__version__",
                "__git_version__",
            }:
                results[key] = val
    assert len(results) == 2, f"Expected 2 version info keys, got {len(results)}"
    return results


def dynamic_version_info():
    """Generate version info file with git metadata."""
    version_file = _root / "version.txt"
    base_version = version_file.read_text().strip()

    package_dir = Path(__file__).parent.parent / "nemo_automodel"
    _version_file = package_dir / "_version.py"

    git_commit_hash = get_git_commit_hash(_root)
    git_tag = get_git_tag(_root)

    if git_commit_hash:
        git_commit_hash = git_commit_hash
    elif _version_file.exists() and not is_git_repo():
        git_commit_hash = get_version_info_match(_version_file).get("__git_version__")
    else:
        git_commit_hash = "gitunknown"

    final_version = base_version

    with open(_version_file, "w") as f:
        f.write('"""Generate version info file with git metadata."""\n')
        need_skip_git_version = os.environ.get("NO_GIT_VERSION", "").lower() in ("1", "true", "yes", "on")
        if git_tag or need_skip_git_version:
            f.write(f'__version__ = "{final_version}"\n')
        else:
            final_version = f"{final_version}+{git_commit_hash}"
            f.write(f'__version__ = "{final_version}"\n')
        f.write(f'__git_version__ = "{git_commit_hash}"\n')
    logger.info(f"Created _version file with version {final_version}")
    return final_version, git_commit_hash


dynamic_version_info()


def get_requires_for_build_wheel(config_settings=None):
    return orig.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return orig.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):
    return orig.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return orig.prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return orig.build_editable(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    return orig.build_sdist(sdist_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return orig.build_wheel(wheel_directory, config_settings, metadata_directory)
