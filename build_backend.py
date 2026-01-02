import subprocess
from pathlib import Path

from setuptools import build_meta as orig

_root = Path(__file__).parent.resolve()


def is_git_repo():
    return (_root / ".git").is_dir()


def get_git_commit_hash(cwd: Path | None, length: int = 10) -> str:
    try:
        cmd = ["git", "rev-parse", f"--short={length}", "HEAD"]
        return subprocess.check_output(cmd, cwd=cwd).strip().decode("utf-8")
    except Exception:
        return ""


def get_git_branch(cwd: Path | None = None) -> str:
    try:
        cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        return subprocess.check_output(cmd, cwd=cwd).strip().decode("utf-8")
    except Exception:
        return ""


def get_git_tag(cwd: Path | None = None) -> str:
    try:
        cmd = ["git", "describe", "--tags", "--match", "v*", "--exact-match"]
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL).strip().decode("utf-8")
    except Exception:
        return ""


def _generate_version_info():
    """Generate version info file with git metadata."""
    version_file = _root / "version.txt"
    if version_file.exists():
        with open(version_file, "r") as f:
            version = f.read().strip()
    else:
        version = "0.0.0"

    # Get git info
    git_commit_hash = get_git_commit_hash(_root)
    git_branch = get_git_branch(_root)
    git_tag = get_git_tag(_root)

    # Create version info in the source tree
    package_dir = Path(__file__).parent / "nemo_automodel"
    _version_file = package_dir / "_version.py"

    # If file exists and not in git repo (installing from sdist), keep existing file
    if _version_file.exists() and not is_git_repo():
        print("The _version file already exists (not in git repo), keeping it")
        return version

    # In git repo (editable) or file doesn't exist, create/update it
    with open(_version_file, "w") as f:
        f.write('"""Build _version for nemo_automodel package."""\n')
        if git_branch.startswith("release") or git_tag:
            # Release version or tag version may be push to PyPI, it shouldn't has git hash suffix which will affect wheel name.
            f.write(f'__version__ = "{version}"\n')
        else:
            f.write(f'__version__ = "{version}+{git_commit_hash}"\n')
        git_commit_hash = git_commit_hash or "unknown"
        f.write(f'__git_version__ = "{git_commit_hash}"\n')
    print(f"Created _version file with version {version}")
    return version


# Generate version info as soon as this module is imported
_generate_version_info()


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
