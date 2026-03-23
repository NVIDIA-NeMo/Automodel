from packaging.version import Version
from typing import Optional

import requests
import tomllib


def get_latest_version(package_name: str) -> Optional[str]:
    """
    Sends api request to pypi to get the latest version of a package.

    Args:
        package_name: Name of the package to request

    Returns:
        latest_version: Latest version string of the package in pypi
    """
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        latest_version = data['info']['version']
        return latest_version
    else:
        return None


def find_lockfile_dependency(package_name: str, lockfile: str = "uv.lock") -> Optional[dict]:
    """
    Finds a specific package defined in the uv.lock file.

    Args:
        package_name: Name of the package to request
        lockfile: Path to the uv lock file

    Returns:
        package: Returns dictionary object of the package if found in uv.lock
    """
    with open(lockfile, "rb") as f:
        data = tomllib.load(f)

    for pkg in data.get("package", []):
        if pkg.get("name") == package_name:
            return pkg
    return None


def write_results(results_dic: dict, path: str = "/tmp/transformers_version_check.sh") -> None:
    """
    Write dictionary into file with + separator.

    Args:
        results_dic: Dictionary to write to file
        path: Path to the output file

    """
    with open(path, "a") as file:
        for key, val in results_dic.items():
            file.write(f"{key}={val}\n")


def main() -> None:
    package = "transformers"
    results = {
        "UPDATE_TRANSFORMERS": False,
        "TRANSFORMERS_VERSION": "",
    }
    print("--------------------------------------------------")
    print(f"Version Report: f{package}")

    # Check pypi for latest version
    latest_version = get_latest_version(package)
    print(f"Pypi latest version: {latest_version}")

    # Check version in uv.lock
    uv_package_version = find_lockfile_dependency(package).get("version")
    print(f"uv.lock package version: {uv_package_version}")

    if Version(latest_version) > Version(uv_package_version):
        print("Pypi has newer version. Updating uv lock file.")
        results["UPDATE_TRANSFORMERS"] = True
        results["TRANSFORMERS_VERSION"] = latest_version

    write_results(results)


if __name__ == "__main__":
    main()
