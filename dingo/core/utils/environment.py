"""
Utilities for documenting the Python environment.
"""

import os
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path


def get_packages() -> list[str]:
    """
    Get a list of installed packages in the current environment.

    Returns:
        A list of installed packages with their versions.
    """

    list_of_packages = []
    for dist in sorted(metadata.distributions(), key=lambda d: d.name.lower()):
        list_of_packages.append(f"{dist.name}=={dist.version}")

    return list_of_packages


def get_python_version() -> str:
    """
    Get the Python interpreter version.

    Returns:
        The Python interpreter version.
    """

    return sys.version


def get_virtual_environment() -> str:
    """
    Get the virtual environment name.

    Returns:
        The virtual environment name.
    """

    try:
        return os.environ["VIRTUAL_ENV"]
    except KeyError:
        return "No virtual environment detected"


def document_environment(target_dir: Path) -> None:
    """
    Document the current environment to a `info_env.txt` file
    inside the given `target_dir`.
    """

    # Get the list of installed packages
    packages = get_packages()

    # Get the Python interpreter version
    python_version = get_python_version()

    # Get the virtual environment name
    virtual_environment = get_virtual_environment()

    # Write the environment to a requirements file
    with open(target_dir / "info_env.txt", "w") as file:
        file.write(f"# Date: {datetime.now().isoformat()}\n")
        file.write(f"# Python version: {python_version}\n")
        file.write(f"# Virtual environment: {virtual_environment}\n")
        for package in packages:
            file.write(f"{package}\n")
