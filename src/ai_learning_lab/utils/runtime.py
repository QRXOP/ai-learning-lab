"""Runtime and dependency metadata for reproducible experiment records."""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version


def package_version(distribution: str) -> str:
    """Return an installed distribution's version, or ``"not-installed"``."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def runtime_metadata() -> dict[str, dict[str, str]]:
    """Describe the platform, Python version, and key dependency versions."""
    return {
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "python_version": platform.python_version(),
        },
        "dependencies": {
            "scikit_learn": package_version("scikit-learn"),
            "numpy": package_version("numpy"),
        },
    }
