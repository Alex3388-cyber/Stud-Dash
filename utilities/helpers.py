"""General helper functions shared across the project."""

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return its Path object."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
