"""Data loading helpers for student performance datasets."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
SAMPLE_DATA_PATH = DATASETS_DIR / "sample_students.csv"


def load_sample_data() -> pd.DataFrame:
    """Load the bundled sample dataset used by the starter dashboard."""
    return pd.read_csv(SAMPLE_DATA_PATH)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset from a user-provided path."""
    return pd.read_csv(Path(path))
