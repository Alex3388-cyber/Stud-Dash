"""ETL extractor — reads uploaded file bytes into a raw DataFrame."""

from __future__ import annotations

import io

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def extract(file_name: str, file_content: bytes) -> pd.DataFrame:
    """Parse file bytes into a raw DataFrame based on extension."""
    ext = _get_extension(file_name)
    if ext == ".csv":
        return _read_csv(file_content)
    if ext in {".xlsx", ".xls"}:
        return _read_excel(file_content, ext)
    raise ValueError(f"Unsupported file type: {ext}. Upload a CSV or Excel file.")


def _get_extension(file_name: str) -> str:
    from pathlib import Path
    return Path(file_name).suffix.lower()


def _read_csv(content: bytes) -> pd.DataFrame:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode CSV file. Try saving as UTF-8.")


def _read_excel(content: bytes, ext: str) -> pd.DataFrame:
    engine = "openpyxl" if ext == ".xlsx" else "xlrd"
    return pd.read_excel(io.BytesIO(content), engine=engine)
