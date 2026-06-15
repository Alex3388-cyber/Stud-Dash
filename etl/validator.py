"""ETL validator — data quality assessment producing a scored quality report."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    total_rows: int
    total_columns: int
    completeness_score: float
    duplicate_rate: float
    outlier_flags: dict[str, int]
    type_mismatch_columns: list[str]
    empty_columns: list[str]
    high_cardinality_columns: list[str]
    overall_score: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "completeness_score": round(self.completeness_score, 2),
            "duplicate_rate": round(self.duplicate_rate, 4),
            "outlier_flags": self.outlier_flags,
            "type_mismatch_columns": self.type_mismatch_columns,
            "empty_columns": self.empty_columns,
            "high_cardinality_columns": self.high_cardinality_columns,
            "overall_score": round(self.overall_score, 2),
            "warnings": self.warnings,
        }


def validate(data: pd.DataFrame) -> QualityReport:
    """Assess data quality and return a scored QualityReport."""
    total_rows = len(data)
    total_columns = len(data.columns)

    if total_rows == 0:
        return QualityReport(0, total_columns, 0.0, 0.0, {}, [], list(data.columns), [], 0.0, ["Dataset is empty"])

    # Completeness: fraction of non-null cells
    completeness_score = float(1 - data.isna().mean().mean()) * 100

    # Duplicate rate
    duplicate_count = int(data.duplicated().sum())
    duplicate_rate = duplicate_count / total_rows

    # Outlier flags: z-score > 3 on numeric columns
    outlier_flags: dict[str, int] = {}
    numeric_cols = data.select_dtypes(include="number").columns
    for col in numeric_cols:
        series = data[col].dropna()
        if len(series) < 4:
            continue
        z = np.abs((series - series.mean()) / (series.std() + 1e-9))
        count = int((z > 3).sum())
        if count:
            outlier_flags[col] = count

    # Columns that are completely empty
    empty_columns = [col for col in data.columns if data[col].isna().all()]

    # High cardinality (>50 unique values in categorical)
    cat_cols = data.select_dtypes(include=["object", "string", "category"]).columns
    high_cardinality = [col for col in cat_cols if data[col].nunique(dropna=True) > 50]

    # Type mismatch: columns that look numeric but have non-numeric values
    type_mismatch: list[str] = []
    for col in data.select_dtypes(include=["object"]).columns:
        converted = pd.to_numeric(data[col], errors="coerce")
        if converted.notna().mean() > 0.5 and data[col].notna().mean() > 0.5:
            type_mismatch.append(col)

    # Build warnings
    warnings: list[str] = []
    if duplicate_rate > 0.05:
        warnings.append(f"{duplicate_count} duplicate rows detected ({duplicate_rate:.1%})")
    if empty_columns:
        warnings.append(f"Empty columns (all null): {', '.join(empty_columns)}")
    if high_cardinality:
        warnings.append(f"High-cardinality columns (>50 unique): {', '.join(high_cardinality)}")
    if type_mismatch:
        warnings.append(f"Columns with mixed numeric/text values: {', '.join(type_mismatch)}")
    if outlier_flags:
        flagged = ", ".join(f"{c}({n})" for c, n in list(outlier_flags.items())[:5])
        warnings.append(f"Statistical outliers (z>3) in: {flagged}")

    # Overall score: weighted average
    completeness_weight = 0.40
    duplicate_weight = 0.20
    outlier_weight = 0.15
    cardinality_weight = 0.10
    mismatch_weight = 0.15

    outlier_penalty = min(1.0, sum(outlier_flags.values()) / max(total_rows, 1) * 10)
    cardinality_penalty = min(1.0, len(high_cardinality) / max(total_columns, 1))
    mismatch_penalty = min(1.0, len(type_mismatch) / max(total_columns, 1))

    overall_score = (
        (completeness_score / 100) * completeness_weight
        + (1 - duplicate_rate) * duplicate_weight
        + (1 - outlier_penalty) * outlier_weight
        + (1 - cardinality_penalty) * cardinality_weight
        + (1 - mismatch_penalty) * mismatch_weight
    ) * 100

    return QualityReport(
        total_rows=total_rows,
        total_columns=total_columns,
        completeness_score=completeness_score,
        duplicate_rate=duplicate_rate,
        outlier_flags=outlier_flags,
        type_mismatch_columns=type_mismatch,
        empty_columns=empty_columns,
        high_cardinality_columns=high_cardinality,
        overall_score=overall_score,
        warnings=warnings,
    )
