"""Data preprocessing utilities for the Student Performance Prediction Dashboard.

The functions in this module prepare uploaded student datasets for future model
training without fitting a machine learning model. They keep a cleaned
human-readable dataset separate from the transformed model-ready feature
matrix so reports and visual exploration never accidentally use encoded or
scaled values.

This version adds stronger preprocessing governance for scalability:

- automatic ID-like column detection
- exclusion of unique identifiers from one-hot encoding
- high-cardinality detection and warnings
- memory-aware sparse encoding
- preprocessing metadata that explains risky columns to the user
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_CARDINALITY_THRESHOLD = 30
DEFAULT_HIGH_UNIQUENESS_RATIO = 0.9


@dataclass(frozen=True)
class PreprocessingSummary:
    """Summary values that explain what happened during preprocessing."""

    original_rows: int
    original_columns: int
    duplicate_rows_removed: int
    rows_after_duplicates: int
    missing_values_before: int
    missing_values_after: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    identifier_columns: list[str]
    excluded_categorical_columns: list[str]
    high_cardinality_columns: list[str]
    warning_messages: list[str]
    cardinality_threshold: int
    encoded_feature_count: int
    processed_rows: int
    processed_columns: int
    feature_matrix_format: str


def normalize_column_name(name: str) -> str:
    """Normalize a column name for ID-like and governance heuristics."""
    return str(name).strip().lower().replace(" ", "").replace("_", "").replace("-", "").replace("/", "")


def get_numeric_columns(data: pd.DataFrame) -> list[str]:
    """Identify numeric columns that should be normalized."""
    return data.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(data: pd.DataFrame) -> list[str]:
    """Identify categorical columns that should be encoded."""
    return data.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()


def remove_duplicate_records(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows and return the cleaned dataframe plus removal count."""
    original_rows = len(data)

    # Keep the first copy of each duplicate record so the dataset keeps one valid row.
    deduplicated = data.drop_duplicates().reset_index(drop=True)
    duplicate_rows_removed = original_rows - len(deduplicated)
    return deduplicated, duplicate_rows_removed


def unique_value_ratio(series: pd.Series) -> float:
    """Return the fraction of non-missing values that are unique."""
    non_missing = series.dropna()
    if non_missing.empty:
        return 0.0
    return float(non_missing.nunique(dropna=True) / len(non_missing))


def is_sequential_integer_series(series: pd.Series) -> bool:
    """Return True when a numeric series behaves like a simple identifier."""
    non_missing = pd.to_numeric(series, errors="coerce").dropna()
    if len(non_missing) < 3:
        return False
    if (non_missing % 1 != 0).any():
        return False

    sorted_values = non_missing.sort_values().astype("int64")
    differences = sorted_values.diff().dropna()
    if differences.empty:
        return False
    return bool(differences.isin([0, 1]).all() and sorted_values.nunique() >= max(3, len(sorted_values) - 1))


def detect_identifier_columns(
    data: pd.DataFrame,
    uniqueness_ratio_threshold: float = DEFAULT_HIGH_UNIQUENESS_RATIO,
) -> list[str]:
    """Detect ID-like columns that should be excluded from model encoding.

    Professional preprocessing should treat unique identifiers as metadata, not
    predictive signal. This function flags explicit ID names and columns whose
    values are nearly all unique.
    """
    explicit_keywords = {
        "id",
        "studentid",
        "studentnumber",
        "studentno",
        "registrationnumber",
        "regnumber",
        "matricnumber",
        "admissionnumber",
        "indexnumber",
        "candidateid",
        "userid",
        "recordid",
        "serialnumber",
    }

    identifier_columns: list[str] = []
    total_rows = len(data)
    for column in data.columns:
        normalized_name = normalize_column_name(column)
        column_values = data[column]
        uniqueness_ratio = unique_value_ratio(column_values)

        explicit_name_match = normalized_name in explicit_keywords
        fuzzy_name_match = any(
            token in normalized_name
            for token in ["studentid", "registration", "matric", "recordid", "serial", "candidateid"]
        ) or normalized_name.endswith("id") or normalized_name.endswith("number") or normalized_name.endswith("no")

        is_almost_unique = total_rows > 0 and uniqueness_ratio >= uniqueness_ratio_threshold
        numeric_sequence_like = column_values.dtype.kind in {"i", "u", "f"} and is_sequential_integer_series(column_values)

        if explicit_name_match or fuzzy_name_match or (is_almost_unique and numeric_sequence_like):
            identifier_columns.append(column)
            continue

        # String columns with near row-level uniqueness usually behave like IDs,
        # registration codes, emails, or names and explode one-hot width.
        if column_values.dtype.name in {"object", "string", "category"} and is_almost_unique:
            identifier_columns.append(column)

    return identifier_columns


def detect_high_cardinality_columns(
    data: pd.DataFrame,
    categorical_columns: list[str],
    cardinality_threshold: int,
) -> list[str]:
    """Detect categorical columns with too many distinct levels for safe one-hot encoding."""
    high_cardinality_columns: list[str] = []
    for column in categorical_columns:
        unique_count = int(data[column].astype("string").dropna().nunique())
        if unique_count > cardinality_threshold:
            high_cardinality_columns.append(column)
    return high_cardinality_columns


def build_preprocessing_warnings(
    identifier_columns: list[str],
    high_cardinality_columns: list[str],
    cardinality_threshold: int,
) -> list[str]:
    """Generate user-facing preprocessing warnings for risky columns."""
    warnings: list[str] = []
    if identifier_columns:
        warnings.append(
            "Identifier-like columns were excluded from encoding to prevent meaningless high-dimensional features: "
            + ", ".join(identifier_columns)
        )
    if high_cardinality_columns:
        warnings.append(
            f"High-cardinality categorical columns above {cardinality_threshold} unique values were excluded from one-hot encoding: "
            + ", ".join(high_cardinality_columns)
        )
    return warnings


def choose_preprocessing_columns(
    data: pd.DataFrame,
    cardinality_threshold: int = DEFAULT_CARDINALITY_THRESHOLD,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    """Split columns into safe numeric/categorical preprocessing groups.

    Returns:
    - numeric columns kept
    - categorical columns kept
    - identifier columns excluded
    - high-cardinality categorical columns excluded
    - excluded categorical columns
    - warning messages
    """
    numeric_columns = get_numeric_columns(data)
    categorical_columns = get_categorical_columns(data)
    identifier_columns = detect_identifier_columns(data)

    safe_numeric_columns = [column for column in numeric_columns if column not in identifier_columns]
    safe_categorical_columns = [column for column in categorical_columns if column not in identifier_columns]
    high_cardinality_columns = detect_high_cardinality_columns(data, safe_categorical_columns, cardinality_threshold)
    safe_categorical_columns = [column for column in safe_categorical_columns if column not in high_cardinality_columns]

    excluded_categorical_columns = [column for column in categorical_columns if column not in safe_categorical_columns]
    warnings = build_preprocessing_warnings(
        identifier_columns=identifier_columns,
        high_cardinality_columns=high_cardinality_columns,
        cardinality_threshold=cardinality_threshold,
    )
    return (
        safe_numeric_columns,
        safe_categorical_columns,
        identifier_columns,
        high_cardinality_columns,
        excluded_categorical_columns,
        warnings,
    )


def build_preprocessing_transformer(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """Build a Scikit-learn transformer for imputation, encoding, and scaling.

    The encoder keeps sparse output enabled so wide categorical spaces remain
    manageable in memory when the dataset grows.
    """
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                # with_mean=False preserves sparse compatibility when the full
                # ColumnTransformer concatenates numeric and encoded features.
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


def get_transformed_feature_names(
    transformer: ColumnTransformer,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> list[str]:
    """Return readable output column names from the fitted transformer."""
    feature_names: list[str] = []

    if numeric_columns:
        feature_names.extend(numeric_columns)

    if categorical_columns:
        encoder = transformer.named_transformers_["categorical"].named_steps["encoder"]
        encoded_names = encoder.get_feature_names_out(categorical_columns).tolist()
        feature_names.extend(encoded_names)

    return feature_names


def build_feature_matrix_dataframe(
    transformed_data,
    feature_names: list[str],
    index: pd.Index,
) -> tuple[pd.DataFrame, str]:
    """Convert transformed output into a previewable dataframe with efficient dtypes."""
    if sparse.issparse(transformed_data):
        feature_matrix = pd.DataFrame.sparse.from_spmatrix(transformed_data, index=index, columns=feature_names)
        return feature_matrix, "sparse"

    feature_matrix = pd.DataFrame(transformed_data, columns=feature_names, index=index)
    numeric_feature_columns = feature_matrix.select_dtypes(include="number").columns
    if len(numeric_feature_columns) > 0:
        feature_matrix.loc[:, numeric_feature_columns] = feature_matrix.loc[:, numeric_feature_columns].astype("float32")
    return feature_matrix, "dense"


def preprocess_dataset(
    data: pd.DataFrame,
    cardinality_threshold: int = DEFAULT_CARDINALITY_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, PreprocessingSummary, ColumnTransformer]:
    """Preprocess a dataframe and return cleaned data, feature matrix, summary, and transformer."""
    original_rows = len(data)
    original_columns = len(data.columns)
    missing_values_before = int(data.isna().sum().sum())

    # Duplicate records are removed before fitting transformers so repeated rows
    # do not distort imputation values or future model behavior.
    deduplicated_data, duplicate_rows_removed = remove_duplicate_records(data)

    (
        numeric_columns,
        categorical_columns,
        identifier_columns,
        high_cardinality_columns,
        excluded_categorical_columns,
        warning_messages,
    ) = choose_preprocessing_columns(deduplicated_data, cardinality_threshold=cardinality_threshold)

    transformer = build_preprocessing_transformer(numeric_columns, categorical_columns)

    if not numeric_columns and not categorical_columns:
        raise ValueError(
            "No numeric or safe categorical columns were found for preprocessing after identifier and cardinality checks."
        )

    transformed_array = transformer.fit_transform(deduplicated_data)
    feature_names = get_transformed_feature_names(transformer, numeric_columns, categorical_columns)
    feature_matrix, matrix_format = build_feature_matrix_dataframe(
        transformed_data=transformed_array,
        feature_names=feature_names,
        index=deduplicated_data.index,
    )
    cleaned_data = deduplicated_data.copy()

    # Missing values in the model-ready matrix should be resolved by the fitted
    # imputers. Sparse dataframes do not support the same fast count path, so we
    # use a conservative column-wise check that still works for both formats.
    missing_values_after = int(feature_matrix.isna().sum().sum())
    summary = PreprocessingSummary(
        original_rows=original_rows,
        original_columns=original_columns,
        duplicate_rows_removed=duplicate_rows_removed,
        rows_after_duplicates=len(deduplicated_data),
        missing_values_before=missing_values_before,
        missing_values_after=missing_values_after,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        identifier_columns=identifier_columns,
        excluded_categorical_columns=excluded_categorical_columns,
        high_cardinality_columns=high_cardinality_columns,
        warning_messages=warning_messages,
        cardinality_threshold=cardinality_threshold,
        encoded_feature_count=max(0, len(feature_names) - len(numeric_columns)),
        processed_rows=len(feature_matrix),
        processed_columns=len(feature_matrix.columns),
        feature_matrix_format=matrix_format,
    )

    return cleaned_data, feature_matrix, summary, transformer


def build_preprocessing_summary_table(summary: PreprocessingSummary) -> pd.DataFrame:
    """Convert preprocessing summary values into a table for display."""
    return pd.DataFrame(
        [
            ("Original rows", summary.original_rows),
            ("Original columns", summary.original_columns),
            ("Duplicate rows removed", summary.duplicate_rows_removed),
            ("Rows after duplicate removal", summary.rows_after_duplicates),
            ("Missing values before preprocessing", summary.missing_values_before),
            ("Missing values after preprocessing", summary.missing_values_after),
            ("Numeric columns normalized", len(summary.numeric_columns)),
            ("Categorical columns encoded", len(summary.categorical_columns)),
            ("Identifier columns excluded", len(summary.identifier_columns)),
            ("High-cardinality columns excluded", len(summary.high_cardinality_columns)),
            ("Cardinality threshold", summary.cardinality_threshold),
            ("Encoded categorical features created", summary.encoded_feature_count),
            ("Processed rows", summary.processed_rows),
            ("Processed columns", summary.processed_columns),
            ("Feature matrix format", summary.feature_matrix_format),
        ],
        columns=["Step", "Result"],
    )
