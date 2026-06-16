"""Dataset lifecycle and orchestration services."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable

import joblib
import pandas as pd
import streamlit as st

from models.classification import train_classification_models
from models.clustering import run_kmeans_clustering
from preprocessing.preprocessor import preprocess_dataset
from utilities.dataset_manager import (
    clear_classification_run,
    clear_clustering_run,
    get_auto_pipeline_report,
    get_auto_pipeline_signature,
    get_cleaned_dataset,
    get_dashboard_dataset,
    get_dataset_name,
    get_dataset_origin,
    get_dataset_signature,
    get_feature_matrix,
    get_modeling_dataset,
    get_raw_dataset,
    get_report_dataset_state,
    set_auto_pipeline_report,
    set_classification_run,
    set_clustering_run,
    set_preprocessing_artifacts,
)
from utilities.schema_mapping import normalize_lookup

_PIPELINE_CACHE_DIR = os.path.join("database", "pipeline_cache")


def _sig_cache_key(signature: object) -> str:
    return hashlib.md5(str(signature).encode()).hexdigest()


def _load_pipeline_disk_cache(signature: object) -> dict | None:
    path = os.path.join(_PIPELINE_CACHE_DIR, f"{_sig_cache_key(signature)}.pkl")
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


def _save_pipeline_disk_cache(signature: object, cache: dict) -> None:
    os.makedirs(_PIPELINE_CACHE_DIR, exist_ok=True)
    path = os.path.join(_PIPELINE_CACHE_DIR, f"{_sig_cache_key(signature)}.pkl")
    try:
        joblib.dump(cache, path)
    except Exception:
        pass


def get_active_kpi_dataset() -> tuple[pd.DataFrame, str, str]:
    """Return the dataset used for Home KPIs from the centralized dataset manager."""
    data, dataset_name, source_label = get_dashboard_dataset()
    if data is None:
        return pd.DataFrame(), "No uploaded dataset", source_label
    return data, dataset_name or "Uploaded dataset", source_label


def get_active_uploaded_dataset() -> tuple[pd.DataFrame | None, str | None]:
    """Return the current raw uploaded dataset."""
    data = get_raw_dataset()
    if data is None or data.empty:
        return None, None
    return data, get_dataset_name()


def get_active_modeling_dataset() -> tuple[pd.DataFrame | None, str | None]:
    """Return the current modeling dataset, preferring cleaned records."""
    data, dataset_name, _source_label = get_modeling_dataset()
    return data, dataset_name


def get_analysis_dataset() -> tuple[pd.DataFrame | None, str | None]:
    """Return the preferred human-readable dataset for exploration and reports."""
    data, dataset_name, _source_label = get_report_dataset_state()
    if data is not None:
        return data, dataset_name
    return None, None


def get_exploration_data_sources() -> dict[str, tuple[pd.DataFrame, str]]:
    """Build available dataset-source options for exploration."""
    data_sources: dict[str, tuple[pd.DataFrame, str]] = {}
    raw_dataset, raw_dataset_name = get_active_uploaded_dataset()
    cleaned_dataset = get_cleaned_dataset()
    feature_matrix = get_feature_matrix()

    if raw_dataset is not None and raw_dataset_name is not None:
        data_sources["Uploaded dataset"] = (raw_dataset, raw_dataset_name)
    if cleaned_dataset is not None and not cleaned_dataset.empty:
        data_sources["Cleaned dataset"] = (cleaned_dataset, f"{get_dataset_name()} (cleaned)")
    if feature_matrix is not None and not feature_matrix.empty:
        data_sources["Feature matrix"] = (feature_matrix, f"{get_dataset_name()} (feature matrix)")
    return data_sources


def get_schema_value(key: str, default=None):
    """Read one value from the active dataset schema mapping."""
    from utilities.dataset_manager import get_schema_mapping

    mapping = get_schema_mapping() or {}
    return mapping.get(key, default)


def numeric_series(data: pd.DataFrame, column: str) -> pd.Series:
    """Read a dataframe column as numeric values when possible."""
    return pd.to_numeric(data[column], errors="coerce")


def has_numeric_values(data: pd.DataFrame, column: str) -> bool:
    """Check whether a column has at least one usable numeric value."""
    return numeric_series(data, column).notna().any()


def calculate_score_pass_threshold(data: pd.DataFrame, score_columns: list[str]) -> float:
    """Infer a reasonable Pass/Fail threshold from the detected score scale."""
    if not score_columns:
        return 60.0

    score_values = data.loc[:, score_columns].apply(pd.to_numeric, errors="coerce")
    max_score = score_values.stack().max()
    if max_score == max_score and float(max_score) <= 1.5:
        return 0.6
    if max_score == max_score and float(max_score) <= 20:
        return 10.0
    return 60.0


def get_score_columns(data: pd.DataFrame) -> list[str]:
    """Return schema-mapped numeric academic score columns."""
    mapped_columns = get_schema_value("score_columns", []) or []
    return [column for column in mapped_columns if column in data.columns and has_numeric_values(data, column)]


def get_attendance_column(data: pd.DataFrame) -> str | None:
    """Return the schema-mapped attendance or absence column."""
    mapped_column = get_schema_value("attendance_column")
    if mapped_column in data.columns and has_numeric_values(data, mapped_column):
        return mapped_column
    return None


def find_existing_pass_fail_target(data: pd.DataFrame):
    """Find and normalize the schema-mapped Pass/Fail target column."""
    from models.classification import FAIL_LABEL, PASS_LABEL, normalize_pass_fail_target

    target_column = get_schema_value("target_column")
    if target_column in data.columns:
        target = normalize_pass_fail_target(data[target_column])
        valid_target = target.dropna()
        if not valid_target.empty and valid_target.isin([PASS_LABEL, FAIL_LABEL]).all():
            return target, target_column
    return None, None


def choose_home_feature_columns(
    data: pd.DataFrame,
    target_column: str | None,
    score_columns: list[str],
) -> list[str]:
    """Choose safe default feature columns for automatic classification retraining."""
    excluded_columns = set(score_columns)
    if target_column:
        excluded_columns.add(target_column)

    feature_columns: list[str] = []
    for column in data.columns:
        normalized = normalize_lookup(column)
        if column in excluded_columns:
            continue
        if normalized == "id" or normalized.endswith("id") or "studentid" in normalized:
            continue
        if has_numeric_values(data, column) or data[column].dtype.name in {"object", "category", "bool"}:
            feature_columns.append(column)
    return feature_columns


def choose_auto_classification_setup(
    data: pd.DataFrame,
) -> tuple[str | None, list[str], list[str], float]:
    """Choose sensible default target and feature columns for automatic retraining."""
    existing_target, target_column = find_existing_pass_fail_target(data)
    if target_column:
        mapped_feature_columns = get_schema_value("classification_feature_columns", []) or []
        feature_columns = [column for column in mapped_feature_columns if column in data.columns and column != target_column]
        if not feature_columns:
            feature_columns = choose_home_feature_columns(data, target_column, [])
        if not feature_columns:
            raise ValueError("No usable feature columns were detected for automatic classification retraining.")
        return target_column, [], feature_columns, 60.0

    score_columns = get_score_columns(data)
    if not score_columns:
        raise ValueError("No Pass/Fail column or score columns were detected for automatic classification retraining.")

    preferred_target_columns = get_schema_value("performance_columns", []) or []
    target_score_columns = [column for column in preferred_target_columns if column in score_columns] or [score_columns[-1]]
    mapped_feature_columns = get_schema_value("classification_feature_columns", []) or []
    feature_columns = [column for column in mapped_feature_columns if column in data.columns and column not in target_score_columns]
    if not feature_columns:
        feature_columns = choose_home_feature_columns(data, None, target_score_columns)
    if not feature_columns:
        feature_columns = [column for column in score_columns if column not in target_score_columns]
    if not feature_columns:
        raise ValueError("No usable feature columns were detected for automatic classification retraining.")

    return None, target_score_columns, feature_columns, calculate_score_pass_threshold(data, target_score_columns)


def choose_auto_clustering_setup(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Choose sensible numeric fields for automatic K-Means refresh."""
    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        raise ValueError("No numeric columns were detected for automatic clustering.")

    mapped_feature_columns = get_schema_value("clustering_feature_columns", []) or []
    feature_columns = [column for column in mapped_feature_columns if column in numeric_columns]
    if not feature_columns:
        feature_columns = numeric_columns[: min(5, len(numeric_columns))]

    mapped_performance_columns = get_schema_value("performance_columns", []) or []
    performance_columns = [column for column in mapped_performance_columns if column in numeric_columns]
    if not performance_columns:
        performance_columns = feature_columns[:1]

    if not feature_columns:
        raise ValueError("No numeric feature columns were detected for automatic clustering.")
    if not performance_columns:
        raise ValueError("No numeric performance columns were detected for automatic clustering.")

    return feature_columns, performance_columns


def run_automatic_dataset_pipeline(
    data: pd.DataFrame,
    dataset_name: str,
    on_progress: Callable[[float, str], None] | None = None,
    signature: object = None,
) -> dict[str, object]:
    """Preprocess the active dataset and retrain automatic analytics models.

    Checks a joblib disk cache keyed by dataset signature before training.
    On a cache hit, models and preprocessed data are restored instantly.
    """
    def _step(pct: float, msg: str) -> None:
        if on_progress is not None:
            on_progress(pct, msg)

    # Fast path: restore from disk cache if available
    if signature is not None:
        _step(0.05, "Checking cached analytics results…")
        cached = _load_pipeline_disk_cache(signature)
        if cached is not None:
            _step(0.40, "Cache hit — restoring analytics results…")
            try:
                set_preprocessing_artifacts(
                    cached["cleaned_dataset"],
                    cached["feature_matrix"],
                    cached["preprocessing_summary"],
                )
                if cached.get("classification_run") is not None:
                    set_classification_run(cached["classification_run"])
                if cached.get("clustering_run") is not None:
                    set_clustering_run(cached["clustering_run"])
                _step(1.00, "Analytics ready")
                return cached["report"]
            except Exception:
                pass  # Cache is corrupt — fall through and retrain

    report: dict[str, object] = {
        "dataset_name": dataset_name,
        "preprocessing": "pending",
        "classification": "pending",
        "clustering": "pending",
    }

    cleaned_dataset = data
    _step(0.05, "Reading dataset schema…")
    try:
        _step(0.10, "Step 1 of 3 — Preprocessing & cleaning dataset…")
        cleaned_dataset, feature_matrix, preprocessing_summary, _transformer = preprocess_dataset(data)
        set_preprocessing_artifacts(cleaned_dataset, feature_matrix, preprocessing_summary)
        report["preprocessing"] = "completed"
        _step(0.34, "Step 1 of 3 — Preprocessing complete")
    except Exception as error:
        report["preprocessing"] = f"skipped: {error}"
        feature_matrix = pd.DataFrame()
        preprocessing_summary = {}

    modeling_data = cleaned_dataset if isinstance(cleaned_dataset, pd.DataFrame) and not cleaned_dataset.empty else data

    classification_run = None
    try:
        _step(0.40, "Step 2 of 3 — Training classification models…")
        target_column, score_columns, feature_columns, pass_threshold = choose_auto_classification_setup(modeling_data)
        classification_run = train_classification_models(
            data=modeling_data,
            feature_columns=feature_columns,
            target_column=target_column,
            score_columns=score_columns,
            pass_threshold=pass_threshold,
            test_size=0.3,
            random_state=42,
            cv_folds=3,
        )
        set_classification_run(classification_run)
        report["classification"] = "completed"
        _step(0.67, "Step 2 of 3 — Classification models ready")
    except Exception as error:
        clear_classification_run()
        report["classification"] = f"skipped: {error}"
        _step(0.67, "Step 2 of 3 — Classification skipped")

    clustering_run = None
    try:
        _step(0.72, "Step 3 of 3 — Running K-Means student clustering…")
        clustering_features, performance_columns = choose_auto_clustering_setup(modeling_data)
        clustering_run = run_kmeans_clustering(
            data=modeling_data,
            feature_columns=clustering_features,
            performance_columns=performance_columns,
            random_state=42,
        )
        set_clustering_run(clustering_run)
        report["clustering"] = "completed"
        _step(1.00, "Pipeline complete — dashboard ready")
    except Exception as error:
        clear_clustering_run()
        report["clustering"] = f"skipped: {error}"
        _step(1.00, "Pipeline complete")

    # Persist to disk so subsequent sessions skip retraining
    if signature is not None:
        _save_pipeline_disk_cache(signature, {
            "cleaned_dataset": cleaned_dataset,
            "feature_matrix": feature_matrix,
            "preprocessing_summary": preprocessing_summary,
            "classification_run": classification_run,
            "clustering_run": clustering_run,
            "report": report,
        })

    return report


def ensure_active_dataset_pipeline(show_spinner: bool = False) -> tuple[bool, dict[str, object] | None]:
    """Run automatic preprocessing and model refresh once per uploaded dataset."""
    data, dataset_name = get_active_uploaded_dataset()
    active_signature = get_dataset_signature()
    if data is None or dataset_name is None or active_signature is None:
        return False, None

    existing_signature = get_auto_pipeline_signature()
    if existing_signature == active_signature:
        return False, get_auto_pipeline_report()

    if show_spinner:
        progress_bar = st.progress(0.0, text="Initializing analytics pipeline…")

        def _on_progress(pct: float, msg: str) -> None:
            progress_bar.progress(min(pct, 1.0), text=msg)

        report = run_automatic_dataset_pipeline(data, dataset_name, on_progress=_on_progress, signature=active_signature)
        set_auto_pipeline_report(active_signature, report)
        progress_bar.empty()
    else:
        report = run_automatic_dataset_pipeline(data, dataset_name, signature=active_signature)
        set_auto_pipeline_report(active_signature, report)

    return True, report


def get_dataset_origin_label() -> str:
    """Return a UI-friendly dataset origin label."""
    dataset_origin = get_dataset_origin()
    return "loaded from upload" if dataset_origin == "upload" else dataset_origin
