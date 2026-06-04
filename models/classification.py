"""Pass/Fail classification models for the Student Performance Prediction Dashboard.

This module upgrades the dashboard's classification workflow to a more
professional academic standard by adding:

- stratified train/test splitting
- leakage-safe preprocessing inside Scikit-learn pipelines
- training-set cross-validation
- probability calibration for final holdout evaluation
- feature importance analysis
- richer evaluation diagnostics and summary outputs
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


PASS_FAIL_TARGET = "Pass/Fail"
PASS_LABEL = "Pass"
FAIL_LABEL = "Fail"
CLASS_LABELS = [FAIL_LABEL, PASS_LABEL]


@dataclass(frozen=True)
class ClassificationResult:
    """Evaluation output for one trained classifier."""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float | None
    average_precision: float | None
    log_loss: float | None
    brier_score: float | None
    confusion_matrix: pd.DataFrame
    cross_validation_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    roc_curve: pd.DataFrame
    precision_recall_curve: pd.DataFrame
    calibration_curve: pd.DataFrame
    probability_calibrated: bool
    calibration_note: str
    cross_validation_note: str
    summary: str


@dataclass(frozen=True)
class ClassificationRun:
    """Complete output from a professional classification run."""

    target_source: str
    preprocessing_summary: str
    feature_columns: list[str]
    removed_leakage_columns: list[str]
    numeric_feature_columns: list[str]
    categorical_feature_columns: list[str]
    train_rows: int
    test_rows: int
    class_distribution: pd.DataFrame
    cv_folds: int | None
    calibration_folds: int | None
    performance_summary: str
    results: list[ClassificationResult]


def find_pass_fail_column(data: pd.DataFrame) -> str | None:
    """Find a Pass/Fail target column even if casing or separators differ."""
    normalized_lookup = {
        column.strip().lower().replace("_", "").replace(" ", "").replace("/", ""): column
        for column in data.columns
    }
    return normalized_lookup.get("passfail")


def normalize_pass_fail_target(target: pd.Series) -> pd.Series:
    """Normalize common Pass/Fail values into consistent labels."""
    normalized = target.astype("string").str.strip().str.lower()
    pass_values = {"pass", "passed", "yes", "y", "true", "1", "successful"}
    fail_values = {"fail", "failed", "no", "n", "false", "0", "unsuccessful"}

    mapped = normalized.map(
        lambda value: PASS_LABEL if value in pass_values else FAIL_LABEL if value in fail_values else pd.NA
    )
    return mapped.astype("string")


def create_pass_fail_target_from_scores(
    data: pd.DataFrame,
    score_columns: list[str],
    pass_threshold: float,
) -> pd.Series:
    """Create a Pass/Fail target by averaging selected score columns."""
    if not score_columns:
        raise ValueError("Select at least one score column to derive the Pass/Fail target.")

    # These score columns define the target and therefore must not be reused as
    # features unless we explicitly remove them later to prevent leakage.
    score_values = data.loc[:, score_columns].apply(pd.to_numeric, errors="coerce")
    if score_values.dropna(how="all").empty:
        raise ValueError("The selected score columns do not contain usable numeric values for Pass/Fail derivation.")

    average_score = score_values.mean(axis=1)
    return average_score.ge(pass_threshold).map({True: PASS_LABEL, False: FAIL_LABEL}).astype("string")


def normalize_column_name(name: str) -> str:
    """Normalize a dataframe column name for flexible matching."""
    return str(name).strip().lower().replace(" ", "").replace("_", "").replace("-", "").replace("/", "")


def find_final_grade_column(data: pd.DataFrame) -> str | None:
    """Find a likely final-grade column when present."""
    normalized_columns = {normalize_column_name(column): column for column in data.columns}
    for candidate in ["g3", "finalgrade", "finalscore", "finalmark"]:
        if candidate in normalized_columns:
            return normalized_columns[candidate]
    return None


def choose_target_score_columns(data: pd.DataFrame, score_columns: list[str]) -> list[str]:
    """Use a final-grade column as target when possible, otherwise use selected scores."""
    final_grade_column = find_final_grade_column(data)
    if final_grade_column and final_grade_column in score_columns:
        return [final_grade_column]
    return score_columns


def sanitize_feature_columns(
    feature_columns: list[str],
    target_column: str | None,
    target_score_columns: list[str],
) -> tuple[list[str], list[str]]:
    """Remove columns that would leak direct target information into the model.

    If Pass/Fail is derived from one or more score columns, those exact columns
    are removed from the feature set automatically even if the user selected
    them. This keeps the evaluation academically honest.
    """
    leakage_columns = set(target_score_columns)
    if target_column:
        leakage_columns.add(target_column)

    cleaned_columns: list[str] = []
    removed_columns: list[str] = []
    for column in feature_columns:
        if column in leakage_columns:
            removed_columns.append(column)
            continue
        if column not in cleaned_columns:
            cleaned_columns.append(column)
    return cleaned_columns, removed_columns


def get_supported_feature_columns(data: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature columns supported by the model pipeline."""
    selected_data = data.loc[:, feature_columns]
    numeric_columns = selected_data.select_dtypes(include="number").columns.tolist()
    categorical_columns = selected_data.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    return numeric_columns, categorical_columns


def build_classification_pipeline(
    classifier,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    """Build a preprocessing + classifier pipeline.

    The preprocessing transformer remains inside the pipeline so each training
    fold learns imputation, encoding, and scaling rules from its own training
    subset only. This is the main guard against preprocessing leakage.
    """
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def build_calibrated_model(
    classifier,
    numeric_columns: list[str],
    categorical_columns: list[str],
    calibration_folds: int,
    random_state: int,
) -> CalibratedClassifierCV:
    """Build a calibrated classifier for final holdout evaluation.

    Sigmoid calibration is used here because it is more stable than isotonic
    calibration on the smaller academic datasets commonly uploaded to the app.
    """
    base_pipeline = build_classification_pipeline(classifier, numeric_columns, categorical_columns)
    calibration_strategy = StratifiedKFold(
        n_splits=calibration_folds,
        shuffle=True,
        random_state=random_state,
    )
    return CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=calibration_strategy,
    )


def validate_classification_target(target: pd.Series, test_size: float) -> None:
    """Ensure the target has enough Pass and Fail examples for training/testing."""
    class_counts = target.value_counts()
    if len(class_counts) < 2:
        raise ValueError("The target must contain both Pass and Fail records.")

    if class_counts.min() < 2:
        raise ValueError("Each class needs at least two records for a train/test split.")

    test_rows = ceil(len(target) * test_size)
    train_rows = len(target) - test_rows
    if test_rows < len(class_counts) or train_rows < len(class_counts):
        raise ValueError(
            "Increase the dataset size or reduce the test split so both classes appear in train and test data."
        )


def choose_cv_folds(target: pd.Series, requested_folds: int) -> int | None:
    """Choose a valid stratified CV fold count for the current training target."""
    minority_class_size = int(target.value_counts().min())
    if minority_class_size < 2:
        return None
    return min(requested_folds, minority_class_size)


def encode_binary_target(target: pd.Series | np.ndarray | list[str]) -> np.ndarray:
    """Convert Pass/Fail labels into binary values for probability metrics."""
    encoded = pd.Series(target, dtype="string").map({FAIL_LABEL: 0, PASS_LABEL: 1})
    if encoded.isna().any():
        raise ValueError("Target values must be normalized to Pass/Fail before evaluation.")
    return encoded.astype(int).to_numpy()


def reorder_probability_matrix(
    probabilities: np.ndarray,
    classes: np.ndarray | list[str],
    ordered_labels: list[str],
) -> np.ndarray:
    """Reorder a probability matrix to match the requested class-label order."""
    class_lookup = {label: index for index, label in enumerate(classes)}
    ordered_indices = [class_lookup[label] for label in ordered_labels]
    return probabilities[:, ordered_indices]


def extract_positive_class_probabilities(
    probabilities: np.ndarray,
    classes: np.ndarray | list[str],
) -> np.ndarray:
    """Return the predicted probability for the positive Pass class."""
    class_lookup = {label: index for index, label in enumerate(classes)}
    if PASS_LABEL not in class_lookup:
        raise ValueError("The calibrated classifier did not expose a Pass probability column.")
    return probabilities[:, class_lookup[PASS_LABEL]]


def positive_probability_from_estimator(estimator, features: pd.DataFrame) -> np.ndarray:
    """Read positive-class probabilities from any fitted estimator with predict_proba."""
    probabilities = estimator.predict_proba(features)
    return extract_positive_class_probabilities(probabilities, estimator.classes_)


def negative_log_loss_scorer(estimator, features: pd.DataFrame, target: pd.Series) -> float:
    """Cross-validation scorer for log loss where lower loss is better."""
    probabilities = estimator.predict_proba(features)
    ordered_probabilities = reorder_probability_matrix(probabilities, estimator.classes_, CLASS_LABELS)
    return -log_loss(target, ordered_probabilities, labels=CLASS_LABELS)


def negative_brier_score_scorer(estimator, features: pd.DataFrame, target: pd.Series) -> float:
    """Cross-validation scorer for the Brier score on Pass probabilities."""
    return -brier_score_loss(encode_binary_target(target), positive_probability_from_estimator(estimator, features))


def roc_auc_scorer(estimator, features: pd.DataFrame, target: pd.Series) -> float:
    """Cross-validation scorer for ROC-AUC using the Pass probability."""
    return roc_auc_score(encode_binary_target(target), positive_probability_from_estimator(estimator, features))


def average_precision_scorer(estimator, features: pd.DataFrame, target: pd.Series) -> float:
    """Cross-validation scorer for PR-AUC using the Pass probability."""
    return average_precision_score(encode_binary_target(target), positive_probability_from_estimator(estimator, features))


def build_cross_validation_scoring() -> dict[str, object]:
    """Build leakage-safe cross-validation scorers for model comparison."""
    return {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, pos_label=PASS_LABEL, zero_division=0),
        "recall": make_scorer(recall_score, pos_label=PASS_LABEL, zero_division=0),
        "f1": make_scorer(f1_score, pos_label=PASS_LABEL, zero_division=0),
        "roc_auc": roc_auc_scorer,
        "average_precision": average_precision_scorer,
        "neg_log_loss": negative_log_loss_scorer,
        "neg_brier_score": negative_brier_score_scorer,
    }


def build_cross_validation_summary(
    estimator,
    features: pd.DataFrame,
    target: pd.Series,
    cv_folds: int | None,
    random_state: int,
) -> pd.DataFrame:
    """Run cross-validation on the training split and summarize the results.

    Cross-validation is run only on the training partition. The test set stays
    untouched until the final holdout evaluation step.
    """
    if cv_folds is None or cv_folds < 2:
        return pd.DataFrame(columns=["Metric", "Mean", "Std"])

    strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_validate(
        estimator,
        features,
        target,
        cv=strategy,
        scoring=build_cross_validation_scoring(),
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )

    metric_map = [
        ("test_accuracy", "Accuracy", False),
        ("test_precision", "Precision", False),
        ("test_recall", "Recall", False),
        ("test_f1", "F1-score", False),
        ("test_roc_auc", "ROC-AUC", False),
        ("test_average_precision", "PR-AUC", False),
        ("test_neg_log_loss", "Log Loss", True),
        ("test_neg_brier_score", "Brier Score", True),
    ]

    rows = []
    for key, label, negate in metric_map:
        values = scores[key]
        mean_value = float(np.mean(values))
        std_value = float(np.std(values, ddof=0))
        if negate:
            mean_value = -mean_value
        rows.append({"Metric": label, "Mean": mean_value, "Std": std_value})

    rows.append({"Metric": "Fit Time (s)", "Mean": float(np.mean(scores["fit_time"])), "Std": float(np.std(scores["fit_time"], ddof=0))})
    rows.append({"Metric": "Score Time (s)", "Mean": float(np.mean(scores["score_time"])), "Std": float(np.std(scores["score_time"], ddof=0))})
    return pd.DataFrame(rows)


def get_cross_validation_metric_summary(
    cross_validation_metrics: pd.DataFrame,
    metric_name: str,
) -> tuple[float | None, float | None]:
    """Return a CV metric mean/std pair from a summary dataframe."""
    metric_row = cross_validation_metrics.loc[cross_validation_metrics["Metric"] == metric_name]
    if metric_row.empty:
        return None, None
    return float(metric_row["Mean"].iloc[0]), float(metric_row["Std"].iloc[0])


def choose_calibration_bins(sample_count: int) -> int:
    """Choose a sensible number of calibration bins for the test-set size."""
    if sample_count < 12:
        return 3
    if sample_count < 25:
        return 4
    return min(8, max(5, sample_count // 10))


def build_roc_curve_table(target: pd.Series, pass_probabilities: np.ndarray) -> pd.DataFrame:
    """Build ROC curve coordinates for plotting."""
    false_positive_rate, true_positive_rate, thresholds = roc_curve(encode_binary_target(target), pass_probabilities)
    return pd.DataFrame(
        {
            "False Positive Rate": false_positive_rate,
            "True Positive Rate": true_positive_rate,
            "Threshold": thresholds,
        }
    )


def build_precision_recall_curve_table(target: pd.Series, pass_probabilities: np.ndarray) -> pd.DataFrame:
    """Build precision-recall curve coordinates for plotting."""
    precision_values, recall_values, thresholds = precision_recall_curve(encode_binary_target(target), pass_probabilities)
    threshold_values = np.append(thresholds, np.nan)
    return pd.DataFrame(
        {
            "Recall": recall_values,
            "Precision": precision_values,
            "Threshold": threshold_values,
        }
    )


def build_calibration_curve_table(target: pd.Series, pass_probabilities: np.ndarray) -> pd.DataFrame:
    """Build calibration-curve coordinates for plotting."""
    probability_true, probability_pred = calibration_curve(
        encode_binary_target(target),
        pass_probabilities,
        n_bins=choose_calibration_bins(len(target)),
        strategy="quantile",
    )
    return pd.DataFrame(
        {
            "Mean Predicted Probability": probability_pred,
            "Observed Pass Rate": probability_true,
        }
    )


def build_feature_importance_table(
    estimator,
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int,
) -> pd.DataFrame:
    """Estimate feature importance with permutation on the holdout set.

    Permutation importance is model-agnostic, which means the dashboard can use
    one consistent interpretation method for Logistic Regression, Decision Tree,
    and Random Forest.
    """
    if features.empty:
        return pd.DataFrame(columns=["Feature", "Importance", "Importance Std"])

    importance = permutation_importance(
        estimator,
        features,
        target,
        scoring=make_scorer(f1_score, pos_label=PASS_LABEL, zero_division=0),
        n_repeats=8,
        random_state=random_state,
        n_jobs=1,
    )
    frame = pd.DataFrame(
        {
            "Feature": features.columns,
            "Importance": importance.importances_mean,
            "Importance Std": importance.importances_std,
        }
    )
    return frame.sort_values(["Importance", "Feature"], ascending=[False, True]).reset_index(drop=True)


def build_model_summary(
    model_name: str,
    accuracy: float,
    f1_value: float,
    roc_auc: float | None,
    brier_score_value: float | None,
    cross_validation_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    probability_calibrated: bool,
    calibration_note: str,
    cross_validation_note: str,
) -> str:
    """Generate a short academic performance summary for one model."""
    cv_f1_mean, cv_f1_std = get_cross_validation_metric_summary(cross_validation_metrics, "F1-score")
    top_feature = None
    if not feature_importance.empty and float(feature_importance["Importance"].iloc[0]) > 0:
        top_feature = str(feature_importance["Feature"].iloc[0])

    summary_parts = [
        f"Holdout accuracy was {accuracy:.1%} with an F1-score of {f1_value:.1%}.",
    ]
    if roc_auc is not None:
        summary_parts.append(f"ROC-AUC reached {roc_auc:.1%}.")
    if cv_f1_mean is not None and cv_f1_std is not None:
        summary_parts.append(f"Training-set cross-validation F1 averaged {cv_f1_mean:.1%} (+/- {cv_f1_std:.1%}).")
    elif cross_validation_note:
        summary_parts.append(cross_validation_note)
    if brier_score_value is not None:
        summary_parts.append(f"The calibrated probability Brier score was {brier_score_value:.3f}.")
    if calibration_note:
        summary_parts.append(calibration_note)
    if top_feature:
        summary_parts.append(f"The most influential holdout feature was `{top_feature}` by permutation importance.")
    return f"{model_name}: " + " ".join(summary_parts)


def evaluate_predictions(
    model_name: str,
    target: pd.Series,
    predictions: pd.Series,
    probabilities: np.ndarray,
    classes: np.ndarray | list[str],
    cross_validation_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    probability_calibrated: bool,
    calibration_note: str,
    cross_validation_note: str,
) -> ClassificationResult:
    """Calculate metrics and evaluation artifacts for one calibrated model."""
    ordered_probabilities = reorder_probability_matrix(probabilities, classes, CLASS_LABELS)
    pass_probabilities = extract_positive_class_probabilities(probabilities, classes)
    encoded_target = encode_binary_target(target)

    matrix = confusion_matrix(target, predictions, labels=CLASS_LABELS)
    matrix_table = pd.DataFrame(
        matrix,
        index=[f"Actual {label}" for label in CLASS_LABELS],
        columns=[f"Predicted {label}" for label in CLASS_LABELS],
    )

    accuracy = accuracy_score(target, predictions)
    precision = precision_score(target, predictions, pos_label=PASS_LABEL, zero_division=0)
    recall = recall_score(target, predictions, pos_label=PASS_LABEL, zero_division=0)
    f1_value = f1_score(target, predictions, pos_label=PASS_LABEL, zero_division=0)
    roc_auc = roc_auc_score(encoded_target, pass_probabilities)
    average_precision = average_precision_score(encoded_target, pass_probabilities)
    test_log_loss = log_loss(target, ordered_probabilities, labels=CLASS_LABELS)
    brier = brier_score_loss(encoded_target, pass_probabilities)

    roc_curve_table = build_roc_curve_table(target, pass_probabilities)
    pr_curve_table = build_precision_recall_curve_table(target, pass_probabilities)
    calibration_table = build_calibration_curve_table(target, pass_probabilities)
    summary = build_model_summary(
        model_name=model_name,
        accuracy=accuracy,
        f1_value=f1_value,
        roc_auc=roc_auc,
        brier_score_value=brier,
        cross_validation_metrics=cross_validation_metrics,
        feature_importance=feature_importance,
        probability_calibrated=probability_calibrated,
        calibration_note=calibration_note,
        cross_validation_note=cross_validation_note,
    )

    return ClassificationResult(
        model_name=model_name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_value,
        roc_auc=roc_auc,
        average_precision=average_precision,
        log_loss=test_log_loss,
        brier_score=brier,
        confusion_matrix=matrix_table,
        cross_validation_metrics=cross_validation_metrics,
        feature_importance=feature_importance,
        roc_curve=roc_curve_table,
        precision_recall_curve=pr_curve_table,
        calibration_curve=calibration_table,
        probability_calibrated=probability_calibrated,
        calibration_note=calibration_note,
        cross_validation_note=cross_validation_note,
        summary=summary,
    )


def build_overall_performance_summary(results: list[ClassificationResult]) -> str:
    """Generate a project-level comparison summary across all trained models."""
    if not results:
        return "No classification models were trained."

    best_holdout_model = max(results, key=lambda result: (result.f1_score, result.accuracy))
    best_calibrated_model = min(
        results,
        key=lambda result: result.brier_score if result.brier_score is not None else float("inf"),
    )

    best_cv_model = None
    best_cv_f1 = float("-inf")
    for result in results:
        cv_f1_mean, _cv_f1_std = get_cross_validation_metric_summary(result.cross_validation_metrics, "F1-score")
        if cv_f1_mean is not None and cv_f1_mean > best_cv_f1:
            best_cv_f1 = cv_f1_mean
            best_cv_model = result

    summary = (
        f"{best_holdout_model.model_name} achieved the strongest holdout F1-score "
        f"({best_holdout_model.f1_score:.1%}) on unseen test data. "
    )
    if best_cv_model is not None and best_cv_f1 > float("-inf"):
        summary += (
            f"{best_cv_model.model_name} produced the best cross-validation F1 estimate "
            f"({best_cv_f1:.1%} on training folds). "
        )
    else:
        summary += "Cross-validation stability estimates were unavailable for this dataset size. "
    summary += (
        f"{best_calibrated_model.model_name} returned the most reliable calibrated probabilities "
        f"with the lowest Brier score ({best_calibrated_model.brier_score:.3f})."
    )
    return summary


def train_classification_models(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str | None = None,
    score_columns: list[str] | None = None,
    pass_threshold: float = 60.0,
    test_size: float = 0.3,
    random_state: int = 42,
    cv_folds: int = 5,
) -> ClassificationRun:
    """Train calibrated classifiers and compare them with professional diagnostics."""
    if target_column:
        target = normalize_pass_fail_target(data[target_column])
        target_source = target_column
        target_score_columns: list[str] = []
    else:
        target_score_columns = choose_target_score_columns(data, score_columns or [])
        target = create_pass_fail_target_from_scores(data, target_score_columns, pass_threshold)
        target_source = f"Derived from {', '.join(target_score_columns)} at threshold {pass_threshold:g}"

    safe_feature_columns, removed_leakage_columns = sanitize_feature_columns(
        feature_columns=feature_columns,
        target_column=target_column,
        target_score_columns=target_score_columns,
    )
    if not safe_feature_columns:
        removed_list = ", ".join(removed_leakage_columns) if removed_leakage_columns else "the selected features"
        raise ValueError(
            f"No valid feature columns remain after leakage prevention removed {removed_list}. "
            "Choose predictor columns that do not directly define the target."
        )

    modeling_data = data.loc[:, safe_feature_columns].copy()
    modeling_data[PASS_FAIL_TARGET] = target

    # Invalid target labels are removed before splitting. Feature missing values
    # remain in place and are learned by the training pipeline later.
    modeling_data = modeling_data.dropna(subset=[PASS_FAIL_TARGET]).reset_index(drop=True)
    target = modeling_data[PASS_FAIL_TARGET]
    features = modeling_data.loc[:, safe_feature_columns]

    numeric_columns, categorical_columns = get_supported_feature_columns(features, safe_feature_columns)
    supported_columns = numeric_columns + categorical_columns
    if not supported_columns:
        raise ValueError("Select at least one numeric or categorical feature column.")

    features = features.loc[:, supported_columns]
    preprocessing_summary = (
        "Each model uses a leakage-safe preprocessing pipeline: numeric fields are "
        "median-imputed and scaled, categorical fields are filled with 'Missing' and one-hot encoded. "
        "Cross-validation is run only on the training split, and probability calibration is fitted on "
        "training data only before final holdout testing."
    )

    validate_classification_target(target, test_size)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    training_minority_class = int(y_train.value_counts().min())
    active_cv_folds = choose_cv_folds(y_train, requested_folds=cv_folds)
    calibration_folds = min(3, training_minority_class) if training_minority_class >= 2 else None

    model_definitions = [
        (
            "Decision Tree Classifier",
            DecisionTreeClassifier(random_state=random_state, max_depth=5, min_samples_leaf=4),
        ),
        (
            "Logistic Regression",
            LogisticRegression(max_iter=2000, random_state=random_state),
        ),
        (
            "Random Forest Classifier",
            RandomForestClassifier(
                n_estimators=250,
                random_state=random_state,
                min_samples_leaf=2,
                n_jobs=-1,
            ),
        ),
    ]

    results: list[ClassificationResult] = []
    for model_name, classifier in model_definitions:
        base_pipeline = build_classification_pipeline(clone(classifier), numeric_columns, categorical_columns)
        cross_validation_metrics = build_cross_validation_summary(
            estimator=base_pipeline,
            features=x_train,
            target=y_train,
            cv_folds=active_cv_folds,
            random_state=random_state,
        )
        if active_cv_folds is None:
            cross_validation_note = (
                "Cross-validation was skipped because the training split does not contain enough records per class."
            )
        else:
            cross_validation_note = ""

        if calibration_folds is not None and calibration_folds >= 2:
            final_model = build_calibrated_model(
                classifier=clone(classifier),
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                calibration_folds=calibration_folds,
                random_state=random_state,
            )
            final_model.fit(x_train, y_train)
            probability_calibrated = True
            calibration_note = (
                f"Pass probabilities were calibrated with sigmoid calibration across {calibration_folds} training folds."
            )
        else:
            final_model = base_pipeline
            final_model.fit(x_train, y_train)
            probability_calibrated = False
            calibration_note = (
                "Probability calibration was skipped because the training split is too small to support a stratified calibration procedure."
            )

        predictions = pd.Series(final_model.predict(x_test), index=y_test.index, name="Prediction").astype("string")
        probabilities = final_model.predict_proba(x_test)

        try:
            feature_importance = build_feature_importance_table(
                estimator=final_model,
                features=x_test,
                target=y_test,
                random_state=random_state,
            )
        except Exception:
            feature_importance = pd.DataFrame(columns=["Feature", "Importance", "Importance Std"])

        results.append(
            evaluate_predictions(
                model_name=model_name,
                target=y_test,
                predictions=predictions,
                probabilities=probabilities,
                classes=final_model.classes_,
                cross_validation_metrics=cross_validation_metrics,
                feature_importance=feature_importance,
                probability_calibrated=probability_calibrated,
                calibration_note=calibration_note,
                cross_validation_note=cross_validation_note,
            )
        )

    class_distribution = target.value_counts().rename_axis("Class").reset_index(name="Records")
    performance_summary = build_overall_performance_summary(results)
    return ClassificationRun(
        target_source=target_source,
        preprocessing_summary=preprocessing_summary,
        feature_columns=supported_columns,
        removed_leakage_columns=removed_leakage_columns,
        numeric_feature_columns=numeric_columns,
        categorical_feature_columns=categorical_columns,
        train_rows=len(x_train),
        test_rows=len(x_test),
        class_distribution=class_distribution,
        cv_folds=active_cv_folds,
        calibration_folds=calibration_folds,
        performance_summary=performance_summary,
        results=results,
    )


def format_metric(mean_value: float | None, std_value: float | None = None) -> str:
    """Format a metric value, optionally as mean +/- std."""
    if mean_value is None:
        return "N/A"
    if std_value is None:
        return f"{mean_value:.4f}"
    return f"{mean_value:.4f} +/- {std_value:.4f}"


def build_metrics_table(results: list[ClassificationResult]) -> pd.DataFrame:
    """Convert model results into a professional comparison table."""
    rows = []
    for result in results:
        cv_accuracy_mean, cv_accuracy_std = get_cross_validation_metric_summary(result.cross_validation_metrics, "Accuracy")
        cv_f1_mean, cv_f1_std = get_cross_validation_metric_summary(result.cross_validation_metrics, "F1-score")
        cv_roc_auc_mean, cv_roc_auc_std = get_cross_validation_metric_summary(result.cross_validation_metrics, "ROC-AUC")
        rows.append(
            {
                "Model": result.model_name,
                "Accuracy": round(result.accuracy, 4),
                "Precision": round(result.precision, 4),
                "Recall": round(result.recall, 4),
                "F1-score": round(result.f1_score, 4),
                "ROC-AUC": round(result.roc_auc, 4) if result.roc_auc is not None else None,
                "PR-AUC": round(result.average_precision, 4) if result.average_precision is not None else None,
                "Log Loss": round(result.log_loss, 4) if result.log_loss is not None else None,
                "Brier Score": round(result.brier_score, 4) if result.brier_score is not None else None,
                "CV Accuracy": format_metric(cv_accuracy_mean, cv_accuracy_std),
                "CV F1-score": format_metric(cv_f1_mean, cv_f1_std),
                "CV ROC-AUC": format_metric(cv_roc_auc_mean, cv_roc_auc_std),
            }
        )
    return pd.DataFrame(rows)
