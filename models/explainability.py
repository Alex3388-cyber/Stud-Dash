"""Explainable AI layer using SHAP for all sklearn pipeline models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExplanationResult:
    feature_names: list[str]
    shap_values: list[float]
    base_value: float
    predicted_value: float
    top_contributors: list[dict]


def explain_prediction(model, features: pd.DataFrame) -> dict | None:
    """Return SHAP values for a prediction, or None if SHAP is unavailable."""
    try:
        import shap
    except ImportError:
        return _fallback_explanation(model, features)

    try:
        pipeline = model if hasattr(model, "named_steps") else getattr(model, "estimator", model)

        if hasattr(pipeline, "named_steps"):
            preprocessor = pipeline.named_steps.get("preprocessor")
            classifier = pipeline.named_steps.get("classifier")
        else:
            preprocessor = None
            classifier = pipeline

        if preprocessor is not None:
            X_transformed = preprocessor.transform(features)
        else:
            X_transformed = features.values

        try:
            explainer = shap.TreeExplainer(classifier)
            shap_vals = explainer.shap_values(X_transformed)
        except Exception:
            try:
                explainer = shap.LinearExplainer(classifier, X_transformed)
                shap_vals = explainer.shap_values(X_transformed)
            except Exception:
                return _fallback_explanation(model, features)

        if isinstance(shap_vals, list):
            vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        else:
            vals = shap_vals[0]

        feat_names = list(features.columns)
        vals_list = vals.tolist() if hasattr(vals, "tolist") else list(vals)

        contributors = sorted(
            [{"feature": f, "shap_value": round(v, 4)} for f, v in zip(feat_names, vals_list)],
            key=lambda x: abs(x["shap_value"]),
            reverse=True,
        )

        return {
            "feature_names": feat_names,
            "shap_values": [round(v, 4) for v in vals_list],
            "top_contributors": contributors[:5],
            "method": "shap_tree" if hasattr(classifier, "estimators_") else "shap_linear",
        }
    except Exception:
        return _fallback_explanation(model, features)


def _fallback_explanation(model, features: pd.DataFrame) -> dict:
    """Permutation-based approximate feature importance when SHAP is unavailable."""
    feat_names = list(features.columns)
    try:
        base_proba = model.predict_proba(features)[0]
        base_pass = float(base_proba[list(model.classes_).index("Pass")]) if "Pass" in list(model.classes_) else float(max(base_proba))
    except Exception:
        return {"feature_names": feat_names, "shap_values": [0.0] * len(feat_names), "top_contributors": [], "method": "unavailable"}

    contributions = []
    for i, col in enumerate(feat_names):
        perturbed = features.copy()
        perturbed[col] = 0.0
        try:
            perturbed_proba = model.predict_proba(perturbed)[0]
            perturbed_pass = float(perturbed_proba[list(model.classes_).index("Pass")]) if "Pass" in list(model.classes_) else float(max(perturbed_proba))
            contribution = base_pass - perturbed_pass
        except Exception:
            contribution = 0.0
        contributions.append(contribution)

    contributors = sorted(
        [{"feature": f, "shap_value": round(v, 4)} for f, v in zip(feat_names, contributions)],
        key=lambda x: abs(x["shap_value"]),
        reverse=True,
    )

    return {
        "feature_names": feat_names,
        "shap_values": [round(v, 4) for v in contributions],
        "top_contributors": contributors[:5],
        "method": "permutation_fallback",
    }


def build_counterfactual(
    model,
    features: pd.DataFrame,
    target_label: str = "Pass",
    steps: int = 20,
) -> list[dict]:
    """Return a simple counterfactual: which feature changes would flip the prediction."""
    suggestions = []
    try:
        current_pred = model.predict(features)[0]
        if current_pred == target_label:
            return [{"message": f"Student is already predicted as {target_label}"}]

        for col in features.columns:
            original_val = float(features[col].iloc[0])
            col_range = np.linspace(original_val * 0.5, original_val * 1.5 + 10, steps)
            for new_val in col_range:
                modified = features.copy()
                modified[col] = new_val
                pred = model.predict(modified)[0]
                if pred == target_label:
                    suggestions.append({
                        "feature": col,
                        "original_value": round(original_val, 2),
                        "suggested_value": round(new_val, 2),
                        "change": round(new_val - original_val, 2),
                    })
                    break
    except Exception:
        pass
    return suggestions[:3]
