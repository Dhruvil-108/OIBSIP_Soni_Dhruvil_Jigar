"""Model evaluation utilities for Iris flower classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


def _project_root() -> Path:
    """Return the project root directory."""

    return Path(__file__).resolve().parents[1]


def _reports_dir(config: Mapping[str, Any]) -> Path:
    """Return the configured reports directory."""

    reports_dir = _project_root() / Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def _plot_confusion_matrix(y_test: Sequence[int], y_pred: Sequence[int], config: Mapping[str, Any]) -> Path:
    """Save a confusion matrix heatmap."""

    reports_dir = _reports_dir(config)
    labels = sorted(config["data"]["label_to_species"].keys())
    class_names = [config["data"]["label_to_species"][label] for label in labels]
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    output_path = reports_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def _plot_roc_auc(y_test: Sequence[int], y_score: np.ndarray, config: Mapping[str, Any]) -> Path:
    """Save a multiclass one-vs-rest ROC-AUC curve."""

    reports_dir = _reports_dir(config)
    labels = sorted(config["data"]["label_to_species"].keys())
    y_test_binarized = label_binarize(y_test, classes=labels)

    plt.figure(figsize=(8, 6))
    for index, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, index], y_score[:, index])
        auc_score = roc_auc_score(y_test_binarized[:, index], y_score[:, index])
        plt.plot(fpr, tpr, label=f"{config['data']['label_to_species'][label]} (AUC={auc_score:.3f})")

    micro_auc = roc_auc_score(y_test_binarized, y_score, multi_class="ovr", average="macro")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"ROC-AUC Curve (macro AUC={micro_auc:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    output_path = reports_dir / "roc_auc_curve.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def _extract_feature_importance(model: Any, X_test: pd.DataFrame, y_test: Sequence[int]) -> np.ndarray:
    """Extract feature importance values using the best available strategy."""

    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_)

    if hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_)
        return np.mean(np.abs(coefficients), axis=0)

    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    return np.asarray(result.importances_mean)


def _plot_feature_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: Sequence[int],
    feature_names: Sequence[str],
    config: Mapping[str, Any],
) -> Path:
    """Save a feature importance plot for the evaluated model."""

    reports_dir = _reports_dir(config)
    importance = _extract_feature_importance(model, X_test, y_test)
    importance_frame = pd.DataFrame({"feature": list(feature_names), "importance": importance})
    importance_frame = importance_frame.sort_values(by="importance", ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=importance_frame, x="importance", y="feature", palette="viridis")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    output_path = reports_dir / "feature_importance.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Mapping[str, Any],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    """Generate evaluation metrics and plots for the best model."""

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    confusion_path = _plot_confusion_matrix(y_test, y_pred, config)
    roc_auc_path = _plot_roc_auc(y_test, y_score, config)
    feature_importance_path = _plot_feature_importance(model, X_test, y_test, feature_names, config)

    report = classification_report(
        y_test,
        y_pred,
        target_names=[config["data"]["label_to_species"][label] for label in sorted(config["data"]["label_to_species"].keys())],
        zero_division=0,
    )

    reports_dir = _reports_dir(config)
    report_path = reports_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    logger.info("Classification report:\n{}", report)
    logger.info("Saved confusion matrix to {}", confusion_path)
    logger.info("Saved ROC-AUC curve to {}", roc_auc_path)
    logger.info("Saved feature importance plot to {}", feature_importance_path)
    logger.info("Saved classification report to {}", report_path)

    return {
        "classification_report": report,
        "confusion_matrix_path": str(confusion_path),
        "roc_auc_path": str(roc_auc_path),
        "feature_importance_path": str(feature_importance_path),
        "report_path": str(report_path),
    }
