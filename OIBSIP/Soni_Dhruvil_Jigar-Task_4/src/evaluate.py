"""Evaluation module for the Email Spam Detection System.

This module provides functions to calculate accuracy, precision, recall,
F1, and ROC-AUC metrics, output the classification report, and generate
plots (confusion matrix and ROC curve) using a non-interactive backend.
"""

import logging
from pathlib import Path
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)
from src.config import Config

logger = logging.getLogger(__name__)


def evaluate_model(
    config: Config,
    model: Any,
    X_test: pd.Series,
    y_test: pd.Series,
    vectorizer: Any
) -> Dict[str, float]:
    """Evaluates the model on test data, prints reports, and generates plots.

    Computes accuracy, precision, recall, F1, and ROC-AUC. Saves classification
    report text and plot images to the reports/ directory.

    Args:
        config (Config): System configuration.
        model (Any): The fitted classifier model.
        X_test (pd.Series): The test set text.
        y_test (pd.Series): The test set labels ('ham' or 'spam').
        vectorizer (Any): The fitted vectorizer.

    Returns:
        Dict[str, float]: Dictionary of metric scores.
    """
    logger.info("Starting model evaluation...")
    
    # Ensure reports directory exists
    reports_dir: Path = config.REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Encode labels to 0 and 1
    y_test_encoded = y_test.map({"ham": 0, "spam": 1})

    # Vectorize test text
    logger.info("Transforming test data...")
    X_test_transformed = vectorizer.transform(X_test)

    # Predictions
    y_pred = model.predict(X_test_transformed)
    
    # Check if model supports predict_proba (it should)
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
    else:
        logger.warning("Model does not support predict_proba; using decision function or fallback.")
        if hasattr(model, "decision_function"):
            y_pred_proba = model.decision_function(X_test_transformed)
        else:
            y_pred_proba = y_pred.astype(float)

    # Calculate metrics
    metrics = {
        "Accuracy": float(accuracy_score(y_test_encoded, y_pred)),
        "Precision": float(precision_score(y_test_encoded, y_pred)),
        "Recall": float(recall_score(y_test_encoded, y_pred)),
        "F1": float(f1_score(y_test_encoded, y_pred)),
        "ROC-AUC": float(roc_auc_score(y_test_encoded, y_pred_proba)),
    }

    # Print and log classification report
    report_str = classification_report(
        y_test_encoded,
        y_pred,
        target_names=["ham", "spam"]
    )
    logger.info(f"\nClassification Report:\n{report_str}")

    # Save classification report to file
    report_file = reports_dir / "classification_report.txt"
    logger.info(f"Saving classification report text to {report_file}...")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== Classification Metrics ===\n")
        for metric_name, val in metrics.items():
            f.write(f"{metric_name}: {val:.4f}\n")
        f.write("\n=== Detailed Classification Report ===\n")
        f.write(report_str)

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["ham", "spam"],
        yticklabels=["ham", "spam"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = reports_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {cm_path}")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test_encoded, y_pred_proba)
    auc_score = metrics["ROC-AUC"]
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = reports_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()
    logger.info(f"ROC curve plot saved to {roc_path}")

    return metrics


if __name__ == "__main__":
    print("Evaluation module loaded successfully.")
