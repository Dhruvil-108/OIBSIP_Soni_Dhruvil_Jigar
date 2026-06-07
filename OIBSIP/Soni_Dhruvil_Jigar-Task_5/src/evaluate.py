"""Evaluation module for the Sales Prediction ML pipeline.

This module evaluates the trained model on the test set, computes regression
metrics, saves the metrics to a CSV report, and generates diagnostic plots
including Actual vs Predicted, Residuals, Feature Importance, and Learning Curves.
"""

import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline

from src.config import config

# Configure logger
logger = logging.getLogger(__name__)


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray, num_features: int) -> dict[str, float]:
    """Computes key regression metrics including Adjusted R2.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        num_features: Number of predictor features.

    Returns:
        dict[str, float]: Dictionary of metric names and their calculated values.
    """
    n = len(y_true)
    p = num_features

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    # Handle edge case where sample size is close to feature count
    if n - p - 1 <= 0:
        adjusted_r2 = r2
        logger.warning(
            "Sample size (%d) too small relative to feature count (%d). Adjusted R2 set to R2.",
            n, p
        )
    else:
        adjusted_r2 = float(1 - (1 - r2) * (n - 1) / (n - p - 1))

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adjusted_r2
    }
    return metrics


def plot_actual_vs_predicted(y_true: pd.Series, y_pred: np.ndarray, save_path: Path) -> None:
    """Generates and saves an Actual vs Predicted scatter plot.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        save_path: Destination file path.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7, color="#1f77b4")
    
    # Reference line for perfect prediction
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="#d62728", linestyle="--", lw=2)

    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs. Predicted Sales")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved Actual vs Predicted plot to %s", save_path)


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, save_path: Path) -> None:
    """Generates and saves a residual analysis plot.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        save_path: Destination file path.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7, color="#2ca02c")
    
    # Horizontal line at zero
    plt.axhline(y=0, color="#d62728", linestyle="--", lw=2)

    plt.xlabel("Predicted Sales")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved Residual plot to %s", save_path)


def plot_feature_importance(model: Pipeline, feature_names: list[str], save_path: Path) -> None:
    """Generates and saves feature importance or coefficient magnitude chart.

    Args:
        model: Fitted Pipeline model.
        feature_names: List of feature names.
        save_path: Destination file path.
    """
    regressor = model.named_steps["regressor"]
    importances = None
    title = "Feature Importance"

    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
    elif hasattr(regressor, "coef_"):
        importances = np.abs(regressor.coef_)
        title = "Feature Coefficients (Magnitude)"

    if importances is None:
        logger.warning("Estimator does not support feature importances or coefficients.")
        return

    # Create Series and sort
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(feat_imp))
    feat_imp.plot(kind="barh", color=colors)

    plt.xlabel("Importance Score" if "Importance" in title else "Coefficient Magnitude")
    plt.ylabel("Features")
    plt.title(title)
    plt.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved Feature Importance plot to %s", save_path)


def plot_learning_curve(model: Pipeline, X: pd.DataFrame, y: pd.Series, save_path: Path) -> None:
    """Generates and saves the model learning curve.

    Args:
        model: Model Pipeline to evaluate.
        X: Feature matrix.
        y: Target variable.
        save_path: Destination file path.
    """
    plt.figure(figsize=(8, 6))
    
    # 5-fold CV learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=config.CV_FOLDS, scoring="r2", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), random_state=config.RANDOM_STATE
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, "o-", color="#1f77b4", label="Training score")
    plt.plot(train_sizes, test_mean, "o-", color="#ff7f0e", label="Cross-validation score")

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="#1f77b4")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="#ff7f0e")

    plt.xlabel("Training Examples")
    plt.ylabel("R2 Score")
    plt.title("Model Learning Curve")
    plt.legend(loc="best")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved Learning Curve plot to %s", save_path)


def evaluate_model(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Runs the complete model evaluation suite.

    Args:
        model: Fitted model Pipeline.
        X_train: Training feature DataFrame.
        y_train: Training target variable.
        X_test: Test feature DataFrame.
        y_test: Test target variable.

    Returns:
        dict[str, float]: Calculated metrics dictionary.
    """
    logger.info("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    num_features = X_test.shape[1]

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, num_features)
    logger.info("Metrics: MAE=%.4f, MSE=%.4f, RMSE=%.4f, R2=%.4f, Adj R2=%.4f",
                metrics["MAE"], metrics["MSE"], metrics["RMSE"], metrics["R2"], metrics["Adjusted_R2"])

    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_csv_path = config.REPORTS_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info("Saved model metrics to %s", metrics_csv_path)

    # Ensure figures directory exists
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_actual_vs_predicted(y_test, y_pred, config.FIGURES_DIR / "actual_vs_predicted.png")
    plot_residuals(y_test, y_pred, config.FIGURES_DIR / "residual_plot.png")
    plot_feature_importance(model, list(X_test.columns), config.FIGURES_DIR / "feature_importance.png")
    plot_learning_curve(model, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), config.FIGURES_DIR / "learning_curve.png")

    return metrics


if __name__ == "__main__":
    import joblib
    from src.model import train_and_tune_pipeline
    
    logging.basicConfig(level=logging.INFO)
    try:
        # Load best model and data splits if already ran, or run training pipeline
        model_path = config.MODEL_SAVE_PATH / "best_model.joblib"
        if not model_path.exists():
            logger.info("No saved model found. Training model now...")
            model, X_train, y_train, X_test, y_test = train_and_tune_pipeline()
        else:
            logger.info("Loading saved model pipeline...")
            model = joblib.load(model_path)
            # Reload dataset to recreate splits
            from src.data_loader import load_dataset
            from src.model import prepare_data
            df = load_dataset()
            X_train, X_test, y_train, y_test, _, _ = prepare_data(df)
            
        evaluate_model(model, X_train, y_train, X_test, y_test)
        print("\nEvaluation successfully completed.")
    except Exception as e:
        logger.error("Error running standalone evaluation: %s", e, exc_info=True)
