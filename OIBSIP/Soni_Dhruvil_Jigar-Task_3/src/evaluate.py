"""Evaluation module for the Car Price Prediction ML system.

This module evaluates all trained models, logs regression metrics (MAE, MSE, RMSE,
R², Adjusted R²), saves them to reports/model_metrics.csv, and generates figures
including actual vs. predicted, residuals, feature importance, learning curve,
and cross-model R² comparisons.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from src.config import config

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray, num_features: int) -> Dict[str, float]:
    """Calculates MAE, MSE, RMSE, R2, and Adjusted R2.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        num_features: Number of features used in the model to compute Adjusted R2.

    Returns:
        A dictionary containing all calculated metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    n = len(y_true)
    p = num_features
    
    # Calculate Adjusted R2, handle division by zero case
    if n - p - 1 > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        adj_r2 = r2
        logger.warning("Adjusted R² denominator <= 0. Set Adjusted R² equal to R².")
        
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adj_r2
    }


def plot_actual_vs_predicted(y_true: pd.Series, y_pred: np.ndarray, save_path: Path) -> None:
    """Generates and saves actual vs predicted scatter plot.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        save_path: Location to save the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7, color="#1a73e8")
    
    # Diagonal baseline
    limits = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(limits, limits, color="#ea4335", linestyle="--", linewidth=2, label="Perfect Fit")
    
    plt.xlabel("Actual Price (Lakhs)", fontsize=11)
    plt.ylabel("Predicted Price (Lakhs)", fontsize=11)
    plt.title("Actual vs Predicted Car Resale Prices", fontsize=13, fontweight="bold", pad=15)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved Actual vs Predicted plot to: %s", save_path)


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, save_path: Path) -> None:
    """Generates and saves residual distribution plot.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        save_path: Location to save the plot.
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color="#34a853", bins=20)
    plt.axvline(0, color="#ea4335", linestyle="--", linewidth=1.5)
    
    plt.xlabel("Residuals (Actual - Predicted)", fontsize=11)
    plt.ylabel("Density / Count", fontsize=11)
    plt.title("Residuals Distribution Plot", fontsize=13, fontweight="bold", pad=15)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved Residuals plot to: %s", save_path)


def plot_feature_importance(pipeline: Any, feature_names: List[str], save_path: Path) -> None:
    """Generates and saves feature importance plot for the top 15 features.

    Handles models that expose either feature_importances_ or coef_.

    Args:
        pipeline: Fitted model Pipeline containing the regressor.
        feature_names: List of raw feature names.
        save_path: Location to save the plot.
    """
    regressor = pipeline.named_steps["regressor"]
    importances = None
    title = "Feature Importance"
    
    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
        title = "Tree-based Feature Importances"
    elif hasattr(regressor, "coef_"):
        importances = np.abs(regressor.coef_)
        title = "Linear Regression Feature Coefficients (Abs)"
        
    if importances is None:
        logger.warning("Could not extract feature importances or coefficients from the model.")
        return

    # Create DataFrame of feature importances
    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feat_df, palette="viridis", hue="Feature", legend=False)
    plt.xlabel("Relative Importance / Weight", fontsize=11)
    plt.ylabel("Feature Name", fontsize=11)
    plt.title(f"Top 15 Features - {title}", fontsize=13, fontweight="bold", pad=15)
    plt.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved Feature Importance plot to: %s", save_path)


def plot_learning_curves(pipeline: Any, X: pd.DataFrame, y: pd.Series, save_path: Path) -> None:
    """Generates and saves learning curves of the best model.

    Args:
        pipeline: Fitted model Pipeline.
        X: Feature dataset.
        y: Target series.
        save_path: Location to save the plot.
    """
    logger.info("Generating learning curves...")
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y, cv=config.N_SPLITS, scoring="r2", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=config.RANDOM_STATE
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, "o-", color="#ea4335", label="Training Score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#ea4335")
    
    plt.plot(train_sizes, val_mean, "s-", color="#1a73e8", label="Validation Score")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#1a73e8")

    plt.xlabel("Training Set Size", fontsize=11)
    plt.ylabel("R² Score", fontsize=11)
    plt.title("Model Learning Curves", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="best")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved Learning Curve plot to: %s", save_path)


def plot_model_comparison(metrics_df: pd.DataFrame, save_path: Path) -> None:
    """Generates and saves model comparison bar chart across all model R² scores.

    Args:
        metrics_df: DataFrame containing all models evaluation metrics.
        save_path: Location to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x="R2", y="Model", data=metrics_df.sort_values("R2", ascending=False), palette="Blues_r", hue="Model", legend=False)
    plt.xlabel("R² Score", fontsize=11)
    plt.ylabel("Model Name", fontsize=11)
    plt.title("Model Comparison - Test R² Score", fontsize=13, fontweight="bold", pad=15)
    plt.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved Model Comparison plot to: %s", save_path)


def evaluate_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
    """Evaluates all saved model pipelines and saves reports and plots.

    Args:
        X_train: Training features DataFrame.
        X_test: Test features DataFrame.
        y_train: Training target Series.
        y_test: Test target Series.
    """
    logger.info("Starting model evaluation pipeline.")
    
    # Identify all saved models
    model_paths = list(config.MODEL_DIR.glob("*.pkl"))
    # Exclude best_model, preprocessor_state, scaler, dropped_features
    exclude_files = ["best_model.pkl", "preprocessor_state.pkl", "scaler.pkl", "dropped_features.pkl"]
    model_files = [p for p in model_paths if p.name not in exclude_files]

    if not model_files:
        raise FileNotFoundError(f"No trained models found in: {config.MODEL_DIR}")

    results = []
    
    # Evaluate each model on test data
    for path in model_files:
        model_name = path.stem
        logger.info("Evaluating model: %s", model_name)
        pipeline = joblib.load(path)
        
        y_pred = pipeline.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred, X_test.shape[1])
        metrics["Model"] = model_name
        results.append(metrics)

    metrics_df = pd.DataFrame(results)
    
    # Save metrics csv
    metrics_csv_path = config.REPORTS_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info("Saved model evaluation metrics summary to: %s", metrics_csv_path)
    logger.info("\nEvaluation Metrics Summary:\n%s", metrics_df.to_string(index=False))

    # Generate Cross-Model Comparison plot
    fig_dir = config.REPORTS_DIR / "figures"
    plot_model_comparison(metrics_df, fig_dir / "model_comparison.png")

    # Load best model for detailed evaluation plots
    best_model_path = config.MODEL_DIR / "best_model.pkl"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found at: {best_model_path}")
        
    logger.info("Generating detailed plots for the best model.")
    best_pipeline = joblib.load(best_model_path)
    
    # Predictions of best model
    best_y_pred = best_pipeline.predict(X_test)
    
    # Generate scatter plot (Actual vs Predicted)
    plot_actual_vs_predicted(y_test, best_y_pred, fig_dir / "actual_vs_predicted.png")
    
    # Generate residuals plot
    plot_residuals(y_test, best_y_pred, fig_dir / "residuals_distribution.png")
    
    # Generate feature importance plot
    plot_feature_importance(best_pipeline, X_test.columns.tolist(), fig_dir / "feature_importance.png")
    
    # Generate learning curves (using combined train and test sets for CV analysis)
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    plot_learning_curves(best_pipeline, X_all, y_all, fig_dir / "learning_curve.png")
    
    logger.info("Model evaluation pipeline completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_loader import load_data
    from src.preprocessor import preprocess
    from src.feature_engineering import engineer_features
    from src.model import split_data
    try:
        raw_df = load_data()
        prep_df = preprocess(raw_df, is_training=False)
        feat_df = engineer_features(prep_df, is_training=False)
        X_train, X_test, y_train, y_test = split_data(feat_df)
        evaluate_pipeline(X_train, X_test, y_train, y_test)
    except Exception as e:
        logger.exception("Evaluation verification failed:")
