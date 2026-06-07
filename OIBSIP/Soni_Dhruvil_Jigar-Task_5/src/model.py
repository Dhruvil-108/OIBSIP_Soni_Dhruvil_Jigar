"""Model training and tuning module for the Sales Prediction ML pipeline.

This module defines the training pipeline, runs baseline models, evaluates them
using 5-fold cross-validation, tunes the top two models using GridSearchCV,
and saves the best overall model pipeline.
"""

import logging
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor

from src.config import config
from src.data_loader import load_dataset
from src.feature_engineering import FeatureEngineer
from src.preprocessor import DataPreprocessor

# Configure logger
logger = logging.getLogger(__name__)


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, DataPreprocessor, FeatureEngineer]:
    """Splits, preprocesses, and engineers features on train/test sets.

    Ensures that preprocessors and feature engineers are fit ONLY on the
    training set to prevent data leakage.

    Args:
        df: Cleaned input DataFrame.

    Returns:
        A tuple of:
            - X_train: Preprocessed and engineered training features (unscaled).
            - X_test: Preprocessed and engineered test features (unscaled).
            - y_train: Training target variable.
            - y_test: Test target variable.
            - preprocessor: Fitted DataPreprocessor instance.
            - engineer: Fitted FeatureEngineer instance.
    """
    logger.info("Splitting dataset into train and test sets...")
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    # 1. Fit and apply Preprocessor on Train, apply on Test
    preprocessor = DataPreprocessor()
    preprocessor.fit(train_df, target_col="Sales")
    train_df_prep = preprocessor.transform(train_df)
    test_df_prep = preprocessor.transform(test_df)

    # 2. Fit and apply Feature Engineer on Train, apply on Test
    engineer = FeatureEngineer(scaler_type="standard")
    engineer.fit(train_df_prep, target_col="Sales")

    # Generate interaction and select final features (without scaling, Pipeline will scale)
    train_eng = engineer.create_interactions(train_df_prep)
    test_eng = engineer.create_interactions(test_df_prep)

    X_train = train_eng[engineer.final_features]
    X_test = test_eng[engineer.final_features]
    y_train = train_df_prep["Sales"]
    y_test = test_df_prep["Sales"]

    # Save fitted preprocessor and feature engineer to models directory
    joblib.dump(preprocessor, config.MODEL_SAVE_PATH / "preprocessor.joblib")
    joblib.dump(engineer, config.MODEL_SAVE_PATH / "feature_engineer.joblib")
    joblib.dump(engineer.scaler, config.MODEL_SAVE_PATH / "scaler.joblib")
    logger.info("Saved fitted preprocessor, feature engineer, and scaler states.")

    return X_train, X_test, y_train, y_test, preprocessor, engineer


def get_baseline_pipelines() -> dict[str, Pipeline]:
    """Creates dictionary of pipelines for baseline models.

    Returns:
        dict[str, Pipeline]: Name to Pipeline mappings.
    """
    scaler = StandardScaler() if config.IQR_THRESHOLD == 1.5 else MinMaxScaler()
    
    pipelines = {
        "Linear Regression": Pipeline(
            [("scaler", scaler), ("regressor", LinearRegression())]
        ),
        "Ridge Regression": Pipeline(
            [("scaler", scaler), ("regressor", Ridge(random_state=config.RANDOM_STATE))]
        ),
        "Lasso Regression": Pipeline(
            [("scaler", scaler), ("regressor", Lasso(random_state=config.RANDOM_STATE))]
        ),
        "Random Forest": Pipeline(
            [("scaler", scaler), ("regressor", RandomForestRegressor(random_state=config.RANDOM_STATE))]
        ),
        "XGBoost": Pipeline(
            [("scaler", scaler), ("regressor", XGBRegressor(random_state=config.RANDOM_STATE))]
        ),
    }
    return pipelines


def train_and_tune_pipeline() -> Tuple[Pipeline, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Executes the complete training, evaluation, and tuning workflow.

    Returns:
        Tuple: Best fitted model Pipeline, X_train, y_train, X_test, y_test.
    """
    # Load and prepare data
    df = load_dataset()
    X_train, X_test, y_train, y_test, preprocessor, engineer = prepare_data(df)

    # Cross-validation setup
    kf = KFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )

    pipelines = get_baseline_pipelines()
    baseline_scores: dict[str, float] = {}

    logger.info("--- Starting Baseline Model Cross-Validation ---")
    for name, pipeline in pipelines.items():
        # CV scoring using R2 metric
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1
        )
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        baseline_scores[name] = mean_score
        logger.info("%s - Mean R2: %.4f (Std: %.4f)", name, mean_score, std_score)

    # Select top 2 models for GridSearchCV
    sorted_models = sorted(baseline_scores.items(), key=lambda x: x[1], reverse=True)
    top_models = [sorted_models[0][0], sorted_models[1][0]]
    logger.info("Top 2 models selected for tuning: %s", top_models)

    # Param grids for grid search
    param_grids: dict[str, dict[str, list[Any]]] = {
        "Random Forest": {
            "regressor__n_estimators": [50, 100, 200],
            "regressor__max_depth": [None, 5, 10],
            "regressor__min_samples_split": [2, 5],
        },
        "XGBoost": {
            "regressor__n_estimators": [50, 100, 200],
            "regressor__max_depth": [3, 5, 7],
            "regressor__learning_rate": [0.01, 0.1, 0.2],
        },
        "Linear Regression": {},
        "Ridge Regression": {
            "regressor__alpha": [0.1, 1.0, 10.0]
        },
        "Lasso Regression": {
            "regressor__alpha": [0.01, 0.1, 1.0]
        }
    }

    best_tuned_score = -float("inf")
    best_pipeline: Optional[Pipeline] = None
    best_model_name = ""

    for model_name in top_models:
        pipeline = pipelines[model_name]
        param_grid = param_grids.get(model_name, {})
        
        if not param_grid:
            # If no parameters to tune (e.g. Linear Regression), just fit baseline
            pipeline.fit(X_train, y_train)
            cv_score = baseline_scores[model_name]
        else:
            logger.info("Tuning %s with GridSearchCV...", model_name)
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=kf, scoring="r2", n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            cv_score = grid_search.best_score_
            pipeline = grid_search.best_estimator_
            logger.info("%s Best Params: %s", model_name, grid_search.best_params_)
            logger.info("%s Tuned CV R2: %.4f", model_name, cv_score)

        if cv_score > best_tuned_score:
            best_tuned_score = cv_score
            best_pipeline = pipeline
            best_model_name = model_name

    assert best_pipeline is not None, "Best pipeline cannot be None."
    logger.info("Best overall model: %s with CV R2 = %.4f", best_model_name, best_tuned_score)

    # Fit best model on full training set
    best_pipeline.fit(X_train, y_train)

    # Save best model
    model_save_path = config.MODEL_SAVE_PATH / "best_model.joblib"
    joblib.dump(best_pipeline, model_save_path)
    logger.info("Best model pipeline saved to: %s", model_save_path)

    return best_pipeline, X_train, y_train, X_test, y_test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        model, X_tr, y_tr, X_te, y_te = train_and_tune_pipeline()
        print("\nModel pipeline training complete.")
        print(f"Features trained on: {list(X_tr.columns)}")
        print(f"Test R2 Score: {model.score(X_te, y_te):.4f}")
    except Exception as e:
        logger.error("Error executing model training standalone: %s", e, exc_info=True)
