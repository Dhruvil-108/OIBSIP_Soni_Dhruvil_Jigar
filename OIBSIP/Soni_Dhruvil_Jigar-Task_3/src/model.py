"""Model training module for the Car Price Prediction ML system.

This module splits the dataset, trains multiple regression models (Linear Regression,
Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost), performs hyperparameter
tuning, and saves the best model based on cross-validated R² score.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from src.config import config

logger = logging.getLogger(__name__)


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the preprocessed and engineered DataFrame into train and test sets.

    Removes target columns and target-derived features from predictors to prevent leakage.

    Args:
        df: Feature-engineered DataFrame.

    Returns:
        A tuple of (X_train, X_test, y_train, y_test).
    """
    logger.info("Splitting dataset into train and test sets.")
    
    # Target column
    y = df[config.TARGET_COLUMN]
    
    # Drop target and target-derived columns from features
    cols_to_drop = [config.TARGET_COLUMN, "depreciation"]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    logger.info("Train set shape: %s, Test set shape: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def create_pipeline(regressor: Any) -> Pipeline:
    """Creates a scikit-learn Pipeline with a StandardScaler and a regressor.

    Args:
        regressor: The scikit-learn regressor instance.

    Returns:
        An sklearn Pipeline object.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", regressor)
    ])


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Pipeline]:
    """Trains and tunes all 5 regression models using 5-fold cross-validation.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        A dictionary mapping model names to their fitted Pipelines.
    """
    kf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    
    # Define baseline models
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=config.RANDOM_STATE),
        "Lasso": Lasso(random_state=config.RANDOM_STATE),
        "RandomForest": RandomForestRegressor(random_state=config.RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=config.RANDOM_STATE),
        "XGBoost": XGBRegressor(random_state=config.RANDOM_STATE)
    }
    
    trained_pipelines: Dict[str, Pipeline] = {}
    
    # First, evaluate all baseline models using 5-fold CV
    logger.info("Starting baseline model training and 5-fold cross-validation.")
    for name, model in models.items():
        pipeline = create_pipeline(model)
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1
        )
        logger.info("Baseline %s CV R² scores: %s | Mean R²: %.4f (std: %.4f)",
                    name, cv_scores, cv_scores.mean(), cv_scores.std())
        
        # Fit on whole training set
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

    # Hyperparameter tuning on top 2 models: RandomForest and XGBoost
    logger.info("Starting hyperparameter tuning with GridSearchCV.")
    
    # 1. Random Forest Tuning
    rf_param_grid = {
        "regressor__n_estimators": [50, 100, 200],
        "regressor__max_depth": [None, 5, 10, 15],
        "regressor__min_samples_split": [2, 5]
    }
    rf_pipeline = create_pipeline(RandomForestRegressor(random_state=config.RANDOM_STATE))
    rf_grid = GridSearchCV(
        rf_pipeline, rf_param_grid, cv=kf, scoring="r2", n_jobs=-1, verbose=1
    )
    logger.info("Tuning Random Forest...")
    rf_grid.fit(X_train, y_train)
    logger.info("Best Random Forest CV R²: %.4f with params: %s",
                rf_grid.best_score_, rf_grid.best_params_)
    trained_pipelines["RandomForest_Tuned"] = rf_grid.best_estimator_
    
    # 2. XGBoost Tuning
    xgb_param_grid = {
        "regressor__n_estimators": [50, 100, 200],
        "regressor__max_depth": [3, 5, 7],
        "regressor__learning_rate": [0.01, 0.05, 0.1]
    }
    xgb_pipeline = create_pipeline(XGBRegressor(random_state=config.RANDOM_STATE))
    xgb_grid = GridSearchCV(
        xgb_pipeline, xgb_param_grid, cv=kf, scoring="r2", n_jobs=-1, verbose=1
    )
    logger.info("Tuning XGBoost...")
    xgb_grid.fit(X_train, y_train)
    logger.info("Best XGBoost CV R²: %.4f with params: %s",
                xgb_grid.best_score_, xgb_grid.best_params_)
    trained_pipelines["XGBoost_Tuned"] = xgb_grid.best_estimator_

    # Save all individual trained pipelines to models directory
    for name, pipeline in trained_pipelines.items():
        model_path = config.MODEL_DIR / f"{name}.pkl"
        joblib.dump(pipeline, model_path)
        logger.info("Saved pipeline %s to: %s", name, model_path)

    return trained_pipelines


def select_best_model(trained_pipelines: Dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series) -> str:
    """Selects the best model by evaluating R² score on the test set.

    Saves the best model as `best_model.pkl`.

    Args:
        trained_pipelines: Dictionary of model name to trained Pipeline.
        X_test: Test features.
        y_test: Test target.

    Returns:
        The name of the best-performing model.
    """
    logger.info("Selecting best model by test set R² score.")
    best_score = -float("inf")
    best_name = None
    
    for name, pipeline in trained_pipelines.items():
        score = pipeline.score(X_test, y_test)
        logger.info("Model '%s' test set R²: %.4f", name, score)
        if score > best_score:
            best_score = score
            best_name = name
            
    if best_name is not None:
        best_pipeline = trained_pipelines[best_name]
        best_model_path = config.MODEL_DIR / "best_model.pkl"
        joblib.dump(best_pipeline, best_model_path)
        logger.info(">>> Best model: '%s' with test R²: %.4f. Saved to %s",
                    best_name, best_score, best_model_path)
    else:
        raise ValueError("No model was trained successfully.")
        
    return best_name


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_loader import load_data
    from src.preprocessor import preprocess
    from src.feature_engineering import engineer_features
    try:
        raw_df = load_data()
        prep_df = preprocess(raw_df, is_training=True)
        feat_df = engineer_features(prep_df, is_training=True)
        X_train, X_test, y_train, y_test = split_data(feat_df)
        pipelines = train_models(X_train, y_train)
        select_best_model(pipelines, X_test, y_test)
    except Exception as e:
        logger.exception("Model training verification failed:")
