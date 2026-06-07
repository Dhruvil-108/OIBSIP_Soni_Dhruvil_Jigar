"""Model training and selection module for the Email Spam Detection System.

This module builds pipelines, runs cross-validation, logs performance scores,
selects the best model by F1-score, fits it on the full training set, and saves it.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
from src.config import Config

logger = logging.getLogger(__name__)


def train_and_select_best_model(
    config: Config,
    X_train: pd.Series,
    y_train: pd.Series,
    vectorizer: Any
) -> Tuple[Any, Dict[str, float]]:
    """Trains three models using cross-validation and saves the best model.

    Models trained: Multinomial Naïve Bayes, Logistic Regression, and Random Forest.
    F1-score is used to select the best model.

    Args:
        config (Config): System configuration.
        X_train (pd.Series): Preprocessed training text.
        y_train (pd.Series): Training labels ('ham' or 'spam').
        vectorizer (Any): Unfitted text vectorizer.

    Returns:
        Tuple[Any, Dict[str, float]]: The best fitted classifier and a dict of CV scores.
    """
    logger.info("Initializing models for training...")
    
    # Map labels to binary values to avoid scoring ambiguities
    y_train_encoded = y_train.map({"ham": 0, "spam": 1})
    
    models = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(
            random_state=config.RANDOM_STATE,
            max_iter=1000
        ),
        "RandomForest": RandomForestClassifier(
            random_state=config.RANDOM_STATE,
            n_estimators=100
        )
    }

    best_f1 = -1.0
    best_model_name = None
    best_pipeline = None
    cv_results = {}

    # Define custom scorer to ensure we score F1 for 'spam' (encoded as 1)
    f1_scorer = make_scorer(f1_score, pos_label=1)

    for name, clf in models.items():
        logger.info(f"Evaluating {name} with 5-fold cross-validation...")
        pipeline = Pipeline([
            ("vectorizer", vectorizer),
            ("classifier", clf)
        ])
        
        # 5-fold cross validation
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train_encoded,
            cv=5,
            scoring=f1_scorer,
            n_jobs=-1
        )
        
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        cv_results[name] = mean_score
        
        logger.info(f"{name} 5-Fold CV F1-Score: {mean_score:.4f} (+/- {std_score:.4f})")

        if mean_score > best_f1:
            best_f1 = mean_score
            best_model_name = name
            best_pipeline = pipeline

    logger.info(f"Best model selected: {best_model_name} with F1-score: {best_f1:.4f}")

    # Fit the best pipeline on the entire training data
    logger.info(f"Fitting {best_model_name} pipeline on the full training dataset...")
    best_pipeline.fit(X_train, y_train_encoded)

    # Save the classifier component
    save_path = config.MODEL_SAVE_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving best model classifier to {save_path}...")
    joblib.dump(best_pipeline.named_steps["classifier"], save_path)

    # Save the vectorizer component separately
    vec_save_path = config.VECTORIZER_SAVE_PATH
    logger.info(f"Saving tfidf vectorizer to {vec_save_path}...")
    joblib.dump(best_pipeline.named_steps["vectorizer"], vec_save_path)

    return best_pipeline.named_steps["classifier"], cv_results


if __name__ == "__main__":
    # Test model selection module placeholder
    print("Model training module loaded successfully.")
