"""Model training pipeline for Iris flower classification."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    import mlflow
except ImportError:  # pragma: no cover - optional dependency fallback
    mlflow = None


def _project_root() -> Path:
    """Return the project root directory."""

    return Path(__file__).resolve().parents[1]


def _build_models(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the full comparison set of classification models."""

    training_config = config["training"]
    model_config = training_config["models"]
    random_state = training_config["random_state"]

    return {
        "logistic_regression": LogisticRegression(
            C=model_config["logistic_regression"]["C"],
            max_iter=model_config["logistic_regression"]["max_iter"],
            random_state=random_state,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=model_config["decision_tree"]["max_depth"],
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=model_config["random_forest"]["n_estimators"],
            max_depth=model_config["random_forest"]["max_depth"],
            random_state=random_state,
        ),
        "svm": SVC(
            C=model_config["svm"]["C"],
            kernel=model_config["svm"]["kernel"],
            probability=model_config["svm"]["probability"],
            random_state=random_state,
        ),
        "knn": KNeighborsClassifier(n_neighbors=model_config["knn"]["n_neighbors"]),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=model_config["gradient_boosting"]["n_estimators"],
            learning_rate=model_config["gradient_boosting"]["learning_rate"],
            random_state=random_state,
        ),
    }


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Train, compare, and persist all requested models."""

    training_config = config["training"]
    models_dir = _project_root() / Path(config["paths"]["models_dir"])
    reports_dir = _project_root() / Path(config["paths"]["reports_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }
    cv = StratifiedKFold(
        n_splits=training_config["cv_folds"],
        shuffle=True,
        random_state=training_config["random_state"],
    )

    if mlflow is not None:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        tracking_uri = (_project_root() / Path(config["paths"]["mlruns_dir"])).resolve().as_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config["project_name"])

    best_model_name = ""
    best_model_score = -np.inf
    best_model = None
    all_results: list[dict[str, Any]] = []

    for model_name, model in _build_models(config).items():
        logger.info("Training model: {}", model_name)
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=training_config["n_jobs"],
            return_train_score=False,
        )

        metrics = {
            metric_name: float(np.mean(cv_results[f"test_{metric_name}"]))
            for metric_name in scoring
        }
        metrics["cv_folds"] = float(training_config["cv_folds"])
        all_results.append({"model_name": model_name, **metrics})

        fitted_model = clone(model).fit(X_train, y_train)
        model_path = models_dir / f"{model_name}.pkl"
        joblib.dump(fitted_model, model_path)

        logger.info(
            "{} -> accuracy={:.4f}, precision={:.4f}, recall={:.4f}, f1={:.4f}",
            model_name,
            metrics["accuracy"],
            metrics["precision_macro"],
            metrics["recall_macro"],
            metrics["f1_macro"],
        )

        if mlflow is not None:
            with mlflow.start_run(run_name=model_name):
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("cv_folds", training_config["cv_folds"])
                mlflow.log_metrics(
                    {
                        "accuracy": metrics["accuracy"],
                        "precision_macro": metrics["precision_macro"],
                        "recall_macro": metrics["recall_macro"],
                        "f1_macro": metrics["f1_macro"],
                    }
                )
                mlflow.sklearn.log_model(fitted_model, artifact_path="model")

        if metrics["f1_macro"] > best_model_score:
            best_model_score = metrics["f1_macro"]
            best_model_name = model_name
            best_model = fitted_model

    if best_model is None:
        raise RuntimeError("No model was trained successfully.")

    best_model_path = _project_root() / Path(config["paths"]["best_model_path"])
    joblib.dump(best_model, best_model_path)
    logger.info("Best model: {} with F1={:.4f}", best_model_name, best_model_score)
    logger.info("Saved best model to {}", best_model_path)

    results_frame = pd.DataFrame(all_results).sort_values(by="f1_macro", ascending=False)
    results_path = reports_dir / "model_comparison.csv"
    results_frame.to_csv(results_path, index=False)
    logger.info("Saved model comparison results to {}", results_path)

    return {
        "best_model_name": best_model_name,
        "best_model_score": best_model_score,
        "results": all_results,
        "best_model_path": str(best_model_path),
        "results_path": str(results_path),
    }
