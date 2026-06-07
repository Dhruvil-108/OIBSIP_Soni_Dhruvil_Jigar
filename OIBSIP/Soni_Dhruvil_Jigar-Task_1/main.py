"""Command-line entry point for the Iris flower classification project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import yaml
from loguru import logger

from src.data_loader import load_raw_data
from src.evaluate import evaluate_model
from src.predict import predict_species
from src.preprocessor import preprocess_data
from src.train import train_models


PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load the project configuration from YAML."""

    resolved_path = config_path or PROJECT_ROOT / "config.yaml"
    with resolved_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def setup_logging(config: Mapping[str, Any]) -> None:
    """Configure loguru for console and file output."""

    logs_dir = PROJECT_ROOT / Path(config["paths"]["logs_dir"])
    log_path = PROJECT_ROOT / Path(config["paths"]["training_log"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {function} | {message}"
    logger.add(sys.stdout, level="INFO", format=log_format)
    logger.add(log_path, level="INFO", format=log_format, rotation="5 MB", retention="7 days")


def run_training(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the full data loading, preprocessing, and model training flow."""

    df = load_raw_data(config)
    preprocess_result = preprocess_data(df, config)
    training_summary = train_models(preprocess_result.X_train, preprocess_result.y_train, config)
    return {
        "training_summary": training_summary,
        "preprocess_result": preprocess_result,
    }


def run_evaluation(config: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the persisted best model on the holdout split."""

    df = load_raw_data(config)
    preprocess_result = preprocess_data(df, config)
    best_model = joblib.load(PROJECT_ROOT / Path(config["paths"]["best_model_path"]))
    return evaluate_model(
        model=best_model,
        X_test=preprocess_result.X_test,
        y_test=preprocess_result.y_test,
        config=config,
        feature_names=preprocess_result.feature_columns,
    )


def run_prediction(config: Mapping[str, Any], raw_input: str) -> dict[str, Any]:
    """Predict a species from a comma-separated measurement string."""

    return predict_species(raw_input, config)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Iris Flower Classification CLI")
    parser.add_argument(
        "--mode",
        choices=("train", "evaluate", "predict"),
        required=True,
        help="Pipeline mode to execute.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help='Comma-separated iris measurements for prediction: "5.1,3.5,1.4,0.2"',
    )
    return parser


def main() -> int:
    """Run the selected pipeline mode."""

    config = load_config()
    setup_logging(config)
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        result = run_training(config)
        logger.info("Training completed successfully: {}", result["training_summary"])
        return 0

    if args.mode == "evaluate":
        result = run_evaluation(config)
        logger.info("Evaluation completed successfully: {}", result["classification_report"])
        return 0

    if args.input is None:
        logger.error("Prediction mode requires --input")
        return 1

    result = run_prediction(config, args.input)
    logger.info(
        "Predicted species: {} | confidence: {:.2f}%",
        result["species"],
        result["confidence"] * 100,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
