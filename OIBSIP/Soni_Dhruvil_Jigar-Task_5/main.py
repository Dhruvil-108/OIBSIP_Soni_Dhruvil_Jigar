"""Main entrypoint for the Sales Prediction ML pipeline.

Provides a CLI for training the pipeline, evaluating model performance,
and generating predictions from JSON input strings. Logs runtime details
to both the console and a file log.
"""

import argparse
import json
import logging
import sys
from typing import Optional

from src.config import config

# Setup module-level logger
logger = logging.getLogger("main")


def setup_logging() -> None:
    """Configures structured logging to console and pipeline log file."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOGS_DIR / "pipeline.log"

    # Define standard format with timestamps
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplication
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # File Handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)


def run_pipeline(args_list: Optional[list[str]] = None) -> None:
    """Parses arguments and executes requested pipeline steps.

    Args:
        args_list: Optional list of command-line arguments to parse.
            If None, parses sys.argv.
    """
    setup_logging()
    logger.info("Initializing Sales Prediction ML Pipeline CLI...")

    parser = argparse.ArgumentParser(
        description="Production-Grade Sales Prediction ML System Pipeline CLI."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run preprocessing, feature engineering, and train/tune regressors."
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the saved model and export metrics/plots."
    )
    parser.add_argument(
        "--predict",
        type=str,
        help="Generate prediction for a JSON input string. "
             "Example: --predict '{\"TV\": 230.1, \"Radio\": 37.8, \"Newspaper\": 69.2}'"
    )

    args = parser.parse_args(args_list)

    if not (args.train or args.evaluate or args.predict):
        parser.print_help()
        logger.warning("No action flags supplied. Exit.")
        sys.exit(1)

    if args.train:
        logger.info("--- Execution Stage: TRAINING ---")
        from src.model import train_and_tune_pipeline
        try:
            train_and_tune_pipeline()
            logger.info("Training stage completed successfully.")
        except Exception as e:
            logger.error("Training stage failed: %s", e, exc_info=True)
            sys.exit(1)

    if args.evaluate:
        logger.info("--- Execution Stage: EVALUATION ---")
        import joblib
        from src.data_loader import load_dataset
        from src.evaluate import evaluate_model
        from src.model import prepare_data

        model_path = config.MODEL_SAVE_PATH / "best_model.joblib"
        if not model_path.exists():
            logger.error("Model artifact not found at %s. Train the model using --train first.", model_path)
            sys.exit(1)

        try:
            model = joblib.load(model_path)
            df = load_dataset()
            X_train, X_test, y_train, y_test, _, _ = prepare_data(df)
            evaluate_model(model, X_train, y_train, X_test, y_test)
            logger.info("Evaluation stage completed successfully.")
        except Exception as e:
            logger.error("Evaluation stage failed: %s", e, exc_info=True)
            sys.exit(1)

    if args.predict:
        logger.info("--- Execution Stage: INFERENCE ---")
        from src.predict import predict

        try:
            input_dict = json.loads(args.predict)
        except json.JSONDecodeError as e:
            logger.error(
                "Invalid JSON string supplied to --predict: %s. Use format: '{\"TV\": 230.1, \"Radio\": 37.8, \"Newspaper\": 69.2}'",
                e
            )
            sys.exit(1)

        try:
            result = predict(input_dict)
            # Log output cleanly
            logger.info("Prediction output: %s", json.dumps(result, indent=2))
        except Exception as e:
            logger.error("Inference stage failed: %s", e, exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
