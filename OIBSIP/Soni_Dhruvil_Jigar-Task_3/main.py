"""Main orchestration script for the Car Price Prediction ML system.

Provides a CLI interface to trigger model training, evaluation, or predict
car price given feature inputs in JSON format.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from src.config import config
from src.data_loader import load_data
from src.preprocessor import preprocess
from src.feature_engineering import engineer_features
from src.model import split_data, train_models, select_best_model
from src.evaluate import evaluate_pipeline
from src.predict import predict


def setup_logging() -> None:
    """Configures structured logging to console and a local pipeline log file."""
    log_file = config.LOGS_DIR / "pipeline.log"
    
    # Root logger configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def run_training() -> None:
    """Runs the full model training and tuning pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("========================================")
    logger.info("Starting training pipeline...")
    logger.info("========================================")
    
    # 1. Load data
    raw_df = load_data()
    
    # 2. Preprocess data
    prep_df = preprocess(raw_df, is_training=True)
    
    # 3. Feature Engineering
    feat_df = engineer_features(prep_df, is_training=True)
    
    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = split_data(feat_df)
    
    # 5. Train & hyperparameter tune models
    trained_pipelines = train_models(X_train, y_train)
    
    # 6. Select and save best model
    best_model_name = select_best_model(trained_pipelines, X_test, y_test)
    
    logger.info("Training pipeline finished. Best model selected: %s", best_model_name)


def run_evaluation() -> None:
    """Runs the model evaluation pipeline, generating metrics and charts."""
    logger = logging.getLogger(__name__)
    logger.info("========================================")
    logger.info("Starting evaluation pipeline...")
    logger.info("========================================")
    
    # 1. Load data
    raw_df = load_data()
    
    # 2. Preprocess using saved training state
    prep_df = preprocess(raw_df, is_training=False)
    
    # 3. Feature Engineering using saved training scaler
    feat_df = engineer_features(prep_df, is_training=False)
    
    # 4. Split data (will produce same split due to fixed random state)
    X_train, X_test, y_train, y_test = split_data(feat_df)
    
    # 5. Run evaluation reports and plotting
    evaluate_pipeline(X_train, X_test, y_train, y_test)
    
    logger.info("Evaluation pipeline completed successfully.")


def run_prediction(input_json: str) -> None:
    """Performs inference using the best trained model on the input JSON.

    Args:
        input_json: A JSON string containing the required car features.
    """
    logger = logging.getLogger(__name__)
    logger.info("========================================")
    logger.info("Starting prediction request...")
    logger.info("========================================")
    
    try:
        input_data = json.loads(input_json)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse prediction input JSON: %s", e)
        sys.exit(1)
        
    try:
        prediction_result = predict(input_data)
        logger.info("Prediction successful!")
        logger.info("Input: %s", input_data)
        logger.info("Result: %s", prediction_result)
        # Output prediction to stdout as raw JSON for external integration
        sys.stdout.write(json.dumps(prediction_result) + "\n")
    except Exception as e:
        logger.exception("Error occurred during prediction:")
        sys.exit(1)


def main() -> None:
    """Main execution orchestrator."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(
        description="Used Car Price Prediction ML System Orchestrator CLI"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Run model training and selection")
    group.add_argument("--evaluate", action="store_true", help="Run model evaluation and save reports")
    group.add_argument("--predict", type=str, metavar="JSON_STRING", help="Predict resale price from a JSON string")
    
    args = parser.parse_args()
    
    if args.train:
        run_training()
    elif args.evaluate:
        run_evaluation()
    elif args.predict:
        run_prediction(args.predict)


if __name__ == "__main__":
    main()
