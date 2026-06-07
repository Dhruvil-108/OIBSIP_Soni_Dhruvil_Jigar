"""Main orchestration script for the Email Spam Detection ML System.

This script parses CLI arguments, initializes logging, and routes requests to train,
evaluate, or run inference on the ML pipeline.
"""

import argparse
import json
import logging
from sklearn.model_selection import train_test_split
from src.config import Config
from src.data_loader import load_data
from src.preprocessor import clean_text
from src.feature_engineering import build_tfidf_vectorizer
from src.model import train_and_select_best_model
from src.evaluate import evaluate_model
from src.predict import predict, load_inference_artifacts

logger = logging.getLogger("main")


def setup_logging(config: Config) -> None:
    """Configures structured logging to console and file.

    Args:
        config (Config): System configuration.
    """
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOGS_DIR / "run.log"

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Reset existing handlers to prevent duplicate logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )
    # Silence matplotlib debug/info logging
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logger.info("Logging configured. Logs are saved to %s", log_file)


def run_train(config: Config) -> None:
    """Executes the data loading, preprocessing, and training pipeline.

    Args:
        config (Config): System configuration.
    """
    logger.info("Starting training pipeline...")
    
    # 1. Load Data
    df = load_data(config)
    
    # 2. Preprocess Data
    df["clean_text"] = clean_text(df["text"])
    
    # 3. Train-Test Split (stratified)
    X = df["clean_text"]
    y = df["label"]
    
    logger.info("Splitting data with test_size=%s, random_state=%s", config.TEST_SIZE, config.RANDOM_STATE)
    X_train, _, y_train, _ = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    # 4. Feature Engineering (initialize TF-IDF vectorizer)
    vectorizer = build_tfidf_vectorizer(config)
    
    # 5. Model Selection and Training (fits and saves classifier and vectorizer)
    best_clf, cv_scores = train_and_select_best_model(config, X_train, y_train, vectorizer)
    
    logger.info("Training pipeline completed. CV Scores: %s", cv_scores)


def run_evaluate(config: Config) -> None:
    """Loads the model and evaluates it on the test dataset.

    Args:
        config (Config): System configuration.
    """
    logger.info("Starting evaluation pipeline...")
    
    # Load model and vectorizer
    try:
        model, vectorizer = load_inference_artifacts(config)
    except FileNotFoundError as e:
        logger.error("Cannot evaluate: %s. Please run training first using --train.", e)
        return

    # Load and preprocess dataset to reconstruct the exact same test partition
    df = load_data(config)
    df["clean_text"] = clean_text(df["text"])
    
    X = df["clean_text"]
    y = df["label"]
    
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    # Run evaluation
    metrics = evaluate_model(config, model, X_test, y_test, vectorizer)
    logger.info("Evaluation pipeline completed. Test metrics: %s", metrics)


def run_predict(email_text: str, config: Config) -> None:
    """Predicts the label of a raw email string.

    Args:
        email_text (str): The raw email text.
        config (Config): System configuration.
    """
    logger.info("Starting prediction/inference...")
    try:
        model, vectorizer = load_inference_artifacts(config)
    except FileNotFoundError as e:
        logger.error("Cannot predict: %s. Please run training first using --train.", e)
        return

    result = predict(email_text, model, vectorizer)
    
    # Output result to console as formatted JSON
    print(json.dumps(result, indent=2))


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Production-Grade Email Spam Detection ML System CLI."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run end-to-end training pipeline and save the best model."
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the saved model on the test partition and plot reports."
    )
    parser.add_argument(
        "--predict",
        type=str,
        help="Predict if the provided email text is Spam or Ham."
    )

    args = parser.parse_args()
    config = Config()
    setup_logging(config)

    if not (args.train or args.evaluate or args.predict is not None):
        parser.print_help()
        return

    if args.train:
        run_train(config)
        
    if args.evaluate:
        run_evaluate(config)
        
    if args.predict is not None:
        run_predict(args.predict, config)


if __name__ == "__main__":
    main()
