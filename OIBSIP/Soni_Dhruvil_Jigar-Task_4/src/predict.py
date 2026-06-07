"""Prediction and inference module for the Email Spam Detection System.

This module provides functions to load the saved classification model and vectorizer
and predict whether an input email text is 'Spam' or 'Ham' along with a confidence score.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
from src.config import Config
from src.preprocessor import clean_text

logger = logging.getLogger(__name__)


def load_inference_artifacts(config: Config) -> Tuple[Any, Any]:
    """Loads the best model classifier and the TF-IDF vectorizer.

    Args:
        config (Config): System configuration.

    Returns:
        Tuple[Any, Any]: (model, vectorizer) loaded artifacts.

    Raises:
        FileNotFoundError: If the model or vectorizer file is missing.
    """
    model_path: Path = config.MODEL_SAVE_PATH
    vec_path: Path = config.VECTORIZER_SAVE_PATH

    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")

    if not vec_path.exists():
        logger.error(f"Vectorizer file not found at {vec_path}")
        raise FileNotFoundError(f"Vectorizer file not found at {vec_path}")

    logger.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    logger.info(f"Loading vectorizer from {vec_path}...")
    vectorizer = joblib.load(vec_path)

    return model, vectorizer


def predict(email_text: str, model: Any = None, vectorizer: Any = None) -> Dict[str, Any]:
    """Predicts if an email text is spam or ham.

    Loads the saved model and vectorizer dynamically if not provided.

    Args:
        email_text (str): The raw text of the email.
        model (Any, optional): Pre-loaded model. Defaults to None.
        vectorizer (Any, optional): Pre-loaded vectorizer. Defaults to None.

    Returns:
        Dict[str, Any]: Prediction dictionary:
            {
                "label": "Spam" or "Ham",
                "confidence": float probability score
            }
    """
    # Load artifacts if not provided
    if model is None or vectorizer is None:
        config = Config()
        model, vectorizer = load_inference_artifacts(config)

    # Clean the input text
    cleaned_series = clean_text(pd.Series([email_text]))
    cleaned_txt = cleaned_series.iloc[0]

    if not cleaned_txt.strip():
        # Edge case: text contains no words after cleaning
        return {"label": "Ham", "confidence": 0.5000}

    # Transform text to features
    vec_text = vectorizer.transform([cleaned_txt])

    # Get class predictions and probabilities
    pred_class = int(model.predict(vec_text)[0])
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec_text)[0]
        confidence = float(probs[pred_class])
    else:
        confidence = 1.0

    label = "Spam" if pred_class == 1 else "Ham"

    logger.info(f"Prediction complete. Label: {label}, Confidence: {confidence:.4f}")
    return {"label": label, "confidence": confidence}


if __name__ == "__main__":
    # Test prediction standalone
    logging.basicConfig(level=logging.INFO)
    try:
        res = predict("Congratulations! You won a $1,000 Walmart Gift Card. Click here to claim.")
        print("Inference Test Result:", res)
    except Exception as ex:
        print("Error during test prediction:", ex)
