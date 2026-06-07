# Email Spam Detection ML System

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

## Project Overview
This project is a production-grade Email Spam Detection Machine Learning system designed to identify and classify spam text messages and emails. The system uses natural language processing (NLP) pipelines to clean, normalize, tokenize, and stem unstructured email content. It extracts features using TF-IDF and bag-of-words methods, trains multiple classifiers (Logistic Regression, Naïve Bayes, and Random Forest), optimizes performance via cross-validation, and serves predictions via a command-line interface. Identifying spam is a vital real-world task that secures inbox safety, protects users against phishing and financial fraud, and optimizes communication efficiency.

## Tech Stack
| Library | Version | Purpose |
| :--- | :--- | :--- |
| **Python** | >=3.10 | Core programming language |
| **pandas** | >=2.0 | Data loading, manipulation, and standardization |
| **numpy** | >=1.24 | Vector/matrix numeric calculations and statistics |
| **scikit-learn** | >=1.3 | Text vectorization, ML models training, pipelines, and evaluation metrics |
| **nltk** | >=3.8 | Natural Language Toolkit for stopword removal and Porter Stemming |
| **matplotlib** | >=3.7 | Visualization engine to plot confusion matrix and ROC curves |
| **seaborn** | >=0.12 | Heatmap visualization helper |
| **joblib** | >=1.3 | Model serialization and deserialization |
| **xgboost** | >=2.0 | High performance gradient boosted trees (additional classifier) |
| **pytest** | >=7.4 | Unit testing framework |

## Project Structure
```
email-spam-detection/
├── data/                     ← CSV already exists here
│   └── spam.csv
├── notebooks/
│   └── 01_EDA.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   ├── best_model.joblib
│   └── tfidf_vectorizer.joblib
├── reports/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── tests/
│   └── test_pipeline.py
├── main.py
├── requirements.txt
└── README.md
```

## Dataset
The project utilizes the SMS Spam Collection dataset (stored in `data/spam.csv`).
- **Columns**:
  - `label` (`v1`): Class identifier: `ham` (legitimate) or `spam` (spam).
  - `text` (`v2`): The raw text message content.
- **Class Distribution Note**: The dataset is heavily imbalanced, containing approximately 86.6% ham messages and 13.4% spam messages. Because of this, our model evaluation focuses on the F1-score rather than raw accuracy.

## Setup & Installation
Follow these commands to clone, set up a virtual environment, and install dependencies:

```bash
git clone <repository_url>
cd email-spam-detection
python -m venv venv
```

To activate the virtual environment:
- **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- **Mac / Linux**:
  ```bash
  source venv/bin/activate
  ```

Install requirements:
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

## How to Run
Use the `main.py` CLI interface to orchestrate the pipeline:

- **Train the Model**:
  ```bash
  python main.py --train
  ```
- **Evaluate on the Test Set**:
  ```bash
  python main.py --evaluate
  ```
- **Predict Single Texts**:
  ```bash
  python main.py --predict "Win a free iPhone now!!!"
  ```

## Model Results
Below is the performance comparison across the trained models during 5-fold cross validation.

| Model | 5-Fold CV F1-Score | Test Accuracy | Test Precision | Test Recall | Test F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Multinomial NB** | 0.8625 | - | - | - | - |
| **Logistic Regression** | 0.8439 | - | - | - | - |
| **Random Forest (Best)** | **0.9009** | **0.9767** | **0.9920** | **0.8322** | **0.9051** |

*Note: Individual test metrics are shown for the best model chosen by F1-score.*

## Contributing
1. Fork the Repository.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License
Distributed under the MIT License. See `LICENSE` for more information.
