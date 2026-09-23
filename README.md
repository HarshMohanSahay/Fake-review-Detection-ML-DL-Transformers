# Fake Review Detection using ML, Deep Learning and Transformers

A multi-class NLP project for detecting **genuine reviews, human-generated fake reviews, and AI-generated fake reviews**.

## Project Overview

The project compares traditional machine learning, deep learning, and transformer-based approaches for fake review detection.

### Classes

- `genuine`
- `fake_human`
- `fake_ai`

## Pipeline

```text
Review Dataset
      ↓
Text Preprocessing
      ↓
TF-IDF / Tokenization
      ↓
Classical ML ──────────────┐
                           │
Deep Learning ─────────────┼──→ Evaluation
                           │
BERT / RoBERTa ────────────┘
      ↓
Ensemble Experiment
```

## Models Implemented

### Classical ML
- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes
- Complement Naive Bayes
- Random Forest

### Deep Learning
- LSTM
- CNN
- CNN-LSTM
- CNN-BiLSTM

### Transformers
- BERT
- Improved BERT
- RoBERTa
- RoBERTa with differential learning rates

### Ensemble
- Logistic Regression + BERT
- Logistic Regression + RoBERTa

## Reported Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression | 93.54% | 0.9355 |
| CNN-BiLSTM | 88.54% | 0.8837 |
| BERT v2 | 92.08% | 0.9206 |
| RoBERTa v2 | 95.83% | 0.9583 |
| **LR + RoBERTa v2 Ensemble** | **97.50%** | **0.9750** |

> Metrics above are the results reported in the original project notebook.

## Repository Structure

```text
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_classical_ml.ipynb
│   ├── 02_text_analysis.ipynb
│   ├── 03_deep_learning.ipynb
│   ├── 04_transformers.ipynb
│   └── 05_ensemble.ipynb
├── results/
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   └── README.md
├── models/
│   └── README.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The dataset is not stored in this repository. Place the required `final_reviews.csv` file inside `data/`.

The expected columns are:

```text
text
label
```

## How to Run

```bash
git clone <your-repository-url>
cd Fake-review-Detection-ML-DL-Transformers

pip install -r requirements.txt
```

Then place the dataset at:

```text
data/final_reviews.csv
```

Open the notebooks in order from `01` to `05`.

## Notes

The project was originally developed and experimented with in Kaggle notebooks. The repository version is organized into separate notebooks so that the workflow is easier to understand and review.
