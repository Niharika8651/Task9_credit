# Credit Card Fraud Detection using Random Forest

## Objective
Detect fraudulent credit card transactions using machine learning while handling imbalanced data.

## Dataset
Synthetic credit card transaction dataset generated using Python.

## Models Used
- Logistic Regression (Baseline)
- Random Forest Classifier

## Evaluation Metrics
- Precision
- Recall
- F1-score

Accuracy is avoided due to class imbalance.

## Key Features
- Transaction Amount
- Transaction Time
- Location Risk
- Device Score
- Previous Fraud Count

## Results
Random Forest outperformed Logistic Regression, especially in recall for fraud cases.

## Files
- `data/credit_card_fraud.csv` – Dataset
- `src/fraud_detection.py` – Code
- `models/random_forest_model.pkl` – Saved model
- `outputs/feature_importance.png` – Feature importance plot
