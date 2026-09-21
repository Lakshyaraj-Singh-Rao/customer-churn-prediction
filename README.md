# Customer Churn Prediction

This project explores a telecom customer dataset and trains classification models to identify customers likely to churn. It includes exploratory data analysis, a class-balanced logistic-regression model with decision-threshold tuning, and an XGBoost model for comparison.

## Project contents

| File | Purpose |
| --- | --- |
| `telecom_churn.csv` | Source dataset. `Churn` is the binary target column. |
| `EDA.ipynb` | Exploratory analysis and visualizations of churn against customer attributes. |
| `logisticRegression.ipynb` | Logistic-regression baseline, evaluation, and threshold selection focused on recall. |
| `XGboost.ipynb` | XGBoost classification model and evaluation. |

## Data

Each row represents a telecom customer. The available predictors include account age, contract renewal, data plan and usage, customer-service calls, calling activity, charges, overage fees, and roaming minutes. The target is:

- `Churn`: `1` when a customer churned and `0` otherwise.

## Approach

1. Load `telecom_churn.csv` and separate `Churn` from the feature columns.
2. Explore churn patterns with count plots, including data-plan status, contract renewal, and customer-service calls.
3. Split the data into stratified training and test sets (80/20, `random_state=42`).
4. Train a logistic-regression model using balanced class weights to account for churn imbalance.
5. Evaluate with a confusion matrix, classification report, precision, recall, and F1 score. The notebook evaluates probability thresholds and selects one that reaches at least 75% recall while maximizing precision.
6. Train and evaluate an XGBoost classifier as an additional model.

## Getting started

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Launch Jupyter and run the notebooks from this project folder so they can find `telecom_churn.csv`:

```powershell
jupyter notebook
```

Recommended notebook order:

1. `EDA.ipynb`
2. `logisticRegression.ipynb`
3. `XGboost.ipynb`

## Notes

- Churn data is imbalanced, so accuracy alone can be misleading; prioritize recall, precision, and F1 score when comparing models.
- No serialized trained-model artifact is currently included. Re-run a notebook to train a model.
