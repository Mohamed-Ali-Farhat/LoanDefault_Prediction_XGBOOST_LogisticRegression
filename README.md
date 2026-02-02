# Loan Default Prediction - Fixed Version


## Overview

This project implements a machine learning pipeline for predicting loan defaults using XGBoost and Logistic Regression models. The dataset contains features related to user behavior, financial activities, and demographics. This is a **fixed version** of the original implementation, addressing issues like data leakage, multicollinearity, and improper data splitting to ensure more reliable and realistic model performance.

The notebook (`LoanDefault_Prediction_XGBOOST_LogisticRegression.ipynb`) performs data loading, exploratory data analysis (EDA), preprocessing, model training, evaluation, and feature importance analysis.

### Key Changes from Original Version
1. **Removed data leakage features** (e.g., `savings_worth`, `existing_debt_amount`, `loan_amount_requested`) to prevent overfitting and ensure model generalizability.
2. **Removed low-value features** due to multicollinearity and weak predictive power.
3. **Proper train/test split** applied before any preprocessing to avoid data leakage.
4. **Added feature importance analysis** using SHAP values for better interpretability.
5. **More realistic evaluation metrics** (e.g., AUC-ROC, Precision-Recall) to handle class imbalance.

## Dataset
- **File**: `dataset.csv` (not included in this repo; assume it's provided or generated separately).
- **Shape**: 10,000 rows × 25 columns.
- **Target Variable**: `loan_default` (binary: 0 = no default, 1 = default).
- **Target Distribution**:
  - No Default: 70.21%
  - Default: 29.79%
- **Key Features** (after preprocessing):
  - `online_txn_count_last_30d`
  - `finance_app_time_pct`
  - `avg_txn_amount`
  - `avg_call_duration_mins`
  - `distinct_contacts_weekly`
  - `monthly_data_usage_gb`
  - `financial_apps_installed`
  - And others (see notebook for full list).

The dataset includes behavioral metrics like app usage, transaction counts, and demographics.

## Requirements
- Python 3.6+
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - shap
  - matplotlib
  - seaborn

Install dependencies using:
```
pip install -r requirements.txt
```
(If `requirements.txt` is not present, create one with the above libraries.)

## Usage
1. **Clone the Repository**:
   ```
   git clone <repo-url>
   cd <repo-folder>
   ```

2. **Prepare the Dataset**:
   - Place `dataset.csv` in the root directory.

3. **Run the Notebook**:
   - Open in Jupyter Notebook or JupyterLab:
     ```
     jupyter notebook LoanDefault_Prediction_XGBOOST_LogisticRegression.ipynb
     ```
   - Execute cells sequentially for data loading, preprocessing, training, and evaluation.

4. **Output**:
   - Prints dataset overview, target distribution, and model metrics.
   - Generates plots (e.g., feature importance via SHAP).
   - Displays top features by importance.

## Models
- **XGBoost**: Gradient boosting classifier for handling non-linear relationships and feature interactions.
- **Logistic Regression**: Baseline linear model for comparison.

Both models are trained on a balanced dataset (handling class imbalance if needed) with hyperparameter tuning via GridSearchCV.

## Evaluation Metrics
- Focus on realistic metrics for imbalanced data:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - AUC-ROC
- Cross-validation used for robust performance estimation.

## Feature Importance
Based on SHAP values from the XGBoost model:

| Feature                  | Importance |
|--------------------------|------------|
| online_txn_count_last_30d| 2.643589  |
| finance_app_time_pct     | 1.211879  |
| loan_amount_requested    | 0.739099  |
| unique_apps_per_day      | 0.585711  |
| avg_txn_amount           | 0.476077  |
| avg_call_duration_mins   | 0.409271  |
| distinct_contacts_weekly | 0.370133  |
| monthly_data_usage_gb    | 0.310450  |
| financial_apps_installed | 0.285593  |
| social_media_pct         | 0.108505  |

(Top 10 features; see notebook for visualization.)

## Results
- XGBoost typically outperforms Logistic Regression in AUC-ROC due to its ability to capture complex patterns.
- Detailed metrics and confusion matrices are printed in the notebook.

## Limitations
- Assumes `dataset.csv` is clean; real-world data may require additional handling for missing values or outliers.
- Model performance depends on the dataset; further tuning or ensemble methods could improve results.
- No deployment code included (e.g., API or web app).

## Contributing
Contributions are welcome! Please fork the repo and submit a pull request.

## Author
- [Your Name or Handle]
- Contact: [your.email@example.com]

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
