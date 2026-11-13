# Bank Customer Churn Prediction

Predict which bank customers are likely to leave (churn), compare five models, and explain the best one with SHAP.

- **Data:** Kaggle "Churn Modelling" dataset - 10,000 customers, about 20% churned.
- **Notebook:** `customer_churn_prediction.ipynb` (all results below are printed in it).

## Method

1. Drop ID columns, encode Gender and Geography.
2. Stratified 80/20 train/test split (8,000 / 2,000 customers).
3. Scale features with StandardScaler (fitted on the training set only).
4. Fix class imbalance with **SMOTE on the training set only** (1,630 -> 6,370 churn examples). The test set is left untouched.
5. Train Logistic Regression, Random Forest, XGBoost, LightGBM and a small neural network (Keras).
6. Explain the XGBoost model with SHAP.

## Results on the test set (2,000 customers)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | 0.856 | 0.657 | 0.607 | **0.631** | **0.870** |
| Neural network | 0.833 | 0.578 | 0.658 | 0.615 | 0.857 |
| LightGBM | 0.859 | 0.693 | 0.548 | 0.612 | 0.855 |
| Random Forest | 0.842 | 0.614 | 0.602 | 0.608 | 0.850 |
| Logistic Regression | 0.716 | 0.389 | 0.698 | 0.500 | 0.777 |

Precision, recall and F1 are for the churn class.

## What I learned

- **XGBoost was the best overall**, with the highest F1 and ROC-AUC.
- **Accuracy alone is misleading here.** Predicting "nobody churns" would already be about 80% accurate. Recall and F1 on the churn class matter more.
- Logistic Regression finds the most churners (recall 0.70) but with many false alarms (precision 0.39).
- **Tuning did not help.** GridSearchCV gave a cross-validation ROC-AUC of 0.96, but the tuned model scored 0.847 on the test set, below the untuned 0.870. The cross-validation score was inflated because it was run on SMOTE-resampled data, where synthetic copies of the same customers appear in both the training and validation folds. A better approach is to put SMOTE inside the cross-validation pipeline.
- **SHAP:** Age and number of products are the two strongest churn drivers, followed by tenure and whether the customer is an active member.

## Tools

Python, Pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE), XGBoost, LightGBM, TensorFlow / Keras, SHAP, Matplotlib, Seaborn.

## Run it

Open `customer_churn_prediction.ipynb` in Google Colab or Jupyter. Download `Churn_Modelling.csv` from Kaggle (the notebook can download it if you upload your own `kaggle.json`).

## Files

| File | What it is |
|---|---|
| `customer_churn_prediction.ipynb` | Full analysis: EDA, preprocessing, five models, SHAP, tuning |
| `README.md` | This file |
