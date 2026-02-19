Bank Customer Churn Prediction
📌 Project Overview

Customer churn is a critical challenge in the banking sector, as the loss of existing customers directly impacts revenue and increases customer acquisition costs. This project develops a machine-learning framework to predict customer churn using demographic, financial, and behavioural customer data.

The objective is to identify customers at high risk of attrition and evaluate multiple predictive models under consistent experimental conditions.

🎯 Objectives

• Analyse factors influencing customer churn
• Build predictive models for churn classification
• Address class imbalance using SMOTE
• Compare classical and advanced machine-learning models
• Apply explainable AI techniques for model interpretation

🧠 Models Implemented

• Logistic Regression
• Random Forest
• XGBoost
• LightGBM
• Artificial Neural Network (ANN)

⚙️ Techniques & Methodology

• Data preprocessing & feature engineering
• Categorical feature encoding
• Feature scaling (StandardScaler)
• Class imbalance handling (SMOTE)
• Model evaluation using:

    ✔ Accuracy
    ✔ Precision
    ✔ Recall
    ✔ F1-score
    ✔ ROC–AUC

• SHAP Explainability for feature importance analysis

📊 Key Findings

• Class imbalance significantly affects churn detection
• SMOTE improves minority-class prediction performance
• XGBoost achieved the most balanced overall performance
• Customer Age, Number of Products, and Activity Status were the most influential churn drivers

🛠 Tools & Technologies

Python
Pandas / NumPy
Scikit-learn
XGBoost
LightGBM
TensorFlow / Keras
SHAP
Matplotlib / Seaborn

📈 Practical Relevance

This project demonstrates how machine-learning models can support proactive customer retention strategies by identifying churn-prone customers and providing interpretable insights for decision-making.

📂 Repository Structure
data/        → Dataset reference  
notebooks/   → Analysis & modelling notebooks  
src/         → Model training scripts  
outputs/     → Figures / evaluation visuals  

🔗 Dataset Source

Kaggle – Bank Customer Churn Modelling Dataset
