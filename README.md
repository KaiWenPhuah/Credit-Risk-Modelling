Credit Risk Modelling

A machine learning project to predict loan defaults using feature engineering, model training (Logistic Regression, Gradient Boosting, LightGBM), and explainability with SHAP. The project also includes a Streamlit app for interactive predictions on new loan data.

Features

Data Preprocessing & Cleaning
- Handle missing values and drop high-missing columns
- Standardize categorical variables and merge rare categories
- Extract features from dates (loan age, credit history length, etc.)
- Encode categorical variables with WoE/IV and one-hot encoding

Feature Engineering
- Create flags for recent payments and credit pulls
- Drop redundant or highly correlated features

Feature selection using Random Forest importance
- Modeling

Train and evaluate multiple models:
- Logistic Regression
- Gradient Boosting
- LightGBM

Metrics used: 
- Accuracy, Recall, F2-Score, AUC
- ROC curve comparisons

Model Explainability
- SHAP summary plots for global feature importance
- SHAP force plots for individual loan predictions

Deployment
Streamlit app to:
- Upload CSV loan data
- Predict default probabilities

Adjust threshold for classification

Download predictions
