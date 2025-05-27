# Survival-Prediction-on-the-Titanic-Dataset-A-Comparative-Study-of-Classification-Models

## Overview

This project explores advanced feature engineering and dimensionality reduction techniques to predict survival on the Titanic dataset. We evaluated the impact of interaction terms, second-order terms, and PCA on the performance of five classification models.

## 📌 Objectives

- Preprocess and clean the Titanic dataset
- Engineer features using interaction and polynomial terms
- Reduce dimensionality with Principal Component Analysis (PCA)
- Build and evaluate classification models:
  - Logistic Regression
  - Naive Bayes
  - Linear Discriminant Analysis (LDA)
  - Quadratic Discriminant Analysis (QDA)
  - k-Nearest Neighbors (KNN)
- Compare model performance across three settings:
  - Main features only
  - Interaction and second-order terms
  - PCA-transformed features
- Validate results using 10-fold cross-validation and ROC/AUC metrics

## 🧠 Methodology

### Data Preprocessing
- Imputed missing values in `Age` and `Embarked`
- Converted categorical variables to factors
- Standardized numeric features
- Removed irrelevant fields (`Name`, `Cabin`, `Ticket`)

### Feature Engineering
- Created interaction terms like `Age × Sex`, `Fare × Pclass`, etc.
- Added second-order terms (e.g., `Age²`, `Fare²`) — later removed due to overfitting
- Performed correlation analysis to inform feature construction

### Dimensionality Reduction
- Applied PCA to numeric variables
- Retained 6 principal components capturing ~86% variance

### Model Building
Each model was trained under:
- Main features
- Interaction & second-order terms
- PCA-transformed features

### Model Evaluation
- 80/20 Train-Test Split
- ROC curves and AUC scores
- 10-Fold Cross-Validation for robustness

## 📊 Results Summary

| Model                   | Main AUC | Interaction AUC | PCA AUC |
|------------------------|----------|------------------|---------|
| Logistic Regression    | 0.838    | **0.856**        | 0.845   |
| Naive Bayes            | 0.808    | 0.802            | **0.844** |
| LDA                    | 0.831    | **0.852**        | 0.843   |
| QDA                    | 0.832    | 0.797            | **0.845** |
| KNN                    | **0.780**| 0.758            | 0.764   |

- **Best Overall Model:** LDA with interaction terms (AUC = 0.864, via cross-validation)
- **Best PCA-Boosted Model:** Naive Bayes
- **Least Effective Algorithm:** KNN (most sensitive to high-dimensional representation)

## 📚 References

- Acun et al. (2018). *A Comparative Study on Machine Learning Techniques Using Titanic Dataset.*
- [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)

## 🧩 Future Improvements

- Test ensemble methods (e.g., Random Forest, XGBoost)
- Automate feature selection with LASSO or RFE
- Tune hyperparameters across all models
- Deploy an interactive web app for real-time prediction

