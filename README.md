# 🏦 Loan Default Prediction using Machine Learning

**Author:** Tamara Melioranskaia  
**Goal:** Develop a machine learning model to predict loan defaults and support better lending decisions in fintech.

---

## 📌 Project Overview

This project focuses on predicting whether a loan applicant is likely to default, using historical data and supervised machine learning techniques. The solution aims to assist credit analysts and fintech platforms in making smarter, data-driven loan decisions.

---

## 🚀 Project Workflow

1. **Data Collection**  
   Retrieved ~260K loan records from a Kaggle public dataset.

2. **Data Cleaning & Validation**  
   - Dropped non-informative columns (e.g. Loan ID)  
   - Removed missing or zero-value entries  
   - Standardized data types

3. **Exploratory Data Analysis (EDA)**  
   - Analyzed distributions, target imbalance, and correlations  
   - Visualized key drivers of loan default

4. **Feature Engineering**  
   - Converted boolean & categorical variables into numeric formats  
   - Applied One-Hot Encoding for multi-class variables  
   - Selected top 10 features using model-based importance and SHAP

5. **Model Selection & Training**  
   - Compared multiple models:  
     `Logistic Regression`, `Random Forest`, `XGBoost`, `LightGBM`  
   - Focused on improving **Recall** for class 1 (defaults)

6. **Hyperparameter Tuning**  
   - Used GridSearchCV and RandomizedSearchCV to optimize models

7. **Model Evaluation**  
   Evaluated using:  
   - **Recall (Class 1)** – captures defaulters  
   - **ROC-AUC** – class separation quality  
   - **PR-AUC** – better metric for imbalanced data  
   - **Accuracy** – overall performance

---

## 📊 Results Summary

| Model               | Accuracy | Recall (Class 1) | ROC-AUC | PR-AUC |
|--------------------|----------|------------------|---------|--------|
| Logistic Regression| 0.88     | 0.70             | 0.75    | 0.31   |
| LightGBM           | 0.70     | 0.67             | 0.76    | 0.33   |
| XGBoost            | 0.72     | 0.63             | 0.74    | 0.31   |
| Random Forest      | 0.88     | 0.06             | 0.74    | 0.29   |

---

## ✅ Selected Model: Logistic Regression

- Best **Recall** for defaulters (~70%)
- Transparent and easy to explain — fits credit scoring needs
- Works well with class balancing
- No overfitting observed (train ≈ test)
- Fast and scalable

---

## 💡 Business Insights

- Finding defaulters early helps reduce losses  
- Logistic Regression with class balancing catches most risky clients  
- Tree models give slightly better overall AUC but lower Recall  
- **High accuracy is not enough** — focus on **Recall** for business impact

---

## 🖥️ Demo Application (Flask)

A simple web interface was built using **Flask**.  
It allows users (e.g. credit officers) to:

- Enter loan applicant information
- Instantly receive the model’s recommendation (approve or decline)

This simulates how the model could be used in real fintech workflows.

---

## 🔮 Future Work

- Try other advanced models (e.g. Neural Networks)  
- Perform deeper hyperparameter tuning  
- Integrate the solution into a real-world fintech system  
- Explore better sampling and cost-sensitive techniques for imbalanced data

---# Project_Loan-Prediction