# 🏦 Predicting Loan Payback – Kaggle Playground Series S5E11

This repository contains my solution for the Kaggle competition  
**“Predicting Loan Payback” – Playground Series Season 5, Episode 11**.

🔗 Competition link:  
https://www.kaggle.com/competitions/playground-series-s5e11

---

## 📌 Problem Statement

Given a synthetic tabular dataset representing borrower and loan information,  
the objective is to **predict the probability that a borrower will pay back their loan**.

- **Task type:** Binary classification (probability prediction)
- **Target column:** `loan_paid_back`
- **Evaluation metric:** ROC-AUC

---

## 🧠 Approach Overview

The solution focuses on a **strong, carefully tuned LightGBM model**, which is well-suited for tabular data.  
Instead of complex deep learning or heavy ensembles, this approach prioritizes:

- Stability  
- Generalization  
- Kaggle leaderboard performance  

Key highlights:
- LightGBM (Gradient Boosted Decision Trees)
- Stratified 5-Fold Cross-Validation
- Multi-Seed Training for variance reduction
- Robust preprocessing for numeric and categorical features
- Class-imbalance handling

---

## ⚙️ Data Preprocessing

### Handling Missing Values
- **Numerical features:** Median imputation
- **Categorical features:** Filled with a constant value (`__MISSING__`)

### Categorical Encoding
- **Ordinal Encoding** is used
- Unseen categories in the test set are safely handled (`unknown_value = -1`)

### Feature Detection
- Object-type columns are treated as categorical
- Low-cardinality integer columns are also treated as categorical

---

## ⚖️ Class Imbalance Handling

Class imbalance is handled using **balanced class weights**, computed from the training data and passed directly to the LightGBM model.

This ensures that minority and majority classes contribute equally during training.

---

## 🚀 Model & Training Strategy

### Model
- **LightGBM (LGBMClassifier)**
- Gradient Boosted Decision Trees

### Training Strategy
- **Stratified 5-Fold Cross-Validation**
- **Multi-Seed Training** (Seeds: `42`, `2024`, `7`)
- Predictions are averaged across:
  - All folds
  - All seeds

This reduces variance and improves generalization on the leaderboard.

---

## 🔧 Key Hyperparameters

```text
learning_rate        = 0.015
num_leaves           = 63
feature_fraction     = 0.7
bagging_fraction     = 0.8
lambda_l1            = 0.3
lambda_l2            = 0.6
n_estimators         = 7000 (with early stopping)
early_stopping_rounds= 200
