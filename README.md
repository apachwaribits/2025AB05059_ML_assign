---

# Student Academic Success Prediction Project

## a. Problem Statement

The objective of this project is to predict student academic outcomes—specifically identifying students at risk of dropping out versus those likely to graduate—using early-semester performance and demographic data.

By leveraging machine learning, institutions can implement proactive intervention strategies to improve student retention rates.

---

## b. Dataset Description *(1 Mark)*

The project utilizes the **Predict Students' Dropout and Academic Success** dataset from the UCI Machine Learning Repository.

* **Dataset Name:** Predict Students' Dropout and Academic Success
* **Problem Type:** Classification
* **Target Variable:** `Target` *(Multi-class: Graduate, Dropout, Enrolled)*
* **Number of Instances:** 4,424
* **Number of Features:** 36

### Class Distribution

* **Graduate:** 2,209
* **Dropout:** 1,421
* **Enrolled:** 794

---

## c. Models Used *(6 Marks)*

The following table summarizes the performance of six classification models across key evaluation metrics:

| ML Model Name            | Accuracy | AUC      | Precision | Recall   | F1 Score | MCC      |
| ------------------------ | -------- | -------- | --------- | -------- | -------- | -------- |
| Logistic Regression      | 0.752542 | 0.869383 | 0.734083  | 0.752542 | 0.736308 | 0.593230 |
| Decision Tree            | 0.716384 | 0.768441 | 0.710379  | 0.716384 | 0.708616 | 0.537111 |
| kNN                      | 0.698305 | 0.791192 | 0.689086  | 0.698305 | 0.687903 | 0.504966 |
| Naive Bayes              | 0.692655 | 0.801522 | 0.680850  | 0.692655 | 0.679823 | 0.494250 |
| Random Forest (Ensemble) | 0.766102 | 0.875226 | 0.750497  | 0.766102 | 0.749871 | 0.616348 |
| XGBoost (Ensemble)       | 0.763842 | 0.875032 | 0.754142  | 0.763842 | 0.755302 | 0.613874 |

---

## Observations on Model Performance *(3 Marks)*

### Logistic Regression

* Strong linear baseline model.
* High AUC (0.869383) indicates strong separability.
* Reliable and interpretable performer.

### Decision Tree

* Moderate performance.
* Lower MCC (0.537111) suggests higher variance.
* Less stable compared to ensemble models.

### kNN

* Accuracy: 0.698305.
* Performance limited in high-dimensional feature space.
* Distance-based learning less effective here.

### Naive Bayes

* Lowest MCC (0.494250).
* Independence assumption likely violated due to correlated academic features.

### Random Forest (Ensemble)

* Highest Accuracy (0.766102).
* Highest MCC (0.616348).
* Very stable and reduced overfitting compared to Decision Tree.

### XGBoost (Ensemble)

* Nearly identical performance to Random Forest.
* Highest AUC (0.875032).
* Excellent at modeling complex, non-linear relationships.

---
