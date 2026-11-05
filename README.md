# Core Machine Learning with Scikit-Learn

This repository contains practical implementations of core machine learning algorithms using **Scikit-Learn**, serving as a solid foundational stepping stone toward mastering Machine Learning and Deep Learning.

---

## 📌 Overview

Understanding fundamental algorithms is essential before diving into complex architectures and deep neural networks. This repository demonstrates how classical machine learning algorithms work in practice using standard preprocessing, training, evaluation, and visualization workflows.

---

## 🧠 Implemented Algorithms & Content

### 1. Supervised Learning

#### Regression
* **Linear Regression** ([`linear_regression.py`](./linear_regression.py))
  * Continuous target prediction on the California Housing dataset.
  * Evaluates $R^2$ score, model coefficients, and intercepts.

* **Logistic Regression** ([`logistic_regression.py`](./logistic_regression.py))
  * Binary classification using the Breast Cancer dataset.
  * Performance evaluation using classification accuracy.

#### Classification
* **K-Nearest Neighbors (KNN)** ([`knn.py`](./knn.py))
  * Multi-class classification using categorical data encoding on the Car Evaluation dataset (`car.data`).
* **Support Vector Machine (SVM)** ([`svm.py`](./svm.py))
  * Classification using Support Vector Classifiers (`SVC`) on the Iris dataset.
* **Decision Tree** ([`decision_tree.py`](./decision_tree.py))
  * Decision tree classification on historical baseball data (`500hits.csv`) evaluating Hall of Fame candidacy.
* **Random Forest** ([`random_forest.py`](./random_forest.py))
  * Ensemble learning with `RandomForestClassifier` (hyperparameter tuning on estimators, split criterion, and depth).
* **Gradient Boosting Classifier (GBC)** ([`gbc.py`](./gbc.py))
  * Ensemble boosting on the Wine dataset with K-fold cross-validation and hyperparameter search (`GridSearchCV`).

---

### 2. Unsupervised Learning

#### Clustering
* **K-Means Clustering** ([`kmeans.py`](./kmeans.py))
  * Unsupervised data grouping on feature-scaled cancer data with cross-tabulation evaluation.

#### Dimensionality Reduction
* **Principal Component Analysis (PCA)** ([`pca.py`](./pca.py))
  * Feature scaling, variance ratio analysis, and 2D/3D visualization of dimensional reduction on the 2022 MLB dataset (`2022mlbteams.csv`).

---

## 📂 Included Datasets

* `car.data`: Car evaluation dataset with categorical attributes.
* `500hits.csv`: Baseball statistics used for decision tree and ensemble classification.
* `2022mlbteams.csv`: MLB season statistics used for PCA dimensionality reduction.
* *Built-in Scikit-Learn Datasets*: California Housing, Breast Cancer, Iris, and Wine datasets.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
```

### Running an Example

Run any script directly using Python:

```bash
# Example: Run Linear Regression
python linear_regression.py

# Example: Run Random Forest Classifier
python random_forest.py

# Example: Run PCA
python pca.py
```

