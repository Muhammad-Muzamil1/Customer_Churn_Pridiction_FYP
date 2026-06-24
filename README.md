# Customer Churn Prediction and Product Recommendation System

## Overview

This project predicts customer churn using Machine Learning and provides personalized product recommendations for customers who are likely to leave. The system combines customer behavioral analysis, churn prediction models, and recommendation techniques to help businesses improve customer retention and increase sales.

The project uses customer transaction data to create RFM-based features (Recency, Frequency, Monetary Value), train multiple machine learning models, identify high-risk customers, and recommend products based on customer similarity.

---

# Problem Statement

Customer retention is one of the most important challenges faced by businesses. Acquiring new customers is often more expensive than retaining existing ones.

This project aims to:

* Identify customers likely to churn.
* Analyze customer purchasing behavior.
* Compare multiple machine learning models.
* Recommend products to high-risk customers.
* Support customer retention strategies through data-driven insights.

---

# Dataset

The dataset contains customer transaction records including:

| Feature     | Description                |
| ----------- | -------------------------- |
| CustomerID  | Unique customer identifier |
| InvoiceNo   | Invoice number             |
| InvoiceDate | Date of transaction        |
| StockCode   | Product identifier         |
| Description | Product name               |
| Quantity    | Quantity purchased         |
| UnitPrice   | Price per unit             |

---

# Data Preprocessing

The following preprocessing steps were performed:

### Data Cleaning

* Removed missing Customer IDs.
* Removed transactions with negative or zero quantity.
* Converted InvoiceDate to datetime format.

### Feature Engineering

Created:

```python
TotalAmount = Quantity × UnitPrice
```

Generated customer-level features:

| Feature         | Description                           |
| --------------- | ------------------------------------- |
| recency_days    | Days since last purchase              |
| frequency       | Number of unique orders               |
| monetary        | Total spending                        |
| unique_products | Number of distinct products purchased |
| total_items     | Total quantity purchased              |
| avg_order_value | Average order value                   |

---

# Churn Definition

A customer is considered churned if:

```python
recency_days > 90
```

Otherwise:

```python
Active Customer
```

Target Variable:

| Value | Meaning          |
| ----- | ---------------- |
| 0     | Active Customer  |
| 1     | Churned Customer |

---

# RFM Analysis

The project uses RFM (Recency, Frequency, Monetary) analysis.

## Recency (R)

Measures how recently a customer purchased.

Higher score = More recent customer.

## Frequency (F)

Measures how often a customer purchases.

Higher score = More loyal customer.

## Monetary (M)

Measures total customer spending.

Higher score = More valuable customer.

Customers are segmented into 5 groups using quantiles.

---

# Machine Learning Pipeline

## Feature Selection

The following features were used:

```python
[
 'R',
 'F',
 'M',
 'frequency',
 'monetary',
 'avg_order_value',
 'unique_products',
 'total_items'
]
```

---

## Train-Test Split

```python
80% Training Data
20% Testing Data
```

Stratified sampling was used to preserve class distribution.

---

## Feature Scaling

Standardization performed using:

```python
StandardScaler()
```

This ensures all features have equal importance.

---

## Handling Class Imbalance

SMOTE (Synthetic Minority Oversampling Technique) was applied.

### Why SMOTE?

Customer churn datasets are often imbalanced.

SMOTE creates synthetic churn examples to improve model learning.

---

# Machine Learning Models

The following algorithms were trained and evaluated:

## 1. Logistic Regression

* Linear classification model
* Uses balanced class weights
* Fast and interpretable

## 2. K-Nearest Neighbors (KNN)

Parameters:

```python
n_neighbors = 7
weights = "distance"
metric = "euclidean"
```

Advantages:

* Simple and intuitive
* Effective for local patterns

## 3. Random Forest

Parameters:

```python
n_estimators = 500
max_depth = 12
```

Advantages:

* Handles non-linearity
* Robust to overfitting
* Provides feature importance

## 4. XGBoost

Parameters:

```python
n_estimators = 500
learning_rate = 0.03
max_depth = 6
```

Advantages:

* High predictive power
* Excellent performance on structured data

---

# Model Evaluation Metrics

Models were evaluated using:

### Accuracy

Measures overall correctness.

### Precision

Measures how many predicted churn customers actually churned.

### Recall

Measures how many actual churn customers were detected.

### F1 Score

Balance between precision and recall.

### ROC-AUC Score

Measures classification performance across all thresholds.

---

# Exploratory Data Analysis (EDA)

## 1. Churned vs Active Customers

Count plot showing:

* Active customers
* Churned customers

Purpose:

* Understand customer distribution.

---

## 2. Customer Churn Distribution

Pie chart showing percentage split between:

* Churned customers
* Active customers

Purpose:

* Visualize churn ratio.

---

## 3. SMOTE Distribution Comparison

Bar chart comparing:

* Before SMOTE
* After SMOTE

Purpose:

* Demonstrate class balancing effectiveness.

---

## 4. Recency vs Churn

Boxplot analysis.

Insights:

* Churned customers generally have significantly higher recency values.
* Customers inactive for longer periods are more likely to churn.

---

## 5. Spending vs Churn

Boxplot showing:

```text
Monetary Value vs Churn
```

Insights:

* Customer spending behavior differs between churned and active customers.

---

## 6. Purchase Frequency vs Churn

Boxplot showing:

```text
Frequency vs Churn
```

Insights:

* Loyal customers purchase more frequently.
* Lower frequency often correlates with churn.

---

## 7. Feature Correlation Heatmap

Correlation matrix among all features.

Purpose:

* Understand feature relationships.
* Identify strong predictors of churn.

---

## 8. Model Performance Comparison

Bar chart comparing:

* Accuracy
* Precision
* Recall
* F1 Score

for all trained models.

Purpose:

* Identify strongest performing algorithm.

---

## 9. ROC-AUC Curve

ROC curve generated for:

* Logistic Regression
* KNN
* Random Forest
* XGBoost

Purpose:

* Compare discrimination capability.

Higher AUC indicates better performance.

---

## 10. Confusion Matrix

Generated for the best-performing model.

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

Purpose:

* Detailed classification analysis.

---

## 11. Precision-Recall Curve

Evaluates model performance under class imbalance.

Useful for churn prediction scenarios.

---

## 12. Feature Importance Analysis

Random Forest feature importance visualization.

Highlights:

* Most influential churn indicators.
* Business-critical customer behaviors.

---

# Churn Probability Prediction

The best-performing model is selected automatically based on:

```python
Highest ROC-AUC Score
```

The model predicts:

```python
Churn Probability
```

for every customer.

Customers with:

```python
Probability ≥ 0.70
```

are classified as high-risk churn customers.

---

# Product Recommendation System

A recommendation engine was developed for churned customers.

## Approach

### User-Item Matrix

Created using:

```python
CustomerID × Product
```

purchase quantities.

### Customer Similarity

Calculated using:

```python
Cosine Similarity
```

### Recommendation Logic

1. Find similar customers.
2. Identify products purchased by similar customers.
3. Exclude products already purchased.
4. Rank products by popularity.
5. Recommend top products.

---

# Recommendation Analysis

## Recommendation Coverage Graph

Shows:

* Customers receiving recommendations
* Customers without recommendations

Purpose:

* Evaluate recommendation system effectiveness.

---

## Top Recommended Products

Bar chart displaying:

```text
Top 15 Recommended Products
```

Purpose:

* Identify products most likely to re-engage churned customers.

---

# Saved Artifacts

The project exports the following files:

| File                      | Purpose                             |
| ------------------------- | ----------------------------------- |
| best_churn_model.joblib   | Best trained churn prediction model |
| best_scaler.joblib        | Feature scaler                      |
| customer_features.csv     | Engineered customer features        |
| churn_recommendations.csv | Final recommendations               |
| product_lookup.csv        | Product mapping                     |
| user_item_matrix.pkl      | Customer-product matrix             |
| item_similarity.pkl       | Product similarity matrix           |

---

# Technologies Used

## Programming Language

* Python

## Data Analysis

* Pandas
* NumPy

## Visualization

* Matplotlib
* Seaborn

## Machine Learning

* Scikit-Learn
* XGBoost
* Imbalanced-Learn (SMOTE)

## Model Persistence

* Joblib
* Pickle

---

# Project Workflow

```text
Raw Transaction Data
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
RFM Analysis
        │
        ▼
Train/Test Split
        │
        ▼
Feature Scaling
        │
        ▼
SMOTE Balancing
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Best Model Selection
        │
        ▼
Churn Prediction
        │
        ▼
Customer Similarity
        │
        ▼
Product Recommendation
        │
        ▼
Retention Strategy
```

---

# Future Improvements

* Real-time churn prediction.
* Streamlit web application deployment.
* Advanced recommendation algorithms.
* Hyperparameter optimization.
* Deep learning-based churn prediction.
* Automated retention campaign generation.
* Customer lifetime value prediction.

---

# Conclusion

This project successfully combines customer churn prediction and product recommendation into a single intelligent retention system. Through RFM analysis, machine learning classification, SMOTE balancing, customer similarity modeling, and recommendation generation, the solution helps businesses identify at-risk customers and proactively engage them with relevant product suggestions.
