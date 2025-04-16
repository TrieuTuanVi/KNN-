# Project Python Machine Learning: Customer Segmentation using KNN (K-Nearest Neighbors) for a Telecom Company

This project applies the **K-Nearest Neighbors (KNN)** algorithm to classify new telecom customers into predefined customer groups based on demographic data such as **region, age, and marital status**. This segmentation helps the company optimize marketing strategies and personalize service offerings.

---

## 🎯 Project Objective

A telecom company has classified its customers into **four distinct groups** based on demographic features.  
Our goal is to develop a **KNN model** that can automatically assign **new customers** to one of these four groups.

The target variable is called `custcat` and has four possible values:

| Value | Customer Group |
|-------|----------------|
| 1     | Basic Service  |
| 2     | E-Service      |
| 3     | Plus Service   |
| 4     | Total Service  |

---

## 📊 Dataset Overview

The dataset contains demographic and account-related information for each customer:

- **Region** - Geographic area of the customer.
- **Age** - Customer age.
- **Marital Status** - Whether the customer is single, married, or divorced.
- **Income** - Annual income.
- **Gender** - Male or Female.
- **Tenure** - Years the customer has used the service.
- **Custcat** - The target class label (1-4) representing the customer group.

---

## ⚙️ Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib & Seaborn
---

## 📝 Methodology

1. **Data Preprocessing**
    - Handle missing values.
    - Normalize feature values for optimal KNN performance.

2. **Splitting Dataset**
    - Train-test split (typically 80% training, 20% testing).

3. **Model Training**
    - Use **K-Nearest Neighbors** algorithm.
    - Hyperparameter tuning for the optimal `K` using accuracy and error plots.

4. **Model Evaluation**
    - Confusion matrix.
    - Accuracy score.
    - Classification report.
