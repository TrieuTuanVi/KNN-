# Project Python - Machine Learning: Customer Segmentation using KNN (K-Nearest Neighbors) for a Telecom Company

This project applies the **K-Nearest Neighbors (KNN)** machine learning algorithm to classify telecom customers into four predefined categories based on demographic and behavioral features. This segmentation allows the company to make data-driven marketing decisions and improve customer service strategies.

---

## 💡 Problem Statement

A telecom company wants to predict the customer segment for new users based on historical data. Customers are classified into four groups depending on their demographics and service usage patterns:

| Value | Customer Group      |
|-------|----------------------|
| 1     | Basic Service        |
| 2     | E-Service            |
| 3     | Plus Service         |
| 4     | Total Service        |

Our task is to build a predictive model using **K-Nearest Neighbors** to automate this customer grouping process.

---

## 📂 Dataset Overview

The dataset consists of multiple customer attributes:

- **Region**: Geographic area (integer value).
- **Age**: Age of the customer.
- **Marital Status**: Whether the customer is single, married, or divorced.
- **Income**: Annual income (USD).
- **Gender**: Male (0) or Female (1).
- **Tenure**: Number of years using the service.
- **Custcat**: Target class (1-4) for customer group.

This dataset is ideal for classification tasks, especially using algorithms like KNN which rely on distance-based similarity.

---

## ⚙️ Project Structure

The project follows a clear machine learning pipeline:

### 1️⃣ Data Exploration

- Loaded the dataset using `pandas` and performed an initial data overview.
- Analyzed class distributions to understand data balance.
- Visualized data using **matplotlib** and **seaborn** for exploratory insights.

### 2️⃣ Data Preprocessing

- Selected relevant features for model training.
- Applied **normalization** to ensure uniform scaling, which is critical for distance-based models like KNN.
- Split the dataset into **training** and **testing sets** (80/20 split).

### 3️⃣ Model Building

- Used `sklearn.neighbors.KNeighborsClassifier` to implement the KNN model.
- Experimented with different `k` values to select the best neighbor count.
- Trained the model on the training data.

### 4️⃣ Model Evaluation

- Predicted labels on the test set.
- Evaluated model performance using:
  - **Accuracy Score**
  - **Confusion Matrix**
  - **Classification Report**
- Visualized the accuracy trend for different `k` values to fine-tune hyperparameters.

---

## 🔥 Model Results

The model shows reliable predictive power, achieving solid accuracy on unseen test data.

Key metrics include:
- **Accuracy Score:** ~ *[Insert your value here]*.
- **Confusion Matrix:** To analyze true positives, false negatives, and misclassifications.
- **Classification Report:** Precision, recall, and F1-score per class.

---

## 🧠 Insights & Benefits

- Identifying customer types helps the company focus resources on high-value groups.
- Segmentation supports targeted marketing campaigns.
- The model can be retrained and improved as more data is collected.

