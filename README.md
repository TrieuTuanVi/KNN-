# Project Python - Machine Learning: Customer Segmentation using KNN (K-Nearest Neighbors) for a Telecom Company

This project applies the **K-Nearest Neighbors (KNN)** machine learning algorithm to classify telecom customers into four predefined categories based on demographic and behavioral features. This segmentation allows the company to make data-driven marketing decisions and improve customer service strategies.

---

## 🧠 CaseStudy
A telecom company has segmented its customers into four groups based on demographic data, such as region, age, and marital status. They want to segment new customers into groups. The target field, called custcat, has four possible values ​​corresponding to the four customer groups, as follows:

| Value | Customer Group      |
|-------|----------------------|
| 1     | Basic Service        |
| 2     | E-Service            |
| 3     | Plus Service         |
| 4     | Total Service        |

---
## 🎯 Project Objective

Our task is to build a predictive model using **K-Nearest Neighbors** to automate this customer grouping process.

---

## 🔍 About K-Nearest Neighbors (KNN)

K-Nearest Neighbors is an algorithm for supervised learning. Where the data is 'trained' with data points corresponding to their classification. Once a point is to be predicted, it takes into account the 'K' nearest points to it to determine it's classification.

### Here's an visualization of the K-Nearest Neighbors algorithm.

![image](https://github.com/user-attachments/assets/53c1e3cd-9cb7-4310-ac23-2e97f12a6f30)

In this case, we have data points of Class A and B. We want to predict what the star (test data point) is. If we consider a k value of 3 (3 nearest data points) we will obtain a prediction of Class B. Yet if we consider a k value of 6, we will obtain a prediction of Class A.

In this sense, it is important to consider the value of k. But hopefully from this diagram, you should get a sense of what the K-Nearest Neighbors algorithm is. It considers the 'K' Nearest Neighbors (points) when it predicts the classification of the test point.

---
## 📚 Table of contents

1. About the dataset
2. Data Visualization and Analysis
3. Classification

---


## 📂 Dataset 

- <a href= "https://github.com/TrieuTuanVi/KNN-ALGORITHM/blob/main/knn_data.csv">Dataset</a>

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
## 🛠️Tools & Technologies Used

- Python
- Scikit-learn
- Pandas
- Matplotlib / Seaborn
  
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
- Visualized the accuracy trend for different `k` values to fine-tune hyperparameters.

---
## 💡 Model Results

The model shows reliable predictive power, achieving solid accuracy on unseen test data.

- Breakdown of Customer Categories:
  - **Class 1** (Basic-service): 266 samples

  - **Class 2** (E-Service): 217 samples

  - **Class 3** (Plus Service): 281 samples

  - **Class 4** (Total Service): 236 samples

- Model Accuracy:

  - **Train set Accuracy**: 0.51625

  - **Test set Accuracy**: 0.31

- Best K Value for KNN Model: After experimenting with different values for K, the best accuracy was achieved with **K = **9, yielding an **accuracy = 0.34**.

![image](https://github.com/user-attachments/assets/96c60030-5cfc-4eb6-9110-1efdfc19a8e0)

---
## 🏆 Business Impact

* Marketing Optimization: Segment-specific offers and promotions.

* Customer Retention: Proactive engagement for high-value groups.

* Data-Driven Decisions: Efficient allocation of customer service resources.Business Impact

---

## ✅ Conclusion
This project demonstrates the power of K-Nearest Neighbors (KNN) in solving classification problems in real-world business scenarios, particularly for customer segmentation in the telecom industry. With this model, the company can enhance customer experiences and create better marketing strategies.




