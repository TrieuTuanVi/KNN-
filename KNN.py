# Import librabries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv('practice_knn_data.csv')

# Data overview
print(df.head())
print(df['custcat'].value_counts())

# Visualize income distribution
df.hist(column='income', bins=50)
plt.show()

# Visualize multiple columns
df[["region", "tenure", "income", "ed"]].hist()
plt.show()

# Feature set
X = df[['region', 'tenure', 'age', 'marital', 'address', 
        'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values

# Labels
y = df['custcat'].values

# Normalize data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# KNN model with k=6
k = 6
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

# Prediction
yhat = neigh.predict(X_test)

# Accuracy
print("Train set Accuracy:", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy:", metrics.accuracy_score(y_test, yhat))

# Try with different Ks
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1, Ks):
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

# Plot accuracy vs K
plt.plot(range(1, Ks), mean_acc, 'g')
plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy', '+/- 3xstd'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print("The best accuracy was with", mean_acc.max(), "with k =", mean_acc.argmax() + 1)
