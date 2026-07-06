from sklearn.cluster import KMeans

import sys
from pathlib import Path
import wandb
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.data_loader import load_preprocessed_data

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

wandb.init(project="intrusion-detection", name="offline-kmeans-run")
wandb.config.update({
    "k": 2,
    "n_init": 10,
    "random_state": 42,
    "algorithm": "Offline K-Means"
})

# generate modell
k = 2 # distinguish between 0 normal and 1 attack

# define kMeans with 2 clusters, n_init=10 to run the algorithm 10 times with different centroid seeds and take the best one, random_state for reproducibility
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)


# load preprocessed data
X_train, X_test, y_train, y_test = load_preprocessed_data(task_type='binary')

# train the model
kmeans.fit(X_train)

# predict cluster labels for training data
train_clusters = kmeans.predict(X_train)

# cluster analysis
cluster_analysis = pd.DataFrame({'cluster': train_clusters, 'Label': y_train})

print(cluster_analysis.groupby("cluster")["Label"].value_counts())

# class label mapping
cluster_to_class = {}

for cluster_id in range(2):
    cluster_labels = y_train[train_clusters == cluster_id]
    majority_class = cluster_labels.mode()[0]  # get the most common class label in the cluster
    cluster_to_class[cluster_id] = majority_class

print(cluster_to_class)


# cluster testdata
test_clusters = kmeans.predict(X_test)

# map cluster labels to class labels
y_pred = pd.Series(test_clusters).map(cluster_to_class)

print("Accuracy:",
      accuracy_score(y_test, y_pred))

print("Precision:",
      precision_score(y_test, y_pred))

print("Recall:",
      recall_score(y_test, y_pred))

print("F1:",
      f1_score(y_test, y_pred))


cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

# cluster centres
print(kmeans.cluster_centers_)


wandb.log({
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-score": f1_score(y_test, y_pred)
})

wandb.finish()
