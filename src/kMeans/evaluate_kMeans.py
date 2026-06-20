import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.data_loader import load_preprocessed_data
from online_kMeans import OnlineKMeans


# -----------------------------
# LOAD DATA
# -----------------------------
X_train, X_test, y_train, y_test = load_preprocessed_data("binary")

# convert everything to numpy (VERY IMPORTANT)
X_train = X_train.to_numpy(dtype=np.float64)
X_test = X_test.to_numpy(dtype=np.float64)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# -----------------------------
# OFFLINE KMEANS (SKLEARN)
# -----------------------------
print("Running Offline KMeans...")

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_train)

train_clusters = kmeans.predict(X_train)

# majority vote mapping
cluster_to_class_off = {}
for c in np.unique(train_clusters):
    mask = (train_clusters == c)
    labels = y_train[mask]
    cluster_to_class_off[c] = pd.Series(labels).mode()[0]

test_clusters = kmeans.predict(X_test)
y_pred_off = pd.Series(test_clusters).map(cluster_to_class_off).to_numpy()


# -----------------------------
# ONLINE KMEANS
# -----------------------------
print("Running Online KMeans...")

online = OnlineKMeans(k=2, fr=1.0)

train_clusters_on = online.fit_stream(X_train)
train_clusters_on = np.array(train_clusters_on)

cluster_to_class_on = {}
for c in np.unique(train_clusters_on):
    mask = (train_clusters_on == c)
    labels = y_train[mask]
    cluster_to_class_on[c] = pd.Series(labels).mode()[0]

# IMPORTANT: new model for test assignment (no retraining leakage)
online_test = OnlineKMeans(k=2, fr=1.0)
test_clusters_on = online_test.fit_stream(X_test)

y_pred_on = pd.Series(test_clusters_on).map(cluster_to_class_on)

# fill missing mappings (edge case safety)
y_pred_on = y_pred_on.fillna(0).to_numpy()


# -----------------------------
# METRICS FUNCTION
# -----------------------------

print("Evaluating models...")
def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "cm": confusion_matrix(y_true, y_pred)
    }


metrics_off = evaluate(y_test, y_pred_off)
metrics_on = evaluate(y_test, y_pred_on)


# -----------------------------
# PRINT RESULTS
# -----------------------------
print("\n=== OFFLINE KMEANS ===")
for k, v in metrics_off.items():
    if k != "cm":
        print(f"{k}: {v:.4f}")
print(metrics_off["cm"])


print("\n=== ONLINE KMEANS ===")
for k, v in metrics_on.items():
    if k != "cm":
        print(f"{k}: {v:.4f}")
print(metrics_on["cm"])


# -----------------------------
# PLOT COMPARISON
# -----------------------------
labels = ["accuracy", "precision", "recall", "f1"]

offline_vals = [metrics_off[l] for l in labels]
online_vals = [metrics_on[l] for l in labels]

x = np.arange(len(labels))

plt.figure(figsize=(8, 5))
plt.bar(x - 0.2, offline_vals, width=0.4, label="Offline KMeans")
plt.bar(x + 0.2, online_vals, width=0.4, label="Online KMeans")

plt.xticks(x, labels)
plt.ylim(0, 1)
plt.title("Offline vs Online KMeans Performance Comparison")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()