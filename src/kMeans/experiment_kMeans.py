from online_kMeans import OnlineKMeans
from offline_kMeans import KMeans
from src.utils.data_loader import load_preprocessed_data
from sklearn.metrics import f1_score, confusion_matrix
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.data_loader import load_preprocessed_data


# load preprocessed data
X_train, X_test, y_train, y_test = load_preprocessed_data("binary")
X_train = X_train.astype(float)
X_test = X_test.astype(float)


# online simulation

model = OnlineKMeans(k=2, fr=1.0)

train_clusters = model.fit_stream(X_train)

cluster_to_class = {}

for c in set(train_clusters):
    labels = y_train[train_clusters == c]
    cluster_to_class[c] = labels.mode()[0]

test_clusters = model.fit_stream(X_test)

y_pred = pd.Series(test_clusters).map(cluster_to_class)

print("F1:", f1_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))