import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import pandas as pd
import numpy as np

import sys
from pathlib import Path
import time
import wandb

# PATH SETUP
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.kMeans.online_kMeans import OnlineKMeans
from src.monitoring.stream_monitor import StreamMonitor
from src.monitoring.drift_detector import DriftDetector
from src.utils.data_loader import load_preprocessed_data


# LOAD DATA
print("\n [INIT] Loading dataset...")
X_train, X_test, y_train, y_test = load_preprocessed_data("binary")

X_stream = X_test.values

print(f"Dataset loaded")
print(f"   ➜ Train shape: {X_train.shape}")
print(f"   ➜ Test shape:  {X_test.shape}")
print(f"   ➜ Stream size: {len(X_stream)} samples\n")


# INIT SYSTEM
wandb.init(project="intrusion-detection", name="online-kmeans-stream")
print("[INIT] Starting OnlineKMeans system...")
model = OnlineKMeans(k=2, fr=1.0)
monitor = StreamMonitor()
drift_detector = DriftDetector()
# --- Log Hyperparameters ---
wandb.config.update({
    "k": model.k,
    "fr": model.fr,
    "drift_threshold": drift_detector.threshold,
    "window_size": monitor.window_size,
    "batch_size": 50,
    "algorithm": "Online K-Means"
})
print(f"   ➜ k = {model.k}")
print(f"   ➜ fr = {model.fr}")
print("   ➜ Drift threshold =", drift_detector.threshold)
print("   ➜ Window size =", monitor.window_size)
print("\nStarting stream processing...\n")



# STREAM PROCESSING
batch_size = 50
total_batches = len(X_stream) // batch_size

for i in range(0, len(X_stream), batch_size):

    batch_idx = i // batch_size
    batch = X_stream[i:i+batch_size]

    print("\n" + "="*60)
    print(f" BATCH {batch_idx}/{total_batches}")
    print(f"   ➜ samples in batch: {len(batch)}")

    # --- clustering ---
    assignments = model.partial_fit_stream(batch)

    print(f"\n Clustering results:")
    unique, counts = np.unique(assignments, return_counts=True)

    for c, n in zip(unique, counts):
        print(f"   ➜ Cluster {c}: {n} samples")

    print(f"   ➜ Total clusters so far: {len(model.centers)}")

    # --- distances ---
    distances = [
        np.min([np.linalg.norm(x - c) for c in model.centers])
        for x in batch
    ]

    print(f"\nDistance stats:")
    print(f"   ➜ mean distance: {np.mean(distances):.4f}")
    print(f"   ➜ max distance:  {np.max(distances):.4f}")
    print(f"   ➜ min distance:  {np.min(distances):.4f}")

    # --- monitoring ---
    monitor.update(model, assignments, distances)

    print(f"\n System state:")
    print(f"   ➜ avg cluster count: {monitor.history['cluster_count'][-1]}")
    print(f"   ➜ entropy (chaos):   {monitor.history['entropy'][-1]:.4f}")
    print(f"   ➜ avg distance:      {monitor.history['avg_distance'][-1]:.4f}")
    wandb.log({
        "Stream/Cluster_Count": monitor.history['cluster_count'][-1],
        "Stream/Entropy": monitor.history['entropy'][-1],
        "Stream/Avg_Distance": monitor.history['avg_distance'][-1],
        "Stream/Batch_Index": batch_idx
    })
    # --- drift detection ---
    if drift_detector.check(monitor):
        print("\n DRIFT DETECTED!")
        print("   ➜ Data distribution has changed significantly")
        print("   ➜ Possible intrusion pattern or concept shift\n")
    else:
        print("\n No drift detected")





# FINAL SUMMARY & EVALUATION
print("\n" + "#"*60)
print(" STREAM FINISHED")
print(f"   ➜ Final clusters: {len(model.centers)}")

print("\n Final system summary:")
print(f"   ➜ avg distance (last): {monitor.history['avg_distance'][-1]:.4f}")
print(f"   ➜ entropy (last):      {monitor.history['entropy'][-1]:.4f}")
print("#"*60 + "\n")



# cluster assignments for test set
test_clusters = model.partial_fit_stream(X_test.values)

# mapping clusters to class labels based on majority voting
cluster_to_class = {}

df = pd.DataFrame({
    "cluster": test_clusters,
    "label": y_test.values
})

cluster_to_class = df.groupby("cluster")["label"].agg(lambda x: x.mode()[0]).to_dict()

y_pred = pd.Series(test_clusters).map(cluster_to_class).fillna(0)

print("\n FINAL EVALUATION OF ONLINE K-MEANS\n")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# --- W&B Final Evaluation Logging ---
wandb.log({
    "Final_Evaluation/Accuracy": accuracy_score(y_test, y_pred),
    "Final_Evaluation/Precision": precision_score(y_test, y_pred),
    "Final_Evaluation/Recall": recall_score(y_test, y_pred),
    "Final_Evaluation/F1-score": f1_score(y_test, y_pred)
})

wandb.finish()

tn, fp, fn, tp = cm.ravel()

print(f"""
True Negatives (normal correctly classified): {tn}
False Positives (false alarms):              {fp}
False Negatives (missed attacks):            {fn}
True Positives (attacks detected):           {tp}
""")