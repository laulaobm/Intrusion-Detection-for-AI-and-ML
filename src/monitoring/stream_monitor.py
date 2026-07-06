import numpy as np
from collections import deque

class StreamMonitor:
    def __init__(self, window_size=200):
        self.window_size = window_size

        self.recent_distances = deque(maxlen=window_size)
        self.history = {
            "cluster_count": [],
            "avg_distance": [],
            "entropy": []
        }

    def entropy(self, assignments):
        values, counts = np.unique(assignments, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-9))

    def update(self, model, assignments, distances):
        self.history["cluster_count"].append(len(model.centers))
        self.history["avg_distance"].append(np.mean(distances))
        self.history["entropy"].append(self.entropy(assignments))

        self.recent_distances.extend(distances)