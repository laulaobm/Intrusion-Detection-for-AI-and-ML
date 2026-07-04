import numpy as np
import random

class OnlineKMeans:
    def __init__(self, k=2, fr=10.0, max_centers = 300):
        self.k = k
        self.fr = fr
        self.max_centers = max_centers
        self.centers = []

    def euclidean(self, a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return np.linalg.norm(a - b)

    def partial_fit_stream(self, X):
        X = np.asarray(X, dtype=np.float64)
        assignments = []

        for x in X:
            if len(self.centers) < self.k:
                self.centers.append(x)
                assignments.append(len(self.centers) - 1)
                continue

            distances = np.array([self.euclidean(x, c) for c in self.centers])
            distances = distances / (np.mean(distances) + 1e-9)
            min_dist = min(distances)

            p = min((min_dist ** 2) / self.fr, 1)

            if len(self.centers) < self.max_centers and random.random() < p:
                self.centers.append(x)
                assignments.append(len(self.centers) - 1)
            else:
                assignments.append(int(np.argmin(distances)))

        return np.array(assignments)