import numpy as np
import random

class OnlineKMeans:
    # k number of start clusters, fr manages how fast new clusters are created
    def __init__(self, k=2, fr=1.0):
        self.k = k
        self.fr = fr
        self.centers = []
    
    def euclidean(self,a, b):
        # change to floats
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return np.linalg.norm(a - b)
    
    def fit_stream(self, X):
        X = np.asarray(X, dtype=np.float64) # float cast
        assignments = []

        for i, x in enumerate(X):
            if len(self.centers) < self.k: # fist k vectors become initial centers
                self.centers.append(x)
                assignments.append(len(self.centers) - 1)
                continue

            # calculate distance to existing centers    
            distances = [self.euclidean(x, center) for center in self.centers]
            min_dist = min(distances)

            # probability of creating new cluster
            p = min((min_dist**2) / self.fr, 1) # formular from paper

            if random.random() < p: # create new cluster
                self.centers.append(x)
                assignments.append(len(self.centers) - 1)
            else: # assign to nearest cluster
                cluster_id = np.argmin(distances)
                assignments.append(cluster_id)

        return np.array(assignments)
