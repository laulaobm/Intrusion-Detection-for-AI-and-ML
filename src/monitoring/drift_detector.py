import numpy as np

class DriftDetector:
    def __init__(self, threshold=1.5):
        self.threshold = threshold

    def check(self, monitor):
        if len(monitor.history["avg_distance"]) < 20:
            return False

        recent = np.mean(monitor.history["avg_distance"][-10:])
        past = np.mean(monitor.history["avg_distance"][-50:-10])

        if past == 0:
            return False

        return (recent / past) > self.threshold