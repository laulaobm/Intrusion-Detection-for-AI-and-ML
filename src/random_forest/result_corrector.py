import numpy as np


class ResultCorrector:
    def __init__(self):
        pass

    def correct_binary(self, X_test_df, y_pred):
        corrected_pred = y_pred.copy()

        benign_override = (
                (corrected_pred == 1) &
                (X_test_df.get('sbytes', 1.0) < 1e-4) &
                (X_test_df.get('dbytes', 1.0) < 1e-4) &
                (X_test_df.get('dur', 1.0) < 1e-4)
        )
        corrected_pred[benign_override] = 0

        return corrected_pred