import numpy as np


class ResultCorrector:
    def __init__(self):
        self.attack_matrix = None

    def correct_binary(self, rf_model, X_test, threshold=0.65):
        probas = rf_model.predict_proba(X_test)[:, 1]
        corrected_pred = (probas >= threshold).astype(int)
        return corrected_pred

    def correct_multiclass(self, rf_model, X_test_df, y_pred):
        pass