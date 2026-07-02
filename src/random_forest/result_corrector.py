import numpy as np


class ResultCorrector:
    def __init__(self):
        self.attack_matrix = None

    def correct_binary(self, rf_model, X_test, threshold=0.65):
        probas = rf_model.predict_proba(X_test)[:, 1]
        corrected_pred = (probas >= threshold).astype(int)
        return corrected_pred

    def correct_multiclass(self, rf_model, X_test_df, label_encoder, confidence_threshold=0.55):
        probas = rf_model.predict_proba(X_test_df)
        max_probas = np.max(probas, axis=1)
        raw_predictions = np.argmax(probas, axis=1)

        corrected_pred = raw_predictions.copy()

        uncertain_mask = max_probas < confidence_threshold

        dos_idx = np.where(label_encoder.classes_ == 'DoS')[0][0]
        recon_idx = np.where(label_encoder.classes_ == 'Reconnaissance')[0][0]

        dos_condition = (X_test_df.get('sload', 0) > 0.95) & (X_test_df.get('dload', 0) > 0.95) & uncertain_mask
        recon_condition = (X_test_df.get('dur', 1) < 1e-4) & (X_test_df.get('spkts', 1) < 1e-4) & uncertain_mask

        corrected_pred[dos_condition] = dos_idx
        corrected_pred[recon_condition] = recon_idx

        return corrected_pred