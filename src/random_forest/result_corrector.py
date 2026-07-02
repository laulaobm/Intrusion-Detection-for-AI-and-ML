import numpy as np


class ResultCorrector:
    def __init__(self):
        self.attack_matrix = None

    def correct_binary(self, rf_model, X_test, threshold=0.65):
        probas = rf_model.predict_proba(X_test)[:, 1]
        corrected_pred = (probas >= threshold).astype(int)
        return corrected_pred

    def correct_multiclass(self, X_test_df, y_pred_encoded, label_encoder):
        corrected_pred = y_pred_encoded.copy()

        # Example logic based on Section 2.1.1
        # High sload/dload often indicates DoS or Generic attacks [cite: 127, 155]
        dos_mask = (X_test_df['sload'] > 0.8) & (X_test_df['dload'] > 0.8)

        # Reconnaissance often has very short duration [cite: 131]
        recon_mask = (X_test_df['dur'] < 0.01) & (X_test_df['spkts'] < 0.02)

        # Mapping rules
        recon_idx = np.where(label_encoder.classes_ == 'Reconnaissance')[0][0]
        dos_idx = np.where(label_encoder.classes_ == 'DoS')[0][0]

        corrected_pred[dos_mask] = dos_idx
        corrected_pred[recon_mask] = recon_idx

        return corrected_pred