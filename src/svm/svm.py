import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    precision_score, recall_score, matthews_corrcoef
)
import wandb

try:
    import cupy as cp
    import cudf
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.svm import SVC as cuSVC

    GPU_AVAILABLE = cp.cuda.is_available()
except Exception:
    GPU_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class GPUMulticlassSVC:
    def __init__(self, C=100.0, gamma=1.0, kernel='rbf'):
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.models = {}
        self.classes_ = None

    def fit(self, X, y):
        import cudf
        if hasattr(y, 'to_numpy'):
            y_np = y.to_numpy()
        else:
            y_np = np.array(y)

        self.classes_ = np.unique(y_np)
        X_gpu = cudf.from_pandas(X.astype(np.float32)) if isinstance(X, pd.DataFrame) else cudf.DataFrame(
            X.astype(np.float32))

        counts = pd.Series(y_np).value_counts()
        total = len(y_np)

        for c in self.classes_:
            y_bin = (y_np == c).astype(np.int32)
            y_gpu = cudf.Series(y_bin)

            class_weight_ratio = total / (len(self.classes_) * counts[c])

            model = cuSVC(C=self.C * class_weight_ratio, gamma=self.gamma, kernel=self.kernel)
            model.fit(X_gpu, y_gpu)
            self.models[c] = model
        return self

    def predict(self, X):
        import cudf
        X_gpu = cudf.from_pandas(X.astype(np.float32)) if isinstance(X, pd.DataFrame) else cudf.DataFrame(
            X.astype(np.float32))

        scores = []
        for c in self.classes_:
            df_score = self.models[c].decision_function(X_gpu)
            if hasattr(df_score, 'to_numpy'):
                df_score = df_score.to_numpy()
            else:
                df_score = cp.asnumpy(df_score)
            scores.append(df_score)

        scores = np.column_stack(scores)
        preds_idx = np.argmax(scores, axis=1)
        return self.classes_[preds_idx]


train_path = '../../data/UNSW_NB15_training-set.csv'
test_path = '../../data/UNSW_NB15_testing-set.csv'

data_dir = '/kaggle/working'
output_dir = '/kaggle/working'
os.makedirs(output_dir, exist_ok=True)

MULTI_SMOTE_MAX_RATIO = 15
MULTI_SMOTE_TARGET_CAP = 20000


def build_capped_smote_strategy(y, max_ratio=MULTI_SMOTE_MAX_RATIO, target_cap=MULTI_SMOTE_TARGET_CAP):
    counts = pd.Series(y).value_counts()
    strategy = {}
    for cls, count in counts.items():
        target = min(target_cap, count * max_ratio)
        target = max(target, count)
        strategy[cls] = int(target)
    return strategy


def run_preprocessing():
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = train_df.drop(columns=['id'])
    test_df = test_df.drop(columns=['id'])

    X_train = train_df.drop(columns=['attack_cat', 'label'])
    y_train_binary = train_df['label']
    y_train_multi = train_df['attack_cat']

    X_test = test_df.drop(columns=['attack_cat', 'label'])
    y_test_binary = test_df['label']
    y_test_multi = test_df['attack_cat']

    categorical_cols = ['proto', 'service', 'state']
    X_train = pd.get_dummies(X_train, columns=categorical_cols)
    X_test = pd.get_dummies(X_test, columns=categorical_cols)

    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    label_encoder = LabelEncoder()
    y_train_multi_encoded = label_encoder.fit_transform(y_train_multi)
    y_test_multi_encoded = label_encoder.transform(y_test_multi)

    scaler = MinMaxScaler()
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64', 'int32', 'float32', 'uint8']).columns

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    smote = SMOTE(random_state=42)
    multi_strategy = build_capped_smote_strategy(y_train_multi_encoded)
    smote_multi = SMOTE(random_state=42, sampling_strategy=multi_strategy)

    X_train_multi_resampled, y_train_multi_resampled = smote_multi.fit_resample(X_train, y_train_multi_encoded)
    X_train_binary_resampled, y_train_binary_resampled = smote.fit_resample(X_train, y_train_binary)

    joblib.dump(scaler, os.path.join(output_dir, 'min_max_scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))

    X_train_multi_resampled.to_parquet(os.path.join(output_dir, 'X_train_multi_resampled.parquet'))
    X_train_binary_resampled.to_parquet(os.path.join(output_dir, 'X_train_binary_resampled.parquet'))
    X_test.to_parquet(os.path.join(output_dir, 'X_test.parquet'))

    pd.Series(y_train_multi_resampled, name='target').to_frame().to_parquet(
        os.path.join(output_dir, 'y_train_multi_resampled.parquet'))
    pd.Series(y_train_binary_resampled, name='target').to_frame().to_parquet(
        os.path.join(output_dir, 'y_train_binary_resampled.parquet'))
    pd.Series(y_test_multi_encoded, name='target').to_frame().to_parquet(
        os.path.join(output_dir, 'y_test_multi.parquet'))
    pd.Series(y_test_binary, name='target').to_frame().to_parquet(os.path.join(output_dir, 'y_test_binary.parquet'))


def ensure_preprocessed_data_exists():
    required_files = [
        'X_test.parquet', 'X_train_binary_resampled.parquet', 'X_train_multi_resampled.parquet',
        'y_train_binary_resampled.parquet', 'y_train_multi_resampled.parquet',
        'y_test_binary.parquet', 'y_test_multi.parquet', 'label_encoder.pkl',
    ]
    missing = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        run_preprocessing()


def load_preprocessed_data(task_type='binary'):
    X_test_loaded = pd.read_parquet(os.path.join(data_dir, 'X_test.parquet'))
    if task_type == 'binary':
        X_train_loaded = pd.read_parquet(os.path.join(data_dir, 'X_train_binary_resampled.parquet'))
        y_train_loaded = pd.read_parquet(os.path.join(data_dir, 'y_train_binary_resampled.parquet'))['target']
        y_test_loaded = pd.read_parquet(os.path.join(data_dir, 'y_test_binary.parquet'))['target']
    elif task_type == 'multi':
        X_train_loaded = pd.read_parquet(os.path.join(data_dir, 'X_train_multi_resampled.parquet'))
        y_train_loaded = pd.read_parquet(os.path.join(data_dir, 'y_train_multi_resampled.parquet'))['target']
        y_test_loaded = pd.read_parquet(os.path.join(data_dir, 'y_test_multi.parquet'))['target']
    return X_train_loaded, X_test_loaded, y_train_loaded, y_test_loaded


RF_N_ESTIMATORS = 250
RF_RANDOM_STATE = 42

RF_TOP_K = 45
RF_TOP_K_MULTI = 55

SVM_KERNEL = 'rbf'
SVM_C = 100.0
SVM_GAMMA = 1.0

ENABLE_SVM_HYPERPARAM_SEARCH = True
SVM_PARAM_GRID = {
    'C': [50, 100, 150, 200, 250, 300],
    'gamma': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
}
VALIDATION_SPLIT_SIZE = 0.2


def select_features_with_rf(X_train, y_train, X_test, top_k=RF_TOP_K):
    feature_names = X_train.columns.tolist()
    if GPU_AVAILABLE:
        X_train_gpu = cudf.from_pandas(X_train.astype(np.float32))
        y_train_gpu = cudf.Series(y_train.to_numpy().astype(np.int32))
        rf = cuRF(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE)
        rf.fit(X_train_gpu, y_train_gpu)
        importances = pd.Series(cp.asnumpy(rf.feature_importances_), index=feature_names)
    else:
        rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        importances = pd.Series(rf.feature_importances_, index=feature_names)

    top_features = importances.sort_values(ascending=False).head(top_k).index.tolist()
    return X_train[top_features], X_test[top_features], top_features, rf


def tune_svm_hyperparameters(X_train, y_train, param_grid=SVM_PARAM_GRID,
                             kernel=SVM_KERNEL, val_size=VALIDATION_SPLIT_SIZE,
                             scoring='macro', task_type='binary'):
    max_tune_samples = 25000
    if len(X_train) > max_tune_samples:
        _, X_tune, _, y_tune = train_test_split(X_train, y_train, test_size=max_tune_samples, random_state=42,
                                                stratify=y_train)
    else:
        X_tune, y_tune = X_train, y_train

    X_tr, X_val, y_tr, y_val = train_test_split(X_tune, y_tune, test_size=val_size, random_state=42, stratify=y_tune)
    best_score = -1.0
    best_params = {'C': param_grid['C'][0], 'gamma': param_grid['gamma'][0]}

    tuning_history = []

    print(f"\n--- SVM Hyperparamete Search (C, gamma) | scoring={scoring} ---")
    for C in param_grid['C']:
        for gamma in param_grid['gamma']:
            if GPU_AVAILABLE:
                if task_type == 'multi':
                    model = GPUMulticlassSVC(kernel=kernel, C=C, gamma=gamma)
                    model.fit(X_tr, y_tr)
                    y_val_pred = model.predict(X_val)
                else:
                    X_tr_gpu = cudf.from_pandas(X_tr.astype(np.float32))
                    y_tr_gpu = cudf.Series(y_tr.to_numpy().astype(np.float32))
                    X_val_gpu = cudf.from_pandas(X_val.astype(np.float32))

                    model = cuSVC(kernel=kernel, C=C * 1.5, gamma=gamma)
                    model.fit(X_tr_gpu, y_tr_gpu)
                    y_val_pred = model.predict(X_val_gpu)
                    y_val_pred = y_val_pred.to_numpy() if hasattr(y_val_pred, 'to_numpy') else cp.asnumpy(y_val_pred)
                    y_val_pred = np.round(y_val_pred).astype(int)
            else:
                model = SVC(kernel=kernel, C=C, gamma=gamma, class_weight='balanced', random_state=42)
                model.fit(X_tr, y_tr)
                y_val_pred = model.predict(X_val)

            score = f1_score(y_val, y_val_pred, average=scoring, zero_division=0)
            print(f"  C={C:<6} gamma={str(gamma):<8} -> {scoring} F1 (Validation): {score:.4f}")

            tuning_history.append({'C': C, 'gamma': gamma, 'F1_Score': score})

            if score > best_score:
                best_score = score
                best_params = {'C': C, 'gamma': gamma}

    print(f"Best parameters ({task_type}): C={best_params['C']}, gamma={best_params['gamma']}")

    try:
        df_plot = pd.DataFrame(tuning_history)
        pivot_table = df_plot.pivot(index='C', columns='gamma', values='F1_Score')

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="viridis",
                    cbar_kws={'label': f'Validation F1 ({scoring})'})
        plt.title(f"SVM Hyperparameter Tuning Grid - Task: {task_type.upper()}", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Gamma", fontsize=12)
        plt.ylabel("C", fontsize=12)
        plt.tight_layout()

        plot_filename = os.path.join(output_dir, f'svm_tuning_{task_type}.png')
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        print(f"Heatmap generated: {plot_filename}")
    except Exception as e:
        print(f"Cannot generate heatmap: {e}")

    return best_params


def train_svm(X_train, y_train, C=SVM_C, gamma=SVM_GAMMA, kernel=SVM_KERNEL, task_type='binary'):
    if GPU_AVAILABLE:
        if task_type == 'multi':
            model = GPUMulticlassSVC(C=C, gamma=gamma, kernel=kernel)
            model.fit(X_train, y_train)
            return model
        else:
            X_train_gpu = cudf.from_pandas(X_train.astype(np.float32))
            y_train_gpu = cudf.Series(y_train.to_numpy().astype(np.float32))
            svm = cuSVC(kernel=kernel, C=C * 1.65, gamma=gamma)
            svm.fit(X_train_gpu, y_train_gpu)
            return svm
    else:
        svm = SVC(kernel=kernel, C=C, gamma=gamma, class_weight='balanced', random_state=42)
        svm.fit(X_train, y_train)
        return svm


def predict_svm(model, X_test, threshold=-0.35):
    if isinstance(model, GPUMulticlassSVC):
        return model.predict(X_test)

    if hasattr(model, "classes_") and len(model.classes_) > 2:
        return model.predict(X_test)

    if "cuml" in str(type(model)):
        X_test_gpu = cudf.from_pandas(X_test.astype(np.float32))
        decision_scores = model.decision_function(X_test_gpu)

        if hasattr(decision_scores, 'to_numpy'):
            decision_scores = decision_scores.to_numpy()
        else:
            decision_scores = cp.asnumpy(decision_scores)

        return (decision_scores > threshold).astype(int)

    decision_scores = model.decision_function(X_test)
    return (decision_scores > threshold).astype(int)


def compute_fpr(y_true, y_pred, average='macro'):
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    fprs, supports = [], []
    for i in range(n_classes):
        FP = cm[:, i].sum() - cm[i, i]
        TN = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fprs.append(FP / (FP + TN) if (FP + TN) > 0 else 0.0)
        supports.append(cm[i, :].sum())
    if average == 'weighted':
        return float(np.average(fprs, weights=supports))
    return float(np.mean(fprs))


def evaluate_model(model, X_test, y_test, label_names=None, average_strategy='macro', task_type='binary'):
    custom_threshold = -0.65 if task_type == 'binary' else 0.0

    if task_type == 'binary':
        y_pred = predict_svm(model, X_test, threshold=custom_threshold)
    else:
        y_pred = predict_svm(model, X_test)

    print("\n--- Final Classification Report (TEST SET) ---")
    print(classification_report(y_test, y_pred, target_names=label_names))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=average_strategy, zero_division=0)
    rec = recall_score(y_test, y_pred, average=average_strategy, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average_strategy, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    fpr = compute_fpr(y_test, y_pred, average=average_strategy)

    print("=" * 50)
    print(f"Accuracy:                          {acc:.4f}")
    print(f"Precision:                         {prec:.4f}")
    print(f"Recall:                            {rec:.4f}")
    print(f"F1-Score:                          {f1:.4f}")
    print(f"MCC (Matthews-Korrelation):        {mcc:.4f}")
    print(f"FPR:                               {fpr:.4f}")

    return {
        "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1-Score": f1, "MCC": mcc, "FPR": fpr
    }


def run_hierarchical_pipeline():
    print(f"\n=============== Task: BINARY (Normal vs. Attack) ===============")
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = load_preprocessed_data(task_type='binary')

    X_train_bin_sel, X_test_bin_sel, top_features_bin, rf_model_bin = select_features_with_rf(
        X_train_bin, y_train_bin, X_test_bin, top_k=RF_TOP_K
    )

    if ENABLE_SVM_HYPERPARAM_SEARCH:
        best_params_bin = tune_svm_hyperparameters(
            X_train_bin_sel, y_train_bin, scoring='macro', task_type='binary'
        )
        svm_model_bin = train_svm(
            X_train_bin_sel, y_train_bin, C=best_params_bin['C'], gamma=best_params_bin['gamma'], task_type='binary'
        )
    else:
        svm_model_bin = train_svm(X_train_bin_sel, y_train_bin, task_type='binary')

    bin_metrics = evaluate_model(svm_model_bin, X_test_bin_sel, y_test_bin, label_names=None, average_strategy='macro',
                                 task_type='binary')

    wandb.log({f"Binary/{k}": v for k, v in bin_metrics.items()})

    if ENABLE_SVM_HYPERPARAM_SEARCH:
        wandb.config.update({"Binary_Best_C": best_params_bin['C'], "Binary_Best_Gamma": best_params_bin['gamma']})
        if os.path.exists(os.path.join(output_dir, 'svm_tuning_binary.png')):
            wandb.log({"Binary_Tuning_Heatmap": wandb.Image(os.path.join(output_dir, 'svm_tuning_binary.png'))})

    joblib.dump(rf_model_bin, os.path.join(output_dir, 'rf_model_binary.pkl'))
    joblib.dump(svm_model_bin, os.path.join(output_dir, 'svm_model_binary.pkl'))
    joblib.dump(top_features_bin, os.path.join(output_dir, 'selected_features_binary.pkl'))

    print(f"\n=============== Task: MULTICLASS (Hierarchical Approach) ===============")
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = load_preprocessed_data(task_type='multi')

    label_encoder = joblib.load(os.path.join(data_dir, 'label_encoder.pkl'))
    label_names = label_encoder.classes_
    normal_encoded_val = label_encoder.transform(['Normal'])[0]

    attack_mask = (y_train_multi != normal_encoded_val)
    X_train_multi_attacks = X_train_multi[attack_mask]
    y_train_multi_attacks = y_train_multi[attack_mask]

    X_train_multi_sel, X_test_multi_sel, top_features_multi, rf_model_multi = select_features_with_rf(
        X_train_multi_attacks, y_train_multi_attacks, X_test_multi, top_k=RF_TOP_K_MULTI
    )

    if ENABLE_SVM_HYPERPARAM_SEARCH:
        best_params_multi = tune_svm_hyperparameters(
            X_train_multi_sel, y_train_multi_attacks, scoring='macro', task_type='multi'
        )
        svm_model_multi = train_svm(
            X_train_multi_sel, y_train_multi_attacks, C=best_params_multi['C'], gamma=best_params_multi['gamma'],
            task_type='multi'
        )
    else:
        svm_model_multi = train_svm(X_train_multi_sel, y_train_multi_attacks, task_type='multi')

    joblib.dump(rf_model_multi, os.path.join(output_dir, 'rf_model_multi.pkl'))
    joblib.dump(svm_model_multi, os.path.join(output_dir, 'svm_model_multi.pkl'))
    joblib.dump(top_features_multi, os.path.join(output_dir, 'selected_features_multi.pkl'))

    y_pred_binary = predict_svm(svm_model_bin, X_test_bin_sel, threshold=-0.65)

    y_pred_final = np.full(len(X_test_multi), normal_encoded_val, dtype=int)

    attack_indices = np.where(y_pred_binary == 1)[0]

    if len(attack_indices) > 0:
        X_test_attack_samples = X_test_multi_sel.iloc[attack_indices]
        y_pred_attacks = predict_svm(svm_model_multi, X_test_attack_samples)
        y_pred_final[attack_indices] = y_pred_attacks

    print("\n--- Final Hierarchical Classification Report (TEST SET) ---")
    print(classification_report(y_test_multi, y_pred_final, target_names=label_names))

    acc = accuracy_score(y_test_multi, y_pred_final)
    prec = precision_score(y_test_multi, y_pred_final, average='macro', zero_division=0)
    rec = recall_score(y_test_multi, y_pred_final, average='macro', zero_division=0)
    f1 = f1_score(y_test_multi, y_pred_final, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_test_multi, y_pred_final)
    fpr = compute_fpr(y_test_multi, y_pred_final, average='macro')

    print("=" * 50)
    print(f"Accuracy:                          {acc:.4f}")
    print(f"Precision:                         {prec:.4f}")
    print(f"Recall:                            {rec:.4f}")
    print(f"F1-Score:                          {f1:.4f}")
    print(f"MCC (Matthews-Korrelation):        {mcc:.4f}")
    print(f"FPR:                               {fpr:.4f}")

    if ENABLE_SVM_HYPERPARAM_SEARCH:
        wandb.config.update({"Multi_Best_C": best_params_multi['C'], "Multi_Best_Gamma": best_params_multi['gamma']})
        if os.path.exists(os.path.join(output_dir, 'svm_tuning_multi.png')):
            wandb.log({"Multi_Tuning_Heatmap": wandb.Image(os.path.join(output_dir, 'svm_tuning_multi.png'))})

    wandb.log({
        "Hierarchical/Accuracy": acc,
        "Hierarchical/Precision": prec,
        "Hierarchical/Recall": rec,
        "Hierarchical/F1-Score": f1,
        "Hierarchical/MCC": mcc,
        "Hierarchical/FPR": fpr
    })


if __name__ == '__main__':
    wandb.init(project="intrusion-detection", name="svm-hierarchical-run")
    wandb.config.update({
        "RF_TOP_K_BINARY": RF_TOP_K,
        "RF_TOP_K_MULTI": RF_TOP_K_MULTI,
        "SVM_KERNEL": SVM_KERNEL,
        "ENABLE_SVM_HYPERPARAM_SEARCH": ENABLE_SVM_HYPERPARAM_SEARCH,
        "algorithm": "Hierarchical SVM"
    })

    ensure_preprocessed_data_exists()
    run_hierarchical_pipeline()

    wandb.finish()