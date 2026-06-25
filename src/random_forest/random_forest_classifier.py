from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
from src.utils.data_loader import load_preprocessed_data


def train_random_forest(X_train, y_train, random_state=42):
    rf_model = RandomForestClassifier(random_state=random_state)
    rf_model.fit(X_train, y_train)
    return rf_model


def evaluate_predictions(y_true, y_pred, is_binary=True):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary' if is_binary else 'weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='binary' if is_binary else 'weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary' if is_binary else 'weighted', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    fpr = 0.0
    if is_binary:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return acc, prec, rec, f1, mcc, fpr


def print_evaluation_metrics(title, metrics, is_binary=True):
    acc, prec, rec, f1, mcc, fpr = metrics
    print(f"--- {title} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall (DR): {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    if is_binary:
        print(f"FPR (FAR): {fpr:.4f}")
    print("\n")


def run_baseline_pipeline(task_type):
    is_binary = (task_type == 'binary')

    X_train, X_test, y_train, y_test = load_preprocessed_data(task_type=task_type)

    rf_model = train_random_forest(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    metrics = evaluate_predictions(y_test, y_pred, is_binary=is_binary)

    title = "Binary Baseline (General Threat Detection)" if is_binary else "Multiclass Baseline (Attack Categorization)"
    print_evaluation_metrics(title, metrics, is_binary=is_binary)

    return rf_model, metrics


if __name__ == "__main__":
    rf_bin_model, bin_metrics = run_baseline_pipeline(task_type='binary')

    rf_multi_model, multi_metrics = run_baseline_pipeline(task_type='multi')