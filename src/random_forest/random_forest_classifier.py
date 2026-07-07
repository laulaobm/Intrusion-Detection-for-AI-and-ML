from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef, ConfusionMatrixDisplay
import joblib
from src.utils.data_loader import load_preprocessed_data
from forest_pruner import run_phase_2
from result_corrector import ResultCorrector
import matplotlib.pyplot as plt
import wandb
def train_random_forest(X_train, y_train, random_state=42, max_depth=20):
    rf_model = RandomForestClassifier(
        random_state=random_state,
        max_depth=max_depth,
        max_samples=0.8,
        n_estimators=100
    )
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

def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation='vertical')
    plt.title(title)
    plt.tight_layout()
    plt.savefig("confusion_matrix_random_forest")
    plt.show()


if __name__ == "__main__":
    wandb.init(project="intrusion-detection", name="random-forest-run")

    wandb.config.update({
        "max_depth": 20,
        "max_samples": 0.8,
        "n_estimators": 100,
        "binary_confidence_threshold": 0.65,
        "multiclass_confidence_threshold": 0.55,
        "binary_sim_threshold": 0.85,
        "multiclass_sim_threshold": 0.90
    })

    rf_bin_model, bin_metrics = run_baseline_pipeline(task_type='binary')
    pruned_bin_model, remaining_bin_trees = run_phase_2(rf_bin_model, task_type='binary', sim_threshold=0.85)

    _, X_test_bin, _, y_test_bin = load_preprocessed_data(task_type='binary')

    corrector = ResultCorrector()
    y_pred_bin_corrected = corrector.correct_binary(pruned_bin_model, X_test_bin, threshold=0.65)

    corrected_bin_metrics = evaluate_predictions(y_test_bin, y_pred_bin_corrected, is_binary=True)

    print(f"Trees remaining in Binary model after pruning: {remaining_bin_trees}")
    print_evaluation_metrics("Corrected Binary Model", corrected_bin_metrics, is_binary=True)

    wandb.log({
        "Binary/Accuracy": corrected_bin_metrics[0],
        "Binary/Precision": corrected_bin_metrics[1],
        "Binary/Recall_DR": corrected_bin_metrics[2],
        "Binary/F1-score": corrected_bin_metrics[3],
        "Binary/MCC": corrected_bin_metrics[4],
        "Binary/FPR_FAR": corrected_bin_metrics[5],
        "Binary/Remaining_Trees": remaining_bin_trees
    })

    rf_multi_model, multi_metrics = run_baseline_pipeline(task_type='multi')
    pruned_multi_model, remaining_multi_trees = run_phase_2(rf_multi_model, task_type='multi', sim_threshold=0.90)

    _, X_test_multi, _, y_test_multi = load_preprocessed_data(task_type='multi')

    label_encoder = joblib.load('../preprocessing/label_encoder.pkl')

    y_pred_multi_pruned = pruned_multi_model.predict(X_test_multi)

    y_pred_multi_corrected = corrector.correct_multiclass(pruned_multi_model, X_test_multi, label_encoder)

    corrected_multi_metrics = evaluate_predictions(y_test_multi, y_pred_multi_corrected, is_binary=False)

    print(f"Trees remaining in Multiclass model after pruning: {remaining_multi_trees}")
    print_evaluation_metrics("Corrected Multiclass Model", corrected_multi_metrics, is_binary=False)

    wandb.log({
        "Multiclass/Accuracy": corrected_multi_metrics[0],
        "Multiclass/Precision": corrected_multi_metrics[1],
        "Multiclass/Recall_DR": corrected_multi_metrics[2],
        "Multiclass/F1-score": corrected_multi_metrics[3],
        "Multiclass/MCC": corrected_multi_metrics[4],
        "Multiclass/Remaining_Trees": remaining_multi_trees
    })


    wandb.finish()