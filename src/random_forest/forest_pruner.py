import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from src.utils.data_loader import load_preprocessed_data


def get_tree_features(tree):
    return set(tree.tree_.feature[tree.tree_.feature >= 0])


def calculate_jaccard_similarity(tree_a, tree_b):
    features_a = get_tree_features(tree_a)
    features_b = get_tree_features(tree_b)

    if not features_a or not features_b:
        return 0.0

    intersection = len(features_a.intersection(features_b))
    union = len(features_a.union(features_b))

    return intersection / union


def get_aligned_probabilities(tree, X_val, n_classes, class_mapping):
    probas = tree.predict_proba(X_val)
    aligned_probas = np.zeros((X_val.shape[0], n_classes))

    for idx, cls in enumerate(tree.classes_):
        if cls in class_mapping:
            aligned_idx = class_mapping[cls]
            if len(probas.shape) == 1:
                aligned_probas[:, aligned_idx] = probas
            else:
                aligned_probas[:, aligned_idx] = probas[:, idx]

    return aligned_probas


def evaluate_tree_auc(tree, X_val, y_val, is_binary, all_classes):
    n_classes = len(all_classes)
    class_mapping = {cls: idx for idx, cls in enumerate(all_classes)}

    predictions = get_aligned_probabilities(tree, X_val, n_classes, class_mapping)

    if is_binary:
        return roc_auc_score(y_val, predictions[:, 1])
    else:
        return roc_auc_score(y_val, predictions, multi_class='ovr')


def prune_random_forest(rf_model, X_val, y_val, is_binary=True, auc_threshold=0.5, sim_threshold=0.8):
    trees = rf_model.estimators_
    n_trees = len(trees)
    all_classes = rf_model.classes_

    auc_scores = np.zeros(n_trees)
    for i, tree in enumerate(trees):
        auc_scores[i] = evaluate_tree_auc(tree, X_val, y_val, is_binary, all_classes)

    trees_to_keep_indices = []
    trees_to_discard_indices = set()

    sorted_indices = np.argsort(auc_scores)[::-1]

    for i in sorted_indices:
        if i in trees_to_discard_indices:
            continue

        if auc_scores[i] < auc_threshold:
            trees_to_discard_indices.add(i)
            continue

        trees_to_keep_indices.append(i)

        for j in sorted_indices:
            if j == i or j in trees_to_discard_indices:
                continue

            similarity = calculate_jaccard_similarity(trees[i], trees[j])

            if similarity > sim_threshold and auc_scores[j] < auc_scores[i]:
                trees_to_discard_indices.add(j)

    rf_model.estimators_ = [trees[i] for i in trees_to_keep_indices]
    rf_model.n_estimators = len(trees_to_keep_indices)

    return rf_model, len(trees_to_keep_indices)


def run_phase_2(rf_model, task_type):
    is_binary = (task_type == 'binary')
    _, X_test, _, y_test = load_preprocessed_data(task_type=task_type)

    X_test_np = X_test.to_numpy()

    pruned_model, remaining_trees = prune_random_forest(
        rf_model,
        X_test_np,
        y_test,
        is_binary=is_binary
    )

    return pruned_model, remaining_trees