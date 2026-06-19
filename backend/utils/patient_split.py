# file: utils/patient_split.py
# description: Patient-level grouping, k-fold splitting, and validation-only threshold selection.
# author: María Victoria Anconetani

import os
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score


def extract_patient_id(filepath):
    """
    Extracts the patient ID from a filename, assuming the `{patient_id}_...` naming
    convention used throughout this project (e.g. "p12_caso_1_0.dcm" -> "p12",
    "p1_control_1.nii.gz" -> "p1").
    """
    return os.path.basename(filepath).split("_")[0]


def patient_kfold_splits(labels, patient_ids, n_splits=5, seed=42):
    """
    Builds patient-grouped, label-stratified k-fold splits: no patient ever appears
    in both the train and validation side of a fold.

    Returns:
        list of (train_idx, val_idx) index arrays into `labels`/`patient_ids`.
    """
    labels = np.asarray(labels)
    patient_ids = np.asarray(patient_ids)

    # Determine each patient's label (consistent across all of a patient's files)
    # to check there are enough patients per class to support n_splits.
    seen = {}
    for label, pid in zip(labels, patient_ids):
        seen.setdefault(pid, label)
    patient_labels = np.array(list(seen.values()))
    min_patients_per_class = np.bincount(patient_labels).min()

    if n_splits > min_patients_per_class:
        raise ValueError(
            f"🚨 n_splits={n_splits} exceeds the number of patients in the smallest "
            f"class ({min_patients_per_class}). Reduce n_splits."
        )

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(labels)), labels, groups=patient_ids))


def assert_no_patient_leakage(pool_patient_ids, test_patient_ids, dataset_name=""):
    """
    Raises if any patient appears in both the train/val pool and the held-out test set.
    """
    overlap = set(pool_patient_ids) & set(test_patient_ids)
    if overlap:
        raise ValueError(
            f"🚨 Patient leakage detected in {dataset_name}: patients {sorted(overlap)} "
            f"appear in both the train/val pool and the test set."
        )


def find_best_threshold(y_true, y_probs, thresholds=None):
    """
    Selects the decision threshold that maximizes F1 score on the given predictions.

    IMPORTANT: must only ever be called with validation (or out-of-fold validation)
    predictions, never with test-set predictions.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    best_threshold, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_threshold = score, t

    return float(best_threshold), float(best_f1)
