# file: train/train_lge.py
# description: Script to train the LGE model using patient-grouped k-fold
#              cross-validation. The held-out test set is never used for early
#              stopping or threshold selection — only for final evaluation
#              (see evaluate_lge.py).
# author: María Victoria Anconetani
# date: 20/02/2025

import json
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

from backend.config import DATA_ROOT
from backend.preprocessing.load_lge_data import LGEDataLoader
from backend.model.lge_model import LGE_CNN
from backend.utils.patient_split import find_best_threshold

N_SPLITS = 5
SEED = 42

# Load dataset
dataset_root = os.path.join(DATA_ROOT, "LGE_prep_nii_divided")
data_loader = LGEDataLoader(dataset_root, seed=SEED)
data_loader.prepare_pools()


def class_weights_for(labels):
    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        print("🚨 WARNING: Only one class found in this fold's training data! Defaulting to equal class weights.")
        return {0: 1.0, 1: 1.0}

    weights = compute_class_weight("balanced", classes=unique_classes, y=labels)
    return {0: weights[0] * 1.2, 1: weights[1] * 1.2}


fold_aucs = []
oof_true, oof_probs = [], []
best_fold = {"auc": -1.0, "model": None}

for fold_idx, train_dataset, val_dataset in data_loader.get_patient_kfold(n_splits=N_SPLITS):
    print(f"\n===== Fold {fold_idx + 1}/{N_SPLITS} =====")

    y_train_fold = np.concatenate([labels for _, labels in train_dataset.as_numpy_iterator()])
    class_weight_dict = class_weights_for(y_train_fold)
    print("✅ Class Weights:", class_weight_dict)

    # Fresh model per fold
    model = LGE_CNN()
    model.model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]

    model.model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )

    y_val = np.concatenate([labels for _, labels in val_dataset.as_numpy_iterator()])
    y_val_probs = model.model.predict(val_dataset, verbose=0).flatten()

    fold_auc = roc_auc_score(y_val, y_val_probs)
    print(f"✅ Fold {fold_idx + 1} Validation AUC: {fold_auc:.4f}")

    fold_aucs.append(fold_auc)
    oof_true.append(y_val)
    oof_probs.append(y_val_probs)

    if fold_auc > best_fold["auc"]:
        best_fold = {"auc": fold_auc, "model": model.model}

print(f"\n✅ Cross-Validation AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

# Select the decision threshold from pooled out-of-fold validation predictions only
# (the held-out test set is never touched here).
oof_true = np.concatenate(oof_true)
oof_probs = np.concatenate(oof_probs)
best_threshold, best_f1 = find_best_threshold(oof_true, oof_probs)
print(f"✅ Threshold selected from out-of-fold validation predictions: {best_threshold:.3f} (F1={best_f1:.4f})")

# Save the best-performing fold's model and the validation-selected threshold
best_fold["model"].save("lge_cnn_model.h5")
with open("lge_cnn_threshold.json", "w") as f:
    json.dump({"threshold": best_threshold}, f)

print("✅ Model Trained & Saved Successfully")
