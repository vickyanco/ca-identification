# file: preprocessing/load_lge_data.py
# description: Utility functions to load and preprocess NIfTI files.
# author: María Victoria Anconetani
# date: 20/02/2025

import os
import numpy as np
import nibabel as nib
import tensorflow as tf

from backend.utils.patient_split import (
    extract_patient_id,
    patient_kfold_splits,
    assert_no_patient_leakage,
)

class LGEDataLoader:
    def __init__(self, dataset_root, input_shape=(172, 192, 12, 1), batch_size=8, seed=42):
        """
        Initializes the data loader for LGE classification.

        Args:
            dataset_root (str): Path to the dataset folder containing train_case, train_control, etc.
            input_shape (tuple): Expected shape of input images.
            batch_size (int): Batch size for the TensorFlow dataset.
            seed (int): Seed for shuffling and k-fold splitting, for reproducibility.
        """
        self.dataset_root = dataset_root
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.seed = seed

        # Define dataset paths
        self.train_case_dir = os.path.join(dataset_root, "train_casos")
        self.train_control_dir = os.path.join(dataset_root, "train_controles")
        self.test_case_dir = os.path.join(dataset_root, "test_casos")
        self.test_control_dir = os.path.join(dataset_root, "test_controles")

        print("✅ LGEDataLoader Initialized")

    def load_nifti(self, filepath):
        """
        Loads a NIfTI file and returns a NumPy array.

        Args:
            filepath (str): Path to the NIfTI file.

        Returns:
            np.array: Processed image data.
        """
        nifti_img = nib.load(filepath)
        img_data = nifti_img.get_fdata()

        # Add channel dimension
        img_data = np.expand_dims(img_data, axis=-1)
        return img_data.astype(np.float32)

    def load_dataset_from_folder(self, folder_path, label):
        """
        Loads all NIfTI files from a folder and assigns a label and patient ID.

        Args:
            folder_path (str): Path to the folder containing NIfTI files.
            label (int): Label for the class (0 or 1).

        Returns:
            list: List of (image, label, patient_id) tuples.
        """
        dataset = []
        for file in os.listdir(folder_path):
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                file_path = os.path.join(folder_path, file)
                dataset.append((self.load_nifti(file_path), label, extract_patient_id(file)))
        return dataset

    def _to_tf_dataset(self, X, y, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            dataset = dataset.shuffle(len(X), seed=self.seed)
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def prepare_pools(self):
        """
        Loads the train/val pool (train_casos + train_controles) and the held-out
        test set (test_casos + test_controles), and verifies that no patient appears
        in both — so the test set stays uncontaminated by patients used for model
        selection (training, early stopping, threshold tuning).
        """
        pool = self.load_dataset_from_folder(self.train_case_dir, 1) + \
            self.load_dataset_from_folder(self.train_control_dir, 0)
        test_data = self.load_dataset_from_folder(self.test_case_dir, 1) + \
            self.load_dataset_from_folder(self.test_control_dir, 0)

        if not pool:
            raise ValueError("🚨 ERROR: No train/val pool data found! Check dataset path and file format.")
        if not test_data:
            raise ValueError("🚨 ERROR: No test data found! Check dataset path and file format.")

        pool_images, pool_labels, pool_patients = zip(*pool)
        test_images, test_labels, test_patients = zip(*test_data)

        assert_no_patient_leakage(pool_patients, test_patients, dataset_name="LGE")

        if len(np.unique(pool_labels)) < 2:
            raise ValueError("🚨 ERROR: Both classes must be present in the train/val pool")

        self.pool_images = np.array(pool_images)
        self.pool_labels = np.array(pool_labels)
        self.pool_patients = np.array(pool_patients)

        self.test_dataset = self._to_tf_dataset(np.array(test_images), np.array(test_labels), shuffle=False)

        print(f"✅ Train/Val Pool: {len(self.pool_images)} images from {len(set(pool_patients))} patients")
        print(f"✅ Held-out Test: {len(test_images)} images from {len(set(test_patients))} patients")

    def get_patient_kfold(self, n_splits=5):
        """
        Yields (fold_idx, train_dataset, val_dataset) for patient-grouped,
        label-stratified k-fold cross-validation over the train/val pool.
        No patient is ever split across the train and val side of a fold,
        and the held-out test set is never involved.
        """
        splits = patient_kfold_splits(self.pool_labels, self.pool_patients, n_splits=n_splits, seed=self.seed)
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_ds = self._to_tf_dataset(self.pool_images[train_idx], self.pool_labels[train_idx])
            val_ds = self._to_tf_dataset(self.pool_images[val_idx], self.pool_labels[val_idx], shuffle=False)
            yield fold_idx, train_ds, val_ds
