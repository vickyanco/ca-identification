# file: utils/loads.py
# description: Utility functions to load and preprocess NIfTI files.
# author: María Victoria Anconetani
# date: 20/02/2025

import os
import numpy as np
import nibabel as nib
import tensorflow as tf

class LGEDataLoader:
    def __init__(self, dataset_root, input_shape=(172, 192, 12, 1), batch_size=8):
        """
        Initializes the data loader for LGE classification.

        Args:
            dataset_root (str): Path to the dataset folder containing train_case, train_control, etc.
            input_shape (tuple): Expected shape of input images.
            batch_size (int): Batch size for the TensorFlow dataset.
        """
        self.dataset_root = dataset_root
        self.input_shape = input_shape
        self.batch_size = batch_size

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
        Loads all NIfTI files from a folder and assigns a label.

        Args:
            folder_path (str): Path to the folder containing NIfTI files.
            label (int): Label for the class (0 or 1).

        Returns:
            list: List of (image, label) tuples.
        """
        dataset = []
        for file in os.listdir(folder_path):
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                file_path = os.path.join(folder_path, file)
                dataset.append((self.load_nifti(file_path), label))
        return dataset

    def prepare_datasets(self):
        """
        Loads and preprocesses the dataset, converting it into TensorFlow datasets.
        """
        # Load datasets
        train_data = self.load_dataset_from_folder(self.train_case_dir, 1) + \
                    self.load_dataset_from_folder(self.train_control_dir, 0)
        test_data = self.load_dataset_from_folder(self.test_case_dir, 1) + \
                    self.load_dataset_from_folder(self.test_control_dir, 0)

        # Shuffle training data
        np.random.shuffle(train_data)

        # Convert to NumPy arrays
        X_train, y_train = zip(*train_data)
        X_test, y_test = zip(*test_data)

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)

        # Convert to TensorFlow datasets
        def convert_to_tf_dataset(X, y):
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            return dataset.shuffle(len(X)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        self.train_dataset = convert_to_tf_dataset(X_train, y_train)
        self.test_dataset = convert_to_tf_dataset(X_test, y_test)

        print(f"✅ Data Loaded: {len(X_train)} training samples, {len(X_test)} test samples")
