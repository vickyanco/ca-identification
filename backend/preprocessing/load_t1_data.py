# file: preprocessing/load_t1_data.py
# description: Utility functions to load and preprocess dcm files.
# author: María Victoria Anconetani
# date: 20/02/2025

import os
import numpy as np
import pydicom
import tensorflow as tf

class DICOMDataLoader:
    def __init__(self, dataset_root, input_shape=(256, 256, 1), batch_size=8):
        """
        Initializes the DICOM data loader for CNN training.

        Args:
            dataset_root (str): Path to the dataset folder containing train_casos, train_controles, etc.
            input_shape (tuple): Expected shape of input images (H, W, Channels).
            batch_size (int): Batch size for TensorFlow dataset.
        """
        self.dataset_root = dataset_root
        self.input_shape = input_shape
        self.batch_size = batch_size

        # Define dataset paths
        self.train_case_dir = os.path.join(dataset_root, "casos_train")
        self.train_control_dir = os.path.join(dataset_root, "controles_train")
        self.test_case_dir = os.path.join(dataset_root, "casos_test")
        self.test_control_dir = os.path.join(dataset_root, "controles_test")

        print("✅ DICOMDataLoader Initialized")

    def load_dicom(self, filepath):
        """
        Loads a DICOM file and returns a preprocessed NumPy array.

        Args:
            filepath (str): Path to the DICOM file.

        Returns:
            np.array: Preprocessed image data.
        """
        ds = pydicom.dcmread(filepath)
        img = ds.pixel_array.astype(np.float32)

        # Apply Z-score normalization
        mean = np.mean(img)
        std = np.std(img)
        if std == 0:
            std = 1  # Prevent division by zero
        img = (img - mean) / std

        # Expand dimensions for CNN input (H, W, 1)
        img = np.expand_dims(img, axis=-1)

        return img

    def load_dataset_from_folder(self, folder_path, label):
        """
        Loads all DICOM files from a folder and assigns a label.

        Args:
            folder_path (str): Path to the folder containing DICOM files.
            label (int): Label for the class (0 or 1).

        Returns:
            list: List of (image, label) tuples.
        """
        dataset = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".dcm"):  # Only process DICOM files
                    file_path = os.path.join(root, file)
                    dataset.append((self.load_dicom(file_path), label))
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