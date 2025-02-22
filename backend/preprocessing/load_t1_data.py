# file: preprocessing/load_t1_data.py
# description: Utility functions to load and preprocess dcm files.
# author: María Victoria Anconetani
# date: 22/02/2025

import os
import numpy as np
import pydicom
import tensorflow as tf

class T1DataLoader:
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
        self.train_case_dir = os.path.join(dataset_root, "train_casos")
        self.train_control_dir = os.path.join(dataset_root, "train_controles")
        self.test_case_dir = os.path.join(dataset_root, "test_casos")
        self.test_control_dir = os.path.join(dataset_root, "test_controles")

        print("✅ T1DataLoader Initialized")

    def load_dicom(self, filepath):
        """
        Loads a DICOM file and returns a preprocessed NumPy array.
        
        Args:
            filepath (str): Path to the DICOM file.
        
        Returns:
            np.array: Preprocessed image data, or None if an error occurs.
        """
        print(f"📂 Trying to read: {filepath}")

        try:
            ds = pydicom.dcmread(filepath)
            img = ds.pixel_array.astype(np.float32)

            # Apply Z-score normalization
            mean, std = np.mean(img), np.std(img)
            img = (img - mean) / (std if std > 0 else 1)  # Prevent division by zero

            # Expand dimensions for CNN input (H, W, 1)
            img = np.expand_dims(img, axis=-1)

            print(f"✅ Successfully loaded: {filepath}")
            return img
        
        except Exception as e:
            print(f"❌ ERROR reading {filepath}: {e}")
            return None

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
        print(f"📂 Checking folder: {folder_path}")

        if not os.path.exists(folder_path):
            print(f"❌ ERROR: Folder not found: {folder_path}")
            return []

        file_count = 0
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Ensure the file has a valid DICOM extension or try reading anyway
                if file.lower().endswith(('.dcm', '')):  
                    img_array = self.load_dicom(file_path)
                    if img_array is not None:
                        dataset.append((img_array, label))
                        file_count += 1

        print(f"✅ Loaded {file_count} images from {folder_path} (Label {label})")
        return dataset

    def prepare_datasets(self):
        """
        Loads and preprocesses the dataset, converting it into TensorFlow datasets.
        """
        # Load datasets
        train_cases = self.load_dataset_from_folder(self.train_case_dir, 1)
        train_controls = self.load_dataset_from_folder(self.train_control_dir, 0)
        test_cases = self.load_dataset_from_folder(self.test_case_dir, 1)
        test_controls = self.load_dataset_from_folder(self.test_control_dir, 0)

        train_data = train_cases + train_controls
        test_data = test_cases + test_controls

        print(f"✅ Loaded Training Data: {len(train_data)} samples")
        print(f"✅ Loaded Test Data: {len(test_data)} samples")

        # Ensure both classes exist in training
        y_train = [label for _, label in train_data]
        unique_classes = np.unique(y_train)

        if len(unique_classes) < 2:
            print(f"🚨 WARNING: Only one class found in training! Unique labels: {unique_classes}")
            print("❌ ERROR: Check dataset split. Ensure both `train_casos` and `train_controles` contain images.")
            exit(1)

        # Shuffle training data
        np.random.shuffle(train_data)

        print(f"✅ Training Samples: {len(train_data)}")
        print(f"✅ Test Samples: {len(test_data)}")

        if not train_data:
            raise ValueError("🚨 ERROR: No training data found! Check dataset path and file format.")
        if not test_data:
            raise ValueError("🚨 ERROR: No test data found! Check dataset path and file format.")

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
