# file: data_loader_augmentation.py
# description: Base class for image preprocessing
# author: María Victoria Anconetani
# date: 12/02/2025

import os
import glob
import nibabel as nib
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

class DataLoader:
    """
    Handles loading, preprocessing, data augmentation, and class balancing 
    for LGE and T1 Mapping data separately.
    """
    def __init__(self, lge_cases_path, lge_controls_path, t1_cases_path, t1_controls_path, 
                 lge_preprocessor, t1_preprocessor, apply_augmentation=True):
        self.lge_cases_path = lge_cases_path
        self.lge_controls_path = lge_controls_path
        self.t1_cases_path = t1_cases_path
        self.t1_controls_path = t1_controls_path
        self.lge_preprocessor = lge_preprocessor
        self.t1_preprocessor = t1_preprocessor
        self.apply_augmentation = apply_augmentation

    def load_nifti(self, file_path):
        """Loads a .nii.gz file and converts it to a NumPy array."""
        nifti_img = nib.load(file_path)
        img_data = nifti_img.get_fdata()
        return img_data

    def preprocess_image(self, image, preprocessor):
        """Applies the specific preprocessing method (LGE or T1)."""
        return preprocessor.preprocess(image)

    def augment_image(self, image):
        """Applies random 3D data augmentation if enabled."""
        if not self.apply_augmentation:
            return image
        
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

        return image.numpy()

    def load_dataset(self, cases_path, controls_path, preprocessor):
        """Loads images from cases and controls, applies preprocessing & augmentation, and assigns labels."""
        case_files = glob.glob(os.path.join(cases_path, "**/*.nii.gz"), recursive=True)
        control_files = glob.glob(os.path.join(controls_path, "**/*.nii.gz"), recursive=True)

        case_images = []
        control_images = []

        # Process Cases (label = 1)
        for file in case_files:
            image = self.load_nifti(file)
            print(f"Loaded Case Image {file} Shape: {image.shape}")  # Debugging shape
            image = self.preprocess_image(image, preprocessor)
            print(f"Preprocessed Case Image {file} Shape: {image.shape}")  # Debugging shape
            image = self.augment_image(image)
            print(f"Augmented Case Image {file} Shape: {image.shape}")  # Debugging shape
            case_images.append(image)

        # Process Controls (label = 0)
        for file in control_files:
            image = self.load_nifti(file)
            print(f"Loaded Control Image {file} Shape: {image.shape}")  # Debugging shape
            image = self.preprocess_image(image, preprocessor)
            print(f"Preprocessed Control Image {file} Shape: {image.shape}")  # Debugging shape
            image = self.augment_image(image)
            print(f"Augmented Control Image {file} Shape: {image.shape}")  # Debugging shape
            control_images.append(image)

        print(f"✅ Final Case Image Shape: {np.array(case_images, dtype=object).shape}")
        print(f"✅ Final Control Image Shape: {np.array(control_images, dtype=object).shape}")

        # Convert to NumPy arrays
        try:
            X = np.array(case_images + control_images, dtype=np.float32)  # Ensure dtype
            y = np.array([1] * len(case_images) + [0] * len(control_images), dtype=np.int32)  # Ensure labels exist
        except ValueError as e:
            print("🚨 Shape Mismatch Detected! Cannot stack images into an array.")
            raise e

        print(f"✅ Final Dataset Shapes: X={X.shape}, y={y.shape}")  # Debugging

        # Compute class weights
        class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

        return X, y, class_weight_dict

    def load_all_data(self):
        """Loads LGE and T1 Mapping data separately and returns both datasets."""
        lge_X, lge_y, lge_class_weights = self.load_dataset(self.lge_cases_path, self.lge_controls_path, self.lge_preprocessor)
        t1_X, t1_y, t1_class_weights = self.load_dataset(self.t1_cases_path, self.t1_controls_path, self.t1_preprocessor)

        return (lge_X, lge_y, lge_class_weights), (t1_X, t1_y, t1_class_weights)
