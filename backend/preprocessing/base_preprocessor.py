# file: base_preprocessor.py
# description: Base class for image preprocessing
# author: María Victoria Anconetani
# date: 12/02/2025

import numpy as np
from scipy.ndimage import zoom
import nibabel as nib
import tensorflow as tf

class BasePreprocessor:
    def __init__(self, target_shape=(128, 128, 10)):
        self.target_shape = target_shape

    def load_nifti(self, file_path):
        """
        Loads a .nii.gz file and converts it to a NumPy array.
        """
        nifti_img = nib.load(file_path)
        img_data = nifti_img.get_fdata()
        return img_data

    def preprocess(self, image):
        raise NotImplementedError("Subclasses must implement preprocess method")
    
    def resize_image(self, image):
        """Resize 3D medical images (height, width, depth) and handle 2D images."""

        # Ensure the image is a NumPy array
        image = np.array(image, dtype=np.float32)

        # 🔹 Remove singleton dimensions (e.g., (256, 218, 1, 8) → (256, 218, 8))
        image = np.squeeze(image)

        # 🔹 Convert 2D images to 3D by repeating the slice
        if len(image.shape) == 2:  # If image is (Height, Width)
            print(f"⚠️ Converting 2D image {image.shape} to 3D")
            image = np.stack([image] * self.target_shape[2], axis=-1)  # Repeat along depth

        # 🔹 Ensure the image is now 3D (Height, Width, Depth)
        if len(image.shape) != 3:
            raise ValueError(f"🚨 Expected 3D image but got shape {image.shape}")

        # 🔹 Resize height & width using TensorFlow
        image_resized = tf.image.resize(image, self.target_shape[:2], method=tf.image.ResizeMethod.BILINEAR).numpy()

        # 🔹 Resize depth separately using scipy.ndimage.zoom
        depth_factor = self.target_shape[2] / image.shape[2]  # Scaling factor for depth
        image_resized = zoom(image_resized, (1, 1, depth_factor), order=1)  # Linear interpolation

        return image_resized