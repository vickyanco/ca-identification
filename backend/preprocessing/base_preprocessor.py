# file: base_preprocessor.py
# description: Base class for image preprocessing
# author: María Victoria Anconetani
# date: 12/02/2025

import numpy as np
from scipy.ndimage import zoom
import nibabel as nib
import tensorflow as tf

class BasePreprocessor:
    def __init__(self, target_shape):
        self.target_shape = target_shape

    def load_nifti(self, file_path):
        """
        Loads a .nii.gz file and converts it to a NumPy array.
        """
        nifti_img = nib.load(file_path)
        img_data = nifti_img.get_fdata()
        return img_data

    def resize_image(self, image):
        """Resize 3D medical images (Height, Width, Depth)."""
        
        # Ensure the image is a NumPy array
        image = np.array(image, dtype=np.float32)

        # 🔹 Remove singleton dimensions (e.g., (256, 218, 1, 8) → (256, 218, 8))
        image = np.squeeze(image)

        # 🔹 Convert 2D images to 3D by repeating the slice
        if len(image.shape) == 2:  
            print(f"⚠️ Converting 2D image {image.shape} to 3D")
            image = np.stack([image] * self.target_shape[2], axis=-1)

        # 🔹 Ensure the image is now 3D (Height, Width, Depth)
        if len(image.shape) != 3:
            raise ValueError(f"🚨 Expected 3D image but got shape {image.shape}")

        # 🔹 Resize height & width using TensorFlow
        image_resized = tf.image.resize(image, self.target_shape[:2], method=tf.image.ResizeMethod.BILINEAR).numpy()

        # 🔹 Resize Depth (Duplicate last slice if needed)
        current_depth = image.shape[2]
        target_depth = self.target_shape[2]
        
        if current_depth < target_depth:
            extra_slices = target_depth - current_depth
            last_slice = image_resized[:, :, -1]  # Get last slice
            for _ in range(extra_slices):
                image_resized = np.concatenate([image_resized, last_slice[..., np.newaxis]], axis=-1)
        
        elif current_depth > target_depth:
            depth_factor = target_depth / current_depth  
            image_resized = zoom(image_resized, (1, 1, depth_factor), order=1)  

        print(f"✅ Final Processed Image Shape: {image_resized.shape}")  # Debugging
        return image_resized

    def preprocess(self, image):
        """Normalize the image using Min-Max scaling."""
        image = self.resize_image(image)
        min_val, max_val = np.min(image), np.max(image)
        return (image - min_val) / (max_val - min_val + 1e-8) if max_val > min_val else np.zeros_like(image)
