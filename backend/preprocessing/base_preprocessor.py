# file: base_preprocessor.py
# description: Base class for image preprocessing
# author: María Victoria Anconetani
# date: 12/02/2025

import nibabel as nib
import tensorflow as tf

class BasePreprocessor:
    def __init__(self, target_shape=(128, 128, 128)):
        self.target_shape = target_shape

    def load_nifti(self, file_path):
        """
        Loads a .nii.gz file and converts it to a NumPy array.
        """
        nifti_img = nib.load(file_path)
        img_data = nifti_img.get_fdata()
        return img_data

    def preprocess(self, file_path):
        raise NotImplementedError("Subclasses must implement preprocess method")
    
    def resize_image(self, image):
        """
        Resizes the image to `target_shape`.
        """
        image = tf.image.resize(image, self.target_shape, method=tf.image.ResizeMethod.BILINEAR)
        return image.numpy()