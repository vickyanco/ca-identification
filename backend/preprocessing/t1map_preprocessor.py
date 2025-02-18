# file: t1map_preprocessing.py
# description: Preprocessing class for T1 mappping images
# author: María Victoria Anconetani
# date: 12/02/2025

import numpy as np
from .base_preprocessor import BasePreprocessor

class T1Preprocessor(BasePreprocessor):
    def __init__(self, target_shape=(256, 218, 8)):
        super().__init__(target_shape)  # Pass the target shape

    def preprocess(self, image):
        """
        Normalize and resize the image.
        """
        image = self.resize_image(image)  # Ensure uniform size
        mean = np.mean(image)
        std = np.std(image)
        return (image - mean) / (std + 1e-8)  # Z-score normalization
