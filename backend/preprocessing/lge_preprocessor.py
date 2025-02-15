# file: lge_preprocessing.py
# description: Preprocessing class for LGE images
# author: María Victoria Anconetani
# date: 12/02/2025

import numpy as np
from .base_preprocessor import BasePreprocessor

class LGEPreprocessor(BasePreprocessor):
    def preprocess(self, image):
        """
        Normalize and resize the image.
        """
        image = self.resize_image(image)  # Ensure uniform size
        return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)  # Min-Max normalization