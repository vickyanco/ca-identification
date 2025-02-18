# file: lge_preprocessing.py
# description: Preprocessing class for LGE images
# author: María Victoria Anconetani
# date: 12/02/2025

import numpy as np
from .base_preprocessor import BasePreprocessor

class LGEPreprocessor(BasePreprocessor):
    def __init__(self, target_shape=(172, 192, 12)):
        super().__init__(target_shape)  # Pass the target shape

    def preprocess(self, image):
        """
        Normalize and resize the image.
        """
        image = self.resize_image(image)  # Ensure uniform size
        min_val, max_val = np.min(image), np.max(image)
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)  # If max == min, set all pixels to 0
        return image
