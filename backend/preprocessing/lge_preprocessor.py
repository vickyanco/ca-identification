# file: lge_preprocessing.py
# description: Preprocessing class for LGE images
# author: María Victoria Anconetani
# date: 12/02/2025

import numpy as np
from .base_preprocessor import BasePreprocessor

class LGEPreprocessor(BasePreprocessor):
    def preprocess(self, file_path):
        image = self.load_nifti(file_path)
        image = self.resize_image(image)

        # Normalización Min-Max (0 a 1)
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        
        return image