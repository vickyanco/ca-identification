# file: t1map_preprocessing.py
# description: Preprocessing class for T1 mappping
# author: María Victoria Anconetani
# date: 12/02/2025

import numpy as np
from .base_preprocessor import BasePreprocessor

class T1Preprocessor(BasePreprocessor):
    def preprocess(self, file_path):
        image = self.load_nifti(file_path)
        image = self.resize_image(image)

        # Z-score normalization
        mean = np.mean(image)
        std = np.std(image)
        image = (image - mean) / (std + 1e-8)

        return image