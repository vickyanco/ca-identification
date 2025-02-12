# file: data_loader.py
# description: Base class for image preprocessing
# author: María Victoria Anconetani
# date: 12/02/2025

import glob
import os
import numpy as np
from preprocessing.lge_preprocessor import LGEPreprocessor
from preprocessing.t1map_preprocessor import T1Preprocessor

def load_dataset(positive_path, negative_path, preprocessor):
    """
    Load NIfTI images from case and control directories, and apply preprocessing.

    Parameters:
    - positive_path: Path to the folder with cases (e.g., "lge/casos/")
    - negative_path: Path to the folder with controls (e.g., "lge/controles/")
    - preprocessor: Instance of LGEPreprocessor or T1Preprocessor

    Returns:
    - X (array of images)
    - y (array of labels)
    """
    # Load positive images (cases)
    pos_files = glob.glob(os.path.join(positive_path, "**/*.nii.gz"), recursive=True)
    pos_images = [preprocessor.preprocess(file) for file in pos_files]
    pos_labels = [1] * len(pos_images)  # Etiqueta 1 para casos

    # Load negative images (controls)
    neg_files = glob.glob(os.path.join(negative_path, "**/*.nii.gz"), recursive=True)
    neg_images = [preprocessor.preprocess(file) for file in neg_files]
    neg_labels = [0] * len(neg_images)  # Etiqueta 0 para controles

    # Combine data and labels
    X = np.array(pos_images + neg_images)
    y = np.array(pos_labels + neg_labels)

    return X, y

# Ejemplo de uso
lge_preprocessor = LGEPreprocessor()
t1_preprocessor = T1Preprocessor()

lge_X, lge_y = load_dataset("lge_casos", "lge_controls", lge_preprocessor)
t1_X, t1_y = load_dataset("t1_mapping_casos", "t1_mapping_controls", t1_preprocessor)

print(f"LGE images: {lge_X.shape}, Labels: {lge_y.shape}")
print(f"T1 images: {t1_X.shape}, Labels: {t1_y.shape}")