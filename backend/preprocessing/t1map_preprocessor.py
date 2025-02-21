# file: t1map_preprocessing.py
# description: Preprocessing class for T1 mappping images
# author: María Victoria Anconetani
# date: 12/02/2025

import os
import numpy as np
import pydicom
import cv2
from tqdm import tqdm

def normalize_zscore(img):
    """ Apply Z-Score Normalization without rescaling. """
    mean = np.mean(img)
    std = np.std(img)

    # Avoid division by zero
    if std > 0:
        img = (img - mean) / std
    else:
        img = img - mean  # If std = 0, just subtract mean

    return img  

def preprocess_dicom_images(source_dir, target_dir, img_size=(256, 256)):
    """
    Walks through the dataset, resizes DICOM images, applies Z-score normalization, and saves as DICOM.

    Parameters:
        source_dir (str): Path to the original dataset.
        target_dir (str): Path where preprocessed images will be saved.
        img_size (tuple): Target size for resizing (default: 256x256).
    """
    os.makedirs(target_dir, exist_ok=True)  # Ensure target directory exists

    for root, dirs, files in os.walk(source_dir):
        relative_path = os.path.relpath(root, source_dir)
        new_folder_path = os.path.join(target_dir, relative_path)
        os.makedirs(new_folder_path, exist_ok=True)

        for file in tqdm(sorted(files), desc=f"Processing {relative_path}"):
            old_file_path = os.path.join(root, file)
            new_file_path = os.path.join(new_folder_path, file)  # Keep the same filename

            try:
                ds = pydicom.dcmread(old_file_path)
                img = ds.pixel_array.astype(np.float32)  # Convert to float
                
                # Apply Z-Score Normalization
                img_norm = normalize_zscore(img)

                # Resize image to 256x256
                img_resized = cv2.resize(img_norm, img_size, interpolation=cv2.INTER_AREA)

                # Convert back to int16 (preserving Z-score values)
                ds.PixelData = img_resized.astype(np.int16).tobytes()
                ds.Rows, ds.Columns = img_size  # Update metadata
                ds.BitsAllocated = 16  # Keep 16-bit for higher precision
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.PhotometricInterpretation = "MONOCHROME2"

                print(f"Min: {np.min(img_resized)}, Max: {np.max(img_resized)}, Mean: {np.mean(img_resized)}")
                print(f"After Z-score: Min: {np.min(img_norm)}, Max: {np.max(img_norm)}")

                ds.save_as(new_file_path)  # Save new DICOM file

            except Exception as e:
                print(f"⚠️ Error processing {old_file_path}: {e}")

# Set source and target directories
source_directory = "E:/CA EN CMR/T1Map_re"
target_directory = "E:/CA EN CMR/T1Map_pre"

# Run the preprocessing function
preprocess_dicom_images(source_directory, target_directory)