# file: lge_preprocessing.py
# description: Preprocessing class for LGE images
# author: María Victoria Anconetani
# date: 12/02/2025

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

def preprocess_nifti(file_path, target_shape=(172, 192, 12)):
    """Loads a NIfTI file, resizes it to the target shape, normalizes with Z- score normalization, and returns a new NIfTI image."""
    
    # Load NIfTI image
    nifti_img = nib.load(file_path)
    img_data = nifti_img.get_fdata()

    # Get current shape
    current_shape = img_data.shape

    # Step 1: Resize depth (Z-Dimension)
    target_slices = target_shape[2]
    if current_shape[2] > target_slices:
        # Trim excess slices (center crop)
        start_idx = (current_shape[2] - target_slices) // 2
        img_data = img_data[:, :, start_idx:start_idx + target_slices]
    elif current_shape[2] < target_slices:
        # Pad slices with zeros
        pad_size = target_slices - current_shape[2]
        img_data = np.pad(img_data, ((0, 0), (0, 0), (0, pad_size)), mode='constant')

    # Step 2: Resize width & height if needed
    zoom_factors = (target_shape[0] / img_data.shape[0], target_shape[1] / img_data.shape[1], 1)
    img_data = zoom(img_data, zoom_factors, order=1)  # Linear interpolation

    # Step 3: Apply Z-Score Normalization (Standardization)
    mean = np.mean(img_data)
    std = np.std(img_data)
    
    # Avoid division by zero
    if std > 0:
        img_data = (img_data - mean) / std
    else:
        img_data = img_data - mean  # If std = 0, just subtract mean

    # Step 4: Convert back to NIfTI format
    new_nifti = nib.Nifti1Image(img_data, affine=nifti_img.affine)

    return new_nifti

def process_nifti_folder(input_folder, output_folder, target_shape=(172, 192, 12)):
    """Processes all NIfTI files in a folder and saves them to a new folder."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    nifti_files = [f for f in os.listdir(input_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]

    for file in nifti_files:
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)

        try:
            processed_nifti = preprocess_nifti(input_path, target_shape=target_shape)
            nib.save(processed_nifti, output_path)  # Save processed image
            print(f"Processed & Saved: {file} -> Shape: {target_shape}")
        except Exception as e:
            print(f"Error processing {file}: {e}")

# Example usage
input_folder = "E:/CA EN CMR/DE_SS_EC_tfi_psir_p2_PSIR_nii"  
output_folder = "E:/CA EN CMR/LGE_prep_nii"  
process_nifti_folder(input_folder, output_folder)