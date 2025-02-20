# file: utils/check_shape.py
# description: Check the shape of a NIfTI file.
# author: María Victoria Anconetani
# date: 19/02/2025

import os
import nibabel as nib

def print_nifti_shapes(folder_path):
    """Prints the shape of all NIfTI (.nii or .nii.gz) files inside a folder."""
    
    # List all files in the directory
    nifti_files = [f for f in os.listdir(folder_path) if f.endswith('.nii') or f.endswith('.nii.gz')]
    
    if not nifti_files:
        print("No NIfTI files found in the folder.")
        return
    
    # Loop through each file and print its shape
    for file in nifti_files:
        file_path = os.path.join(folder_path, file)
        try:
            nifti_image = nib.load(file_path)  # Load the NIfTI file
            shape = nifti_image.get_fdata().shape  # Get the shape of the image
            print(f"File: {file}, Shape: {shape}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

# Example usage
folder_path = "E:/CA EN CMR/DE_SS_EC_tfi_psir_p2_PSIR_nii"  
print_nifti_shapes(folder_path)

