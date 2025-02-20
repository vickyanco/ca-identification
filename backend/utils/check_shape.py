# file: utils/check_shape.py
# description: Check the shape of a NIfTI file.
# author: María Victoria Anconetani
# date: 19/02/2025

import nibabel as nib
import os

# Path to your NIfTI file
nii_file_2 = r"E:/CA EN CMR/T1Map T1 4cam_nii/p26_control_2.nii.gz"
nii_file = r"E:/CA EN CMR/DE_SS_EC_tfi_psir_p2_PSIR_nii/p3_caso_real.nii.gz"

# Load NIfTI file
nii_img = nib.load(nii_file)

# Get shape
print(f"File: {os.path.basename(nii_file)}")
print(f"NIfTI shape: {nii_img.shape}")  # This should be (X, Y, Z), where Z > 1 for a 3D volume
