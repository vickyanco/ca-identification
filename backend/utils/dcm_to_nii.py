# file: utils/dcm_to_nii.py
# description: Convert DICOM files to NIfTI format using dcm2niix.
# author: María Victoria Anconetani
# date: 19/02/2025

import os
import subprocess

def convert_all_to_nifti(input_folder, output_folder):
    """
    Recursively searches for DICOM folders and converts them to NIfTI using dcm2niix,
    saving all output files directly into a single folder while keeping the original folder names.

    Parameters:
        input_folder (str): Root folder containing DICOM files organized in subfolders.
        output_folder (str): Destination folder for all converted NIfTI files.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    for root, dirs, files in os.walk(input_folder):  # Recursively find all subdirectories
        if not files:  # Skip empty folders
            continue

        # Extract the last folder name to use as the output file name
        folder_name = os.path.basename(root)

        print(f"📂 Converting DICOM series in: {root} (saving as {folder_name}.nii.gz)")

        subprocess.run([
            "dcm2niix",
            "-o", output_folder,  # Save all files in the same directory
            "-f", folder_name,     # Name output file after the folder
            "-z", "y",             # Compress output to .nii.gz
            root                   # Input DICOM folder
        ], check=True)

if __name__ == "__main__":
    input_folder_2 = r"E:/CA EN CMR/T1Map T1 4cam_dcm"
    input_folder = r"E:/CA EN CMR/DE_SS_EC_tfi_psir_p2_PSIR_dcm"
    output_folder_2 = r"E:/CA EN CMR/T1Map T1 4cam_nii"
    output_folder = r"E:/CA EN CMR/DE_SS_EC_tfi_psir_p2_PSIR_nii"   

    convert_all_to_nifti(input_folder, output_folder)
    print("✅ NIfTI conversion completed! All files are in:", output_folder)