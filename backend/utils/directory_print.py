# file: utils/rename_slices.py
# description: Utility function to load DICOM slices and rename them uniquely.
# author: María Victoria Anconetani
# date: 21/02/2025

import os
import shutil

def duplicate_and_rename(source_dir, target_dir):
    """
    Duplicates a folder structure while renaming DICOM files uniquely.

    Parameters:
        source_dir (str): The original dataset directory.
        target_dir (str): The new directory where the renamed dataset will be stored.
    """
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Walk through all directories in the source dataset
    for root, dirs, files in os.walk(source_dir):
        # Compute relative path to maintain folder structure
        relative_path = os.path.relpath(root, source_dir)
        new_folder_path = os.path.join(target_dir, relative_path)

        # Create corresponding folder in the target directory
        os.makedirs(new_folder_path, exist_ok=True)

        # Extract patient and study ID from the folder name
        path_parts = relative_path.split(os.sep)  # Split by folder separator
        if len(path_parts) < 2:
            continue  # Skip root level

        patient_id = path_parts[-1].split("_")[0]  # Extract patient ID (e.g., "p12")
        study_id = path_parts[-1]  # Full study name (e.g., "p12_caso_1")

        # Rename and copy files
        for idx, file in enumerate(sorted(files)):
            old_file_path = os.path.join(root, file)
            new_file_name = f"{patient_id}_{study_id}_{idx}.dcm"  # Ensure unique naming
            new_file_path = os.path.join(new_folder_path, new_file_name)

            # Copy and rename file
            shutil.copy2(old_file_path, new_file_path)
            print(f"Copied: {old_file_path} -> {new_file_path}")

# Set source and target directories
source_directory = "E:/CA EN CMR/T1Map T1 4cam_dcm_div"
target_directory = "E:/CA EN CMR/T1Map_re"

# Run the duplication process
duplicate_and_rename(source_directory, target_directory)