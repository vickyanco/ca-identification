# file: preprocessing/print.py
# description: Count files in subfolders (to check data distribution).
# author: María Victoria Anconetani
# date: 19/02/2025

import os

def count_files_in_subfolders(directory):
    for root, dirs, files in os.walk(directory):
        file_count = len(files)
        print(f"📂 {root} - {file_count} files")

# Set the directory you want to explore
directory_path = "DE_SS_EC_tfi_psir_p2_PSIR_dcm"

# Call the function
count_files_in_subfolders(directory_path)

