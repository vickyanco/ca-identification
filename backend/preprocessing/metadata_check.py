# file: preprocessing/metadata_check.py
# description: Extract metadata from DICOM files.
# author: María Victoria Anconetani
# date: 19/02/2025

import pydicom
import os

dicom_folder = r"E:/CA EN CMR/T1Map T1 4cam_dcm/p1_control_1"  # Change to an actual folder

dicom_files = sorted([f for f in os.listdir(dicom_folder) if os.path.isfile(os.path.join(dicom_folder, f))])

for dicom_file in dicom_files:
    dicom_path = os.path.join(dicom_folder, dicom_file)
    ds = pydicom.dcmread(dicom_path)

    print(f"\n📂 File: {dicom_file}")
    print(f"📋 Series Description: {ds.get('SeriesDescription', 'N/A')}")
    print(f"🔢 Instance Number: {ds.get('InstanceNumber', 'N/A')}")
    print(f"📍 Image Position (Patient): {ds.get('ImagePositionPatient', 'N/A')}")
    print(f"⏳ Temporal Position Index: {ds.get('TemporalPositionIndex', 'N/A')}")
    print(f"📏 Slice Thickness: {ds.get('SliceThickness', 'N/A')} mm")
    print(f"📐 Spacing Between Slices: {ds.get('SpacingBetweenSlices', 'N/A')} mm")
    print(f"⏱ Inversion Time (TI): {ds.get((0x0018, 0x0082), 'N/A')}")  # TI value
    print(f"🔄 Image Orientation: {ds.get('ImageOrientationPatient', 'N/A')}")
    print(f"📡 Echo Time (TE): {ds.get((0x0018, 0x0081), 'N/A')}")  # Echo time
    print(f"🕒 Acquisition Time: {ds.get('AcquisitionTime', 'N/A')}")



