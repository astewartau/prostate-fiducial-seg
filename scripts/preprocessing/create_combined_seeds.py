#!/usr/bin/env python3
"""
Create combined seeds file from individual GS1, GS2, GS3 marker files.
This combines the three gold seed markers into a single multi-label file.
"""

import nibabel as nib
import numpy as np
import glob
import os
import argparse

parser = argparse.ArgumentParser(description='Create combined seeds files from individual GS ROIs')
parser.add_argument('--data-dir', default='data/train', help='Path to data directory containing subject folders')
args = parser.parse_args()

bids_dir = args.data_dir
subject_dirs = sorted(glob.glob(os.path.join(bids_dir, "z*")))

for subject_dir in subject_dirs:
    subject_id = os.path.basename(subject_dir)
    roi_dir = os.path.join(subject_dir, "roi_niftis_mri_space")

    if not os.path.exists(roi_dir):
        print(f"Skipping {subject_id}: no roi_niftis_mri_space directory")
        continue

    # Check if seeds file already exists
    seeds_path = os.path.join(roi_dir, f"{subject_id}_seeds.nii.gz")
    if os.path.exists(seeds_path):
        continue

    # Find GS marker files
    gs_files = {}
    for gs_num in [1, 2, 3]:
        # Try both naming conventions: GS1_01 and GS1
        gs_pattern1 = os.path.join(roi_dir, f"{subject_id}_roi_GS{gs_num}_01_MR.nii.gz")
        gs_pattern2 = os.path.join(roi_dir, f"{subject_id}_roi_GS{gs_num}_MR.nii.gz")

        if os.path.exists(gs_pattern1):
            gs_files[gs_num] = gs_pattern1
        elif os.path.exists(gs_pattern2):
            gs_files[gs_num] = gs_pattern2

    # Need at least one GS marker to create seeds file
    if not gs_files:
        print(f"Skipping {subject_id}: no GS marker files found")
        continue

    # Find prostate segmentation (CTV_High)
    prostate_files = [
        os.path.join(roi_dir, f"{subject_id}_roi_CTV_High_MR.nii.gz"),
        os.path.join(roi_dir, f"{subject_id}_roi_CTV_Low_MR.nii.gz"),
        os.path.join(roi_dir, f"{subject_id}_roi_CTVp_MR.nii.gz"),
    ]
    prostate_file = None
    for pf in prostate_files:
        if os.path.exists(pf):
            prostate_file = pf
            break

    if not prostate_file:
        print(f"Skipping {subject_id}: no prostate segmentation (CTV) found")
        continue

    print(f"Creating seeds file for {subject_id} from {len(gs_files)} markers + prostate")

    # Load prostate to get shape and affine
    prostate_img = nib.load(prostate_file)
    combined_seeds = np.zeros(prostate_img.shape, dtype=np.uint8)

    # Add prostate as label 2
    prostate_data = prostate_img.get_fdata()
    combined_seeds[prostate_data > 0] = 2

    # Add gold seed markers as label 1 (overwrites prostate where they overlap)
    for gs_num, gs_path in sorted(gs_files.items()):
        gs_img = nib.load(gs_path)
        gs_data = gs_img.get_fdata()
        combined_seeds[gs_data > 0] = 1

    # Save combined seeds file
    seeds_nii = nib.Nifti1Image(combined_seeds, prostate_img.affine, prostate_img.header)
    nib.save(seeds_nii, seeds_path)
    print(f"  Saved: {seeds_path}")

print("\nDone creating combined seeds files!")
