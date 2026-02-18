#!/usr/bin/env python3
"""
Shared data loading utility for the prostate fiducial marker segmentation project.

Builds a pandas DataFrame of subject paths from a BIDS-like directory structure.
"""

import glob
import os

import pandas as pd

# Standard columns kept from the raw scan
COLUMNS_TO_KEEP = ['subject_id', 'MRI', 'MRI_homogeneity-corrected', 'CT', 'seeds', 'Prostate']

# Default input and segmentation columns
INFILE_COLS = ['MRI_homogeneity-corrected']
SEG_COL = 'seeds'


def build_subject_dataframe(data_dir, infile_cols=None, seg_col=None, verbose=True):
    """
    Build a DataFrame of subject file paths from a BIDS-like directory.

    Args:
        data_dir: Path to directory containing z{ID}/ subject folders.
        infile_cols: List of required input columns (default: ['MRI_homogeneity-corrected']).
        seg_col: Required segmentation column (default: 'seeds').
        verbose: Print skipped subjects and summary.

    Returns:
        df: DataFrame with one row per usable subject.
        infile_cols: The input column names used.
        seg_col: The segmentation column name used.
    """
    if infile_cols is None:
        infile_cols = list(INFILE_COLS)
    if seg_col is None:
        seg_col = SEG_COL

    paths_list = []
    subject_dirs = sorted(glob.glob(os.path.join(data_dir, "z*")))

    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)

        roi_dir = os.path.join(subject_dir, "roi_niftis_mri_space")
        if not os.path.exists(roi_dir):
            if verbose:
                print(f"Skipping {subject_id}: no roi_niftis_mri_space directory")
            continue

        nii_paths = glob.glob(os.path.join(subject_dir, "*.nii")) + \
                    glob.glob(os.path.join(subject_dir, "*.nii.gz"))
        nii_paths += glob.glob(os.path.join(roi_dir, "*.nii")) + \
                     glob.glob(os.path.join(roi_dir, "*.nii.gz"))

        paths_dict = {}
        for nii_path in nii_paths:
            paths_dict['subject_id'] = subject_id
            file_name = os.path.basename(nii_path).split('.')[0]

            if 'GS1' in file_name:
                segmentation_name = "GS1"
            elif 'GS2' in file_name:
                segmentation_name = "GS2"
            elif 'GS3' in file_name:
                segmentation_name = "GS3"
            else:
                segmentation_name = "_".join(file_name.split('_')[1:])

            if segmentation_name == "roi_CTV_High_MR":
                segmentation_name = "Prostate"

            paths_dict[segmentation_name] = nii_path

        paths_list.append(paths_dict)

    df = pd.DataFrame(paths_list)
    df = df.where(pd.notnull(df), None)

    # Prostate fallback: CTV_High -> CTV_Low -> CTVp
    for index, row in df.iterrows():
        if row.get('Prostate') is None:
            if row.get('roi_CTV_Low_MR') is not None:
                df.at[index, 'Prostate'] = row['roi_CTV_Low_MR']
            elif row.get('roi_CTVp_MR') is not None:
                df.at[index, 'Prostate'] = row['roi_CTVp_MR']

    # Keep standard columns
    cols = [c for c in COLUMNS_TO_KEEP if c in df.columns]
    df = df[cols]

    # Filter for required files
    for col in infile_cols + [seg_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame. Available: {list(df.columns)}")
        df = df[df[col].notna()].reset_index(drop=True)

    if verbose:
        print(f"Loaded {len(df)} subjects from {data_dir}")

    return df, infile_cols, seg_col
