#!/usr/bin/env python3
"""
Reconstruct the exact fold_id to validation subject mapping for both old and new models.
This replicates the exact data loading logic from train_one_model.py.
"""

import glob
import os
import pandas as pd
from sklearn.model_selection import KFold

def build_subject_dataframe(bids_dirs, exclude_invalid=False):
    """Build dataframe exactly as train_one_model.py does."""
    paths_list = []

    for bids_dir in bids_dirs:
        subject_dirs = sorted(glob.glob(os.path.join(bids_dir, "z*")))

        for subject_dir in subject_dirs:
            subject_id = os.path.basename(subject_dir)

            # Skip invalid segmentation folder if requested
            if exclude_invalid and subject_id == "z1438488-Invalid-Segmentation":
                continue

            # Check if roi_niftis_mri_space/ directory exists
            roi_dir = os.path.join(subject_dir, "roi_niftis_mri_space")
            if not os.path.exists(roi_dir):
                continue

            # Get all nifti files
            nii_paths = glob.glob(os.path.join(subject_dir, "*.nii")) + \
                        glob.glob(os.path.join(subject_dir, "*.nii.gz"))

            # Get all nifti files in the roi_niftis_mri_space directory
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

            # Append the dictionary to the list
            paths_list.append(paths_dict)

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(paths_list)
    # Replace NaN values with None
    df = df.where(pd.notnull(df), None)

    # For rows with no Prostate segmentation, if there is a roi_CTV_Low_MR segmentation, use it as Prostate
    for index, row in df.iterrows():
        if row.get('Prostate') is None and row.get('roi_CTV_Low_MR') is not None:
            df.at[index, 'Prostate'] = row['roi_CTV_Low_MR']

    for index, row in df.iterrows():
        if row.get('Prostate') is None and row.get('roi_CTVp_MR') is not None:
            df.at[index, 'Prostate'] = row['roi_CTVp_MR']

    # Remove all columns except
    columns_to_keep = ['subject_id', 'MRI', 'MRI_homogeneity-corrected', 'CT', 'seeds', 'Prostate']
    df = df[columns_to_keep]

    # Keep only rows that have both the input and segmentation files.
    infile_cols = ['MRI_homogeneity-corrected']
    seg_col = 'seeds'

    # Remove rows with missing values in the required columns
    for col in infile_cols + [seg_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame.")
        df = df[df[col].notna()].reset_index(drop=True)

    return df

def create_fold_mapping(df, random_state=42):
    """Create mapping from fold_id to validation subject."""
    k_folds = len(df)
    kf = KFold(n_splits=k_folds, random_state=random_state, shuffle=True)

    fold_mapping = {}
    for fold_id, (train_index, valid_index) in enumerate(kf.split(df)):
        assert len(valid_index) == 1, f"Expected one validation subject, but got {len(valid_index)}."
        validation_subject = df.iloc[valid_index[0]]['subject_id']
        fold_mapping[fold_id] = validation_subject

    return fold_mapping

def main():
    print("=" * 80)
    print("RECONSTRUCTING FOLD MAPPINGS")
    print("=" * 80)
    print()

    # Build old dataset (includes z1438488)
    print("Building OLD dataset (as used by old models)...")
    old_df = build_subject_dataframe(["bids-2025-2"], exclude_invalid=False)
    print(f"Old dataset: {len(old_df)} subjects")
    print(f"Subjects: {sorted(old_df['subject_id'].tolist())}")
    print()

    # Build new dataset (excludes z1438488)
    print("Building NEW dataset (as used by new models)...")
    new_df = build_subject_dataframe(["bids-2025-2"], exclude_invalid=True)
    print(f"New dataset: {len(new_df)} subjects")
    print(f"Subjects: {sorted(new_df['subject_id'].tolist())}")
    print()

    # Create fold mappings
    print("Creating fold mappings...")
    old_fold_mapping = create_fold_mapping(old_df, random_state=42)
    new_fold_mapping = create_fold_mapping(new_df, random_state=42)

    print(f"\nOld models: {len(old_fold_mapping)} folds")
    print(f"New models: {len(new_fold_mapping)} folds")
    print()

    # Find common validation subjects
    old_subjects = set(old_fold_mapping.values())
    new_subjects = set(new_fold_mapping.values())
    common_subjects = old_subjects & new_subjects

    print(f"Common validation subjects: {len(common_subjects)}")
    print(f"Old-only subjects: {len(old_subjects - new_subjects)}")
    print(f"New-only subjects: {len(new_subjects - old_subjects)}")
    print()

    if old_subjects - new_subjects:
        print(f"Old-only subjects (should be z1438488): {sorted(old_subjects - new_subjects)}")

    # Create reverse mapping: subject -> fold_id
    old_subject_to_fold = {v: k for k, v in old_fold_mapping.items()}
    new_subject_to_fold = {v: k for k, v in new_fold_mapping.items()}

    # Show some examples
    print("\nExample mappings:")
    print("-" * 80)
    print(f"{'Subject':<15} {'Old Fold ID':<15} {'New Fold ID':<15}")
    print("-" * 80)
    for subject in sorted(list(common_subjects))[:10]:
        old_fold = old_subject_to_fold.get(subject, "N/A")
        new_fold = new_subject_to_fold.get(subject, "N/A")
        print(f"{subject:<15} {old_fold:<15} {new_fold:<15}")

    # Save mappings to files for use by comparison script
    print("\nSaving fold mappings...")

    with open("old_fold_mapping.txt", "w") as f:
        for fold_id, subject in sorted(old_fold_mapping.items()):
            f.write(f"{fold_id}\t{subject}\n")

    with open("new_fold_mapping.txt", "w") as f:
        for fold_id, subject in sorted(new_fold_mapping.items()):
            f.write(f"{fold_id}\t{subject}\n")

    with open("common_subjects.txt", "w") as f:
        for subject in sorted(common_subjects):
            old_fold = old_subject_to_fold[subject]
            new_fold = new_subject_to_fold[subject]
            f.write(f"{subject}\t{old_fold}\t{new_fold}\n")

    print("Saved:")
    print("  - old_fold_mapping.txt")
    print("  - new_fold_mapping.txt")
    print("  - common_subjects.txt")
    print()

    # Check if z1438488 had invalid segmentation
    if "z1438488-Invalid-Segmentation" in old_subjects:
        print(f"✓ Confirmed: z1438488-Invalid-Segmentation is in old models")
        print(f"  Old fold ID: {old_subject_to_fold['z1438488-Invalid-Segmentation']}")

if __name__ == "__main__":
    main()
