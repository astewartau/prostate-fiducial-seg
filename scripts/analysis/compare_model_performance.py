#!/usr/bin/env python3
"""
Compare overall average performance between old models (trained on 60 subjects including
invalid z1438488) and new models (trained on 93 subjects excluding z1438488).

For OLD models: Reconstruct the data split to determine validation subjects.
For NEW models: Read validation_metadata from checkpoint files.

For each completed model, we extract the best epoch metrics from training logs and compare
aggregate statistics.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

def reconstruct_old_dataset():
    """
    Reconstruct the 60-subject dataset as it existed for old model training.
    This includes z1438488 (from FILES/ subdirectory) which was valid at the time.
    Excludes subjects with NEW marker (these were added after old training).
    Returns a sorted list of subject IDs that would have been used.
    """
    bids_dirs = ["bids-2025-2"]
    paths_list = []

    for bids_dir in bids_dirs:
        subject_dirs = sorted(glob.glob(os.path.join(bids_dir, "z*")))

        for subject_dir in subject_dirs:
            subject_id = os.path.basename(subject_dir)

            # Skip subjects with NEW marker (added after old training)
            if os.path.exists(os.path.join(subject_dir, "NEW")):
                continue

            # For z1438488, check if FILES/roi_niftis_mri_space exists (old location)
            if subject_id == "z1438488-Invalid-Segmentation":
                roi_dir = os.path.join(subject_dir, "FILES", "roi_niftis_mri_space")
                if os.path.exists(roi_dir):
                    # Check for required files in FILES/
                    nii_paths = glob.glob(os.path.join(subject_dir, "FILES", "*.nii*"))
                    nii_paths += glob.glob(os.path.join(roi_dir, "*.nii*"))
                else:
                    continue
            else:
                # Normal check for roi_niftis_mri_space
                roi_dir = os.path.join(subject_dir, "roi_niftis_mri_space")
                if not os.path.exists(roi_dir):
                    continue

                nii_paths = glob.glob(os.path.join(subject_dir, "*.nii*"))
                nii_paths += glob.glob(os.path.join(roi_dir, "*.nii*"))

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

    # Apply same filters as in train_one_model.py
    columns_to_keep = ['subject_id', 'MRI', 'MRI_homogeneity-corrected', 'CT', 'seeds', 'Prostate']
    df = df[columns_to_keep]

    infile_cols = ['MRI_homogeneity-corrected']
    seg_col = 'seeds'

    for col in infile_cols + [seg_col]:
        df = df[df[col].notna()].reset_index(drop=True)

    # Return sorted list of subject IDs
    return sorted(df['subject_id'].tolist())

def create_old_fold_mapping(subject_list, random_state=42):
    """
    Recreate the LOOCV fold mapping using the same logic as train_one_model.py.
    Returns dict: fold_id -> validation_subject_id
    """
    k_folds = len(subject_list)
    kf = KFold(n_splits=k_folds, random_state=random_state, shuffle=True)

    fold_mapping = {}
    for fold_id, (train_index, valid_index) in enumerate(kf.split(subject_list)):
        # Should be exactly 1 validation subject
        assert len(valid_index) == 1, f"Expected 1 validation subject, got {len(valid_index)}"
        validation_subject = subject_list[valid_index[0]]
        fold_mapping[fold_id] = validation_subject

    return fold_mapping

def load_checkpoint_metadata(checkpoint_path):
    """
    Load validation_metadata from a checkpoint file.
    Returns None if torch is not available or metadata is missing.
    """
    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint.get('validation_metadata', None)
    except ImportError:
        print("WARNING: PyTorch not available, cannot load checkpoint metadata")
        return None
    except Exception as e:
        print(f"WARNING: Error loading checkpoint {checkpoint_path}: {e}")
        return None

def find_completed_models(directory):
    """
    Find models that have both best and final .pth files.
    Returns dict mapping fold_id -> {best_file, final_file}
    """
    completed = {}

    # Find all best and final files
    best_files = glob.glob(os.path.join(directory, "*-best.pth"))
    final_files = glob.glob(os.path.join(directory, "*-final.pth"))

    # Extract fold IDs from filenames
    for best_file in best_files:
        # Pattern: T1-YYYYMMDD-HHMMSS-{fold_id}-best.pth
        match = re.search(r'-(\d+)-best\.pth$', best_file)
        if match:
            fold_id = int(match.group(1))
            # Check if corresponding final file exists
            final_pattern = re.sub(r'-best\.pth$', '-final.pth', best_file)
            if os.path.exists(final_pattern):
                completed[fold_id] = {
                    'best_file': best_file,
                    'final_file': final_pattern
                }

    return completed

def find_output_file(directory, fold_id, best_file):
    """
    Find the .out file corresponding to this model training run.
    Match by fold_id in the output content.
    """
    # Look for .out files
    out_files = glob.glob(os.path.join(directory, "*.out"))

    # Try to find .out file that contains this fold info
    for out_file in out_files:
        try:
            with open(out_file, 'r') as f:
                # Read first 100 lines to find the fold identifier
                first_lines = []
                for i, line in enumerate(f):
                    if i >= 100:
                        break
                    first_lines.append(line)
                first_content = ''.join(first_lines)

                # Check if this output is for the right fold
                if f"=== T1-{fold_id} ===" in first_content:
                    return out_file
        except (UnicodeDecodeError, Exception):
            continue

    return None

def parse_metrics_from_output(out_file):
    """
    Parse training metrics from .out file.
    Returns metrics from the epoch with minimum validation loss (best epoch).
    """
    if not out_file or not os.path.exists(out_file):
        return None

    try:
        with open(out_file, 'r') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        return None

    # Parse all epochs
    epochs = []
    for i, line in enumerate(lines):
        # Look for epoch line: "Epoch X/Y, Train Loss: Z, Valid Loss: W"
        epoch_match = re.search(r'Epoch (\d+)/\d+, Train Loss: ([\d.]+), Valid Loss: ([\d.]+)', line)
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            train_loss = float(epoch_match.group(2))
            valid_loss = float(epoch_match.group(3))

            # Next line should have metrics
            if i + 1 < len(lines):
                metrics_line = lines[i + 1]
                metrics_match = re.search(
                    r'Actual seed markers: (\d+), True positive: (\d+), False negative: (\d+), False positive: (\d+)',
                    metrics_line
                )
                if metrics_match:
                    epochs.append({
                        'epoch': epoch_num,
                        'train_loss': train_loss,
                        'valid_loss': valid_loss,
                        'actual_markers': int(metrics_match.group(1)),
                        'true_positive': int(metrics_match.group(2)),
                        'false_negative': int(metrics_match.group(3)),
                        'false_positive': int(metrics_match.group(4))
                    })

    if not epochs:
        return None

    # Find epoch with minimum validation loss (this is the "best" epoch)
    best_epoch = min(epochs, key=lambda x: x['valid_loss'])

    return best_epoch

def compute_derived_metrics(metrics_list):
    """
    Compute derived metrics (sensitivity, precision) from raw counts.
    """
    results = []
    for m in metrics_list:
        sensitivity = m['true_positive'] / m['actual_markers'] if m['actual_markers'] > 0 else 0
        precision = m['true_positive'] / (m['true_positive'] + m['false_positive']) if (m['true_positive'] + m['false_positive']) > 0 else 0

        results.append({
            **m,
            'sensitivity': sensitivity,
            'precision': precision
        })

    return results

def main():
    old_models_dir = "old-models-pre-NEW-data"
    new_models_dir = "."  # Current directory

    print("=" * 80)
    print("COMPARING OLD vs NEW MODEL PERFORMANCE")
    print("=" * 80)
    print()
    print("Old models: trained on 60 subjects (including invalid z1438488)")
    print("New models: trained on 93 subjects (excluding invalid z1438488)")
    print()

    # ========== OLD MODELS: Reconstruct dataset and fold mapping ==========
    print("Reconstructing old dataset...")
    old_subject_list = reconstruct_old_dataset()
    print(f"  Old dataset size: {len(old_subject_list)} subjects")

    if len(old_subject_list) != 60:
        print(f"WARNING: Expected 60 old subjects, but found {len(old_subject_list)}")
        print("  This may indicate a data reconstruction issue.")

    print("\nCreating old fold mapping...")
    old_fold_mapping = create_old_fold_mapping(old_subject_list, random_state=42)
    print(f"  Created {len(old_fold_mapping)} fold mappings")

    # Find completed models in both directories
    print("\nFinding completed models...")
    old_completed = find_completed_models(old_models_dir)
    new_completed = find_completed_models(new_models_dir)

    print(f"Old models completed: {len(old_completed)} / 60 folds")
    print(f"New models completed: {len(new_completed)} / 93 folds")
    print()

    if len(old_completed) == 0:
        print("ERROR: No completed old models found!")
        return

    if len(new_completed) == 0:
        print("ERROR: No completed new models found!")
        return

    # ========== Extract metrics from OLD models ==========
    print("Extracting metrics from old models...")
    old_metrics = []
    old_fold_ids = []
    old_subjects = []

    for fold_id in sorted(old_completed.keys()):
        validation_subject = old_fold_mapping.get(fold_id, "UNKNOWN")
        out_file = find_output_file(old_models_dir, fold_id, old_completed[fold_id]['best_file'])
        metrics = parse_metrics_from_output(out_file)

        if metrics:
            old_metrics.append(metrics)
            old_fold_ids.append(fold_id)
            old_subjects.append(validation_subject)
            print(f"  Fold {fold_id} (validation: {validation_subject}): ✓")
        else:
            print(f"  Fold {fold_id} (validation: {validation_subject}): ✗ (metrics not found)")

    # ========== Extract metrics from NEW models ==========
    print()
    print("Extracting metrics from new models...")
    new_metrics = []
    new_fold_ids = []
    new_subjects = []

    for fold_id in sorted(new_completed.keys()):
        # Load validation metadata from checkpoint
        metadata = load_checkpoint_metadata(new_completed[fold_id]['best_file'])

        if metadata:
            validation_subject = metadata.get('subject_id', 'UNKNOWN')
        else:
            validation_subject = 'UNKNOWN (no metadata)'

        out_file = find_output_file(new_models_dir, fold_id, new_completed[fold_id]['best_file'])
        metrics = parse_metrics_from_output(out_file)

        if metrics:
            new_metrics.append(metrics)
            new_fold_ids.append(fold_id)
            new_subjects.append(validation_subject)
            print(f"  Fold {fold_id} (validation: {validation_subject}): ✓")
        else:
            print(f"  Fold {fold_id} (validation: {validation_subject}): ✗ (metrics not found)")

    print()
    print(f"Successfully extracted metrics from {len(old_metrics)} old models and {len(new_metrics)} new models")
    print()

    if len(old_metrics) == 0 or len(new_metrics) == 0:
        print("ERROR: Insufficient metrics extracted for comparison!")
        return

    # Compute derived metrics
    old_metrics = compute_derived_metrics(old_metrics)
    new_metrics = compute_derived_metrics(new_metrics)

    # Create DataFrames
    old_df = pd.DataFrame(old_metrics)
    new_df = pd.DataFrame(new_metrics)

    # Add fold IDs and validation subjects
    old_df['fold_id'] = old_fold_ids
    old_df['validation_subject'] = old_subjects
    new_df['fold_id'] = new_fold_ids
    new_df['validation_subject'] = new_subjects

    print("=" * 80)
    print("OVERALL PERFORMANCE COMPARISON")
    print("=" * 80)
    print()

    # Metrics to compare
    metrics_to_compare = [
        ('Validation Loss', 'valid_loss', 'lower_is_better'),
        ('True Positives', 'true_positive', 'higher_is_better'),
        ('False Negatives', 'false_negative', 'lower_is_better'),
        ('False Positives', 'false_positive', 'lower_is_better'),
        ('Sensitivity (Recall)', 'sensitivity', 'higher_is_better'),
        ('Precision', 'precision', 'higher_is_better'),
    ]

    results_summary = []

    for metric_name, col, direction in metrics_to_compare:
        old_vals = old_df[col].values
        new_vals = new_df[col].values

        old_mean = np.mean(old_vals)
        new_mean = np.mean(new_vals)
        old_std = np.std(old_vals, ddof=1)
        new_std = np.std(new_vals, ddof=1)
        old_median = np.median(old_vals)
        new_median = np.median(new_vals)

        # Independent samples t-test (since we're comparing different sets of models)
        if len(old_vals) > 1 and len(new_vals) > 1:
            t_stat, p_value = stats.ttest_ind(old_vals, new_vals)

            # Also compute Mann-Whitney U test (non-parametric alternative)
            u_stat, u_p_value = stats.mannwhitneyu(old_vals, new_vals, alternative='two-sided')
        else:
            t_stat, p_value = np.nan, np.nan
            u_stat, u_p_value = np.nan, np.nan

        # Determine if improvement
        if direction == 'higher_is_better':
            improvement = new_mean > old_mean
            improvement_pct = ((new_mean - old_mean) / old_mean * 100) if old_mean != 0 else 0
        else:  # lower_is_better
            improvement = new_mean < old_mean
            improvement_pct = ((old_mean - new_mean) / old_mean * 100) if old_mean != 0 else 0

        print(f"\n{metric_name}:")
        print(f"  Old models (n={len(old_vals)}): {old_mean:.4f} ± {old_std:.4f} (median: {old_median:.4f})")
        print(f"  New models (n={len(new_vals)}): {new_mean:.4f} ± {new_std:.4f} (median: {new_median:.4f})")
        print(f"  Difference: {new_mean - old_mean:+.4f} ({improvement_pct:+.1f}%)")

        if direction == 'higher_is_better':
            print(f"  {'✓ IMPROVEMENT' if improvement else '✗ DEGRADATION'}")
        else:
            print(f"  {'✓ IMPROVEMENT' if improvement else '✗ DEGRADATION'}")

        if not np.isnan(p_value):
            significance = ""
            if p_value < 0.001:
                significance = "***"
            elif p_value < 0.01:
                significance = "**"
            elif p_value < 0.05:
                significance = "*"
            else:
                significance = "ns"

            print(f"  t-test: t={t_stat:.3f}, p={p_value:.4f} {significance}")
            print(f"  Mann-Whitney U: U={u_stat:.1f}, p={u_p_value:.4f}")

        results_summary.append({
            'metric': metric_name,
            'old_mean': old_mean,
            'old_std': old_std,
            'new_mean': new_mean,
            'new_std': new_std,
            'difference': new_mean - old_mean,
            'improvement_pct': improvement_pct,
            'p_value': p_value,
            'significant': p_value < 0.05 if not np.isnan(p_value) else False
        })

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    # Count improvements
    improvements = sum(1 for r in results_summary if r['improvement_pct'] > 0)
    significant_improvements = sum(1 for r in results_summary if r['improvement_pct'] > 0 and r['significant'])

    print(f"Metrics showing improvement: {improvements} / {len(results_summary)}")
    print(f"Statistically significant improvements (p<0.05): {significant_improvements} / {len(results_summary)}")
    print()
    print("Significance levels: * p<0.05, ** p<0.01, *** p<0.001, ns=not significant")
    print()

    # Save detailed results
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv("model_comparison_summary.csv", index=False)

    old_df.to_csv("old_models_metrics.csv", index=False)
    new_df.to_csv("new_models_metrics.csv", index=False)

    print("Saved results:")
    print("  - model_comparison_summary.csv (aggregate statistics)")
    print("  - old_models_metrics.csv (individual old model metrics)")
    print("  - new_models_metrics.csv (individual new model metrics)")

if __name__ == "__main__":
    main()
