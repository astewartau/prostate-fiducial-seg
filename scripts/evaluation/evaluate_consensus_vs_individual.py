#!/usr/bin/env python3
"""
Evaluate and compare consensus vs individual model performance.

For each test subject:
1. Load ground truth segmentation
2. Load consensus segmentation
3. Load all individual model segmentations
4. Compute metrics (TP, FP, FN) using connected component analysis
5. Compare consensus vs individual models
"""

import os
import glob
import numpy as np
import nibabel as nib
import scipy.ndimage
import pandas as pd
from pathlib import Path


def load_nifti(path):
    """Load NIfTI file and return data array"""
    return nib.load(path).get_fdata()


def process_volume(pred_vol, targ_vol, structure):
    """Count markers using connected component labeling"""
    _, pred_nlabels = scipy.ndimage.label(pred_vol, structure=structure)
    _, targ_nlabels = scipy.ndimage.label(targ_vol, structure=structure)

    overlap = np.logical_and(pred_vol == targ_vol, pred_vol == 1)
    _, n_overlaps = scipy.ndimage.label(overlap, structure=structure)

    return pred_nlabels, targ_nlabels, n_overlaps


def compute_metrics(pred, targ):
    """Compute TP, FP, FN for marker detection"""
    structure = np.ones((3, 3, 3), dtype=bool)

    # Extract only seed markers (class 1), not prostate (class 2)
    pred_marker = (pred == 1).astype(np.int32)
    targ_marker = (targ == 1).astype(np.int32)

    # Apply dilation (matching training code)
    pred_marker = scipy.ndimage.binary_dilation(pred_marker)
    targ_marker = scipy.ndimage.binary_dilation(targ_marker)

    p_n, t_n, n_overlap = process_volume(pred_marker, targ_marker, structure)

    false_negative = t_n - n_overlap
    false_positive = p_n - n_overlap

    return {
        "actual_markers": t_n,
        "true_positive": n_overlap,
        "false_negative": false_negative,
        "false_positive": false_positive
    }


def evaluate_subject(subject_id, gt_path, consensus_seg_path, individual_seg_dir):
    """Evaluate consensus and individual models for a single subject"""
    print(f"\n{'='*80}")
    print(f"Evaluating subject: {subject_id}")
    print(f"{'='*80}")

    # Load ground truth
    print(f"Loading ground truth: {gt_path}")
    gt = load_nifti(gt_path)

    # Count ground truth seed markers (class 1 only)
    structure = np.ones((3, 3, 3), dtype=bool)
    gt_seeds = (gt == 1).astype(np.int32)
    gt_dilated = scipy.ndimage.binary_dilation(gt_seeds)
    _, gt_marker_count = scipy.ndimage.label(gt_dilated, structure=structure)
    print(f"Ground truth seed markers: {gt_marker_count}")

    results = []

    # Evaluate consensus
    if os.path.exists(consensus_seg_path):
        print(f"\nEvaluating consensus segmentation...")
        consensus_seg = load_nifti(consensus_seg_path)
        metrics = compute_metrics(consensus_seg, gt)

        result = {
            'subject_id': subject_id,
            'method': 'consensus',
            'model_name': 'consensus_top3',
            **metrics
        }
        results.append(result)

        print(f"  TP: {metrics['true_positive']}, "
              f"FP: {metrics['false_positive']}, "
              f"FN: {metrics['false_negative']}")
    else:
        print(f"WARNING: Consensus segmentation not found at {consensus_seg_path}")

    # Evaluate individual models
    if os.path.exists(individual_seg_dir):
        print(f"\nEvaluating individual models...")
        model_dirs = sorted(glob.glob(os.path.join(individual_seg_dir, "*")))

        for model_dir in model_dirs:
            if not os.path.isdir(model_dir):
                continue

            model_name = os.path.basename(model_dir)
            seg_path = os.path.join(model_dir, "pred_seeds.nii.gz")

            if not os.path.exists(seg_path):
                print(f"  WARNING: Segmentation not found for {model_name}")
                continue

            seg = load_nifti(seg_path)
            metrics = compute_metrics(seg, gt)

            result = {
                'subject_id': subject_id,
                'method': 'individual',
                'model_name': model_name,
                **metrics
            }
            results.append(result)
    else:
        print(f"WARNING: Individual segmentation directory not found at {individual_seg_dir}")

    return results


def main():
    print("="*80)
    print("CONSENSUS vs INDIVIDUAL MODEL EVALUATION")
    print("="*80)

    # Test subjects info
    test_subjects = [
        {
            'subject_id': 'z0197000',
            'gt_path': 'bids-2025-2/z0197000/roi_niftis_mri_space/z0197000_seeds.nii.gz',
            'consensus_seg': 'consensus_vs_individual_results/subject_z0197000_consensus/consensus_top3_segmentation.nii.gz',
            'individual_dir': 'consensus_vs_individual_results/subject_z0197000_individual/'
        },
        {
            'subject_id': 'z1451978',
            'gt_path': 'bids-2025-2/z1451978/roi_niftis_mri_space/z1451978_seeds.nii.gz',
            'consensus_seg': 'consensus_vs_individual_results/subject_z1451978_consensus/consensus_top3_segmentation.nii.gz',
            'individual_dir': 'consensus_vs_individual_results/subject_z1451978_individual/'
        },
        {
            'subject_id': 'z2877562',
            'gt_path': 'bids-2025-2/z2877562/roi_niftis_mri_space/z2877562_seeds.nii.gz',
            'consensus_seg': 'consensus_vs_individual_results/subject_z2877562_consensus/consensus_top3_segmentation.nii.gz',
            'individual_dir': 'consensus_vs_individual_results/subject_z2877562_individual/'
        }
    ]

    all_results = []

    for subject_info in test_subjects:
        results = evaluate_subject(
            subject_info['subject_id'],
            subject_info['gt_path'],
            subject_info['consensus_seg'],
            subject_info['individual_dir']
        )
        all_results.extend(results)

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Calculate derived metrics
    df['sensitivity'] = df['true_positive'] / df['actual_markers']
    df['precision'] = df['true_positive'] / (df['true_positive'] + df['false_positive'])
    df['precision'] = df['precision'].fillna(0)  # Handle division by zero

    # Save detailed results
    df.to_csv('consensus_vs_individual_results/detailed_results.csv', index=False)
    print(f"\n\nSaved detailed results to: consensus_vs_individual_results/detailed_results.csv")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for subject_id in df['subject_id'].unique():
        subject_df = df[df['subject_id'] == subject_id]

        print(f"\n{'='*80}")
        print(f"Subject: {subject_id}")
        print(f"{'='*80}")

        # Consensus results
        consensus = subject_df[subject_df['method'] == 'consensus']
        if len(consensus) > 0:
            c = consensus.iloc[0]
            print(f"\nConsensus (top-3 selection):")
            print(f"  TP: {c['true_positive']}, FP: {c['false_positive']}, FN: {c['false_negative']}")
            print(f"  Sensitivity: {c['sensitivity']:.4f}, Precision: {c['precision']:.4f}")

        # Individual models statistics
        individual = subject_df[subject_df['method'] == 'individual']
        if len(individual) > 0:
            print(f"\nIndividual models (n={len(individual)}):")
            print(f"  TP: {individual['true_positive'].mean():.2f} ± {individual['true_positive'].std():.2f}")
            print(f"  FP: {individual['false_positive'].mean():.2f} ± {individual['false_positive'].std():.2f}")
            print(f"  FN: {individual['false_negative'].mean():.2f} ± {individual['false_negative'].std():.2f}")
            print(f"  Sensitivity: {individual['sensitivity'].mean():.4f} ± {individual['sensitivity'].std():.4f}")
            print(f"  Precision: {individual['precision'].mean():.4f} ± {individual['precision'].std():.4f}")

            # Best individual model
            best_idx = individual['true_positive'].idxmax()
            best = individual.loc[best_idx]
            print(f"\n  Best individual model: {best['model_name']}")
            print(f"    TP: {best['true_positive']}, FP: {best['false_positive']}, FN: {best['false_negative']}")
            print(f"    Sensitivity: {best['sensitivity']:.4f}, Precision: {best['precision']:.4f}")

    # Overall comparison
    print(f"\n{'='*80}")
    print("OVERALL COMPARISON")
    print(f"{'='*80}")

    consensus_all = df[df['method'] == 'consensus']
    individual_all = df[df['method'] == 'individual']

    print(f"\nConsensus (n={len(consensus_all)} subjects):")
    print(f"  Sensitivity: {consensus_all['sensitivity'].mean():.4f} ± {consensus_all['sensitivity'].std():.4f}")
    print(f"  Precision: {consensus_all['precision'].mean():.4f} ± {consensus_all['precision'].std():.4f}")
    print(f"  FP per subject: {consensus_all['false_positive'].mean():.2f} ± {consensus_all['false_positive'].std():.2f}")

    print(f"\nIndividual models (n={len(individual_all)} predictions):")
    print(f"  Sensitivity: {individual_all['sensitivity'].mean():.4f} ± {individual_all['sensitivity'].std():.4f}")
    print(f"  Precision: {individual_all['precision'].mean():.4f} ± {individual_all['precision'].std():.4f}")
    print(f"  FP per prediction: {individual_all['false_positive'].mean():.2f} ± {individual_all['false_positive'].std():.2f}")

    # Summary by subject
    summary_data = []
    for subject_id in df['subject_id'].unique():
        subject_df = df[df['subject_id'] == subject_id]

        consensus = subject_df[subject_df['method'] == 'consensus']
        individual = subject_df[subject_df['method'] == 'individual']

        if len(consensus) > 0 and len(individual) > 0:
            c = consensus.iloc[0]
            summary_data.append({
                'subject_id': subject_id,
                'consensus_sensitivity': c['sensitivity'],
                'consensus_precision': c['precision'],
                'consensus_fp': c['false_positive'],
                'individual_sensitivity_mean': individual['sensitivity'].mean(),
                'individual_precision_mean': individual['precision'].mean(),
                'individual_fp_mean': individual['false_positive'].mean(),
                'individual_sensitivity_std': individual['sensitivity'].std(),
                'individual_precision_std': individual['precision'].std(),
                'individual_fp_std': individual['false_positive'].std(),
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('consensus_vs_individual_results/summary_by_subject.csv', index=False)
    print(f"\n\nSaved summary by subject to: consensus_vs_individual_results/summary_by_subject.csv")
    print()


if __name__ == "__main__":
    main()
