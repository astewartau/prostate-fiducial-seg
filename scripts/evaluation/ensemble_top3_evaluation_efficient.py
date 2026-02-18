#!/usr/bin/env python3
"""
Memory-efficient ensemble evaluation with top-3 confidence selection.

This version processes models sequentially to avoid loading all models into memory at once.
"""

import glob
import os
import re
import sys
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torchio as tio
import scipy.ndimage
from torch.utils.data import DataLoader


# --- Model definition (must match train_one_model.py) ---
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, channels=(16, 32, 64, 128, 256)):
        super(UNet3D, self).__init__()
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        prev_channels = in_channels
        for ch in channels:
            self.encoders.append(self.conv_block(prev_channels, ch))
            prev_channels = ch
        self.bottleneck = self.conv_block(prev_channels, prev_channels * 2)
        rev_channels = list(reversed(channels))
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        cur_channels = prev_channels * 2
        for ch in rev_channels:
            self.upconvs.append(nn.ConvTranspose3d(cur_channels, ch, kernel_size=2, stride=2))
            self.decoders.append(self.conv_block(ch * 2, ch))
            cur_channels = ch
        self.final_conv = nn.Conv3d(cur_channels, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc_feats = []
        for encoder in self.encoders:
            x = encoder(x)
            enc_feats.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upconv, decoder, enc in zip(self.upconvs, self.decoders, reversed(enc_feats)):
            x = upconv(x)
            if x.shape != enc.shape:
                diffZ = enc.size()[2] - x.size()[2]
                diffY = enc.size()[3] - x.size()[3]
                diffX = enc.size()[4] - x.size()[4]
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2,
                              diffZ // 2, diffZ - diffZ // 2])
            x = torch.cat([enc, x], dim=1)
            x = decoder(x)
        return self.final_conv(x)


# --- Custom transforms ---
class PadToCompatibleSize(tio.Transform):
    def __init__(self, min_factor=32):
        super().__init__()
        self.min_factor = min_factor

    def apply_transform(self, subject):
        for key in subject.keys():
            if isinstance(subject[key], (tio.ScalarImage, tio.LabelMap)):
                vol = subject[key].data
                target_shape = []
                for dim in vol.shape:
                    compatible_dim = ((dim + self.min_factor - 1) // self.min_factor) * self.min_factor
                    target_shape.append(compatible_dim)
                target_shape = tuple(target_shape)
                if vol.shape != target_shape:
                    padded_vol = F.pad(vol, [0, target_shape[2] - vol.shape[2],
                                             0, target_shape[1] - vol.shape[1],
                                             0, target_shape[0] - vol.shape[0]])
                    if isinstance(subject[key], tio.ScalarImage):
                        subject[key] = tio.ScalarImage(tensor=padded_vol, affine=subject[key].affine)
                    else:
                        subject[key] = tio.LabelMap(tensor=padded_vol, affine=subject[key].affine)
        return subject


class MergeInputChannels(tio.Transform):
    def __init__(self, infile_cols):
        super().__init__()
        self.infile_cols = infile_cols

    def apply_transform(self, subject):
        input_tensors = []
        for col in self.infile_cols:
            if col not in subject:
                continue
            tensor = subject[col].data
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            input_tensors.append(tensor)
        if not input_tensors:
            return subject
        merged = torch.cat(input_tensors, dim=0)
        subject['image'] = tio.ScalarImage(tensor=merged, affine=subject[self.infile_cols[0]].affine)
        subject['mask'] = subject['seg']
        return subject


# --- Data loading ---
def build_dataset(data_dir="data/train"):
    """Build dataset exactly as train_one_model.py does."""
    bids_dirs = [data_dir]
    paths_list = []

    for bids_dir in bids_dirs:
        subject_dirs = sorted(glob.glob(os.path.join(bids_dir, "z*")))

        for subject_dir in subject_dirs:
            subject_id = os.path.basename(subject_dir)

            roi_dir = os.path.join(subject_dir, "roi_niftis_mri_space")
            if not os.path.exists(roi_dir):
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

    for index, row in df.iterrows():
        if row.get('Prostate') is None and row.get('roi_CTV_Low_MR') is not None:
            df.at[index, 'Prostate'] = row['roi_CTV_Low_MR']

    for index, row in df.iterrows():
        if row.get('Prostate') is None and row.get('roi_CTVp_MR') is not None:
            df.at[index, 'Prostate'] = row['roi_CTVp_MR']

    columns_to_keep = ['subject_id', 'MRI', 'MRI_homogeneity-corrected', 'CT', 'seeds', 'Prostate']
    df = df[columns_to_keep]

    infile_cols = ['MRI_homogeneity-corrected']
    seg_col = 'seeds'

    for col in infile_cols + [seg_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame.")
        df = df[df[col].notna()].reset_index(drop=True)

    return df, infile_cols


# --- Metrics computation ---
def process_volume(pred_vol, targ_vol, structure):
    _, pred_nlabels = scipy.ndimage.label(pred_vol, structure=structure)
    _, targ_nlabels = scipy.ndimage.label(targ_vol, structure=structure)

    overlap = np.logical_and(pred_vol == targ_vol, pred_vol == 1)
    _, n_overlaps = scipy.ndimage.label(overlap, structure=structure)

    return pred_nlabels, targ_nlabels, n_overlaps


def compute_metrics(pred, targ):
    """Compute marker-based detection metrics."""
    structure = np.ones((3, 3, 3), dtype=bool)
    total_pred_marker_count = 0
    total_targ_marker_count = 0
    total_overlap_count = 0

    pred_marker = (pred == 1).astype(np.int32)
    targ_marker = (targ == 1).astype(np.int32)

    pred_marker = scipy.ndimage.binary_dilation(pred_marker)
    targ_marker = scipy.ndimage.binary_dilation(targ_marker)

    for i in range(pred_marker.shape[0]):
        p_vol = pred_marker[i]
        t_vol = targ_marker[i]
        p_n, t_n, n_overlap = process_volume(p_vol, t_vol, structure)
        total_pred_marker_count += p_n
        total_targ_marker_count += t_n
        total_overlap_count += n_overlap

    false_negative = total_targ_marker_count - total_overlap_count
    false_positive = total_pred_marker_count - total_overlap_count

    return {
         "actual_markers": total_targ_marker_count,
         "true_positive": total_overlap_count,
         "false_negative": false_negative,
         "false_positive": false_positive
    }


def select_top3_markers(prob_map, threshold=0.3):
    """
    Select top 3 markers from probability map based on confidence.

    Args:
        prob_map: numpy array of shape [D, H, W] with marker probabilities
        threshold: confidence threshold for initial detection

    Returns:
        binary mask with top 3 markers selected
    """
    # Threshold to get candidate regions
    candidate_mask = (prob_map > threshold).astype(np.uint8)

    # Find connected components
    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, num_components = scipy.ndimage.label(candidate_mask, structure=structure)

    if num_components == 0:
        # No markers detected, return empty
        return np.zeros_like(prob_map, dtype=np.int32)

    # Calculate mean confidence for each component
    component_confidences = []
    for comp_id in range(1, num_components + 1):
        comp_mask = (labeled == comp_id)
        mean_conf = prob_map[comp_mask].mean()
        component_confidences.append((comp_id, mean_conf))

    # Sort by confidence and take top 3
    component_confidences.sort(key=lambda x: x[1], reverse=True)
    top3_components = [comp_id for comp_id, _ in component_confidences[:3]]

    # Create final prediction with top 3 components
    final_mask = np.zeros_like(labeled, dtype=np.int32)
    for comp_id in top3_components:
        final_mask[labeled == comp_id] = 1

    return final_mask


def find_completed_models(directory):
    """Find models that have both best and final .pth files (indicating completed training)."""
    completed = {}
    best_files = glob.glob(os.path.join(directory, "*-best.pth"))

    for best_file in best_files:
        match = re.search(r'-(\d+)-best\.pth$', best_file)
        if match:
            fold_id = int(match.group(1))

            # Check if corresponding final checkpoint exists
            final_file = best_file.replace('-best.pth', '-final.pth')
            if os.path.exists(final_file):
                completed[fold_id] = best_file
            else:
                print(f"  Skipping fold {fold_id}: no final checkpoint found (still training)")

    return completed


def main():
    parser = argparse.ArgumentParser(description='Memory-efficient ensemble evaluation')
    parser.add_argument('--model_dir', type=str, default='models/loocv',
                        help='Directory containing model checkpoints')
    parser.add_argument('--data-dir', type=str, default='data/train',
                        help='Directory containing training data')
    parser.add_argument('--num_models', type=int, default=3,
                        help='Number of models to use in ensemble')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Confidence threshold for initial marker detection')
    parser.add_argument('--output', type=str, default='results/ensemble_top3_results.csv',
                        help='Output CSV file for results')
    args = parser.parse_args()

    print("=" * 80)
    print("MEMORY-EFFICIENT ENSEMBLE EVALUATION WITH TOP-3 MARKER SELECTION")
    print("=" * 80)
    print(f"Model directory: {args.model_dir}")
    print(f"Number of models in ensemble: {args.num_models}")
    print(f"Confidence threshold: {args.threshold}")
    print()

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    print()

    # Build dataset
    print("Loading dataset...")
    df, infile_cols = build_dataset(args.data_dir)
    print(f"Total valid subjects: {len(df)}")
    print()

    # Build evaluation transforms
    eval_transforms = tio.Compose([
        PadToCompatibleSize(min_factor=32),
        tio.ZNormalization(),
        MergeInputChannels(infile_cols)
    ])

    # Build TorchIO subjects
    subjects = []
    for _, row in df.iterrows():
        subject_dict = {}
        for col in infile_cols:
            subject_dict[col] = tio.ScalarImage(row[col])
        subject_dict['seg'] = tio.LabelMap(row['seeds'])
        subject_dict['subject_id'] = row['subject_id']
        subject = tio.Subject(**subject_dict)
        subjects.append(subject)

    subjects_dataset = tio.SubjectsDataset(subjects, transform=eval_transforms)

    class TorchIODatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, subjects_dataset):
            self.subjects_dataset = subjects_dataset

        def __len__(self):
            return len(self.subjects_dataset)

        def __getitem__(self, index):
            subject = self.subjects_dataset[index]
            image = subject['image'].data
            mask = subject['mask'].data.squeeze(0).long()
            return image, mask

    eval_dataset = TorchIODatasetWrapper(subjects_dataset)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Find completed models
    print("Finding completed models...")
    completed_models = find_completed_models(args.model_dir)
    print(f"Found {len(completed_models)} completed models")

    if len(completed_models) < args.num_models:
        print(f"ERROR: Need at least {args.num_models} models, but only found {len(completed_models)}")
        return

    # Select first N models for ensemble
    selected_folds = sorted(completed_models.keys())[:args.num_models]
    print(f"Using models from folds: {selected_folds}")
    print()

    # Cache all images and masks to avoid reloading
    print("Caching dataset...")
    all_images = []
    all_masks = []
    for images, masks in eval_loader:
        all_images.append(images)
        all_masks.append(masks)
    print(f"Cached {len(all_images)} samples")
    print()

    # Collect probabilities from each model sequentially
    print("Running models sequentially to collect probabilities...")
    all_model_probs = []
    individual_preds = {}

    for fold_id in selected_folds:
        best_file = completed_models[fold_id]
        print(f"Processing fold {fold_id}: {best_file}")

        # Load model
        model = UNet3D(in_channels=len(infile_cols), out_channels=3).to(device)
        model.load_state_dict(torch.load(best_file, map_location=device))
        model.eval()

        # Collect probabilities
        fold_probs = []
        fold_preds = []

        with torch.no_grad():
            for idx, (images, masks) in enumerate(zip(all_images, all_masks)):
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                fold_probs.append(probs.cpu().numpy())

                # Also get standard argmax prediction
                preds = torch.argmax(outputs, dim=1)
                fold_preds.append(preds.cpu().numpy())

                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{len(all_images)} samples")

        all_model_probs.append(fold_probs)
        individual_preds[fold_id] = np.concatenate(fold_preds, axis=0)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()
        print()

    # Average probabilities across models and apply top-3 selection
    print("Computing ensemble predictions with top-3 selection...")
    all_preds_ensemble = []

    for idx in range(len(all_images)):
        # Average probabilities from all models for this sample
        sample_probs = [model_probs[idx] for model_probs in all_model_probs]
        avg_prob = np.mean(sample_probs, axis=0)  # Shape: [1, 3, D, H, W]

        # Get marker class probability
        marker_prob = avg_prob[0, 1, :, :, :]  # Shape: [D, H, W]

        # Select top 3 markers
        top3_mask = select_top3_markers(marker_prob, threshold=args.threshold)
        all_preds_ensemble.append(top3_mask[np.newaxis, :, :, :])

    all_preds_ensemble = np.concatenate(all_preds_ensemble, axis=0)
    all_targs = np.concatenate([m.numpy() for m in all_masks], axis=0)
    print()

    # Compute metrics
    print("Computing metrics...")
    print()

    # Ensemble metrics
    print("=" * 80)
    print(f"ENSEMBLE (Top-3 Selection from {args.num_models} models)")
    print("=" * 80)
    ensemble_metrics = compute_metrics(all_preds_ensemble, all_targs)
    ensemble_sensitivity = ensemble_metrics['true_positive'] / ensemble_metrics['actual_markers'] if ensemble_metrics['actual_markers'] > 0 else 0
    ensemble_precision = ensemble_metrics['true_positive'] / (ensemble_metrics['true_positive'] + ensemble_metrics['false_positive']) if (ensemble_metrics['true_positive'] + ensemble_metrics['false_positive']) > 0 else 0

    print(f"Actual markers: {ensemble_metrics['actual_markers']}")
    print(f"True positives: {ensemble_metrics['true_positive']}")
    print(f"False negatives: {ensemble_metrics['false_negative']}")
    print(f"False positives: {ensemble_metrics['false_positive']}")
    print(f"Sensitivity: {ensemble_sensitivity:.4f}")
    print(f"Precision: {ensemble_precision:.4f}")
    print()

    # Individual model metrics
    results = []
    results.append({
        'model': 'ensemble_top3',
        'fold_ids': str(selected_folds),
        'actual_markers': ensemble_metrics['actual_markers'],
        'true_positive': ensemble_metrics['true_positive'],
        'false_negative': ensemble_metrics['false_negative'],
        'false_positive': ensemble_metrics['false_positive'],
        'sensitivity': ensemble_sensitivity,
        'precision': ensemble_precision
    })

    for fold_id in selected_folds:
        print("=" * 80)
        print(f"INDIVIDUAL MODEL: Fold {fold_id}")
        print("=" * 80)

        metrics = compute_metrics(individual_preds[fold_id], all_targs)
        sensitivity = metrics['true_positive'] / metrics['actual_markers'] if metrics['actual_markers'] > 0 else 0
        precision = metrics['true_positive'] / (metrics['true_positive'] + metrics['false_positive']) if (metrics['true_positive'] + metrics['false_positive']) > 0 else 0

        print(f"Actual markers: {metrics['actual_markers']}")
        print(f"True positives: {metrics['true_positive']}")
        print(f"False negatives: {metrics['false_negative']}")
        print(f"False positives: {metrics['false_positive']}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Precision: {precision:.4f}")
        print()

        results.append({
            'model': f'fold_{fold_id}',
            'fold_ids': str([fold_id]),
            'actual_markers': metrics['actual_markers'],
            'true_positive': metrics['true_positive'],
            'false_negative': metrics['false_negative'],
            'false_positive': metrics['false_positive'],
            'sensitivity': sensitivity,
            'precision': precision
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"Saved results to: {args.output}")
    print()

    # Summary comparison
    print("=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    individual_sensitivities = [r['sensitivity'] for r in results[1:]]
    individual_precisions = [r['precision'] for r in results[1:]]

    print(f"Ensemble Sensitivity: {ensemble_sensitivity:.4f}")
    print(f"Individual Models (mean ± std): {np.mean(individual_sensitivities):.4f} ± {np.std(individual_sensitivities):.4f}")
    print()
    print(f"Ensemble Precision: {ensemble_precision:.4f}")
    print(f"Individual Models (mean ± std): {np.mean(individual_precisions):.4f} ± {np.std(individual_precisions):.4f}")
    print()

    if ensemble_sensitivity > np.mean(individual_sensitivities):
        print("✓ Ensemble improves sensitivity!")
    else:
        print("✗ Ensemble does not improve sensitivity")

    if ensemble_precision > np.mean(individual_precisions):
        print("✓ Ensemble improves precision!")
    else:
        print("✗ Ensemble does not improve precision")


if __name__ == "__main__":
    main()
