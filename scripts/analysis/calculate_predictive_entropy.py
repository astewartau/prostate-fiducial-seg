#!/usr/bin/env python3
"""
Calculate predictive entropy for validation subjects.

Predictive entropy measures how uncertain the model is about its predictions.
High entropy = uncertain (probabilities spread across classes)
Low entropy = confident (one class dominates)

For a 3-class segmentation with softmax outputs [p1, p2, p3]:
H = -Σ p_i * log(p_i)

This is the gold standard for uncertainty quantification in deep learning.
"""

import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from scipy.stats import entropy
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt

# Import model architecture from training script
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

def build_subject_dataframe(bids_dirs):
    """Build dataframe of subjects."""
    paths_list = []

    for bids_dir in bids_dirs:
        subject_dirs = sorted(glob.glob(os.path.join(bids_dir, "z*")))

        for subject_dir in subject_dirs:
            subject_id = os.path.basename(subject_dir)

            # Skip excluded subjects
            if os.path.exists(os.path.join(subject_dir, "EXCLUDE_INVALID_SEGMENTATION")):
                print(f"Skipping {subject_id}: marked as EXCLUDE_INVALID_SEGMENTATION")
                continue

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

    # Handle Prostate segmentation fallbacks
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
        if col in df.columns:
            df = df[df[col].notna()].reset_index(drop=True)

    return df

def calculate_voxelwise_entropy(model, subject, device, infile_cols):
    """
    Calculate voxel-wise predictive entropy.
    Returns entropy map and statistics.
    """
    model.eval()

    # Prepare transforms
    transforms = tio.Compose([
        PadToCompatibleSize(min_factor=32),
        tio.ZNormalization(),
        MergeInputChannels(infile_cols)
    ])

    # Apply transforms
    subject_transformed = transforms(subject)

    # Get image
    image = subject_transformed['image'].data.unsqueeze(0).to(device)
    mask = subject_transformed['mask'].data.squeeze(0).numpy()

    # Run inference
    with torch.no_grad():
        logits = model(image)
        # Get probabilities via softmax
        probs = F.softmax(logits, dim=1)  # [1, 3, D, H, W]

    # Move to CPU and convert to numpy
    probs_np = probs.squeeze(0).cpu().numpy()  # [3, D, H, W]

    # Calculate entropy per voxel
    # entropy = -Σ p_i * log(p_i)
    # scipy.stats.entropy does this along axis 0
    entropy_map = entropy(probs_np, axis=0)  # [D, H, W]

    # Get prediction
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    # Calculate statistics
    # Only consider voxels within the prostate region for focused analysis
    prostate_mask = mask == 2
    marker_mask = mask == 1

    stats = {
        'mean_entropy': np.mean(entropy_map),
        'max_entropy': np.max(entropy_map),
        'std_entropy': np.std(entropy_map),
        'median_entropy': np.median(entropy_map),
        # Focused statistics within prostate
        'mean_entropy_prostate': np.mean(entropy_map[prostate_mask]) if prostate_mask.any() else 0,
        'max_entropy_prostate': np.max(entropy_map[prostate_mask]) if prostate_mask.any() else 0,
        # Focused on marker locations
        'mean_entropy_markers': np.mean(entropy_map[marker_mask]) if marker_mask.any() else 0,
        'max_entropy_markers': np.max(entropy_map[marker_mask]) if marker_mask.any() else 0,
        # High uncertainty voxels
        'frac_high_uncertainty': np.mean(entropy_map > 0.5),  # Fraction with entropy > 0.5
        'frac_very_high_uncertainty': np.mean(entropy_map > 0.8),  # Fraction with entropy > 0.8
    }

    return entropy_map, pred, probs_np, mask, stats

def find_model_file(models_dir, fold_id):
    """Find the best model file for a given fold."""
    # Look for pattern: T1-*-{fold_id}-best.pth
    pattern = os.path.join(models_dir, f"T1-*-{fold_id}-best.pth")
    matches = glob.glob(pattern)

    if not matches:
        return None

    # Check if final model exists (means training completed)
    best_file = matches[0]
    final_file = best_file.replace('-best.pth', '-final.pth')

    if not os.path.exists(final_file):
        return None  # Training not completed

    return best_file

def main():
    # Configuration
    bids_dirs = ["/scratch/user/uqaste15/data/2024-prostate/bids-2025-2"]
    models_dir = "/scratch/user/uqaste15/data/2024-prostate"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    infile_cols = ['MRI_homogeneity-corrected']
    num_classes = 3
    random_state = 42

    print("=" * 80)
    print("CALCULATING PREDICTIVE ENTROPY FOR VALIDATION SUBJECTS")
    print("=" * 80)
    print()
    print(f"Device: {device}")
    print()

    # Build subject dataframe
    print("Building subject dataframe...")
    df = build_subject_dataframe(bids_dirs)
    print(f"Total subjects: {len(df)}")
    print()

    # Create fold-to-subject mapping
    k_folds = len(df)
    kf = KFold(n_splits=k_folds, random_state=random_state, shuffle=True)

    # Build subjects list
    subjects = []
    for _, row in df.iterrows():
        subject_dict = {}
        for col in infile_cols:
            subject_dict[col] = tio.ScalarImage(row[col])
        subject_dict['seg'] = tio.LabelMap(row['seeds'])
        subject = tio.Subject(**subject_dict)
        subjects.append(subject)

    # Calculate entropy for each fold
    entropy_results = []

    for fold_id, (train_index, valid_index) in enumerate(kf.split(df)):
        valid_subject_id = df.iloc[valid_index[0]]['subject_id']

        # Find model file
        model_file = find_model_file(models_dir, fold_id)

        if not model_file:
            print(f"Fold {fold_id} ({valid_subject_id}): No completed model found")
            continue

        print(f"Fold {fold_id} ({valid_subject_id}): Loading model...")

        # Load model
        model = UNet3D(in_channels=len(infile_cols), out_channels=num_classes).to(device)
        try:
            model.load_state_dict(torch.load(model_file, map_location=device))
        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")
            continue

        # Get validation subject
        valid_subject = subjects[valid_index[0]]

        # Calculate entropy
        print(f"  Calculating entropy...")
        try:
            entropy_map, pred, probs, mask, stats = calculate_voxelwise_entropy(
                model, valid_subject, device, infile_cols
            )

            stats['fold_id'] = fold_id
            stats['subject_id'] = valid_subject_id
            stats['model_file'] = os.path.basename(model_file)

            entropy_results.append(stats)

            print(f"  ✓ Mean entropy: {stats['mean_entropy']:.4f}")
            print(f"    Max entropy: {stats['max_entropy']:.4f}")
            print(f"    Mean entropy (prostate): {stats['mean_entropy_prostate']:.4f}")
            print(f"    High uncertainty: {stats['frac_high_uncertainty']*100:.1f}%")
            print()

        except Exception as e:
            print(f"  ✗ Failed to calculate entropy: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not entropy_results:
        print("No entropy results computed!")
        return

    # Create results dataframe
    results_df = pd.DataFrame(entropy_results)
    results_df = results_df.sort_values('mean_entropy', ascending=False)

    print("=" * 80)
    print("ENTROPY ANALYSIS RESULTS")
    print("=" * 80)
    print()

    # Display top uncertain subjects
    print("Top 10 Most Uncertain Subjects (by mean entropy):")
    print("-" * 80)
    print(f"{'Fold':<6} {'Subject':<15} {'Mean H':<10} {'Max H':<10} {'Mean H (Prost)':<15} {'High Unc %':<12}")
    print("-" * 80)

    for _, row in results_df.head(10).iterrows():
        print(f"{row['fold_id']:<6.0f} {row['subject_id']:<15} "
              f"{row['mean_entropy']:<10.4f} {row['max_entropy']:<10.4f} "
              f"{row['mean_entropy_prostate']:<15.4f} "
              f"{row['frac_high_uncertainty']*100:<12.1f}")

    print()
    print("=" * 80)

    # Save results
    output_csv = "predictive_entropy_analysis.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"Detailed results saved to: {output_csv}")

    # Create visualization
    if len(results_df) >= 3:
        print("\nGenerating visualization...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Distribution of mean entropy
        axes[0, 0].hist(results_df['mean_entropy'], bins=20, edgecolor='black')
        axes[0, 0].set_xlabel('Mean Predictive Entropy')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Mean Entropy')
        axes[0, 0].axvline(results_df['mean_entropy'].mean(),
                          color='red', linestyle='--', label='Mean')
        axes[0, 0].legend()

        # 2. Mean vs Max entropy
        axes[0, 1].scatter(results_df['mean_entropy'], results_df['max_entropy'],
                          alpha=0.6, s=100)
        axes[0, 1].set_xlabel('Mean Entropy')
        axes[0, 1].set_ylabel('Max Entropy')
        axes[0, 1].set_title('Mean vs Max Entropy')
        axes[0, 1].grid(True, alpha=0.3)

        # Annotate top 5
        for _, row in results_df.head(5).iterrows():
            axes[0, 1].annotate(row['subject_id'],
                              (row['mean_entropy'], row['max_entropy']),
                              fontsize=8, alpha=0.7)

        # 3. Top subjects by mean entropy
        top_n = min(15, len(results_df))
        top_subjects = results_df.head(top_n)
        axes[1, 0].barh(range(top_n), top_subjects['mean_entropy'])
        axes[1, 0].set_yticks(range(top_n))
        axes[1, 0].set_yticklabels(top_subjects['subject_id'], fontsize=8)
        axes[1, 0].set_xlabel('Mean Entropy')
        axes[1, 0].set_title(f'Top {top_n} Most Uncertain Subjects')
        axes[1, 0].invert_yaxis()

        # 4. Fraction of high uncertainty voxels
        axes[1, 1].scatter(results_df['mean_entropy'],
                          results_df['frac_high_uncertainty'] * 100,
                          alpha=0.6, s=100)
        axes[1, 1].set_xlabel('Mean Entropy')
        axes[1, 1].set_ylabel('% High Uncertainty Voxels (H>0.5)')
        axes[1, 1].set_title('Mean Entropy vs High Uncertainty Coverage')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_plot = "predictive_entropy_analysis.png"
        plt.savefig(output_plot, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_plot}")
        plt.close()

    print()
    print("=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    print()
    print("Predictive entropy measures model uncertainty:")
    print("  - Low entropy (~0.0): Model is very confident")
    print("  - High entropy (~1.1 for 3 classes): Model is very uncertain")
    print()
    print("Subjects with high mean entropy are challenging for the model.")
    print("These may represent:")
    print("  - Ambiguous cases (difficult even for humans)")
    print("  - Out-of-distribution examples")
    print("  - Data quality issues")
    print("  - Registration/alignment problems")
    print()

if __name__ == "__main__":
    main()
