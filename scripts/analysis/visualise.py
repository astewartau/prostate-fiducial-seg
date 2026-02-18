#!/usr/bin/env python
import os
import argparse
import glob
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchio as tio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import UNet3D

# --- Function to find nearest compatible dimensions for UNet ---
def find_nearest_compatible_size(input_shape, min_factor=32):
    """Find nearest larger size divisible by min_factor (pad-only approach)"""
    compatible_shape = []
    for dim in input_shape:
        compatible_dim = ((dim + min_factor - 1) // min_factor) * min_factor
        compatible_shape.append(compatible_dim)
    return tuple(compatible_shape)

# --- Crop/pad utility matching training ---
def pad_or_crop_numpy(vol, target):
    # Crop
    slices = []
    for i in range(3):
        diff = vol.shape[i] - target[i]
        if diff > 0:
            start = diff // 2
            slices.append(slice(start, start + target[i]))
        else:
            slices.append(slice(0, vol.shape[i]))
    vol_c = vol[slices[0], slices[1], slices[2]]
    # Pad
    pad_width = []
    for i in range(3):
        diff = target[i] - vol_c.shape[i]
        if diff > 0:
            before = diff // 2
            after = diff - before
            pad_width.append((before, after))
        else:
            pad_width.append((0, 0))
    return np.pad(vol_c, pad_width, mode='constant', constant_values=0)

# --- Data loading ---
from utils.data_loading import build_subject_dataframe

# --- Run inference on a single image ---
def run_inference(model, input_path, device):
    """Run inference on a single MRI image and return prostate segmentation"""
    img = tio.ScalarImage(input_path)
    orig_shape = img.data.numpy()[0].shape
    compatible_shape = find_nearest_compatible_size(orig_shape)
    sample = tio.ZNormalization()(tio.CropOrPad(compatible_shape)(img))
    x = sample.data.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        seg = out.argmax(dim=1).cpu().numpy()[0]

    # Restore original dimensions
    full_seg = pad_or_crop_numpy(seg, orig_shape)
    
    # Extract prostate segmentation (class 2)
    prostate_seg = (full_seg == 2).astype(np.uint8)
    
    return img.data.numpy()[0], prostate_seg

# --- Load and process ground truth segmentation ---
def load_ground_truth_segmentation(prostate_path, original_shape):
    """Load and process ground truth prostate segmentation"""
    if prostate_path is None or not os.path.exists(prostate_path):
        return None
    
    try:
        gt_img = nib.load(prostate_path)
        gt_data = gt_img.get_fdata()
        
        # Ensure same shape as original (may need resizing/resampling)
        if gt_data.shape != original_shape:
            # Simple approach: if shapes don't match, try to pad/crop
            gt_data = pad_or_crop_numpy(gt_data, original_shape)
        
        # Convert to binary mask (handle different label values)
        gt_seg = (gt_data > 0).astype(np.uint8)
        return gt_seg
    except Exception as e:
        print(f"    Warning: Could not load ground truth segmentation: {str(e)}")
        return None

# --- Save individual visualization ---
def save_individual_prediction(mri_slice, seg_slice, subject_id, output_dir, gt_slice=None):
    """Save individual subject prediction with side-by-side comparison"""
    # Create side-by-side figure
    if gt_slice is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        ax2 = None
    
    # Left panel: Prediction
    ax1.imshow(mri_slice, cmap='gray', alpha=0.8)
    if seg_slice.sum() > 0:  # Only overlay if segmentation exists
        ax1.imshow(np.ma.masked_where(seg_slice == 0, seg_slice), 
                  cmap='Reds', alpha=0.5)
    ax1.set_title(f'Prediction - {subject_id}', fontsize=14)
    ax1.axis('off')
    
    # Right panel: Ground Truth (if available)
    if ax2 is not None and gt_slice is not None:
        ax2.imshow(mri_slice, cmap='gray', alpha=0.8)
        if gt_slice.sum() > 0:  # Only overlay if ground truth exists
            ax2.imshow(np.ma.masked_where(gt_slice == 0, gt_slice), 
                      cmap='Blues', alpha=0.5)
        ax2.set_title(f'Ground Truth - {subject_id}', fontsize=14)
        ax2.axis('off')
    elif ax2 is not None:  # Ground truth was expected but not available
        ax2.text(0.5, 0.5, 'Ground Truth\nNot Available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.axis('off')
    
    plt.tight_layout()
    
    # Save individual image
    individual_dir = os.path.join(output_dir, 'individual_predictions')
    os.makedirs(individual_dir, exist_ok=True)
    individual_path = os.path.join(individual_dir, f'{subject_id}_comparison.png')
    plt.savefig(individual_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    return individual_path

# --- Visualization function ---
def visualize_predictions(df, model, device, output_dir, max_subjects=None):
    """Visualize center slices of all predictions - saves both individual and combined images"""
    if max_subjects:
        df = df.head(max_subjects)
    
    n_subjects = len(df)
    cols = min(6, n_subjects)  # Max 6 columns
    rows = (n_subjects + cols - 1) // cols
    
    # Create combined overview figure
    fig = plt.figure(figsize=(4*cols, 4*rows))
    gs = GridSpec(rows, cols, figure=fig)
    
    # Track successful predictions
    successful_predictions = []
    
    for idx, (_, row) in enumerate(df.iterrows()):
        subject_id = row['subject_id']
        input_path = row['MRI_homogeneity-corrected']
        
        print(f"Processing {subject_id}...")
        
        try:
            # Run inference
            mri_vol, prostate_seg = run_inference(model, input_path, device)
            
            # Load ground truth segmentation
            prostate_path = row.get('Prostate')
            gt_seg = load_ground_truth_segmentation(prostate_path, mri_vol.shape)
            
            # Get center slice
            center_slice_idx = mri_vol.shape[2] // 2  # Assuming axial slices
            mri_slice = mri_vol[:, :, center_slice_idx]
            seg_slice = prostate_seg[:, :, center_slice_idx]
            gt_slice = gt_seg[:, :, center_slice_idx]
            
            # Save individual prediction with comparison
            individual_path = save_individual_prediction(mri_slice, seg_slice, subject_id, output_dir, gt_slice)
            print(f"  Comparison image saved: {individual_path}")
            
            # Add to combined overview (showing prediction only)
            row_idx = idx // cols
            col_idx = idx % cols
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            # Display MRI with prostate overlay
            ax.imshow(mri_slice, cmap='gray', alpha=0.8)
            if seg_slice.sum() > 0:  # Only overlay if segmentation exists
                ax.imshow(np.ma.masked_where(seg_slice == 0, seg_slice), 
                         cmap='Reds', alpha=0.5)
            
            ax.set_title(f'{subject_id}', fontsize=10)
            ax.axis('off')
            
            successful_predictions.append(subject_id)
            
        except Exception as e:
            print(f"Error processing {subject_id}: {str(e)}")
            # Create empty subplot for failed cases
            row_idx = idx // cols
            col_idx = idx % cols
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.text(0.5, 0.5, f'{subject_id}\nError', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save the combined overview
    os.makedirs(output_dir, exist_ok=True)
    overview_path = os.path.join(output_dir, 'prostate_predictions_overview.png')
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    print(f"\nCombined overview saved: {overview_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total subjects processed: {len(df)}")
    print(f"  Successful predictions: {len(successful_predictions)}")
    print(f"  Individual images saved in: {os.path.join(output_dir, 'individual_predictions')}")
    print(f"  Combined overview saved as: {overview_path}")
    
    plt.show()

# --- Main function ---
def main():
    parser = argparse.ArgumentParser(description="Run batch inference and visualize prostate predictions")
    parser.add_argument('--model', required=True, help="Path to .pth model file")
    parser.add_argument('--data-dir', default='data/train', help="Path to data directory with subject folders")
    parser.add_argument('--output', default='./visualizations', help="Output directory for visualizations")
    parser.add_argument('--max_subjects', type=int, help="Maximum number of subjects to process (for testing)")
    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    net = UNet3D(in_channels=1, out_channels=3).to(device)
    state = torch.load(args.model, map_location=device)
    net.load_state_dict(state)
    net.eval()
    print("Model loaded successfully!")

    # Build dataframe
    print("Building dataframe...")
    df, _, _ = build_subject_dataframe(args.data_dir)
    
    if len(df) == 0:
        print("No subjects found!")
        return

    # Run batch inference and visualization
    print("Running batch inference and creating visualizations...")
    visualize_predictions(df, net, device, args.output, args.max_subjects)
    
    print("Done!")

if __name__ == '__main__':
    main()