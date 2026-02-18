#!/usr/bin/env python3
"""
Consensus inference script for prostate marker segmentation.

Takes multiple trained models and a single input volume, produces:
1. Individual segmentations from each model
2. Individual probability maps from each model
3. Consensus segmentation using top-3 marker selection from averaged probabilities

The consensus approach leverages the domain knowledge that there are always
exactly 3 markers in each volume.
"""

import os
import argparse
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import torchio as tio
import scipy.ndimage
from pathlib import Path


import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import UNet3D


def find_nearest_compatible_size(input_shape, min_factor=32):
    """Find nearest larger size divisible by min_factor (pad-only approach)"""
    compatible_shape = []
    for dim in input_shape:
        compatible_dim = ((dim + min_factor - 1) // min_factor) * min_factor
        compatible_shape.append(compatible_dim)
    return tuple(compatible_shape)


def pad_or_crop_numpy(vol, target):
    """Crop/pad volume to target shape"""
    # Crop
    slices = []
    for i in range(3):
        diff = vol.shape[i] - target[i]
        if diff > 0:
            start = diff // 2
            slices.append(slice(start, start + target[i]))
        else:
            slices.append(slice(0, vol.shape[i]))
    vol_c = vol[tuple(slices)]

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


def select_top_n_markers(probability_map, n_markers=3, structure=None, threshold=0.1):
    """
    Select top N markers from probability map based on connected component analysis.

    Args:
        probability_map: 3D numpy array of probabilities for marker class
        n_markers: Number of top markers to keep (default: 3)
        structure: Structuring element for connected component labeling
        threshold: Minimum probability to consider a voxel as candidate (default: 0.1)

    Returns:
        Binary segmentation with only top N markers
    """
    if structure is None:
        structure = np.ones((3, 3, 3), dtype=bool)
    binary_mask = (probability_map > threshold).astype(np.int32)

    # Label connected components
    labeled_array, num_features = scipy.ndimage.label(binary_mask, structure=structure)

    if num_features == 0:
        print("  WARNING: No markers found above threshold")
        return np.zeros_like(probability_map, dtype=np.uint8)

    if num_features <= n_markers:
        print(f"  Found {num_features} markers (≤ {n_markers}), keeping all")
        return binary_mask.astype(np.uint8)

    # Calculate mean probability for each component
    component_scores = []
    for label_id in range(1, num_features + 1):
        mask = (labeled_array == label_id)
        mean_prob = probability_map[mask].mean()
        component_scores.append((label_id, mean_prob))

    # Sort by mean probability (descending) and keep top N
    component_scores.sort(key=lambda x: x[1], reverse=True)
    top_n_labels = [label_id for label_id, _ in component_scores[:n_markers]]

    print(f"  Found {num_features} candidate markers, selecting top {n_markers}")
    print(f"  Top {n_markers} marker probabilities: {[score for _, score in component_scores[:n_markers]]}")

    # Create output mask with only top N markers
    output_mask = np.zeros_like(probability_map, dtype=np.uint8)
    for label_id in top_n_labels:
        output_mask[labeled_array == label_id] = 1

    return output_mask


def run_inference(model, input_tensor, device):
    """
    Run inference and return both segmentation and probability maps.

    Returns:
        seg: Argmax segmentation (3D array with class labels)
        prob_maps: Softmax probabilities (4D array: [num_classes, H, W, D])
    """
    with torch.no_grad():
        outputs = model(input_tensor)

        # Get segmentation (argmax)
        seg = outputs.argmax(dim=1).cpu().numpy()[0]

        # Get probability maps (softmax)
        prob_maps = F.softmax(outputs, dim=1).cpu().numpy()[0]  # Shape: [num_classes, H, W, D]

    return seg, prob_maps


def main():
    parser = argparse.ArgumentParser(
        description="Consensus inference for prostate marker segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use all models in current directory
  python consensus_inference.py --models *.pth --input scan.nii.gz --output results/

  # Use specific models
  python consensus_inference.py --models model1.pth model2.pth model3.pth --input scan.nii.gz --output results/

  # Specify number of markers (default: 3)
  python consensus_inference.py --models *.pth --input scan.nii.gz --output results/ --n-markers 3
        """
    )
    parser.add_argument('--models', nargs='+', required=True,
                        help="Paths to model checkpoint files (.pth)")
    parser.add_argument('--input', required=True,
                        help="Input NIfTI file (MRI volume)")
    parser.add_argument('--output', required=True,
                        help="Output directory for results")
    parser.add_argument('--n-markers', type=int, default=3,
                        help="Number of top markers to select in consensus (default: 3)")
    parser.add_argument('--threshold', type=float, default=0.1,
                        help="Probability threshold for marker detection (default: 0.1)")

    args = parser.parse_args()

    # Expand glob patterns in model paths
    model_paths = []
    for pattern in args.models:
        expanded = glob.glob(pattern)
        if expanded:
            model_paths.extend(expanded)
        else:
            # If no glob match, treat as literal path
            if os.path.exists(pattern):
                model_paths.append(pattern)

    model_paths = sorted(list(set(model_paths)))  # Remove duplicates and sort

    if len(model_paths) == 0:
        print("ERROR: No model files found!")
        return

    print("=" * 80)
    print("CONSENSUS INFERENCE FOR PROSTATE MARKER SEGMENTATION")
    print("=" * 80)
    print(f"Input volume: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Number of models: {len(model_paths)}")
    print(f"Number of markers to select: {args.n_markers}")
    print()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load and preprocess input volume
    print("Loading input volume...")
    orig_nifti = nib.load(args.input)
    affine, header = orig_nifti.affine, orig_nifti.header

    img = tio.ScalarImage(args.input)
    orig_shape = img.data.numpy()[0].shape
    compatible_shape = find_nearest_compatible_size(orig_shape)

    print(f"  Original shape: {orig_shape}")
    print(f"  Compatible shape: {compatible_shape}")

    # Apply preprocessing (matching training)
    sample = tio.ZNormalization()(tio.CropOrPad(compatible_shape)(img))
    input_tensor = sample.data.unsqueeze(0).to(device)
    print()

    # Run inference with each model
    print(f"Running inference with {len(model_paths)} models...")
    all_segmentations = []
    all_probability_maps = []  # Will store probability maps for class 1 (seeds)

    for i, model_path in enumerate(model_paths):
        model_name = os.path.basename(model_path)
        print(f"\n[{i+1}/{len(model_paths)}] Processing {model_name}")

        # Load model
        try:
            net = UNet3D(in_channels=1, out_channels=3).to(device)
            checkpoint = torch.load(model_path, map_location=device)

            # Handle both raw state dict and checkpoint with metadata
            if 'model_state_dict' in checkpoint:
                net.load_state_dict(checkpoint['model_state_dict'])
            else:
                net.load_state_dict(checkpoint)

            net.eval()
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            print(f"  Skipping this model")
            continue

        # Run inference
        seg, prob_maps = run_inference(net, input_tensor, device)

        # Crop/pad back to original shape
        seg_full = pad_or_crop_numpy(seg, orig_shape)
        prob_maps_full = np.stack([
            pad_or_crop_numpy(prob_maps[c], orig_shape)
            for c in range(prob_maps.shape[0])
        ], axis=0)

        all_segmentations.append(seg_full)
        all_probability_maps.append(prob_maps_full[1])  # Class 1 = seeds/markers

        # Save individual outputs
        model_prefix = model_name.replace('.pth', '')

        # Save full segmentation (all classes)
        seg_path = os.path.join(args.output, f'{model_prefix}_seg.nii.gz')
        nib.save(nib.Nifti1Image(seg_full.astype(np.uint8), affine, header), seg_path)

        # Save probability map for seeds (class 1)
        prob_path = os.path.join(args.output, f'{model_prefix}_prob_seeds.nii.gz')
        nib.save(nib.Nifti1Image(prob_maps_full[1].astype(np.float32), affine, header), prob_path)

        print(f"  Saved segmentation: {seg_path}")
        print(f"  Saved probability map: {prob_path}")

    if len(all_probability_maps) == 0:
        print("\nERROR: No models successfully processed!")
        return

    print()
    print("=" * 80)
    print("COMPUTING CONSENSUS SEGMENTATION")
    print("=" * 80)

    # Average probability maps across all models
    print(f"Averaging probability maps from {len(all_probability_maps)} models...")
    avg_probability_map = np.mean(all_probability_maps, axis=0)

    # Save averaged probability map
    avg_prob_path = os.path.join(args.output, 'consensus_avg_probability.nii.gz')
    nib.save(nib.Nifti1Image(avg_probability_map.astype(np.float32), affine, header), avg_prob_path)
    print(f"Saved averaged probability map: {avg_prob_path}")
    print()

    # Select top N markers from averaged probability map
    print(f"Selecting top {args.n_markers} markers from averaged probabilities...")
    consensus_seg = select_top_n_markers(
        avg_probability_map,
        n_markers=args.n_markers,
        structure=np.ones((3, 3, 3), dtype=bool),
        threshold=args.threshold
    )

    # Save consensus segmentation
    consensus_path = os.path.join(args.output, 'consensus_top3_segmentation.nii.gz')
    nib.save(nib.Nifti1Image(consensus_seg, affine, header), consensus_path)
    print(f"\nSaved consensus segmentation: {consensus_path}")

    # Summary statistics
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Models processed: {len(all_probability_maps)}")
    print(f"Number of voxels in consensus segmentation: {np.sum(consensus_seg > 0)}")
    print(f"Mean probability in selected markers: {avg_probability_map[consensus_seg > 0].mean():.4f}")
    print(f"Max probability in selected markers: {avg_probability_map[consensus_seg > 0].max():.4f}")
    print(f"Min probability in selected markers: {avg_probability_map[consensus_seg > 0].min():.4f}")
    print()
    print(f"All outputs saved to: {args.output}")
    print()


if __name__ == '__main__':
    main()
