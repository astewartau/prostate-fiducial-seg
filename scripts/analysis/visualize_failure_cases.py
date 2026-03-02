#!/usr/bin/env python3
"""
Visualize failure cases from the production model consensus evaluation.

For each subject with FN or FP markers, generates a figure with:
- One row per failure point (FN or FP)
- Each row: axial/coronal/sagittal MRI alone | same slices with overlays
- Color coding: orange=ground truth (missed), blue=true positive, reddish purple=false positive
"""

import os
import sys
import json
import glob

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import torchio as tio
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import UNet3D


def find_nearest_compatible_size(input_shape, min_factor=32):
    return tuple(
        ((dim + min_factor - 1) // min_factor) * min_factor
        for dim in input_shape
    )


def pad_or_crop_numpy(vol, target):
    slices = []
    for i in range(3):
        diff = vol.shape[i] - target[i]
        if diff > 0:
            start = diff // 2
            slices.append(slice(start, start + target[i]))
        else:
            slices.append(slice(0, vol.shape[i]))
    vol_c = vol[tuple(slices)]
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


def select_top_n_markers(probability_map, n_markers=3, threshold=0.1):
    structure = np.ones((3, 3, 3), dtype=bool)
    binary_mask = (probability_map > threshold).astype(np.int32)
    labeled_array, num_features = scipy.ndimage.label(binary_mask, structure=structure)
    if num_features == 0:
        return np.zeros_like(probability_map, dtype=np.uint8)
    if num_features <= n_markers:
        return binary_mask.astype(np.uint8)
    component_scores = []
    for label_id in range(1, num_features + 1):
        mask = (labeled_array == label_id)
        mean_prob = probability_map[mask].mean()
        component_scores.append((label_id, mean_prob))
    component_scores.sort(key=lambda x: x[1], reverse=True)
    top_labels = [lid for lid, _ in component_scores[:n_markers]]
    output_mask = np.zeros_like(probability_map, dtype=np.uint8)
    for lid in top_labels:
        output_mask[labeled_array == lid] = 1
    return output_mask


def find_failures(pred_seg, targ_seg):
    """Identify FN and FP component centroids and labeled arrays."""
    structure = np.ones((3, 3, 3), dtype=bool)

    pred_marker = scipy.ndimage.binary_dilation(
        (pred_seg == 1).astype(np.int32)
    ).astype(np.int32)
    targ_marker = scipy.ndimage.binary_dilation(
        (targ_seg == 1).astype(np.int32)
    ).astype(np.int32)

    pred_labeled, pred_n = scipy.ndimage.label(pred_marker, structure=structure)
    targ_labeled, targ_n = scipy.ndimage.label(targ_marker, structure=structure)

    detected_targets = set()
    matched_preds = set()
    for t_id in range(1, targ_n + 1):
        t_mask = (targ_labeled == t_id)
        overlapping_preds = set(pred_labeled[t_mask]) - {0}
        if overlapping_preds:
            detected_targets.add(t_id)
            matched_preds.update(overlapping_preds)

    # Find centroids of failures
    failures = []

    # False negatives: target components not detected
    for t_id in range(1, targ_n + 1):
        if t_id not in detected_targets:
            centroid = scipy.ndimage.center_of_mass(targ_labeled == t_id)
            centroid = tuple(int(round(c)) for c in centroid)
            failures.append({
                'type': 'FN',
                'centroid': centroid,
                'label': f'FN: missed marker at ({centroid[0]}, {centroid[1]}, {centroid[2]})',
            })

    # False positives: predicted components not matched to any target
    for p_id in range(1, pred_n + 1):
        if p_id not in matched_preds:
            centroid = scipy.ndimage.center_of_mass(pred_labeled == p_id)
            centroid = tuple(int(round(c)) for c in centroid)
            failures.append({
                'type': 'FP',
                'centroid': centroid,
                'label': f'FP: false detection at ({centroid[0]}, {centroid[1]}, {centroid[2]})',
            })

    return failures, pred_labeled, targ_labeled, detected_targets, matched_preds


def make_overlay_rgb(mri_slice, targ_mask_slice, pred_mask_slice, tp_mask_slice):
    """Create an RGB image with colored overlays on grayscale MRI."""
    # Normalize MRI to [0, 1]
    vmin, vmax = np.percentile(mri_slice[mri_slice > 0], [1, 99]) if np.any(mri_slice > 0) else (0, 1)
    if vmax <= vmin:
        vmax = vmin + 1
    mri_norm = np.clip((mri_slice - vmin) / (vmax - vmin), 0, 1)

    # Create RGB from grayscale
    rgb = np.stack([mri_norm, mri_norm, mri_norm], axis=-1)

    alpha = 0.45

    # Colorblind-friendly palette (Wong, Nature Methods 2011)
    # Orange for ground truth (FN markers)
    fn_mask = targ_mask_slice & ~tp_mask_slice
    if np.any(fn_mask):
        rgb[fn_mask] = (1 - alpha) * rgb[fn_mask] + alpha * np.array([0.90, 0.62, 0.00])

    # Blue for true positives
    if np.any(tp_mask_slice):
        rgb[tp_mask_slice] = (1 - alpha) * rgb[tp_mask_slice] + alpha * np.array([0.00, 0.45, 0.70])

    # Reddish purple for false positives
    fp_mask = pred_mask_slice & ~tp_mask_slice
    if np.any(fp_mask):
        rgb[fp_mask] = (1 - alpha) * rgb[fp_mask] + alpha * np.array([0.80, 0.47, 0.65])

    return np.clip(rgb, 0, 1)


def plot_failure_figure(subject_id, mri_data, pred_seg, targ_seg, failures,
                        pred_labeled, targ_labeled, detected_targets, matched_preds,
                        output_path, half_width=30):
    """Generate a figure with one row per failure point."""
    n_failures = len(failures)
    if n_failures == 0:
        return

    # Build binary masks for overlay
    # Target mask (dilated for visibility)
    targ_mask = scipy.ndimage.binary_dilation(
        (targ_seg == 1).astype(np.int32), iterations=1
    ).astype(bool)
    # Pred mask (dilated for visibility)
    pred_mask = scipy.ndimage.binary_dilation(
        (pred_seg == 1).astype(np.int32), iterations=1
    ).astype(bool)
    # TP mask: overlap of target and prediction
    tp_mask = targ_mask & pred_mask

    fig, axes = plt.subplots(n_failures, 6, figsize=(24, 4 * n_failures))
    if n_failures == 1:
        axes = axes[np.newaxis, :]

    plane_names = ['Axial', 'Coronal', 'Sagittal']

    for row, failure in enumerate(failures):
        ci, cj, ck = failure['centroid']
        color = '#E69F00' if failure['type'] == 'FN' else '#CC79A7'

        # Define slices and crop windows for each plane
        slices_plain = []
        slices_overlay = []
        for plane_idx, (plane_label, center_coord) in enumerate(
            zip(plane_names, [ci, cj, ck])
        ):
            if plane_idx == 0:  # Axial: fix i
                mri_sl = mri_data[ci, :, :]
                targ_sl = targ_mask[ci, :, :]
                pred_sl = pred_mask[ci, :, :]
                tp_sl = tp_mask[ci, :, :]
                # Crop around j, k
                j0 = max(0, cj - half_width)
                j1 = min(mri_data.shape[1], cj + half_width)
                k0 = max(0, ck - half_width)
                k1 = min(mri_data.shape[2], ck + half_width)
                mri_sl = mri_sl[j0:j1, k0:k1]
                targ_sl = targ_sl[j0:j1, k0:k1]
                pred_sl = pred_sl[j0:j1, k0:k1]
                tp_sl = tp_sl[j0:j1, k0:k1]
            elif plane_idx == 1:  # Coronal: fix j
                mri_sl = mri_data[:, cj, :]
                targ_sl = targ_mask[:, cj, :]
                pred_sl = pred_mask[:, cj, :]
                tp_sl = tp_mask[:, cj, :]
                i0 = max(0, ci - half_width)
                i1 = min(mri_data.shape[0], ci + half_width)
                k0 = max(0, ck - half_width)
                k1 = min(mri_data.shape[2], ck + half_width)
                mri_sl = mri_sl[i0:i1, k0:k1]
                targ_sl = targ_sl[i0:i1, k0:k1]
                pred_sl = pred_sl[i0:i1, k0:k1]
                tp_sl = tp_sl[i0:i1, k0:k1]
            else:  # Sagittal: fix k
                mri_sl = mri_data[:, :, ck]
                targ_sl = targ_mask[:, :, ck]
                pred_sl = pred_mask[:, :, ck]
                tp_sl = tp_mask[:, :, ck]
                i0 = max(0, ci - half_width)
                i1 = min(mri_data.shape[0], ci + half_width)
                j0 = max(0, cj - half_width)
                j1 = min(mri_data.shape[1], cj + half_width)
                mri_sl = mri_sl[i0:i1, j0:j1]
                targ_sl = targ_sl[i0:i1, j0:j1]
                pred_sl = pred_sl[i0:i1, j0:j1]
                tp_sl = tp_sl[i0:i1, j0:j1]

            # Plain MRI
            ax_plain = axes[row, plane_idx]
            vmin_p = np.percentile(mri_sl[mri_sl > 0], 1) if np.any(mri_sl > 0) else 0
            vmax_p = np.percentile(mri_sl[mri_sl > 0], 99) if np.any(mri_sl > 0) else 1
            ax_plain.imshow(mri_sl.T, cmap='gray', origin='lower',
                           vmin=vmin_p, vmax=vmax_p, aspect='equal')
            ax_plain.set_xticks([])
            ax_plain.set_yticks([])
            if row == 0:
                ax_plain.set_title(f'{plane_label} (MRI)', fontsize=11)

            # Overlay
            ax_overlay = axes[row, plane_idx + 3]
            overlay_rgb = make_overlay_rgb(mri_sl, targ_sl, pred_sl, tp_sl)
            ax_overlay.imshow(overlay_rgb.transpose(1, 0, 2), origin='lower', aspect='equal')
            ax_overlay.set_xticks([])
            ax_overlay.set_yticks([])
            if row == 0:
                ax_overlay.set_title(f'{plane_label} (overlay)', fontsize=11)

        # Row label
        axes[row, 0].set_ylabel(failure['label'], fontsize=10, color=color,
                                fontweight='bold', rotation=0, labelpad=10,
                                ha='right', va='center')

    # Legend
    legend_elements = [
        Patch(facecolor=(0.90, 0.62, 0.00, 0.6), label='Ground truth (missed)'),
        Patch(facecolor=(0.00, 0.45, 0.70, 0.6), label='True positive'),
        Patch(facecolor=(0.80, 0.47, 0.65, 0.6), label='False positive'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11,
               frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'{subject_id} — Failure Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.08, 0.03, 1, 0.96])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    model_dir = 'models/production'
    data_dir = 'data/test/prepared'
    splits_path = 'data/splits.json'
    output_dir = 'results/failure_cases'

    print("=" * 80)
    print("FAILURE CASE VISUALIZATION")
    print("=" * 80)

    with open(splits_path) as f:
        splits = json.load(f)
    all_subjects = sorted(splits['val'] + splits['test'])

    model_paths = sorted(glob.glob(os.path.join(model_dir, '*-best.pth')))
    print(f"Models: {len(model_paths)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load models
    models = []
    for mp in model_paths:
        net = UNet3D(in_channels=1, out_channels=3).to(device)
        ckpt = torch.load(mp, map_location=device)
        if 'model_state_dict' in ckpt:
            net.load_state_dict(ckpt['model_state_dict'])
        else:
            net.load_state_dict(ckpt)
        net.eval()
        models.append(net)
    print(f"Loaded {len(models)} models\n")

    n_figures = 0
    for i, subject_id in enumerate(all_subjects):
        subject_dir = os.path.join(data_dir, subject_id)
        mri_path = os.path.join(subject_dir, f"{subject_id}_MRI_homogeneity-corrected.nii")
        seeds_path = os.path.join(subject_dir, "roi_niftis_mri_space", f"{subject_id}_seeds.nii.gz")

        if not os.path.exists(mri_path) or not os.path.exists(seeds_path):
            continue

        # Run inference
        img = tio.ScalarImage(mri_path)
        orig_shape = img.data.numpy()[0].shape
        compatible_shape = find_nearest_compatible_size(orig_shape)
        sample = tio.ZNormalization()(tio.CropOrPad(compatible_shape)(img))
        input_tensor = sample.data.unsqueeze(0).to(device)

        seeds_nii = nib.load(seeds_path)
        seeds_data = seeds_nii.get_fdata().astype(np.int32)
        mri_data = nib.load(mri_path).get_fdata()

        seed_probs = []
        with torch.no_grad():
            for net in models:
                outputs = net(input_tensor)
                prob_maps = F.softmax(outputs, dim=1).cpu().numpy()[0]
                seed_prob = pad_or_crop_numpy(prob_maps[1], orig_shape)
                seed_probs.append(seed_prob)

        avg_prob = np.mean(seed_probs, axis=0)
        consensus_seg = select_top_n_markers(avg_prob, n_markers=3, threshold=0.1)

        # Check for failures
        failures, pred_labeled, targ_labeled, detected_targets, matched_preds = \
            find_failures(consensus_seg, seeds_data)

        if len(failures) == 0:
            continue

        print(f"[{i+1}/{len(all_subjects)}] {subject_id}: {len(failures)} failure(s)")
        for f in failures:
            print(f"  {f['label']}")

        output_path = os.path.join(output_dir, f"{subject_id}_failures.png")
        plot_failure_figure(
            subject_id, mri_data, consensus_seg, seeds_data,
            failures, pred_labeled, targ_labeled, detected_targets, matched_preds,
            output_path
        )
        n_figures += 1

    print(f"\nGenerated {n_figures} failure case figures in {output_dir}/")


if __name__ == '__main__':
    main()
