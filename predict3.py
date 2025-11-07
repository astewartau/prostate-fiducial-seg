#!/usr/bin/env python3
"""
predict3.py - Consensus inference with top-3 marker selection

Adapted from predict2.py to use consensus approach:
- Takes multiple models as input
- Averages probability maps across models
- Selects top 3 markers based on domain knowledge
"""
import os
import argparse
import glob
import torch
import nibabel as nib
import torchio as tio
import numpy as np
import scipy.ndimage

import torch.nn as nn
import torch.nn.functional as F

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
        self.final_conv = nn.Conv3d(cur_channels, 3, kernel_size=1)
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


class MergeInputChannels(tio.Transform):
    def __init__(self, infile_cols):
        super().__init__()
        self.infile_cols = infile_cols
    def apply_transform(self, subject):
        tensors = []
        for col in self.infile_cols:
            img = subject.get(col)
            if img is None: continue
            data = img.data
            if data.ndim == 3:
                data = data.unsqueeze(0)
            tensors.append(data)
        merged = torch.cat(tensors, dim=0)
        subject['image'] = tio.ScalarImage(
            tensor=merged, affine=subject[self.infile_cols[0]].affine
        )
        return subject

def find_nearest_compatible_size(input_shape, min_factor=32):
    """Find nearest larger size divisible by min_factor (pad-only approach)"""
    compatible_shape = []
    for dim in input_shape:
        # Round up to nearest multiple of min_factor
        compatible_dim = ((dim + min_factor - 1) // min_factor) * min_factor
        compatible_shape.append(compatible_dim)
    return tuple(compatible_shape)

def get_crop_pad_params(orig_shape, proc_shape, target_shape=(100,100,64)):
    crop_params, pad_params = [None]*3, [None]*3
    for ax in range(3):
        od, pd = orig_shape[ax], proc_shape[ax]
        td = target_shape[ax]
        if od >= td:
            start = (od - td)//2
            crop_params[ax] = (start, start+td)
        else:
            crop_params[ax] = None
        if od < td:
            diff = td - od
            left = diff//2
            pad_params[ax] = (left, diff-left)
        else:
            pad_params[ax] = None
    return crop_params, pad_params

def invert_crop_pad(arr, orig_shape, crop_params, pad_params):
    # first undo padding
    sl = []
    for ax in range(3):
        if pad_params[ax]:
            l,_ = pad_params[ax]
            sl.append(slice(l, l+orig_shape[ax]))
        else:
            sl.append(slice(None))
    truncated = arr[sl[0], sl[1], sl[2]]
    # then place into zeros of orig shape at crop location
    out = np.zeros(orig_shape, dtype=truncated.dtype)
    sl_out = []
    for ax in range(3):
        if crop_params[ax]:
            s,e = crop_params[ax]
            sl_out.append(slice(s,e))
        else:
            sl_out.append(slice(None))
    out[sl_out[0], sl_out[1], sl_out[2]] = truncated
    return out

def select_top_n_markers(probability_map, n_markers=3, threshold=0.1):
    """
    Select top N markers from probability map based on connected component analysis.

    Args:
        probability_map: 3D numpy array of probabilities for marker class
        n_markers: Number of top markers to keep (default: 3)
        threshold: Probability threshold for marker detection

    Returns:
        Binary segmentation with only top N markers
    """
    structure = np.ones((3, 3, 3), dtype=bool)

    # Threshold probability map to get candidate regions
    binary_mask = (probability_map > threshold).astype(np.int32)

    # Label connected components
    labeled_array, num_features = scipy.ndimage.label(binary_mask, structure=structure)

    if num_features == 0:
        print(f"  WARNING: No markers found above threshold {threshold}")
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
    print(f"  Top {n_markers} marker probabilities: {[f'{score:.4f}' for _, score in component_scores[:n_markers]]}")

    # Create output mask with only top N markers
    output_mask = np.zeros_like(probability_map, dtype=np.uint8)
    for label_id in top_n_labels:
        output_mask[labeled_array == label_id] = 1

    return output_mask

def parse_args():
    p = argparse.ArgumentParser(
        description='3D-UNet consensus inference with top-3 marker selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use all models matching pattern
  python predict3.py -i scan.nii -m "models/*-best.pth" -o output/

  # Use specific models
  python predict3.py -i scan.nii -m model1.pth model2.pth model3.pth -o output/
        """
    )
    p.add_argument('-i','--input', required=True, help='Input NIfTI file')
    p.add_argument('-m','--models', nargs='+', required=True, help='Trained .pth files (supports wildcards)')
    p.add_argument('-o','--output_dir', required=True, help='Where to save outputs')
    p.add_argument('--n-markers', type=int, default=3, help='Number of top markers to select (default: 3)')
    p.add_argument('--threshold', type=float, default=0.1, help='Probability threshold (default: 0.1)')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

def main():
    args = parse_args()

    # Expand glob patterns in model paths
    model_paths = []
    for pattern in args.models:
        expanded = glob.glob(pattern)
        if expanded:
            model_paths.extend(expanded)
        else:
            if os.path.exists(pattern):
                model_paths.append(pattern)

    model_paths = sorted(list(set(model_paths)))

    if len(model_paths) == 0:
        print("ERROR: No model files found!")
        return

    print("="*80)
    print("CONSENSUS INFERENCE WITH TOP-3 MARKER SELECTION")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of models: {len(model_paths)}")
    print(f"Number of markers to select: {args.n_markers}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print()

    # --- 1) load input & record original shape/header ---
    img = nib.load(args.input)
    orig_np = img.get_fdata().astype(np.float32)
    orig_shape = orig_np.shape
    hdr = img.header
    print(f"Original shape: {orig_shape}")

    # Calculate dynamic compatible size (rounds to nearest multiple of 32)
    compatible_shape = find_nearest_compatible_size(orig_shape, min_factor=32)
    print(f"Compatible shape (padded to multiples of 32): {compatible_shape}")

    # --- 2) build the TorchIO pipeline with dynamic sizing ---
    infile_cols = ['t1_corrected']
    transforms = tio.Compose([
        tio.CropOrPad(compatible_shape),
        tio.ZNormalization(),
        MergeInputChannels(infile_cols),
    ])

    # --- 3) wrap in a Subject and preprocess ---
    subj = tio.Subject(**{infile_cols[0]: tio.ScalarImage(args.input)})
    subj_t = transforms(subj)
    proc_np = subj_t['image'].data.squeeze(0).numpy()
    print(f"Processed shape: {proc_np.shape}")

    # figure out crop/pad params
    crop_params, pad_params = get_crop_pad_params(orig_shape, proc_np.shape, compatible_shape)

    # --- 4) Run inference with all models and collect probability maps ---
    print(f"\nRunning inference with {len(model_paths)} models...")
    all_probs = []

    tensor = torch.from_numpy(proc_np).unsqueeze(0).to(device)

    for i, model_path in enumerate(model_paths):
        model_name = os.path.basename(model_path)
        print(f"  [{i+1}/{len(model_paths)}] {model_name}")

        try:
            model = UNet3D(in_channels=len(infile_cols), out_channels=3).to(device)
            checkpoint = torch.load(model_path, map_location=device)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.eval()

            with torch.no_grad():
                out = model(tensor.unsqueeze(0))[0]  # [3, 100, 100, 64]
                probs = F.softmax(out, dim=0).cpu().numpy()
                all_probs.append(probs)

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    if len(all_probs) == 0:
        print("\nERROR: No models successfully processed!")
        return

    print(f"\nSuccessfully processed {len(all_probs)} models")

    # --- 5) Average probability maps across models ---
    print("\nAveraging probability maps...")
    avg_probs = np.mean(all_probs, axis=0)  # [3, 100, 100, 64]

    # --- 6) Save averaged probabilities ---
    print("Saving averaged probability maps...")
    for i in range(avg_probs.shape[0]):
        inv = invert_crop_pad(avg_probs[i], orig_shape, crop_params, pad_params)
        nib.save(nib.Nifti1Image(inv, img.affine, hdr),
                 os.path.join(args.output_dir, f"consensus_prob_class{i}.nii.gz"))

    # --- 7) Apply top-N marker selection to seed probability map ---
    print(f"\nSelecting top {args.n_markers} markers from averaged seed probabilities...")
    seed_probs = avg_probs[1]  # Class 1 = seeds

    # Select top N markers
    top_n_seeds = select_top_n_markers(seed_probs, n_markers=args.n_markers, threshold=args.threshold)

    # Invert to original shape
    top_n_seeds_inv = invert_crop_pad(top_n_seeds, orig_shape, crop_params, pad_params)

    # Save consensus segmentation
    nib.save(nib.Nifti1Image(top_n_seeds_inv, img.affine, hdr),
             os.path.join(args.output_dir, f"consensus_top{args.n_markers}_seeds.nii.gz"))

    # Also save standard argmax segmentation for comparison
    seg = np.argmax(avg_probs, axis=0).astype(np.uint8)
    seg_inv = invert_crop_pad(seg, orig_shape, crop_params, pad_params)
    nib.save(nib.Nifti1Image(seg_inv, img.affine, hdr),
             os.path.join(args.output_dir, "consensus_argmax_segmentation.nii.gz"))

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Models processed: {len(all_probs)}")
    print(f"Consensus top-{args.n_markers} segmentation saved to:")
    print(f"  {os.path.join(args.output_dir, f'consensus_top{args.n_markers}_seeds.nii.gz')}")
    print(f"Other outputs saved to: {args.output_dir}")
    print()
    print("Done.", flush=True)

if __name__ == '__main__':
    main()
