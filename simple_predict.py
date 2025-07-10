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
from sklearn.model_selection import KFold

# --- 3D UNet (matching training) ---
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, channels=(16,32,64,128,256)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self._conv_block(prev_ch, ch))
            prev_ch = ch
        self.bottleneck = self._conv_block(prev_ch, prev_ch*2)
        rev = list(reversed(channels))
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        cur_ch = prev_ch*2
        for ch in rev:
            self.upconvs.append(nn.ConvTranspose3d(cur_ch, ch, kernel_size=2, stride=2))
            self.decoders.append(self._conv_block(ch*2, ch))
            cur_ch = ch
        self.final_conv = nn.Conv3d(cur_ch, out_channels, kernel_size=1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = []
        for enc in self.encoders:
            x = enc(x)
            feats.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, enc_feat in zip(self.upconvs, self.decoders, reversed(feats)):
            x = up(x)
            if x.shape != enc_feat.shape:
                dz = enc_feat.size(2) - x.size(2)
                dy = enc_feat.size(3) - x.size(3)
                dx = enc_feat.size(4) - x.size(4)
                x = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2, dz//2, dz-dz//2])
            x = torch.cat([enc_feat, x], dim=1)
            x = dec(x)
        return self.final_conv(x)

# --- Helper to reconstruct the bids dataframe and fetch default input ---
def get_default_input(model_path):
    base = os.path.basename(model_path)
    parts = base.split('-')
    try:
        fold_id = int(parts[-2])
    except:
        raise ValueError(f"Cannot parse fold id from '{base}'")
    bids_dirs = ["bids-2025-2"]
    records = []
    for bids_dir in bids_dirs:
        for subj in sorted(glob.glob(os.path.join(bids_dir, "z*"))):
            sub = os.path.basename(subj)
            roi_dir = os.path.join(subj, "roi_niftis_mri_space")
            if not os.path.isdir(roi_dir):
                continue
            nii_list = glob.glob(os.path.join(subj, "*.nii*")) + glob.glob(os.path.join(roi_dir, "*.nii*"))
            rec = {"subject_id": sub}
            for p in nii_list:
                fn = os.path.basename(p)
                if 'MRI_homogeneity-corrected' in fn:
                    rec['MRI_homogeneity-corrected'] = p
            if 'MRI_homogeneity-corrected' in rec:
                records.append(rec)
    df = pd.DataFrame(records)
    df = df[df['MRI_homogeneity-corrected'].notna()].reset_index(drop=True)
    kf = KFold(n_splits=len(df), random_state=42, shuffle=True)
    splits = list(kf.split(df))
    if fold_id < 0 or fold_id >= len(splits):
        raise IndexError(f"Fold id {fold_id} out of range")
    _, valid_idx = splits[fold_id]
    vid = valid_idx[0]
    return df.loc[vid, 'MRI_homogeneity-corrected']

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

# --- Inference ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help="Path to .pth model file")
    parser.add_argument('--input', help="Input NIfTI; defaults to validation case if omitted.")
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    if not args.input:
        args.input = get_default_input(args.model)
        print(f"No input specified; using validation case: {args.input}")

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet3D(in_channels=1, out_channels=3).to(device)
    state = torch.load(args.model, map_location=device)
    net.load_state_dict(state)
    net.eval()

    orig = nib.load(args.input)
    affine, header = orig.affine, orig.header

    img = tio.ScalarImage(args.input)
    orig_shape = img.data.numpy()[0].shape
    sample = tio.ZNormalization()(tio.CropOrPad((100,100,64))(img))
    x = sample.data.unsqueeze(0).to(device)

    with torch.no_grad():
        out = net(x)
        seg = out.argmax(dim=1).cpu().numpy()[0]

    full = pad_or_crop_numpy(seg, orig_shape)
    seeds = (full == 1).astype(np.uint8)
    prot  = (full == 2).astype(np.uint8)

    nib.save(nib.Nifti1Image(full.astype(np.uint8), affine, header), os.path.join(args.output, 'pred_full.nii.gz'))
    nib.save(nib.Nifti1Image(seeds, affine, header), os.path.join(args.output, 'pred_seeds.nii.gz'))
    nib.save(nib.Nifti1Image(prot, affine, header), os.path.join(args.output, 'pred_prostate.nii.gz'))

if __name__=='__main__':
    main()
