#!/usr/bin/env python3
import sys
import pandas as pd
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import os

def apply_homogeneity_correction(image_data):
    sitk_image = sitk.GetImageFromArray(image_data)
    mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    corrected_image = sitk.N4BiasFieldCorrection(sitk_image, mask)
    corrected_data = sitk.GetArrayFromImage(corrected_image)
    return corrected_data

def append_suffix_to_filename(filename, suffix):
    if filename.endswith('.nii.gz'):
        base = filename[:-7]
        return base + suffix + '.nii.gz'
    else:
        base, ext = os.path.splitext(filename)
        return base + suffix + ext

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Apply N4 homogeneity correction to a NIfTI file')
    parser.add_argument('input', help='Input NIfTI file')
    parser.add_argument('-o', '--output', default=None, help='Output path (default: append _homogeneity-corrected)')
    args = parser.parse_args()

    t1_file = args.input
    new_file = args.output or append_suffix_to_filename(t1_file, '_homogeneity-corrected')

    if os.path.exists(new_file):
        print(f"File {new_file} already exists. Skipping.")
        sys.exit(0)

    nii = nib.load(t1_file)
    data = nii.get_fdata()
    corrected_data = apply_homogeneity_correction(data)
    corrected_img = nib.Nifti1Image(corrected_data, nii.affine, nii.header)
    nib.save(corrected_img, new_file)

    print(f"Saved corrected image to {new_file}")
