#!/usr/bin/env python3
"""
Prepare test data for evaluation/validation.

End-to-end pipeline for each test subject:
1. Find RS (RTSTRUCT), RE (registration), and MRI NIfTI files
2. Extract GS and CTV ROIs from RTSTRUCT → MRI space
3. Apply N4 homogeneity correction to MRI
4. Create combined seeds file (class 1 = markers, class 2 = prostate)

Output structure matches training data format:
    data/test/prepared/z{ID}/
    ├── z{ID}_MRI.nii
    ├── z{ID}_MRI_homogeneity-corrected.nii
    └── roi_niftis_mri_space/
        ├── z{ID}_seeds.nii.gz
        ├── z{ID}_roi_GS1_MR.nii.gz (or GS1_01)
        ├── z{ID}_roi_GS2_MR.nii.gz
        ├── z{ID}_roi_GS3_MR.nii.gz
        └── z{ID}_roi_CTV_High_MR.nii.gz
"""

import argparse
import glob
import os
import shutil
import sys
import traceback

import nibabel as nib
import numpy as np

# Add parent directory so we can import from preprocessing modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_rois import (
    load_registration_transform,
    infer_ct_geometry_from_rs,
    extract_roi_with_sitk,
)
from homogeneity_correction import apply_homogeneity_correction


def find_rs_file(dicom_dir, subject_id):
    """Find RS (RTSTRUCT) DICOM file. Handles variable naming conventions."""
    # Glob for any RS file matching this subject
    candidates = sorted(glob.glob(os.path.join(dicom_dir, f"RS.{subject_id}.*.dcm")))
    # Filter out .0001.dcm duplicates (prefer the base version)
    base_candidates = [c for c in candidates if not c.endswith(".0001.dcm")]
    if base_candidates:
        return base_candidates[0]
    if candidates:
        return candidates[0]
    return None


def find_re_file(dicom_dir, subject_id):
    """Find RE (registration) DICOM file."""
    candidates = sorted(glob.glob(os.path.join(dicom_dir, f"RE.{subject_id}.*.dcm")))
    base_candidates = [c for c in candidates if not c.endswith(".0001.dcm")]
    if base_candidates:
        return base_candidates[0]
    if candidates:
        return candidates[0]
    return None


def find_mri_nifti(bids_dir, subject_id):
    """Find MRI NIfTI from BIDS structure."""
    # Try any session directory and subdirectory (extra_data/, anat/, etc.)
    subdirs = sorted(glob.glob(os.path.join(bids_dir, f"sub-{subject_id}", "ses-*", "*")))
    for subdir in subdirs:
        if not os.path.isdir(subdir):
            continue
        candidates = sorted(glob.glob(os.path.join(subdir, "*.nii")))
        if candidates:
            return candidates[0]
        candidates = sorted(glob.glob(os.path.join(subdir, "*.nii.gz")))
        if candidates:
            return candidates[0]
    return None


def process_subject(subject_id, dicom_dir, bids_dir, output_base, force=False):
    """Process one test subject end-to-end."""
    import pydicom

    print(f"\n{'='*70}")
    print(f"Processing subject: {subject_id}")
    print(f"{'='*70}")

    subject_dicom_dir = os.path.join(dicom_dir, subject_id)
    output_dir = os.path.join(output_base, subject_id)
    roi_dir = os.path.join(output_dir, "roi_niftis_mri_space")

    # Check if already processed
    seeds_path = os.path.join(roi_dir, f"{subject_id}_seeds.nii.gz")
    hc_path = os.path.join(output_dir, f"{subject_id}_MRI_homogeneity-corrected.nii")
    if not force and os.path.exists(seeds_path) and os.path.exists(hc_path):
        print(f"  Already processed (seeds and HC exist). Use --force to reprocess.")
        return True

    # --- Find required files ---
    rs_path = find_rs_file(subject_dicom_dir, subject_id)
    if not rs_path:
        print(f"  ERROR: No RS file found in {subject_dicom_dir}")
        return False

    re_path = find_re_file(subject_dicom_dir, subject_id)
    if not re_path:
        print(f"  ERROR: No RE file found in {subject_dicom_dir}")
        return False

    mri_path = find_mri_nifti(bids_dir, subject_id)
    if not mri_path:
        print(f"  ERROR: No MRI NIfTI found in BIDS for {subject_id}")
        return False

    print(f"  RS:  {os.path.basename(rs_path)}")
    print(f"  RE:  {os.path.basename(re_path)}")
    print(f"  MRI: {mri_path}")

    # --- Create output directories ---
    os.makedirs(roi_dir, exist_ok=True)

    # --- Copy MRI to output directory ---
    output_mri = os.path.join(output_dir, f"{subject_id}_MRI.nii")
    if not os.path.exists(output_mri):
        # Handle both .nii and .nii.gz source
        if mri_path.endswith(".nii.gz"):
            import gzip
            print(f"  Decompressing MRI to output...")
            with gzip.open(mri_path, 'rb') as f_in:
                with open(output_mri, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            print(f"  Copying MRI to output...")
            shutil.copy2(mri_path, output_mri)

    # --- Extract ROIs ---
    print(f"\n  Loading registration transform...")
    try:
        reg_transform = load_registration_transform(re_path)
    except Exception as e:
        print(f"  ERROR: Failed to load registration: {e}")
        return False

    print(f"  Loading RTSTRUCT...")
    rs = pydicom.dcmread(rs_path)
    all_roi_names = [roi.ROIName for roi in rs.StructureSetROISequence]
    print(f"  Found ROIs: {all_roi_names}")

    # Select ROIs to extract (GS markers + CTV)
    chosen_rois = set()
    for name in all_roi_names:
        if "CTV" in name:
            chosen_rois.add(name)

    for base in ["GS1", "GS2", "GS3"]:
        # Extract both base and _01 variants if present (some have empty contours)
        if base in all_roi_names:
            chosen_rois.add(base)
        if f"{base}_01" in all_roi_names:
            chosen_rois.add(f"{base}_01")

    if not chosen_rois:
        print(f"  ERROR: No GS or CTV ROIs found in RTSTRUCT")
        return False

    print(f"  Extracting {len(chosen_rois)} ROIs: {sorted(chosen_rois)}")

    # Infer CT geometry
    spacing, origin, size, unique_z = infer_ct_geometry_from_rs(rs_path)
    print(f"  CT geometry: size={size}, spacing=[{spacing[0]:.1f}, {spacing[1]:.1f}, {spacing[2]:.1f}]")

    # Extract each ROI
    gs_count = 0
    has_ctv = False
    for roi_name in sorted(chosen_rois):
        output_path = os.path.join(roi_dir, f"{subject_id}_roi_{roi_name.replace(' ', '_')}_MR.nii.gz")
        try:
            mask = extract_roi_with_sitk(rs, output_mri, reg_transform, roi_name,
                                         spacing, origin, size, output_path)
            if mask is not None:
                if roi_name.startswith("GS"):
                    gs_count += 1
                elif "CTV" in roi_name:
                    has_ctv = True
        except Exception as e:
            print(f"  ERROR extracting '{roi_name}': {e}")
            traceback.print_exc()

    if gs_count == 0:
        print(f"  WARNING: No GS markers extracted!")
    if not has_ctv:
        print(f"  WARNING: No CTV/prostate segmentation extracted!")

    # --- N4 Homogeneity Correction ---
    if not os.path.exists(hc_path) or force:
        print(f"\n  Applying N4 homogeneity correction...")
        try:
            nii = nib.load(output_mri)
            data = nii.get_fdata()
            corrected_data = apply_homogeneity_correction(data)
            corrected_img = nib.Nifti1Image(corrected_data, nii.affine, nii.header)
            nib.save(corrected_img, hc_path)
            print(f"  Saved: {hc_path}")
        except Exception as e:
            print(f"  ERROR during N4 correction: {e}")
            traceback.print_exc()
            return False
    else:
        print(f"\n  HC already exists, skipping.")

    # --- Create Combined Seeds ---
    if not os.path.exists(seeds_path) or force:
        print(f"\n  Creating combined seeds file...")
        try:
            _create_combined_seeds(subject_id, roi_dir, seeds_path)
        except Exception as e:
            print(f"  ERROR creating seeds: {e}")
            traceback.print_exc()
            return False
    else:
        print(f"\n  Seeds file already exists, skipping.")

    print(f"\n  Subject {subject_id} DONE")
    return True


def _create_combined_seeds(subject_id, roi_dir, seeds_path):
    """Create combined seeds.nii.gz from individual GS and CTV ROIs."""
    # Find GS marker files
    gs_files = {}
    for gs_num in [1, 2, 3]:
        for pattern in [
            f"{subject_id}_roi_GS{gs_num}_01_MR.nii.gz",
            f"{subject_id}_roi_GS{gs_num}_MR.nii.gz",
        ]:
            path = os.path.join(roi_dir, pattern)
            if os.path.exists(path):
                gs_files[gs_num] = path
                break

    if not gs_files:
        print(f"    WARNING: No GS marker files found for seeds creation")
        return

    # Find prostate/CTV file
    ctv_patterns = [
        f"{subject_id}_roi_CTV_High_MR.nii.gz",
        f"{subject_id}_roi_CTV_Low_MR.nii.gz",
        f"{subject_id}_roi_CTVp_MR.nii.gz",
    ]
    prostate_file = None
    for pattern in ctv_patterns:
        path = os.path.join(roi_dir, pattern)
        if os.path.exists(path):
            prostate_file = path
            break

    if not prostate_file:
        print(f"    WARNING: No CTV file found for seeds creation")
        return

    # Build combined label volume
    prostate_img = nib.load(prostate_file)
    combined = np.zeros(prostate_img.shape, dtype=np.uint8)

    # Class 2: prostate
    prostate_data = prostate_img.get_fdata()
    combined[prostate_data > 0] = 2

    # Class 1: gold seed markers (overwrites prostate where they overlap)
    for gs_num, gs_path in sorted(gs_files.items()):
        gs_data = nib.load(gs_path).get_fdata()
        combined[gs_data > 0] = 1

    seeds_nii = nib.Nifti1Image(combined, prostate_img.affine, prostate_img.header)
    nib.save(seeds_nii, seeds_path)
    print(f"    Saved seeds ({len(gs_files)} markers + prostate): {seeds_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare test data for evaluation")
    parser.add_argument("--dicom-dir", default="data/test/dicoms",
                        help="Directory containing z{ID}/ DICOM folders")
    parser.add_argument("--bids-dir", default="data/test/bids",
                        help="BIDS directory with sub-z{ID}/ NIfTI files")
    parser.add_argument("--output-dir", default="data/test/prepared",
                        help="Output directory for prepared subjects")
    parser.add_argument("--subjects", nargs="*",
                        help="Specific subject IDs to process (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Force reprocessing even if output exists")
    args = parser.parse_args()

    # Find subjects to process
    if args.subjects:
        subject_ids = args.subjects
    else:
        subject_dirs = sorted(glob.glob(os.path.join(args.dicom_dir, "z*")))
        subject_ids = [os.path.basename(d) for d in subject_dirs]

    if not subject_ids:
        print(f"No subjects found in {args.dicom_dir}")
        sys.exit(1)

    print(f"Preparing {len(subject_ids)} test subjects")
    print(f"  DICOM dir:  {args.dicom_dir}")
    print(f"  BIDS dir:   {args.bids_dir}")
    print(f"  Output dir: {args.output_dir}")

    # Process each subject
    results = {}
    for subject_id in subject_ids:
        try:
            success = process_subject(subject_id, args.dicom_dir, args.bids_dir,
                                      args.output_dir, force=args.force)
            results[subject_id] = success
        except Exception as e:
            print(f"\n  FATAL ERROR processing {subject_id}: {e}")
            traceback.print_exc()
            results[subject_id] = False

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    succeeded = [s for s, ok in results.items() if ok]
    failed = [s for s, ok in results.items() if not ok]
    print(f"  Succeeded: {len(succeeded)}/{len(results)}")
    if failed:
        print(f"  Failed: {failed}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
