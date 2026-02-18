#!/usr/bin/env python3
"""
Extract ROIs from RTSTRUCT without CT DICOMs - using SimpleITK

This version uses SimpleITK which handles DICOM coordinate systems properly.
SimpleITK uses LPS (DICOM) convention internally, avoiding manual coordinate conversions.

Workflow:
1. Load contour points from RS file (in CT patient coordinates, LPS)
2. Create a mask in CT patient space using inferred geometry
3. Apply registration transform using SimpleITK
4. Resample to MRI space
"""

import pydicom
import numpy as np
import SimpleITK as sitk
import os

def load_registration_transform(reg_path):
    """
    Load the CT-to-MRI registration transform from RE DICOM file.
    Returns SimpleITK AffineTransform.
    """
    reg = pydicom.dcmread(reg_path)

    for i, reg_seq in enumerate(reg.RegistrationSequence):
        if hasattr(reg_seq, 'MatrixRegistrationSequence'):
            for mat_reg in reg_seq.MatrixRegistrationSequence:
                if hasattr(mat_reg, 'MatrixSequence'):
                    for mat in mat_reg.MatrixSequence:
                        if hasattr(mat, 'FrameOfReferenceTransformationMatrix'):
                            matrix_flat = mat.FrameOfReferenceTransformationMatrix
                            matrix = np.array([float(v) for v in matrix_flat]).reshape(4, 4)

                            # Skip identity matrices
                            if not np.allclose(matrix, np.eye(4)):
                                print(f"  Found non-identity transform in Registration {i}")
                                print(f"    Translation: {matrix[:3, 3]}")

                                # Create SimpleITK AffineTransform
                                aff = sitk.AffineTransform(3)
                                aff.SetMatrix(matrix[:3, :3].flatten())
                                aff.SetTranslation(matrix[:3, 3])
                                return aff

    raise ValueError("No non-identity registration transform found")

def infer_ct_geometry_from_rs(rs_path):
    """
    Infer CT geometry from RS contours.
    Returns: spacing, origin, size
    """
    rs = pydicom.dcmread(rs_path)

    # Collect all contour points
    all_points = []
    all_z = []

    for roi_contour in rs.ROIContourSequence:
        if hasattr(roi_contour, 'ContourSequence'):
            for contour in roi_contour.ContourSequence:
                points = np.array(contour.ContourData).reshape(-1, 3)
                all_points.append(points)
                all_z.append(points[0, 2])

    all_points = np.vstack(all_points)
    unique_z = np.unique(np.array(all_z))
    unique_z = np.sort(unique_z)

    # Infer spacing
    z_spacing = np.median(np.diff(unique_z))

    # For XY spacing, we need to estimate
    # Use a reasonable default based on typical CT resolution
    xy_spacing = 1.0  # mm - typical CT pixel spacing

    spacing = [xy_spacing, xy_spacing, z_spacing]

    # Calculate origin and size
    x_min, y_min, z_min = all_points.min(axis=0)
    x_max, y_max, z_max = all_points.max(axis=0)

    # Add some margin
    margin = 10  # mm
    x_min -= margin
    y_min -= margin
    x_max += margin
    y_max += margin

    origin = [x_min, y_min, z_min]

    # Calculate size in voxels
    size = [
        int(np.ceil((x_max - x_min) / spacing[0])),
        int(np.ceil((y_max - y_min) / spacing[1])),
        len(unique_z)
    ]

    return spacing, origin, size, unique_z

def extract_roi_with_sitk(rs, mri_path, reg_transform, roi_name, spacing, origin, size, output_path=None):
    """
    Extract a single ROI using SimpleITK for coordinate handling.
    """
    # Find ROI by name
    roi_number = None
    for roi_struct in rs.StructureSetROISequence:
        if roi_struct.ROIName == roi_name:
            roi_number = roi_struct.ROINumber
            break

    if roi_number is None:
        print(f"  ⚠️  ROI '{roi_name}' not found in RTSTRUCT")
        return None

    # Get contours for this ROI
    contours = None
    for roi_contour in rs.ROIContourSequence:
        if roi_contour.ReferencedROINumber == roi_number:
            contours = roi_contour.ContourSequence if hasattr(roi_contour, 'ContourSequence') else None
            break

    if not contours:
        print(f"  ⚠️  No contours found for {roi_name}")
        return None

    # Create empty mask in CT space
    ct_mask = sitk.Image(size, sitk.sitkUInt8)
    ct_mask.SetSpacing(spacing)
    ct_mask.SetOrigin(origin)
    # SimpleITK default direction is identity, which works for axial scans

    # Convert to numpy for rasterization
    ct_array = sitk.GetArrayFromImage(ct_mask)  # Note: returns ZYX ordering

    # Rasterize contours
    from skimage.draw import polygon

    slices_processed = 0
    points_outside = 0

    for contour in contours:
        # Get points in CT patient space (LPS coordinates)
        ct_points = np.array(contour.ContourData).reshape(-1, 3)

        # Convert from physical coordinates to voxel indices using CT geometry
        # Manual conversion: (point - origin) / spacing
        voxel_coords = (ct_points - origin) / spacing
        voxel_coords = np.round(voxel_coords).astype(int)

        # Get Z slice
        z_slice = int(np.median(voxel_coords[:, 2]))

        if 0 <= z_slice < size[2]:
            # Rasterize polygon on this slice
            # ct_array is [Z, Y, X] but voxel_coords is [X, Y, Z]
            rr, cc = polygon(voxel_coords[:, 1], voxel_coords[:, 0], (size[1], size[0]))
            valid = (rr >= 0) & (rr < size[1]) & (cc >= 0) & (cc < size[0])
            ct_array[z_slice, rr[valid], cc[valid]] = 1
            slices_processed += 1
        else:
            points_outside += len(voxel_coords)

    # Convert back to SimpleITK image
    ct_mask = sitk.GetImageFromArray(ct_array)
    ct_mask.SetSpacing(spacing)
    ct_mask.SetOrigin(origin)

    # Load MRI reference image
    mri_img = sitk.ReadImage(mri_path)

    # Resample CT mask to MRI space using registration transform
    resampled = sitk.Resample(
        ct_mask,
        mri_img,
        reg_transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8
    )

    # Get statistics
    stats_filter = sitk.StatisticsImageFilter()
    stats_filter.Execute(resampled)
    voxel_count = int(stats_filter.GetSum())

    # Count non-zero slices
    array = sitk.GetArrayFromImage(resampled)
    nonzero_slices = np.where(array.sum(axis=(1, 2)) > 0)[0]

    print(f"  ✓ Extracted '{roi_name}':")
    print(f"      {voxel_count} voxels on {len(nonzero_slices)} MRI slices")
    print(f"      Processed {slices_processed} contours in CT space")

    # Save if requested
    if output_path:
        sitk.WriteImage(resampled, output_path)
        print(f"      Saved to: {output_path}")

    return resampled

def process_subject(subject_dir):
    """Process one subject and extract all ROIs"""
    subject = os.path.basename(subject_dir)
    print(f"\n{'='*70}")
    print(f"Processing subject: {subject}")
    print(f"{'='*70}")

    # Paths
    rs_path = os.path.join(subject_dir, f"RS.{subject}.CT.dcm")
    mri_path = os.path.join(subject_dir, f"{subject}_MRI.nii")
    reg_path = os.path.join(subject_dir, f"RE.{subject}.REGISTRATION.dcm")
    output_dir = os.path.join(subject_dir, "roi_niftis_mri_space")

    # Check files exist
    if not os.path.exists(rs_path):
        print(f"✗ RS file not found: {rs_path}")
        return
    if not os.path.exists(mri_path):
        print(f"✗ MRI file not found: {mri_path}")
        return
    if not os.path.exists(reg_path):
        print(f"✗ Registration file not found: {reg_path}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load registration transform
    print("\nLoading registration transform...")
    try:
        reg_transform = load_registration_transform(reg_path)
    except Exception as e:
        print(f"✗ Failed to load registration transform: {e}")
        return

    # Load RTSTRUCT to get all ROI names
    print("\nLoading RTSTRUCT...")
    rs = pydicom.dcmread(rs_path)
    all_roi_names = [roi.ROIName for roi in rs.StructureSetROISequence]

    print(f"Found {len(all_roi_names)} ROIs in RTSTRUCT")

    # Filter ROIs to match old workflow behavior (script2.py from bids-2025-2):
    # - Any ROI containing "CTV"
    # - GS1_01 (or GS1), GS2_01 (or GS2), GS3_01 (or GS3)
    chosen_rois = set()
    for name in all_roi_names:
        if "CTV" in name:
            chosen_rois.add(name)

    targets = ["GS1", "GS2", "GS3"]
    for base in targets:
        if f"{base}_01" in all_roi_names:
            chosen_rois.add(f"{base}_01")
        elif base in all_roi_names:
            chosen_rois.add(base)

    if not chosen_rois:
        print(f"✗ No GS or CTV ROIs found; skipping subject")
        return

    print(f"Selected {len(chosen_rois)} ROIs to extract:")
    for roi_name in sorted(chosen_rois):
        print(f"  - {roi_name}")

    # Infer CT geometry once from all contours
    print("\nInferring CT geometry from contours...")
    spacing, origin, size, unique_z = infer_ct_geometry_from_rs(rs_path)
    print(f"  CT geometry: size={size}, spacing={spacing}, origin={origin[:3]}")

    # Extract selected ROIs
    print(f"\nExtracting ROIs using SimpleITK:")
    extracted_count = 0
    for roi_name in sorted(chosen_rois):
        # Use same naming convention as old workflow: {subject}_roi_{roi_name}_MR.nii.gz
        output_path = os.path.join(output_dir, f"{subject}_roi_{roi_name.replace(' ', '_')}_MR.nii.gz")
        try:
            mask = extract_roi_with_sitk(rs, mri_path, reg_transform, roi_name, spacing, origin, size, output_path)
            if mask is not None:
                extracted_count += 1
        except Exception as e:
            print(f"  ✗ Failed to extract '{roi_name}': {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"Summary: Successfully extracted {extracted_count}/{len(chosen_rois)} ROIs")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    import sys

    # Allow processing specific subject or all subjects in current directory
    if len(sys.argv) > 1:
        # Process specific subject passed as argument
        subject = sys.argv[1]
        if not subject.startswith('z'):
            subject = f"z{subject}"
        process_subject(subject)
    else:
        # Process all subject directories (z*) in current directory
        import glob
        subject_dirs = sorted(glob.glob("z*"))

        if not subject_dirs:
            print("No subject directories (z*) found in current directory")
            print("Usage: python extract_rois.py [subject_id]")
            print("  or run from directory containing z* folders")
            sys.exit(1)

        print(f"Found {len(subject_dirs)} subject(s) to process")
        for subject_dir in subject_dirs:
            process_subject(subject_dir)
