# Prostate Gold Fiducial Marker Segmentation

## Problem Statement

Automatically segment gold fiducial markers (gold seeds) in T1-weighted prostate MRI using deep learning. Patients undergoing prostate radiotherapy have 3 gold seed markers implanted. Accurate automatic segmentation aids treatment planning.

## Directory Structure

```
2024-prostate/
├── scripts/
│   ├── training/           # Model training pipeline
│   ├── inference/          # Prediction/consensus inference
│   ├── evaluation/         # Metrics and comparison
│   ├── preprocessing/      # Data prep (ROI extraction, N4 correction, seed creation)
│   ├── analysis/           # Visualization, statistics, entropy
│   └── utils/              # Helper functions
├── data/
│   ├── train/              # Training data (~93 usable subjects with T1 + seeds)
│   └── test/               # New 34-subject held-out test set (DICOMs + BIDS)
├── models/
│   └── loocv/              # LOOCV model checkpoints (T1-*.pth)
├── results/                # Evaluation outputs, CSVs, comparison results
├── logs/                   # SLURM job outputs
└── archive/                # Old models, legacy scripts, loss curves, pilot data
```

## Labeling Scheme (3-class)

- **Class 0**: Background
- **Class 1**: Gold seed markers (primary target)
- **Class 2**: Prostate (CTV contour)

## Model Architecture

Custom 3D UNet (`UNet3D` in `scripts/training/train_one_model.py`):
- Encoder channels: (16, 32, 64, 128, 256) + bottleneck (512)
- 3-class output, single-channel input (homogeneity-corrected T1w MRI)
- Loss: Combined Cross-Entropy (weighted [1, 5, 1]) + 0.5 * Dice loss
- Optimizer: Adam, lr=0.003, CosineAnnealingLR
- Training: up to 1300 epochs, early stopping (patience=500)
- Augmentation: RandomIntensityShift (gamma), RandomFlip, RandomAffine(90deg), ZNormalization
- Padding: volumes padded to nearest multiple of 32

## Datasets

### Training Data: `data/train/`

- ~98 subject directories, ~93 usable (some missing T1 or invalid segmentation)
- Subject naming: `z{ID}` (e.g., `z0110968`)
- Per-subject files:
  - `z{ID}_MRI_homogeneity-corrected.nii` -- N4 bias-corrected T1w (model input)
  - `roi_niftis_mri_space/z{ID}_seeds.nii.gz` -- combined 3-class segmentation
  - `roi_niftis_mri_space/z{ID}_roi_GS{1,2,3}_MR.nii.gz` -- individual marker ROIs

### Test Data: `data/test/`

- **34 subjects** total: 9 validation + 25 held-out test (see `data/splits.json`)
- BIDS format: `data/test/bids/sub-z{ID}/ses-na/extra_data/*.nii`
- Raw DICOMs: `data/test/dicoms/z{ID}/` (includes MR, RS, and RE files)
- Prepared data: `data/test/prepared/z{ID}/` (same format as training data)
- Prepare via: `python scripts/preprocessing/prepare_test_data.py` or `sbatch scripts/preprocessing/prepare_test_data.slurm`

## Training Approach

### LOOCV (historical)
- Leave-one-out cross-validation: one model per subject
- 95 best-checkpoint models in `models/loocv/`
- Run via: `scripts/training/train_all_models_via_slurm.sh` (submits SLURM jobs from project root)

### Consensus Inference (`scripts/inference/consensus_inference.py`)
Best-performing approach:
1. Run all models on input volume
2. Average marker-class probability maps across models
3. Select top-3 connected components by mean probability
4. Output: segmentation with exactly 3 marker regions

### Production Models
- 4 models trained on ALL 93 training subjects with different random seeds (42, 123, 456, 789)
- Validated on 9 held-out subjects from the test set (see `data/splits.json`)
- 25 remaining test subjects are clean held-out test set
- Run via: `bash scripts/training/train_production_models.sh` (from project root)
- Train command: `python scripts/training/train_one_model.py T1 --mode production --data-dir data/train --val-dir data/test/prepared --val-subjects data/val_subjects.txt --seed 42 --output-dir models/production`
- Models saved to `models/production/`

## Evaluation Metrics

Connected-component-based with binary dilation:
- **Sensitivity**: fraction of real markers detected
- **Precision**: fraction of predicted markers that are real
- Historical LOOCV performance: ~89% sensitivity, ~82% precision (individual models)

## Key Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/training/train_one_model.py` | Train one model | `python scripts/training/train_one_model.py T1 --mode production --seed 42 --data-dir data/train --val-dir data/test/prepared --val-subjects data/val_subjects.txt --output-dir models/production` |
| `scripts/training/train_production_models.sh` | Submit 4 production training jobs | `bash scripts/training/train_production_models.sh` |
| `scripts/inference/consensus_inference.py` | Consensus inference | `python scripts/inference/consensus_inference.py --models models/production/*.pth --input scan.nii --output results/` |
| `scripts/evaluation/ensemble_top3_evaluation_efficient.py` | Evaluate ensemble metrics | `python scripts/evaluation/ensemble_top3_evaluation_efficient.py --model_dir models/loocv --data-dir data/train` |
| `scripts/preprocessing/prepare_test_data.py` | Prepare test subjects end-to-end | `python scripts/preprocessing/prepare_test_data.py --dicom-dir data/test/dicoms --bids-dir data/test/bids --output-dir data/test/prepared` |
| `scripts/preprocessing/extract_rois.py` | Extract ROIs from RTSTRUCT | Imported by `prepare_test_data.py` |
| `scripts/preprocessing/homogeneity_correction.py` | N4 bias correction | `python scripts/preprocessing/homogeneity_correction.py --input scan.nii --output corrected.nii` |
| `scripts/preprocessing/create_combined_seeds.py` | Create seeds.nii.gz | `python scripts/preprocessing/create_combined_seeds.py --data-dir data/train` |

## Environment

- Conda env: `prostate39` (Python 3.9)
- Key packages: PyTorch (CUDA 12.1), torchio, nibabel, scipy, pandas, sklearn
- HPC: QRIS Bunya cluster, SLURM, A100 GPUs
- Account: `a_ai_collab`
- All scripts should be run from the project root (`data/2024-prostate/`)
