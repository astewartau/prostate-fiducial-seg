#!/usr/bin/env bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --account=a_ai_collab
#SBATCH --job-name=prep-test
#SBATCH --output=logs/prep_test_%j.out
#SBATCH --error=logs/prep_test_%j.err

export PYTHONUNBUFFERED=1

echo "Starting test data preparation on $(hostname) at $(date)"

# Set up conda
export CONDA_HOME=$HOME/miniconda3
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate prostate39

echo "Python: $(which python) ($(python --version))"

cd "$SLURM_SUBMIT_DIR"

python scripts/preprocessing/prepare_test_data.py \
    --dicom-dir data/test/dicoms \
    --bids-dir data/test/bids \
    --output-dir data/test/prepared

echo "Finished at $(date)"
