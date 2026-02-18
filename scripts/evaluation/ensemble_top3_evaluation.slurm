#!/usr/bin/env bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=a_barth
#SBATCH --export=PATH,TERM,HOME,LANG
#SBATCH --output=ensemble_eval_%j.out
#SBATCH --error=ensemble_eval_%j.err
#SBATCH --job-name=ensemble_eval

echo "Starting ensemble evaluation on $(hostname) at $(date)"

# Set the location of Miniconda
export CONDA_HOME=$HOME/miniconda3
echo "Using Miniconda from $CONDA_HOME"

# Source the conda profile script
source "$CONDA_HOME/etc/profile.d/conda.sh"
echo "Sourced $CONDA_HOME/etc/profile.d/conda.sh"

# Update the library path for the environment
export LD_LIBRARY_PATH="$CONDA_HOME/envs/prostate/lib:$LD_LIBRARY_PATH"
echo "Updated LD_LIBRARY_PATH to $LD_LIBRARY_PATH"

# Activate the conda environment
echo "Activating conda environment"
conda activate prostate39

# Check the Python path
echo "Python path is $(which python)"

# Print the Python executable and version
echo "Python version is $(python --version)"

# Run the ensemble evaluation (will use CPU)
echo "Starting ensemble evaluation on CPU..."
python ensemble_top3_evaluation_efficient.py --num_models 3 --threshold 0.3

echo "Ensemble evaluation completed at $(date)"
