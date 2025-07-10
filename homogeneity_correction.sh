#!/usr/bin/env bash
#SBATCH --job-name=t1w_homogeneity
#SBATCH --array=0-60           # Adjust to match subject count - 1
#SBATCH --output=t1w_homogeneity_%A_%a.out
#SBATCH --error=t1w_homogeneity_%A_%a.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load software modules
source ~/.bashrc

# Load necessary modules 
conda activate prostate39

cd /home/uqaste15/data/2024-prostate/

python homogeneity_correction.py $SLURM_ARRAY_TASK_ID
