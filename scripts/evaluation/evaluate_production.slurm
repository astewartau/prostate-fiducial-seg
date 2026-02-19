#!/usr/bin/env bash
#SBATCH --partition=gpu_cuda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=a_ai_collab
#SBATCH --output=logs/eval_prod_%j.out
#SBATCH --error=logs/eval_prod_%j.err

export PYTHONUNBUFFERED=1

# Set up conda
export CONDA_HOME=$HOME/miniconda3
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate prostate39

echo "Starting evaluation on $(hostname) at $(date)"
echo "Python: $(which python)"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

cd "$SLURM_SUBMIT_DIR"

python scripts/evaluation/evaluate_production_models.py \
    --model-dir models/production \
    --data-dir data/test/prepared \
    --splits data/splits.json \
    --subset test \
    --output results/production_test_eval.csv

echo "Finished at $(date)"
