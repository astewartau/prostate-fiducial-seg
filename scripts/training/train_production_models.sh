#!/bin/bash
# Train 4 production models with different random seeds.
# All models train on all 93 training subjects and validate on 9 held-out subjects.
# Run from project root: bash scripts/training/train_production_models.sh

mkdir -p models/production logs

for SEED in 42 123 456 789; do
    echo "Submitting production model with seed=${SEED}"
    sbatch --job-name="prod-seed${SEED}" \
           --export=ALL,SEED=$SEED \
           scripts/training/train_production.slurm
done
