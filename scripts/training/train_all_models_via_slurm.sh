#!/bin/bash

for MODEL_NAME in T1; do
    for FOLD_ID in {0..92}; do # 94 subjects total (LOOCV)
        echo $MODEL_NAME-$FOLD_ID
        sbatch --job-name="${MODEL_NAME}-${FOLD_ID}" --export=ALL,MODEL_NAME=$MODEL_NAME,FOLD_ID=$FOLD_ID scripts/training/train_one_model.slurm
    done
done

