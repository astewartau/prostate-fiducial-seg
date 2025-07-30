#!/bin/bash

for MODEL_NAME in T1; do
    for FOLD_ID in {0..1}; do # 59
        echo $MODEL_NAME-$FOLD_ID
        sbatch --job-name="${MODEL_NAME}-${FOLD_ID}" --export=ALL,MODEL_NAME=$MODEL_NAME,FOLD_ID=$FOLD_ID train_one_model.slurm
    done
done

