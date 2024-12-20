#!/bin/bash


# custom config
DATA="data/"
TRAINER=PromptSRC

DATASET=$1
SEED=$2

CFG=vit_b16_c2_ep20_batch4_4+4ctx_cross_datasets-B128
SHOTS=16

for SEED in 1 2 3
    do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}."
    else
        echo "Run this job and save the output to ${DIR}"

        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done