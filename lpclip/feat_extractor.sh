# sh feat_extractor.sh
DATA="/home/jingchen/promtsrc/data/"
OUTPUT='./clip_feat/'
SEED=1

# oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet
for DATASET in imagenet
do
    for SPLIT in train
    do
        python feat_extractor.py \
        --split ${SPLIT} \
        --root ${DATA} \
        --seed ${SEED} \
        --dataset-config-file ../configs/datasets/${DATASET}.yaml \
        --config-file ../configs/trainers/CoOp/vit_b16.yaml \
        --output-dir ${OUTPUT} \
        --eval-only
    done
done
