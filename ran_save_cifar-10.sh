#!/bin/bash

# デフォルトの引数値を設定
SEED=42
MODEL="resnet18k"
EPOCH=2000
DATASET="cifar10"
GRAY_SCALE=false
BATCH=128
LR=0.0001
OPTIMIZER="adam"
MOMENTUM=0.0
LOSS="cross_entropy"
GPU=3
NUM_WORKERS=4
WANDB=true
WANDB_PROJECT="kobayashi_emnist"
WANDB_ENTITY="dsml-kernel24"
USE_SAVE_DATA=false

# ループさせたい model_width の値を配列で指定
model_width_list=(1)

# 固定の label_noise_rate
label_noise_rate=0.2

for model_width in "${model_width_list[@]}"; do
  python dd_scracth_cifar-10_save.py \
    --fix_seed $SEED \
    --model $MODEL \
    --model_width $model_width \
    --epoch $EPOCH \
    --dataset $DATASET \
    --label_noise_rate $label_noise_rate \
    $(if [ "$GRAY_SCALE" = true ]; then echo "--gray_scale"; fi) \
    --batch $BATCH \
    --lr $LR \
    --optimizer $OPTIMIZER \
    --momentum $MOMENTUM \
    --loss $LOSS \
    --gpu $GPU \
    --num_workers $NUM_WORKERS \
    $(if [ "$WANDB" = true ]; then echo "--wandb"; fi) \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY\
    --use_saved_data $USE_SAVE_DATA\

done
