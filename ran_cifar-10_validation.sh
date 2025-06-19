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

# GPUについては、複数使用する場合はカンマ区切りで指定（例: GPU0とGPU1を使用）
GPUS="0,1"

NUM_WORKERS=4
WANDB=true
WANDB_PROJECT="kobayashi_emnist"
WANDB_ENTITY="dsml-kernel24"

# 固定の label_noise_rate
label_noise_rate=0.2

# Pythonスクリプトの実行（スクリプト名は実際のファイル名に合わせてください）
python dd_cifar-10_validation.py \
    --fix_seed ${SEED} \
    --model ${MODEL} \
    --epoch ${EPOCH} \
    --dataset ${DATASET} \
    --gray_scale ${GRAY_SCALE} \
    --batch_size ${BATCH} \
    --lr ${LR} \
    --optimizer ${OPTIMIZER} \
    --momentum ${MOMENTUM} \
    --loss ${LOSS} \
    --num_workers ${NUM_WORKERS} \
    --wandb ${WANDB} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_entity ${WANDB_ENTITY} \
    --label_noise_rate ${label_noise_rate} \
    --gpus ${GPUS}
