#!/bin/bash

# Default arguments
SEED=42
EPOCH=4000
DATASET="cifar10"
GRAY_SCALE=false
BATCH_SIZE=128
LR=0.0001
OPTIMIZER="adam"
MOMENTUM=0.0
LOSS="cross_entropy"
NUM_WORKERS=4
WANDB=true
WANDB_PROJECT="kobayashi_distance2"
WANDB_ENTITY="dsml-kernel24"

# 単一のモデル、パラメータ設定で実行
MODEL="resnet18k"
LABEL_NOISE_RATE=0.0
WEIGHT_NOISY=1.0
WEIGHT_CLEAN=1.0
VAR=0
GPU=0
MODE="no_noise"
DISTANCE_METRIC="euclidean"

# ここに複数のwidthを指定（必要に応じて変更）
#MODEL_WIDTH_LIST=(1 2 8 10)
MODEL_WIDTH_LIST=(64)
# widthごとにループ実行
for MODEL_WIDTH in "${MODEL_WIDTH_LIST[@]}"; do
  echo "[Running] MODEL_WIDTH=$MODEL_WIDTH"

  python alpha_interpolation.py \
    --fix_seed $SEED \
    --model $MODEL \
    --model_width $MODEL_WIDTH \
    --epoch $EPOCH \
    --dataset $DATASET \
    --target color \
    --label_noise_rate $LABEL_NOISE_RATE \
    --variance $VAR \
    --correlation 0.5 \
    $(if [ "$GRAY_SCALE" = true ]; then echo '--gray_scale'; fi) \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --optimizer $OPTIMIZER \
    --momentum $MOMENTUM \
    --loss $LOSS \
    --num_workers $NUM_WORKERS \
    --gpu $GPU \
    $(if [ "$WANDB" = true ]; then echo '--wandb'; fi) \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    --weight_noisy $WEIGHT_NOISY \
    --weight_clean $WEIGHT_CLEAN \
    --mode $MODE \
    --distance_metric $DISTANCE_METRIC

  echo "[✓ Done] MODEL_WIDTH=$MODEL_WIDTH"
  echo "------------------------------------"
done
