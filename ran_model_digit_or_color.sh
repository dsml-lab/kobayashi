#!/bin/bash

# Default arguments
SEED=42
MODEL_WIDTH=1
EPOCH=2000
DATASET="colored_emnist"
GRAY_SCALE=false
BATCH_SIZE=256
LR=0.01
OPTIMIZER="sgd"
MOMENTUM=0.0
LOSS="cross_entropy"
NUM_WORKERS=4
WANDB=true
WANDB_PROJECT="color_or_digit_only"
WANDB_ENTITY="dsml-kernel24"

# 単一のモデル、パラメータ設定で実行
MODEL="cnn_5layers"
LABEL_NOISE_RATE=0.0
WEIGHT_NOISY=1.0
WEIGHT_CLEAN=1.0
GPU=0

# ループするVARの値を定義（必要に応じて値を変更してください）
VAR_VALUES=(0)

for VAR in "${VAR_VALUES[@]}"; do
  echo "Running with VAR=${VAR}"

  # コマンドを実行
  python dd_scratch_models_color_or_digit.py \
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
    --gpu $GPU \
    --num_workers $NUM_WORKERS \
    $(if [ "$WANDB" = true ]; then echo '--wandb'; fi) \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    --weight_noisy $WEIGHT_NOISY \
    --weight_clean $WEIGHT_CLEAN
done
