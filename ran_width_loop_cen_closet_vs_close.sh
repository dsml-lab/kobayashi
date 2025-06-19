#!/bin/bash

# Default arguments
SEED=42  # 固定のシード値
EPOCH=1000
DATASET="distribution_colored_emnist"
GRAY_SCALE=false
BATCH_SIZE=256
LR=0.01
OPTIMIZER="sgd"
MOMENTUM=0.0
LOSS="cross_entropy"
NUM_WORKERS=4
WANDB=true
WANDB_PROJECT="kobayashi_distance2"
WANDB_ENTITY="dsml-kernel24"

# 単一のモデル、パラメータ設定で実行
MODEL="cnn_5layers_cus"
LABEL_NOISE_RATE=0.2
WEIGHT_NOISY=1.0
WEIGHT_CLEAN=1.0
VAR=1000
GPU=1
MODE1="noise"
MODE2="no_noise"

# widthの値を配列で指定
WIDTH_VALUES=(10)  # 必要な幅の値を設定
N1_VALUES=(31)
N2=0

# widthごとにループ
for WIDTH in "${WIDTH_VALUES[@]}"; do
  echo "Running experiments with width=$WIDTH"
  # n1の値ごとにプログラムを実行
  for N1 in "${N1_VALUES[@]}"; do
    echo "Running program with width=$WIDTH, n1=$N1"
    python cen_closet_vs_close.py \
      --fix_seed $SEED \
      --model $MODEL \
      --model_width $WIDTH \
      --epoch $EPOCH \
      --dataset $DATASET \
      --target combined \
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
      --n2 $N2 \
      --n1 $N1 \
      --mode1 $MODE1 \
      --mode2 $MODE2
  done
done