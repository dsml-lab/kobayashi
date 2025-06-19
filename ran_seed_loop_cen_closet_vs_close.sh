#!/bin/bash

# Default arguments
MODEL_WIDTH=4
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
MODEL="cnn_5layers"
# noise 0.0 or 0.2 or 0.5 
LABEL_NOISE_RATE=0.2
WEIGHT_NOISY=1.0
WEIGHT_CLEAN=1.0
VAR=0
GPU=2
MODE1="noise"
MODE2="no_noise"
# SEED値の配列

#SEED_VALUES=(42)
#SEED_VALUES=(45)

#SEED_VALUES=(43 44 45 46)
SEED_VALUES=(47 48 49 50 51)

#SEED_VALUES=(43 44 44)
#SEED_VALUES=(46 47)
#SEED_VALUES=(49 50)
#SEED_VALUES=(48 51)

#SEED_VALUES=(47 48 49 50 51)
#SEED_VALUES=(48 49 50 51)

#N1_VALUES=(62) #C_C 83
#N1_VALUES=(25 67 80 39) #C_C
N1_VALUES=(41 59 63)
#N1_VALUES=(36 39)
N2=0

# SEEDごとにループ
for SEED in "${SEED_VALUES[@]}"; do
  echo "Running experiments with seed=$SEED"
  # n1の値ごとにプログラムを実行
  for N1 in "${N1_VALUES[@]}"; do
    echo "Running program with seed=$SEED, n1=$N1"
    python cen_closet_vs_close.py \
      --fix_seed $SEED \
      --model $MODEL \
      --model_width $MODEL_WIDTH \
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
