#!/bin/bash

# Default arguments
SEED=42
MODEL_WIDTH=1
EPOCH=2000
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
LABEL_NOISE_RATE=0.1
WEIGHT_NOISY=1.0
WEIGHT_CLEAN=1.0
VAR=1000
GPU=1

MODE1="no_noise"
MODE2="no_noise"
# n2の値を配列で指定
#N2_VALUES=(10 20 30 40 50 60 70 80 90)
N2_VALUES=(2 3 4 5 6 7 8 9 10 20 30 40 50 60 70)
N1=0
# n2の値ごとにプログラムを実行
for N2 in "${N2_VALUES[@]}"; do
  echo "Running program with n2=$N2"
  python dd_scrach_center_distance_save.py \
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
    --n2 $N2\
    --n1 $N1\
    --mode1 $MODE1\
    --mode2 $MODE2
done
