#!/bin/bash

# Default arguments
SEED=42
MODEL_WIDTH=1
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
LABEL_NOISE_RATE=0.0
WEIGHT_NOISY=1.0
WEIGHT_CLEAN=1.0
VAR=1000
GPU=2
MODE1="no_noise"
MODE2="no_noise"
# n2の値を配列で指定
#N1_VALUES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 )
N1_VALUES=(51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100)
#N1_VALUES=(31 32 33 34 35 36 37 38 39 41 42 43 44 45 46 47 48 49 51 52 53 54 55 56 57 58 59 61 62 63 64 65 66 67 68 69 71 72 73 74 75 76 77 78 79 81 82 83 84 85 86 87 88 89 91 92 93 94 95 96 97 98 99)
#N1_VALUES=(11 12 13 14 15 16 17 18 19 21 22 23 24 25 26 27 28 29)
#N1_VALUES=(4 5 6 7 )
#N1_VALUES=(8 9 10 20)
#N1_VALUES=(30 40 50 60)
#N1_VALUES=(25 67 80 39)
N2=0
# n2の値ごとにプログラムを実行
for N1 in "${N1_VALUES[@]}"; do
  echo "Running program with n1=$N1"
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
    --n2 $N2\
    --n1 $N1\
    --mode1 $MODE1\
    --mode2 $MODE2
done
