#!/bin/bash

# 実験設定（必要に応じて変更）
SEED=42
MODEL="resnet18k"
MODEL_WIDTH=4
DATASET="distribution_colored_emnist"
LR=0.01
BATCH_SIZE=256
EPOCH=160
LABEL_NOISE_RATE=0.2
OPTIMIZER="sgd"
MOMENTUM=0.0
GPU=0
WEIGHT_NOISY=1.0
WEIGHT_CLEAN=1.0
NUM_WORKERS=4
LOSS="cross_entropy"
USE_SAVED_DATA=false

#-----------------カラー用-----------
VARIANCE=0
CORR=0.5
#-----------------
WANDB=true
WANDB_PROJECT="kobayashi_save_model"
WANDB_ENTITY="dsml-kernel24"

# 引数を使ってPythonファイルを実行
python dd_scratch_model_save.py \
  --fix_seed $SEED \
  --model $MODEL \
  --model_width $MODEL_WIDTH \
  --epoch $EPOCH \
  --dataset $DATASET \
  --target "combined" \
  --label_noise_rate $LABEL_NOISE_RATE \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --optimizer $OPTIMIZER \
  --momentum $MOMENTUM \
  --loss $LOSS \
  --gpu $GPU \
  --num_workers $NUM_WORKERS \
  $(if $WANDB; then echo "--wandb"; fi) \
  $(if [ "$GRAY_SCALE" = true ]; then echo '--gray_scale'; fi) \
  --wandb_project $WANDB_PROJECT \
  --wandb_entity $WANDB_ENTITY

# SEED=42
# MODEL="resnet18k"
# MODEL_WIDTH=64
# DATASET="cifar10"
# LR=0.0001
# BATCH_SIZE=128
# EPOCH=4000
# LABEL_NOISE_RATE=0.2
# OPTIMIZER="adam"
# MOMENTUM=0.0
# GPU=0
# WEIGHT_NOISY=1.0
# WEIGHT_CLEAN=1.0
# NUM_WORKERS=4
# LOSS="cross_entropy"
# USE_SAVED_DATA=false
# #-----------------カラー用-----------
# VARIANCE=0
# CORR=0.5
# #-----------------
# WANDB=true
# WANDB_PROJECT="kobayashi_save_model"
# WANDB_ENTITY="dsml-kernel24"