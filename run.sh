#!/bin/bash

# デフォルトの引数値を設定
SEED=42
MODEL="resnet18"
MODEL_WIDTH=1
EPOCH=1000
DATASET="colored_emnist"
LABEL_NOISE_RATE=0.0
GRAY_SCALE=false
TARGET="combined"
BATCH=128
LR=0.0001
OPTIMIZER="adam"
MOMENTUM=0.0
LOSS="cross_entropy"
GPU=2
NUM_WORKERS=4
WANDB=true
WANDB_PROJECT="dd_scratch_models_2"
WANDB_ENTITY="dsml-kernel24"

# 引数を使ってPythonファイルを実行
python dd_scratch_models_2.py \
  --fix_seed $SEED \
  --model $MODEL \
  --model_width $MODEL_WIDTH \
  --epoch $EPOCH \
  --dataset $DATASET \
  --target $TARGET \
  --label_noise_rate $LABEL_NOISE_RATE \
  $(if $GRAY_SCALE; then echo "--gray_scale"; fi) \
  --batch $BATCH \
  --lr $LR \
  --optimizer $OPTIMIZER \
  --momentum $MOMENTUM \
  --loss $LOSS \
  --gpu $GPU \
  --num_workers $NUM_WORKERS \
  $(if $WANDB; then echo "--wandb"; fi) \
  --wandb_project $WANDB_PROJECT \
  --wandb_entity $WANDB_ENTITY
