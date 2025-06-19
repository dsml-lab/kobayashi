#!/bin/bash

# デフォルトの引数値を設定
SEED=42
MODEL="resnet18k"
MODEL_WIDTH=1
EPOCH=2000
DATASET="cifar10"
LABEL_NOISE_RATE=0.2
BATCH_SIZE=128
#LR=0.0001
LR=0.01
OPTIMIZER="adam"
#OPTIMIZER="adam"
MOMENTUM=0.0
LOSS="cross_entropy"
GPU=3
NUM_WORKERS=4
WANDB=true
WANDB_PROJECT="kobayashi_save_model"
WANDB_ENTITY="dsml-kernel24"

# 引数を使ってPythonファイルを実行
python dd_cifar-10.py \
  --fix_seed $SEED \
  --model $MODEL \
  --model_width $MODEL_WIDTH \
  --epoch $EPOCH \
  --dataset $DATASET \
  --label_noise_rate $LABEL_NOISE_RATE \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --optimizer $OPTIMIZER \
  --momentum $MOMENTUM \
  --loss $LOSS \
  --gpu $GPU \
  --num_workers $NUM_WORKERS \
  $(if $WANDB; then echo "--wandb"; fi) \
  --wandb_project $WANDB_PROJECT \
  --wandb_entity $WANDB_ENTITY