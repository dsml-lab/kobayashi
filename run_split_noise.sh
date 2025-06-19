#!/bin/bash

# デフォルトの引数値を設定
SEED=42
MODEL_WIDTH=1
EPOCH=2000
DATASET="emnist_digits"
GRAY_SCALE=false
BATCH=256
LR=0.01
OPTIMIZER="sgd"
MOMENTUM=0.0
LOSS="cross_entropy"
NUM_WORKERS=4
WANDB=true
WANDB_PROJECT="dd_distribution_color_emnist_split_lossweight_all"
WANDB_ENTITY="dsml-kernel24"
MODEL="cnn_5layers"
LABEL_NOISE_RATE=0.5

# コマンドの定義
cmd1="python dd_scratch_models_for_spread_noise.py \
  --fix_seed $SEED \
  --model $MODEL \
  --model_width $MODEL_WIDTH \
  --epoch $EPOCH \
  --dataset $DATASET \
  --target combined \
  --label_noise_rate $LABEL_NOISE_RATE \
  --variance 100 \
  --correlation 0.5 \
  $(if [ "$GRAY_SCALE" = true ]; then echo '--gray_scale'; fi) \
  --batch $BATCH \
  --lr $LR \
  --optimizer $OPTIMIZER \
  --momentum $MOMENTUM \
  --loss $LOSS \
  --num_workers $NUM_WORKERS \
  $(if [ "$WANDB" = true ]; then echo '--wandb'; fi) \
  --wandb_project $WANDB_PROJECT \
  --wandb_entity $WANDB_ENTITY"

# コマンドの実行
eval $cmd1
