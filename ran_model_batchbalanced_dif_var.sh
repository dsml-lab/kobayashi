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
WANDB_PROJECT="kobayshi_color_mnist"
WANDB_ENTITY="dsml-kernel24"

# 単一のモデル、パラメータ設定で実行
MODEL="cnn_5layers"
LABEL_NOISE_RATE=0.0
WEIGHT_NOISY=1.0
WEIGHT_CLEAN=1.0
GPU=0
TRAIN_VAR=1000
TEST_VAR=0
# コマンドを実行
python dd_scratch_models_split_lossweight_batchblanced_dif_var.py \
  --fix_seed $SEED \
  --model $MODEL \
  --model_width $MODEL_WIDTH \
  --epoch $EPOCH \
  --dataset $DATASET \
  --target combined \
  --label_noise_rate $LABEL_NOISE_RATE \
  --train_variance $TRAIN_VAR \
  --test_variance $TEST_VAR\
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