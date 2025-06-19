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
BATCH=12
LR=0.0001
OPTIMIZER="adam"
MOMENTUM=0.0
LOSS="cross_entropy"
GPU=2
NUM_WORKERS=4
WANDB=true
WANDB_PROJECT="kobayashi_save_model"
WANDB_ENTITY="dsml-kernel24"
# Define command
cmd1="python dd_scratch_models_split_lossweight.py \
  --fix_seed $SEED \
  --model $MODEL \
  --model_width $MODEL_WIDTH \
  --epoch $EPOCH \
  --dataset $DATASET \
  --target combined \
  --label_noise_rate $LABEL_NOISE_RATE \
  --variance 100 \
  --correlation 0.5 \
  $(if [ \"$GRAY_SCALE\" = true ]; then echo '--gray_scale'; fi) \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --optimizer $OPTIMIZER \
  --momentum $MOMENTUM \
  --loss $LOSS \
  --num_workers $NUM_WORKERS \
  $(if [ \"$WANDB\" = true ]; then echo '--wandb'; fi) \
  --wandb_project $WANDB_PROJECT \
  --wandb_entity $WANDB_ENTITY \
  --weight_noisy $WEIGHT_NOISY \
  --weight_clean $WEIGHT_CLEAN"
# Run command
eval $cmd1
