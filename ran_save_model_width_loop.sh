#!/bin/bash

# Default arguments
SEED=42
EPOCH=1000
DATASET="distribution_colored_emnist"
GRAY_SCALE=false
BATCH_SIZE=256
LR=0.01
OPTIMIZER="sgd"
MOMENTUM=0.0
LOSS="cross_entropy"
NUM_WORKERS=4
WANDB=false
WANDB_PROJECT="kobayashi_save_model"
WANDB_ENTITY="dsml-kernel24"
USE_SAVE_DATA=true
MODEL="cnn_5layers_cus"
LABEL_NOISE_RATE=0.2
WEIGHT_NOISY=1.0
WEIGHT_CLEAN=1.0
VAR=1000
GPU=2

# MODEL_WIDTH の値を設定（例: 1, 2, 4, 8）
MODEL_WIDTH_VALUES=(1)

# ループして各 MODEL_WIDTH で実行
for MODEL_WIDTH in "${MODEL_WIDTH_VALUES[@]}"; do
  echo "Running with MODEL_WIDTH=$MODEL_WIDTH"
  python dd_scratch_model_save_batch_shufle.py \
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
    --use_saved_data $USE_SAVE_DATA \
    --weight_clean $WEIGHT_CLEAN
  echo "Finished running MODEL_WIDTH=$MODEL_WIDTH"
done
