#!/bin/bash

# Default arguments
SEED=42
MODEL_WIDTH=1
EPOCH=2000
DATASET="colored_emnist"
GRAY_SCALE=false
BATCH_SIZE=256  # Changed variable name from BATCH to BATCH_SIZE
LR=0.01
OPTIMIZER="sgd"
MOMENTUM=0.0
LOSS="cross_entropy"
NUM_WORKERS=4
WANDB=true
WANDB_PROJECT="dd_distribution_color_emnist_split_lossweight_batchbalanced_color_distribute"
WANDB_ENTITY="dsml-kernel24"
# MODEL="cnn_5layers"

# パラメータの組み合わせ
MODELS=("cnn_5layers")
LABEL_NOISE_RATES=(0.2)
WEIGHT_NOISY_LIST=(1.0)
WEIGHT_CLEAN_LIST=(1.0)
VARIANCE=(0)

# コマンドの配列を初期化
commands=()

# 全てのパラメータの組み合わせを生成
for MODEL in "${MODELS[@]}"; do
  for LABEL_NOISE_RATE in "${LABEL_NOISE_RATES[@]}"; do
    for WEIGHT_NOISY in "${WEIGHT_NOISY_LIST[@]}"; do
      for WEIGHT_CLEAN in "${WEIGHT_CLEAN_LIST[@]}"; do
        for VAR in "${VARIANCE[@]}"; do
          cmd="python dd_scratch_models_split_lossweight_batchblanced.py \
            --fix_seed $SEED \
            --model $MODEL \
            --model_width $MODEL_WIDTH \
            --epoch $EPOCH \
            --dataset $DATASET \
            --target combined \
            --label_noise_rate $LABEL_NOISE_RATE \
            --variance $VAR \
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
          commands+=("$cmd")
        done
      done
    done
  done
done

# コマンドを並列実行
printf "%s\n" "${commands[@]}" | parallel -j 8
