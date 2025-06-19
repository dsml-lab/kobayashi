#!/bin/bash

# デフォルトの引数値を設定
SEED=42
MODEL="resnet18"  # "cnn_2layers", "cnn_5layers", "resnet18"
MODEL_WIDTH=1
EPOCH=1000
DATASET="colored_emnist" # "mnist", "emnist", "cifar10", "cifar100", "tinyImageNet", "colored_emnist"
GRAY_SCALE=false
BATCH=256
LR=0.0001
OPTIMIZER="adam"
MOMENTUM=0.9
LOSS="cross_entropy"
GPU=0
NUM_WORKERS=4
WANDB=true
WANDB_PROJECT="dd_color_emnist"
WANDB_ENTITY="dsml-kernel24"

# targetとlabel_noise_rateのリスト
targets=("digit" "color" "combined")
label_noise_rates=(0 0.1 0.2 0.4 0.8)

# ループで各組み合わせを実行
for target in "${targets[@]}"; do
  for label_noise_rate in "${label_noise_rates[@]}"; do
    python dd_scratch_models.py \
      --fix_seed $SEED \
      --model $MODEL \
      --model_width $MODEL_WIDTH \
      --epoch $EPOCH \
      --dataset $DATASET \
      --target $target \
      --label_noise_rate $label_noise_rate \
      $(if [ "$GRAY_SCALE" = true ]; then echo "--gray_scale"; fi) \
      --batch $BATCH \
      --lr $LR \
      --optimizer $OPTIMIZER \
      --momentum $MOMENTUM \
      --loss $LOSS \
      --gpu $GPU \
      --num_workers $NUM_WORKERS \
      $(if [ "$WANDB" = true ]; then echo "--wandb"; fi) \
      --wandb_project $WANDB_PROJECT \
      --wandb_entity $WANDB_ENTITY
  done
done
