#!/bin/bash

# デフォルトの引数値を設定
SEED=42
MODEL_WIDTH=1
EPOCH=1000
DATASET="distribution_colored_emnist" # "mnist", "emnist", "cifar10", "cifar100", "tinyImageNet", "colored_emnist", "distribution_colore"
GRAY_SCALE=false
BATCH=256
LR=0.01
OPTIMIZER="sgd"
MOMENTUM=0.0
LOSS="cross_entropy"
NUM_WORKERS=4
WANDB=true
WANDB_PROJECT="dd_distribution_color_emnist"
WANDB_ENTITY="dsml-kernel24"

# model, target, and label_noise_rateのリスト
models=("cnn_2layers" "cnn_5layers" "resnet18")
targets=("combined") 
label_noise_rates=(0 0.1 0.2 0.4 0.8)
gpus=(0 1 2 3)
variances=("100" "1000" "10000" "100000")
CORRELATION=0.5

# すべての組み合わせを保存する配列
cmds=()

# GPUインデックスの初期化
gpu_index=0

# ループで各組み合わせを生成
for model in "${models[@]}"; do
  for target in "${targets[@]}"; do
    for label_noise_rate in "${label_noise_rates[@]}"; do
      for variance in "${variances[@]}"; do
        # GPUをラウンドロビンで割り当て
        gpu=${gpus[$gpu_index]}
        cmd="python dd_scratch_models_combine.py \
          --fix_seed $SEED \
          --model $model \
          --model_width $MODEL_WIDTH \
          --epoch $EPOCH \
          --dataset $DATASET \
          --target $target \
          --label_noise_rate $label_noise_rate \
          --variance $variance \
          --correlation $CORRELATION \
          $(if [ "$GRAY_SCALE" = true ]; then echo '--gray_scale'; fi) \
          --batch $BATCH \
          --lr $LR \
          --optimizer $OPTIMIZER \
          --momentum $MOMENTUM \
          --loss $LOSS \
          --gpu $gpu \
          --num_workers $NUM_WORKERS \
          $(if [ "$WANDB" = true ]; then echo '--wandb'; fi) \
          --wandb_project $WANDB_PROJECT \
          --wandb_entity $WANDB_ENTITY"
        cmds+=("$cmd")

        # 次のGPUに移動
        gpu_index=$(( (gpu_index + 1) % ${#gpus[@]} ))
      done
    done
  done
done

# Parallel
parallel --jobs 32 ::: "${cmds[@]}"