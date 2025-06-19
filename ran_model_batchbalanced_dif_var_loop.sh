#!/bin/bash

# Default arguments
SEED=42
MODEL_WIDTH=1
EPOCH=2000
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

# train_varとtest_varの候補
train_var_list=(31622 10000 3162 1000 100 0)
test_var_list=(31622 10000 3162 1000 100 0)

# 使用可能なGPUのリスト
gpu_list=(1 3)

# 何ジョブ同時に実行するか(=上記gpu_listの数)
max_parallel=2

count=0

for TRAIN_VAR in "${train_var_list[@]}"; do
  for TEST_VAR in "${test_var_list[@]}"; do
    
    # GPU割り当て (count に応じて 1 or 3)
    GPU=${gpu_list[$((count % max_parallel))]}

    echo "==== Train var: $TRAIN_VAR, Test var: $TEST_VAR, GPU: $GPU ===="

    python dd_scratch_models_split_lossweight_batchblanced_dif_var.py \
      --fix_seed "$SEED" \
      --model "$MODEL" \
      --model_width "$MODEL_WIDTH" \
      --epoch "$EPOCH" \
      --dataset "$DATASET" \
      --target combined \
      --label_noise_rate "$LABEL_NOISE_RATE" \
      --train_variance "$TRAIN_VAR" \
      --test_variance "$TEST_VAR" \
      --correlation 0.5 \
      $(if [ "$GRAY_SCALE" = true ]; then echo '--gray_scale'; fi) \
      --batch_size "$BATCH_SIZE" \
      --lr "$LR" \
      --optimizer "$OPTIMIZER" \
      --momentum "$MOMENTUM" \
      --loss "$LOSS" \
      --gpu "$GPU" \
      --num_workers "$NUM_WORKERS" \
      $(if [ "$WANDB" = true ]; then echo '--wandb'; fi) \
      --wandb_project "$WANDB_PROJECT" \
      --wandb_entity "$WANDB_ENTITY" \
      --weight_noisy "$WEIGHT_NOISY" \
      --weight_clean "$WEIGHT_CLEAN" &

    ((count++))

    # 2つ目のジョブを投げたらここで一旦待つ
    if [ $((count % max_parallel)) -eq 0 ]; then
      wait
    fi

  done
done

# 最後に投げたジョブが残っていれば、ここで終了を待つ
wait
