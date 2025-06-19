#!/bin/bash

#シード値のリスト
#SEEDS=(42 43 44 45 46 47)
SEEDS=(47 48 49 50 51)

#SEEDS=(42 43)
#SEEDS=(44 45)
#SEEDS=(46 47 48)
#SEEDS=(49 50 51 )
#SEEDS=(47)
#SEEDS=(52 53 54 55 56)
#SEEDS=(57 58 59 60 61)

#SEEDS=(51 52)
#SEEDS=(52 53)
#その他の固定引数
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
WANDB=false
WANDB_PROJECT="kobayashi_save_model"
WANDB_ENTITY="dsml-kernel24"
USE_SAVE_DATA=true  # 既に保存されたデータを使用するためにtrueに設定
WEIGHT_NOISY=1.0
WEIGHT_CLEAN=1.0
VAR=1000
GPU=3
# 単一のモデル、パラメータ設定で実行
MODEL="cnn_5layers"
LABEL_NOISE_RATE=0.0

# 各シード値でループ
for SEED in "${SEEDS[@]}"
do
  echo "---------------------------------------"
  echo "Running with SEED=${SEED}"
  echo "---------------------------------------"

  # 実行するコマンド
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
    --weight_clean $WEIGHT_CLEAN \
  # 実行後の処理（必要に応じて追加）
  # 例: ログの保存、エラーハンドリングなど
done

echo "All runs completed."
