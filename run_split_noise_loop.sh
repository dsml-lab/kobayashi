#!/bin/bash

# デフォルトの引数値を設定
SEEDS=(123 456 789 314)
MODEL_WIDTH=1
EPOCH=1000
DATASET="small_emnist_digits"
GRAY_SCALE=false
BATCH=256
LR=0.01
OPTIMIZER="sgd"
MOMENTUM=0.0
LOSS="cross_entropy"
NUM_WORKERS=4
WANDB=true
WANDB_PROJECT="dd_small_emnist_digits_split_seeds"
WANDB_ENTITY="dsml-kernel24"
LABEL_NOISE_RATE=0.4  # ラベルノイズ率のリスト
CORRELATION=0.5  # 固定値
models=("cnn_2layers" "cnn_5layers") # モデルのリスト

# すべての組み合わせを保存する配列
cmds=()

# ループで各組み合わせを生成
for SEED in "${SEEDS[@]}"; do
    for MODEL in "${models[@]}"; do
        cmd="python dd_small_dataset.py \
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
        cmds+=("$cmd")
    done
done

# 順番にコマンドを実行
for cmd in "${cmds[@]}"; do
    echo "Executing: $cmd"
    eval "$cmd"
done
