#!/bin/bash

# argsの設定

# シード値
fix_seed=42

# 分散と相関の設定
variance=10000
correlation=0.5

# データセットのパラメータ
DATASET="distribution_colored_emnist"
TARGET="combined"
GRAY_SCALE=false

# Pythonプログラムに引数を渡す
python pca.py \
  --fix_seed $fix_seed \
  --variance $variance \
  --correlation $correlation \
  --dataset $DATASET \
  --target $TARGET \
  $(if [ "$GRAY_SCALE" = true ]; then echo "--gray_scale"; fi)
