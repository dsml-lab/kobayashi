import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSVファイルの読み込み
file_path = "/workspace/save_model/Colored_EMSNIT/noise_rate=0_sigma=0/seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/csv/training_metrics.csv"
df = pd.read_csv(file_path)

# フォント設定
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 20
#plt.rcParams["figure.figsize"] = (13, 10)
# データの抽出
epochs = df["epoch"]

# エラー率（割合）への変換
test_error = (100 - df.iloc[:, 15]) / 100
train_error = (100 - df.iloc[:, 5]) / 100
train_error_noisy = (100 - df.iloc[:, 6]) / 100
train_error_clean = (100 - df.iloc[:, 7]) / 100

# Lossの抽出
train_loss = df["train_loss"]
test_loss = df["test_loss"]

# プロットの作成：2行1列のサブプロット
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 11))

#【上段】エラー率（割合）のプロット
ax1.plot(epochs, test_error, label="test", color="red", linestyle="-", linewidth=3)
ax1.plot(epochs, train_error, label="train", color="blue", linestyle="-", linewidth=3)
# ax1.plot(epochs, train_error_noisy, label="train noisy", color="darkblue", linestyle="-", linewidth=3)
# ax1.plot(epochs, train_error_clean, label="train clean", color="cyan", linestyle="-", linewidth=3)
ax1.legend(loc="upper right", fontsize=30)
ax1.set_xscale("log")
ax1.set_xticklabels([])
ax1.set_ylabel("error", fontsize=30)
ax1.yaxis.grid(True, which="both", linestyle="--", linewidth=0.3)
ax1.set_yticks([0.0, 0.2, 0.4, 0.6,0.8, 1.0])

# 縦線の追加（必要に応じて）
vertical_epochs = [1,30,60,140,1000]
for epoch in vertical_epochs:
    ax1.axvline(x=epoch, color="black", linestyle="-", linewidth=0.9, zorder=0)

ax1.tick_params(axis="both", labelsize=30)

#【下段】Train Clean / Noisy Error（割合）のプロット
ax2.plot(epochs, train_error_noisy, label="train noisy", color="darkblue", linestyle="-", linewidth=3)
ax2.plot(epochs, train_error_clean, label="train clean", color="cyan", linestyle="-", linewidth=3)
ax2.set_yticks([0.0, 0.2, 0.4, 0.6,0.8, 1.0])

ax2.set_xscale("log")
ax2.set_xlabel("epoch", fontsize=30)
ax2.set_ylabel("error", fontsize=30)
ax2.yaxis.grid(True, which="both", linestyle="--", linewidth=0.3)
ax2.legend(loc="upper right", fontsize=30)

# 縦線の追加（必要に応じて）
for epoch in vertical_epochs:
    ax2.axvline(x=epoch, color="black", linestyle="-", linewidth=0.9, zorder=0)

ax2.tick_params(axis="both", labelsize=30)

# レイアウト調整と保存
plt.tight_layout()
plt.savefig("/workspace/miru_vizualize/colored_emnist_learning_curve.svg", format='svg')
print("workspace/miru_vizualize/colored_emnist_learning_curve.svg")
