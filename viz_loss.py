import os
import pandas as pd
import matplotlib.pyplot as plt
import re

# ルートディレクトリ（必要に応じてパスを調整）
root_dir = 'save_model/cifar10/noise_0.2/'
output_dir = os.path.join('./fig_loss')
os.makedirs(output_dir, exist_ok=True)

# 幅の値を抽出する正規表現
pattern = re.compile(r'seed_42width(\d+)_resnet18k_cifar10')

# ディレクトリ内のサブフォルダを走査
for subdir in os.listdir(root_dir):
    match = pattern.match(subdir)
    if match:
        width = match.group(1)
        csv_path = os.path.join(root_dir, subdir, 'csv', 'training_metrics.csv')
        if os.path.exists(csv_path):
            # データ読み込み
            df = pd.read_csv(csv_path)

            # 描画
            plt.figure()
            plt.plot(df['epoch'], df['avg_loss_noisy'], label='avg_loss_noisy')
            plt.plot(df['epoch'], df['avg_loss_clean'], label='avg_loss_clean')
            plt.xscale('log')
            plt.xlabel('Epoch (log scale)')
            plt.ylabel('Average Loss')
            plt.ylim(-0.01,0.14)
            plt.title(f'Average Loss (width={width})')
            plt.legend()
            plt.grid(True, which="both", ls="--")

            # 保存
            save_path = os.path.join(output_dir, f'avg_loss_width{width}.png')
            plt.savefig(save_path)
            plt.close()

print("保存完了しました。")
