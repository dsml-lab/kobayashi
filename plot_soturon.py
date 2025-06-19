# plot_label_ratios_smoothed.py
import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
from smoothing import moving_average  # smoothing.pyから関数をインポート

# ===== スムージングを行うかどうかを設定 =====
use_smoothing =  True # Trueでスムージングを行う, Falseで行わない

# ===== ファイルやパラメータの設定 =====
file_dir = "alpha_test/closet_cnn_5layers_distribution_colored_emnist_variance1000_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/6/4"
file_directory = os.path.join(file_dir, "csv")
file_pattern = os.path.join(file_directory, 'alpha_log_epoch_*.csv')
epoch_pattern = re.compile(r'alpha_log_epoch_(\d+)\.csv')

label_column = 'predicted_color'  # ラベルを判定するカラム名
label1 = 6  # 例: 数字の0
label2 = 4  # 例: 数字の2
window_size = 5  # スムージング用のウィンドウサイズ（必要に応じて調整）

# ===== 1. ファイルの読み込みと統合 =====
file_paths = glob.glob(file_pattern)
df_list = []

for file in file_paths:
    match = epoch_pattern.search(os.path.basename(file))
    if match:
        epoch_number = int(match.group(1))
    else:
        print(f"ファイル名からエポック番号を抽出できませんでした: {file}")
        continue

    df = pd.read_csv(file)
    df['epoch'] = epoch_number
    df_list.append(df)

if not df_list:
    raise ValueError("指定されたパターンに一致するファイルが見つかりませんでした。")

full_df = pd.concat(df_list, ignore_index=True)
full_df = full_df.sort_values('epoch')

# ===== 2. 指定ラベルの割合を計算 =====
grouped = full_df.groupby('epoch')
total_counts = grouped.size().reset_index(name='total')
label1_counts = grouped.apply(lambda x: (x[label_column] == label1).sum()).reset_index(name='label1_count')
label2_counts = grouped.apply(lambda x: (x[label_column] == label2).sum()).reset_index(name='label2_count')

ratio_df = total_counts.merge(label1_counts, on='epoch').merge(label2_counts, on='epoch')
ratio_df['label1_ratio'] = (ratio_df['label1_count'] / ratio_df['total']) * 100
ratio_df['label2_ratio'] = (ratio_df['label2_count'] / ratio_df['total']) * 100
ratio_df = ratio_df.sort_values('epoch')

# ===== 3. スムージングの適用／非適用を切り替え =====
if use_smoothing:
    ratio_df['label1_ratio_plot'] = moving_average(ratio_df['label1_ratio'], window_size)
    ratio_df['label2_ratio_plot'] = moving_average(ratio_df['label2_ratio'], window_size)
else:
    ratio_df['label1_ratio_plot'] = ratio_df['label1_ratio']
    ratio_df['label2_ratio_plot'] = ratio_df['label2_ratio']

# ===== 4. 折れ線グラフのプロット =====
plt.style.use('ggplot')  # スタイルを設定（必要に応じて変更）
plt.figure(figsize=(12, 6))

# カラーパレットから色を取得
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
color1 = color_cycle[0]
color2 = color_cycle[1]

if use_smoothing:
    # 元のデータを透明度を下げてプロット
    plt.plot(ratio_df['epoch'],
             ratio_df['label1_ratio'],
             label=f' Clean Original',
             color=color1,
             alpha=0.3,
             linewidth=1)
    
    plt.plot(ratio_df['epoch'],
             ratio_df['label2_ratio'],
             label=f'Noisy Original',
             color=color2,
             alpha=0.3,
             linewidth=1)
    
    # スムージング後のデータを不透明にプロット
    plt.plot(ratio_df['epoch'],
             ratio_df['label1_ratio_plot'],
             label=f'Clean Smoothed',
             color=color1,
             alpha=1.0,
             linewidth=2)
    
    plt.plot(ratio_df['epoch'],
             ratio_df['label2_ratio_plot'],
             label=f'Noisy Smoothed',
             color=color2,
             alpha=1.0,
             linewidth=2)
else:
    # スムージングを適用しない場合のプロット
    plt.plot(ratio_df['epoch'],
             ratio_df['label1_ratio_plot'],
             label=f'Label {label1} Ratio',
             color=color1,
             linewidth=2)
    
    plt.plot(ratio_df['epoch'],
             ratio_df['label2_ratio_plot'],
             label=f'Label {label2} Ratio',
             color=color2,
             linewidth=2)

plt.xscale('log')  # エポック軸をログスケールに
plt.title(f'Ratio of Predicted Labels {label1} and {label2} over Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Ratio (%)', fontsize=14)
plt.legend(title='Labels', fontsize=12, title_fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# ===== 5. 保存ファイル名をスムージングの有無で変更 =====
if use_smoothing:
    file_name = f"viz_smoothed_{label_column}.png"
else:
    file_name = f"viz_no_smoothing_{label_column}.png"

save_dir = os.path.join(file_dir, "fig_and_log")
os.makedirs(save_dir, exist_ok=True)  # ディレクトリがない場合は作成
save_path = os.path.join(save_dir, file_name)

plt.savefig(save_path)

print(f"グラフを保存しました: {save_path}")
