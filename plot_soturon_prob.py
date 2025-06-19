# plot_probability_sums.py
import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
from smoothing import moving_average  # smoothing.pyから関数をインポート

# ===== スムージングを行うかどうかを設定 =====
use_smoothing = False  # Trueでスムージングを行う, Falseで行わない

# ===== ファイルやパラメータの設定 =====
file_dir = "alpha_test/closet_cnn_5layers_distribution_colored_emnist_variance1000_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/no_noise_no_noise/3/3"
file_directory = os.path.join(file_dir, "csv")
file_pattern = os.path.join(file_directory, 'alpha_log_epoch_*.csv')
epoch_pattern = re.compile(r'alpha_log_epoch_(\d+)\.csv')

# 例: digit_probability_0 と digit_probability_5 の合計値をプロットする
# ※ color_probability_0 なども同様に取り扱い可能
prefix = "digit"  # "color" に変えると color_probability_n になる
label1 = "3"
label2 = "3"
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
print(full_df.columns)
print(full_df.head())
# ===== 2. 指定した確率カラムの「合計」値をエポックごとに集計 =====
col_name1 = f"color_probability_3"
col_name2 = f"color_probability_3"

# epochでグループ化して、それぞれの確率カラムを合計
grouped_sum = full_df.groupby('epoch')[[col_name1, col_name2]].sum().reset_index()
grouped_sum = grouped_sum.sort_values('epoch')

# ===== 3. スムージングの適用／非適用を切り替え =====
if use_smoothing:
    grouped_sum['sum1_plot'] = moving_average(grouped_sum[col_name1], window_size)
    grouped_sum['sum2_plot'] = moving_average(grouped_sum[col_name2], window_size)
else:
    grouped_sum['sum1_plot'] = grouped_sum[col_name1]
    grouped_sum['sum2_plot'] = grouped_sum[col_name2]

# ===== 4. 折れ線グラフのプロット =====
plt.style.use('ggplot')  # スタイルを設定（必要に応じて変更）
plt.figure(figsize=(12, 6))

# label1 の合計をプロット
plt.plot(grouped_sum['epoch'],
         grouped_sum['sum1_plot'],
         label=f'{col_name1} (sum)',
         linewidth=2)

# label2 の合計をプロット
plt.plot(grouped_sum['epoch'],
         grouped_sum['sum2_plot'],
         label=f'{col_name2} (sum)',
         linewidth=2)

plt.xscale('log')  # エポック軸をログスケールに
plt.title(f'Sum of {col_name1} and {col_name2} over Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Sum of Probability', fontsize=14)
plt.legend(title='Probability Columns', fontsize=12, title_fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# ===== 5. 保存ファイル名をスムージングの有無で変更 =====
if use_smoothing:
    file_name = f"viz_smoothed_{col_name1}_{col_name2}.png"
else:
    file_name = f"viz_no_smoothing_{col_name1}_{col_name2}.png"

save_dir = os.path.join(file_dir, "fig_and_log")
os.makedirs(save_dir, exist_ok=True)  # ディレクトリがない場合は作成
save_path = os.path.join(save_dir, file_name)

plt.savefig(save_path)
plt.show()

print(f"グラフを保存しました: {save_path}")
