#rm processed_data_cache.pkl

import os
import re
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- code1 の evaluate_label_changes を利用できる前提 ---
from viz_colored_alpha_eval2 import evaluate_label_changes
# smoothing用関数
from smoothing import moving_average


def extract_variance_and_noise(dir_path: str):
    """
    ディレクトリパスから variance と label_noise を抽出。
    例: /workspace/...variance3162...LabelNoiseRate0.2...
         → variance="3162", label_noise="0.2"
    """
    variance_match = re.search(r'variance(\d+)', dir_path)
    variance = variance_match.group(1) if variance_match else None

    noise_match = re.search(r'LabelNoiseRate([\d\.]+)', dir_path)
    label_noise = noise_match.group(1) if noise_match else None

    return variance, label_noise


def get_wandb_run_df(variance: str, label_noise: str, entity='dsml-kernel24', project='kobayashi_save_model'):
    """
    W&B の run_name を組み立てて取得し、DataFrame で返す。
    返す DataFrame のエラーカラムは 0~100(%) に変換済み。
    """
    run_name = f"cnn_5layers_distribution_colored_emnist_variance{variance}_combined_lr0.01_batch256_epoch2000_LabelNoiseRate{label_noise}_Optimsgd_Momentum0.0"

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    target_run = None
    for r in runs:
        if r.name == run_name:
            target_run = r
            break

    if not target_run:
        print(f"Warning: Run not found in W&B for run_name={run_name}")
        return pd.DataFrame()

    df = target_run.history(samples=1000, x_axis="epoch")

    # accuracy → error(%) 変換
    for col in df.columns:
        if 'accuracy' in col:
            error_col = col.replace('accuracy', 'error')
            df[error_col] = 100 - pd.to_numeric(df[col], errors='coerce')

    return df


def get_label_change_df(dir_path: str, target: str, smoothing=True, smoothing_window=5):
    """
    code1の evaluate_label_changes で非スムージングデータを取得後、
    スムージング前・後の2列をDataFrameにまとめる:
       label_change_{target}_raw
       label_change_{target}
    """
    raw_scores_dict = evaluate_label_changes(
        directory=dir_path,
        mode='alpha',  # epochとみなす
        target=target,
        smoothing=False,
        y_scale='ratio'
    )
    sorted_epochs = sorted(raw_scores_dict.keys())
    raw_scores = [raw_scores_dict[e] for e in sorted_epochs]

    if smoothing:
        smooth_scores = moving_average(raw_scores, window_size=smoothing_window)
    else:
        smooth_scores = raw_scores

    df_label_change = pd.DataFrame({
        'epoch': sorted_epochs,
        f'label_change_{target}_raw': raw_scores,
        f'label_change_{target}': smooth_scores
    })
    return df_label_change


def prepare_merged_data(dir_path: str, target: str,
                        label_change_smoothing=True, smoothing_window=5):
    """
    1) dir_path から variance, label_noise を取得
    2) W&B ログ (df_wandb) を取得
    3) label_change DF を取得
    4) マージ (epoch)
    5) 返す (df_merged, variance, label_noise)
    """
    variance, label_noise = extract_variance_and_noise(dir_path)
    if not variance or not label_noise:
        print(f"[WARN] Could not parse variance/noise from {dir_path}")
        return None, None, None, None

    df_wandb = get_wandb_run_df(variance, label_noise)
    if df_wandb.empty:
        print(f"[WARN] No W&B data for var={variance}, noise={label_noise}")
        return None, None, None, None

    df_label_change = get_label_change_df(
        dir_path, target=target,
        smoothing=label_change_smoothing,
        smoothing_window=smoothing_window
    )
    if df_label_change.empty:
        print(f"[WARN] No label-change data in {dir_path}")
        return None, None, None, None

    # epochをintに
    df_wandb['epoch'] = df_wandb['epoch'].astype(int)
    df_label_change['epoch'] = df_label_change['epoch'].astype(int)

    df_merged = pd.merge(df_wandb, df_label_change, on='epoch', how='inner').sort_values(by='epoch')
    if df_merged.empty:
        print(f"[WARN] Merged data is empty for {dir_path}")
        return None, None, None, None

    return df_merged, variance, label_noise, dir_path


def plot_all_in_one(dirs, target='digit', label_change_smoothing=True,
                    smoothing_window=5, output_pdf="combined_plots.pdf"):
    """
    複数のディレクトリを列方向に並べ、1枚の図(4行×N列)で比較する。
    左軸 -> Error(0~100), 右軸 -> Label Change(-0.1~0.15)
    スムージング有効時は、青(透明)でスムージング前＋赤(破線)でスムージング後を描画。
    """

    # 1) 各ディレクトリからマージ済みデータを準備
    data_list = []
    for d in dirs:
        df_merged, variance, label_noise, _ = prepare_merged_data(
            d, target=target,
            label_change_smoothing=label_change_smoothing,
            smoothing_window=smoothing_window
        )
        if df_merged is not None:
            data_list.append((df_merged, variance, label_noise))

    # 有効データがなければ終了
    if not data_list:
        print("No valid data to plot.")
        return

    n_dirs = len(data_list)

    # 2) Figure作成: 4行×n_dirs列
    fig, axes = plt.subplots(nrows=4, ncols=n_dirs, figsize=(10*n_dirs, 15), sharex=False)
    # 行ごとに描画するエラー系カラム
    row_columns = [
        ['test_error', 'test_error_color_total', 'test_error_digit_total'],
        ['train_error', 'train_error_color_total', 'train_error_digit_total'],
        ['train_error_clean', 'train_error_noisy'],
        ['train_error_color_clean', 'train_error_digit_clean',
         'train_error_color_noisy', 'train_error_digit_noisy']
    ]

    # 3) 各列(=各ディレクトリ)ごとに4行を埋める
    for col_idx, (df_merged, variance, label_noise) in enumerate(data_list):
        # カラムタイトル
        axes[0, col_idx].set_title(f"var={variance}, noise={label_noise}", fontsize=14)

        label_change_raw_col = f"label_change_{target}_raw"
        label_change_col = f"label_change_{target}"

        for row_idx in range(4):
            # axesが2次元配列の場合、n_dirs=1だと次元が落ちるので対応
            if n_dirs == 1:
                ax = axes[row_idx]
            else:
                ax = axes[row_idx, col_idx]
            ax2 = ax.twinx()

            # 左軸 -> エラー系
            error_cols = row_columns[row_idx]
            for c in error_cols:
                if c in df_merged.columns:
                    ax.plot(df_merged['epoch'], df_merged[c], label=c)

            # 右軸 -> label change
            if label_change_raw_col in df_merged.columns:
                ax2.plot(
                    df_merged['epoch'], df_merged[label_change_raw_col],
                    label=label_change_raw_col, color='blue', alpha=0.3
                )
            if label_change_col in df_merged.columns:
                if label_change_smoothing:
                    # スムージング後は別色
                    ax2.plot(
                        df_merged['epoch'], df_merged[label_change_col],
                        label=label_change_col, color='red', linestyle='--'
                    )
                else:
                    # スムージング無効時は1本だけ(blue)
                    ax2.plot(
                        df_merged['epoch'], df_merged[label_change_col],
                        label=label_change_col, color='blue', linestyle='--'
                    )

            # 軸設定
            ax.set_xscale('log')
            ax.set_ylim(0, 100)
            ax2.set_ylim(-0.01, 0.05)
            ax.grid(True, which="both", linestyle='--', linewidth=0.5)

            # xラベルは最下行のみ表示
            if row_idx < 3:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Epoch (log scale)")

            ax.set_ylabel("Error (%)")
            ax2.set_ylabel("Label Change")

            # 凡例
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            if lines_1 or lines_2:
                ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

     # 保存先ディレクトリを指定
    save_dir = "pdf/clean_noisy/1000_0.5"
    # ディレクトリが存在しなければ作成
    os.makedirs(save_dir, exist_ok=True)

    # PDFファイルを保存
    save_path = os.path.join(save_dir, output_pdf)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close(fig)
    print(f"Saved combined figure as {save_path}")

if __name__ == "__main__":
    # 例: 3つのディレクトリを指定
    list_of_indices = ["1","2","3","4","5","6","7","30","40"]
    list = ["67","83","31","32","37"]
    t=0
    for i in list_of_indices:
        s = list[t]
        dirs_to_process = [
            f"/workspace/alpha_test/test_closet_cnn_5layers_distribution_colored_emnist_variance1000_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.5_Optimsgd_Momentum0.0/no_noise_no_noise/{i}/{i}",
            f"/workspace/alpha_test/test_closet_cnn_5layers_distribution_colored_emnist_variance1000_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.5_Optimsgd_Momentum0.0/noise_no_noise/{i}/{s}",
        ]
        t=t+1
    
        # PDFファイル名を各iで分けたい場合:
        output_pdf_name = f"clean_noisy_digit_plot_all_in_one_{i}.pdf"
        
        # plot_all_in_oneを呼び出す
        plot_all_in_one(
            dirs=dirs_to_process,
            target='digit',
            label_change_smoothing=True,
            smoothing_window=5,
            output_pdf=output_pdf_name
        )
    




"""
# 例: 3つのディレクトリを指定
    list_of_indices = ["2","3","4","5","7","20","30","40"]
    #
    for i in list_of_indices:
        dirs_to_process = [
            f"/workspace/alpha_test/closet_cnn_5layers_distribution_colored_emnist_variance1000_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.0_Optimsgd_Momentum0.0/no_noise_no_noise/{i}/{i}",
            f"/workspace/alpha_test/closet_cnn_5layers_distribution_colored_emnist_variance1000_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/no_noise_no_noise/{i}/{i}",
            f"/workspace/alpha_test/test_closet_cnn_5layers_distribution_colored_emnist_variance1000_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.5_Optimsgd_Momentum0.0/no_noise_no_noise/{i}/{i}"
        ]
    
        # PDFファイル名を各iで分けたい場合:
        output_pdf_name = f"clean_clean_digit_plot_all_in_one_{i}.pdf"
        
        # plot_all_in_oneを呼び出す
        plot_all_in_one(
            dirs=dirs_to_process,
            target='digit',
            label_change_smoothing=True,
            smoothing_window=5,
            output_pdf=output_pdf_name
        )
"""    

"""    # まとめて1枚のPDFに出力
    plot_all_in_one(
        dirs=dirs_to_process,
        target='digit',
        label_change_smoothing=True,
        smoothing_window=5,
        output_pdf="digit_plot_all_in_one.pdf"  # 好みのファイル名に変更
    )
"""