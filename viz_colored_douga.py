import matplotlib
matplotlib.use('Agg')  # 非表示バックエンド（Agg）を利用して描画負荷を低減

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from itertools import cycle
from matplotlib.lines import Line2D  # ダミー凡例作成用
import numpy as np
import cv2  # OpenCVを利用
from multiprocessing import Process

# フォントや図サイズの設定
plt.rcParams["font.size"] = 14
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["figure.dpi"] = 200

def load_data(data_dir, targets='digit'):
    csv_dir = os.path.join(data_dir, "csv")
    if targets == 'combined':
        files = sorted(glob.glob(os.path.join(csv_dir, "raw_probabilities_epoch_*.csv")))
    else:
        files = sorted(glob.glob(os.path.join(csv_dir, "alpha_log_epoch_*.csv")))
    
    data = {}
    for file in files:
        epoch = int(os.path.basename(file).split('_')[-1].split('.')[0])
        data[epoch] = pd.read_csv(file)
    return data

def extract_digit_for_digit_mode(name):
    if len(name) == 1:
        return 0
    else:
        return int(name[0])

def extract_digit_for_color_mode(name):
    return int(name[-1])  # ディレクトリ名の末尾の数字を使用

def get_highlight_targets(data_dir, targets):
    current_dir_name = os.path.basename(data_dir)
    parent_dir_name = os.path.basename(os.path.dirname(data_dir))
    if targets == 'combined':
        parent_val = int(parent_dir_name)
        current_val = int(current_dir_name)
    elif targets == 'digit':
        parent_val = extract_digit_for_digit_mode(parent_dir_name)
        current_val = extract_digit_for_digit_mode(current_dir_name)
    elif targets == 'color':
        parent_val = extract_digit_for_color_mode(parent_dir_name)
        current_val = extract_digit_for_color_mode(current_dir_name)
    else:
        raise ValueError("targets must be 'digit', 'color', or 'combined'.")
    return [parent_val, current_val]

def target_plot_probabilities(
    data_dir, 
    targets='digit', 
    video_output="output.mp4",   # 出力ファイル名の基本部分
    save_dir=None,               # 動画の保存先ディレクトリ
    use_opencv=True,             # OpenCV を使って動画出力するか
    show_legend=True,
    epoch_start=None, 
    epoch_end=None, 
    epoch_step=None
):
    """
    指定した data_dir 内の CSV ファイルからデータを読み込み、各エポックごとにグラフを更新し、
    Blitting および Agg バックエンドを利用して動画ファイルとして出力します。
    
    ラインは blitting で高速更新し、タイトルは毎回通常描画で更新することで、以前のタイトルが重なる問題を防ぎます。
    """
    data = load_data(data_dir, targets)
    epochs = sorted(data.keys())
    if epoch_start is not None:
        epochs = [e for e in epochs if e >= epoch_start]
    if epoch_end is not None:
        epochs = [e for e in epochs if e <= epoch_end]
    if epoch_step is not None and epoch_step > 1:
        epochs = epochs[::epoch_step]
    if len(epochs) == 0:
        print(f"No CSV files found in {data_dir} within the specified epoch range. Skipping...")
        return

    initial_epoch = epochs[0]
    df_first = data[initial_epoch]
    
    if save_dir is None:
        save_dir = os.path.join(data_dir, "fig_and_log")
    os.makedirs(save_dir, exist_ok=True)
    parent_name = os.path.basename(os.path.dirname(data_dir))
    current_name = os.path.basename(data_dir)
    video_file_name = f"{parent_name}_{current_name}_{video_output}"
    video_output_path = os.path.join(save_dir, video_file_name)
    
    highlight_targets = get_highlight_targets(data_dir, targets)

    # グラフ作成
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    lines = []
    alpha_values = df_first['alpha']
    other_colors = cycle([
        'green', 'orange', 'purple', 'brown', 
        'gray', 'pink', 'olive', 'cyan', 'lime', 'navy'
    ])
    if targets == 'combined':
        for t in range(100):
            column_name = f'probability_{t}'
            if column_name in df_first.columns:
                if t == highlight_targets[0]:
                    color = 'blue'
                    linestyle = "-"
                    linewidth = 2.0
                    label = "Clean Label"
                elif t == highlight_targets[1]:
                    color = 'red'
                    linestyle = "-"
                    linewidth = 2.0
                    label = "Noisy Label"
                else:
                    color = next(other_colors)
                    linestyle = "--"
                    linewidth = 1.0
                    label = None
                line, = ax.plot(alpha_values, df_first[column_name], 
                                color=color, linestyle=linestyle, alpha=0.9, linewidth=linewidth, 
                                label=label)
                lines.append(line)
    else:
        for t in range(10):
            column_name = f'{targets}_probability_{t}'
            if column_name in df_first.columns:
                if t in highlight_targets:
                    color = 'blue' if t == highlight_targets[0] else 'red'
                    label = "Clean Label" if t == highlight_targets[0] else "Noisy Label"
                    line, = ax.plot(alpha_values, df_first[column_name], 
                                    color=color, linewidth=2.0, label=label)
                else:
                    color = next(other_colors)
                    line, = ax.plot(alpha_values, df_first[column_name], 
                                    color=color, alpha=0.3, linewidth=0.5, label=None)
                lines.append(line)
    marker_size = 20
    # マーカー描画（data_dir の文字列に基づく）
    if "no_noise_no_noise" in data_dir:
        ax.plot(0.0, 0.0, marker='o', color='blue', markersize=marker_size, zorder=5)
        ax.plot(1.0, 0.0, marker='o', color='blue', markersize=marker_size, zorder=5)
    elif "noise_no_noise" in data_dir:
        ax.plot(0.0, 0.0, marker='o', color='blue', markersize=marker_size, zorder=5)
        ax.plot(1.0, 0.0, marker='o', color='red', markersize=marker_size, zorder=5)
    else:
        ax.plot(0.0, 0.0, marker='o', color='blue', markersize=marker_size, zorder=5)
        ax.plot(1.0, 0.0, marker='o', color='red', markersize=marker_size, zorder=5)
    ax.axvline(x=0.0, color='grey', linestyle='-', linewidth=1.0)
    ax.axvline(x=1.0, color='grey', linestyle='-', linewidth=1.0)
    ax.set_ylabel('Probability', fontsize=22)
    # タイトルは blitting の対象外として通常更新
    ax.set_title(f'Probability for Epoch {initial_epoch}', fontsize=22)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([r'$x_0 \in X_C$', r'$x_1 \in X_C$'], fontsize=30)
    if show_legend:
        clean_line = Line2D([0], [0], color='blue', linewidth=2.0, label='Clean Label')
        noisy_line = Line2D([0], [0], color='red', linewidth=2.0, label='Noisy Label')
        other_line = Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Others')
        ax.legend(handles=[clean_line, noisy_line, other_line],
                  loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0, fontsize=16)
        fig.subplots_adjust(right=0.8)

    # update 関数：ラインのみ blitting で更新；タイトルは各フレームで通常更新
    def update(epoch):
        df_epoch = data[epoch]
        for t, line in enumerate(lines):
            column_name = f'probability_{t}' if targets == 'combined' else f'{targets}_probability_{t}'
            if column_name in df_epoch.columns:
                line.set_ydata(df_epoch[column_name])
        return lines

    # 初回描画後、背景キャッシュを作成（タイトルは blitting の対象外）
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)

    width, height = fig.canvas.get_width_height()
    new_width, new_height = width // 2, height // 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 5
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (new_width, new_height))

    for epoch in epochs:
        update(epoch)
        # 背景キャッシュからラインのみ再描画
        fig.canvas.restore_region(background)
        for line in lines:
            ax.draw_artist(line)
        fig.canvas.blit(ax.bbox)
        fig.canvas.flush_events()
        
        # タイトルは通常更新（blitting の対象外）
        ax.set_title(f'Probability for Epoch {epoch}', fontsize=22)
        fig.canvas.draw_idle()
        
        # 1フレーム分の画像取得
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape((height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        video_writer.write(image_resized)
        
        # 次フレームのため、再キャッシュ（ここは全体再描画でOK）
        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(ax.bbox)
    
    video_writer.release()
    print(f"Video saved as {video_output_path}")
    plt.close(fig)

def run_in_subprocess(d, save_dir):
    """
    各ディレクトリごとに target_plot_probabilities を実行し、処理終了後にメモリ解放のためサブプロセスを利用。
    """
    target_plot_probabilities(
        data_dir=d,
        targets="combined",
        video_output="output.mp4",
        save_dir=save_dir,
        use_opencv=True,
        show_legend=True,
        epoch_start=1,
        epoch_end=1000,
        epoch_step=1
    )

def main_1():
    """
    main_1：各ディレクトリごとにサブプロセスを作成して動画生成を行います。
    """
    dirs=[
        #"alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/no_noise_no_noise/94/94",
        "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/no_noise_no_noise/39/39",
        "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/no_noise_no_noise/80/80",
        # "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/no_noise_no_noise/67/67",

    ]
    # dirs = [
    #     "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/59/88",
    #     "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/87/6",
    #     "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/36/9",
    #     "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/61/62",
    #     "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/92/45",
    #     "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/35/48",
    #     "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/39/17",
    #     "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/58/23",
    #     "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/11/40",
    #     "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/15/98"
    # ]
    #smalle
    # dirs = [
    #     "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/11/40",
    #     "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/19/52",
    #     "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/91/23",
    #     "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/71/53",
    #     "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/1/12",
    #     "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/17/64",
    #     "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/16/7",
    #     "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/31/54",
    #     "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/13/78",
    #     "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/61/62"
    # ]
    #smalle
    # dirs = [
    #large
    # dirs = [
    # "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/20/63",
    # "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/25/10",
    # "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/55/66",
    # "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/80/63",
    # "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/30/94",
    # "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/88/55",
    # "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/48/1",
    # "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/45/75",
    # "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/35/48",
    # "alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/23/88"
    #]  
    save_dir = "/workspace/miru_vizualize/douga/C_C"
    processes = []
    for d in dirs:
        print(f"Starting processing for: {d}")
        p = Process(target=run_in_subprocess, args=(d, save_dir))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("All processes completed.")

if __name__ == "__main__":
    main_1()
