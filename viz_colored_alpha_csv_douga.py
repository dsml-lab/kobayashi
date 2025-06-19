import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation, FFMpegWriter
from itertools import cycle
from matplotlib.lines import Line2D  # ダミー凡例作成用
import numpy as np

# フォントや図サイズの設定例（適宜変更してください）
plt.rcParams["font.size"] = 14
plt.rcParams["figure.figsize"] = [12,8]
plt.rcParams["figure.dpi"] = 200

def load_data(data_dir, targets='digit'):
    """
    CSVファイルの読み込み先を data_dir 内の csv ディレクトリに変更。
    targetsがcombinedの場合は、raw_probabilities_epoch_{epoch}.csvを読み込む。
    """
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
        tens_digit = 0
    else:
        tens_digit = int(name[0])
    return tens_digit

def extract_digit_for_color_mode(name):
    return int(name[-1])  # ディレクトリ名の末尾の数字を使用

def get_highlight_targets(data_dir, targets):
    current_dir_name = os.path.basename(data_dir)
    parent_dir_name = os.path.basename(os.path.dirname(data_dir))

    if targets == 'combined':
        # combinedの場合は単純にディレクトリ名を数値として扱う
        parent_val = int(parent_dir_name)
        current_val = int(current_dir_name)
        print(f"parent_val: {parent_val}, current_val: {current_val}")
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
    gif_output="output.mp4",   # 出力ファイル拡張子をmp4に変更
    gif=False, 
    show_legend=True,
    epoch_start=None, 
    epoch_end=None, 
    epoch_step=None
):
    """
    - 指定した data_dir 下の csvディレクトリにある alpha_log_epoch_*.csv を読み込み、
      指定範囲の epoch について、alpha と {targets}_probability_(0..9) のプロットを行う
    - gif=True ならすべてのepochをアニメーション化し動画ファイルとして出力し、表示は行わない
    - gif=False ならインタラクティブにスライダーで epoch を切り替え表示し、動画保存ボタンも用意
    - show_legend=False にすると凡例を非表示にできる
    - targets='combined'の場合、raw_probabilities_epoch_{epoch}.csvを使用して
      digit×colorの組み合わせ（0-99）の確率を表示
    """
    data = load_data(data_dir, targets)
    epochs = sorted(data.keys())

    # epoch範囲指定
    if epoch_start is not None:
        epochs = [e for e in epochs if e >= epoch_start]
    if epoch_end is not None:
        epochs = [e for e in epochs if e <= epoch_end]

    # epoch間隔指定
    if epoch_step is not None and epoch_step > 1:
        epochs = epochs[::epoch_step]

    if len(epochs) == 0:
        print(f"No CSV files found in {data_dir} within the specified epoch range. Skipping...")
        return

    initial_epoch = epochs[0]
    df = data[initial_epoch]

    # fig_and_logディレクトリを data_dir 内に作成
    fig_and_log_dir = os.path.join(data_dir, "fig_and_log")
    os.makedirs(fig_and_log_dir, exist_ok=True)
    gif_output_path = os.path.join(fig_and_log_dir, gif_output)

    # highlight対象取得
    highlight_targets = get_highlight_targets(data_dir, targets)

    # グラフの作成
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    lines = []
    
    # 最初のエポックのデータを使用して初期化
    df_first = data[epochs[0]]
    alpha_values = df_first['alpha']

    # "その他"用の色をサイクルさせる（プロット自体は元カラー）
    other_colors = cycle([
        'green', 'orange', 'purple', 'brown', 
        'gray', 'pink', 'olive', 'cyan', 'lime', 'navy'
    ])

    if targets == 'combined':
        parent_val = highlight_targets[0]
        current_val = highlight_targets[1]
        
        for t in range(100):
            column_name = f'probability_{t}'
            if column_name in df_first.columns:
                if t == parent_val:
                    color = 'blue'
                    alpha = 1.0
                    linewidth = 2.0
                    label = "Clean Label"
                elif t == current_val:
                    color = 'red'
                    alpha = 1.0
                    linewidth = 2.0
                    label = "Noisy Label"
                else:
                    color = next(other_colors)
                    alpha = 0.9
                    linewidth = 1.0
                    label = None
                line, = ax.plot(alpha_values, df_first[column_name], 
                                color=color, alpha=alpha, linewidth=linewidth, 
                                label=label)
                lines.append(line)
    else:
        for t in range(10):
            column_name = f'{targets}_probability_{t}'
            if column_name in df_first.columns:
                if t in highlight_targets:
                    if t == highlight_targets[0]:
                        color = 'blue'
                        label = "Clean Label"
                    elif t == highlight_targets[1]:
                        color = 'red'
                        label = "Noisy Label"
                    line, = ax.plot(alpha_values, df_first[column_name], 
                                    color=color, linewidth=2.0, label=label)
                else:
                    color = next(other_colors)
                    line, = ax.plot(alpha_values, df_first[column_name], 
                                    color=color, alpha=0.3, linewidth=0.5, label=None)
                lines.append(line)

    # 縦線 (x=0, x=1) は判例（例示）として黒の点線
    ax.axvline(x=0.0, color='black', linestyle='--', linewidth=1.5)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5)

    # 軸ラベル、範囲などの設定
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(f'{targets} Probability')
    ax.set_title(f'{targets} Probability for Epoch {initial_epoch}')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(np.arange(-0.5, 1.6, 0.5))

    # カスタム凡例の設定：Clean Label, Noisy Label, Other Label（その他はダミー曲線で黒の点線）
    if show_legend:
        clean_line = Line2D([0], [0], color='blue', linewidth=2.0, label='Clean Label')
        noisy_line = Line2D([0], [0], color='red', linewidth=2.0, label='Noisy Label')
        other_line = Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Other Label')
        ax.legend(handles=[clean_line, noisy_line, other_line],
                  loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
        fig.subplots_adjust(right=0.8)

    # アニメーション用のupdate関数
    def update(epoch):
        df_epoch = data[epoch]
        ax.set_title(f'{targets} Probability for Epoch {epoch}')
        for t, line in enumerate(lines):
            column_name = f'probability_{t}' if targets == 'combined' else f'{targets}_probability_{t}'
            if column_name in df_epoch.columns:
                line.set_ydata(df_epoch[column_name])
        return lines

    if gif:
        # 動画ファイルとして保存する場合
        def animate(frame):
            epoch = epochs[frame]
            return update(epoch)

        anim = FuncAnimation(fig, animate, frames=len(epochs), interval=200, blit=True)
        anim.save(gif_output_path, writer=FFMpegWriter(fps=8))
        print(f"Video saved as {gif_output_path}")
        plt.close(fig)

    else:
        # スライダーとボタンを表示 (対話的)
        ax_epoch = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        epoch_slider = Slider(ax_epoch, 'Epoch', epochs[0], epochs[-1], 
                              valinit=initial_epoch, valstep=epochs)

        # ボタンラベルを "Save Video" に変更
        ax_button = plt.axes([0.85, 0.025, 0.1, 0.04])
        button = Button(ax_button, 'Save Video')

        def slider_update(val):
            epoch = int(epoch_slider.val)
            update(epoch)
            fig.canvas.draw_idle()

        epoch_slider.on_changed(slider_update)

        def save_video(event):
            def animate(frame):
                epoch = epochs[frame]
                return update(epoch)

            anim = FuncAnimation(fig, animate, frames=len(epochs), interval=200, blit=True)
            anim.save(gif_output_path, writer=FFMpegWriter(fps=8, codec="mpeg4", extra_args=["-qscale", "0"]))

            print(f"Video saved as {gif_output_path}")

        button.on_clicked(save_video)

        plt.show()


def process_all_subdirs(
    root_dir, 
    targets='digit', 
    gif=False, 
    selected_dirs=None, 
    epoch_start=None, 
    epoch_end=None, 
    epoch_step=None
):
    """
    ルートディレクトリ(root_dir)以下のサブディレクトリそれぞれについて
    target_plot_probabilities を呼び出す。
    selected_dirs が指定されていれば、そのディレクトリ名のみを対象にする。
    """
    all_subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    if selected_dirs is not None:
        subdirs_to_process = [d for d in all_subdirs if d in selected_dirs]
    else:
        subdirs_to_process = all_subdirs

    for name in subdirs_to_process:
        sub_dir = os.path.join(root_dir, name)
        target_plot_probabilities(
            sub_dir, 
            targets=targets, 
            gif_output=f"{targets}.mp4",  # 拡張子をmp4に変更
            gif=gif, 
            epoch_start=epoch_start, 
            epoch_end=epoch_end, 
            epoch_step=epoch_step
        )


# 使用例 (main部などは環境に合わせて適宜書き換えてください)
def main():
    list_dirs = ["10"]
    # 例: ディレクトリパスを適宜変更してください
    dir = "/workspace/alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/0/95"
    target_plot_probabilities(
            data_dir=dir,
            targets="combined",
            gif=True,
            show_legend=True,  # 凡例を表示
            gif_output="output.mp4",  # 出力ファイルをmp4に変更
            epoch_start=0,
            epoch_end=1000,
            epoch_step=1
        )

if __name__ == "__main__":
    main()
