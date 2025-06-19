import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path  # pathlibをインポート

# フォントサイズ、図のサイズ、解像度を指定
plt.rcParams["font.size"] = 14
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["figure.dpi"] = 100  # 解像度を向上

def load_data(data_dir):
    """
    指定されたディレクトリ内のCSVファイルを読み込み、エポックごとにデータを辞書に格納します。
    """
    files = sorted(glob.glob(os.path.join(data_dir, "alpha_log_epoch_*.csv")))
    data = {}
    for file in files:
        epoch = int(os.path.basename(file).split('_')[-1].split('.')[0])
        data[epoch] = pd.read_csv(file)
    return data

def extract_digit(name):
    """
    ディレクトリ名から数字を抽出します。
    """
    try:
        digit = int(name)
        return digit
    except ValueError:
        raise ValueError(f"Directory name '{name}' is not a valid integer.")

def get_highlight_targets(dir_path):
    """
    ハイライト対象の値を取得します。
    'no_noise_no_noise' ディレクトリ以下のすべてのラベルを抽出し、整数のリストとして返します。
    例: 'no_noise_no_noise/0/2/' の場合、[0, 2] を返します。
    """
    # pathlibを使用してパスを分割
    path = Path(dir_path)
    parts = path.parts

    try:
        no_noise_index = parts.index("no_noise_no_noise")
    except ValueError:
        raise ValueError(f"Directory path '{dir_path}' does not contain 'no_noise_no_noise'.")

    # 'no_noise_no_noise' の後の部分をすべて取得
    labels = parts[no_noise_index + 1:]

    highlight_targets = []
    for label in labels:
        try:
            digit = extract_digit(label)
            highlight_targets.append(digit)
        except ValueError:
            print(f"Warning: Label '{label}' is not a valid integer and will be skipped.")

    return highlight_targets

def target_plot_probabilities(sub_dir, gif_output="output.gif", gif=False, epoch_start=None, epoch_end=None, epoch_step=None):
    """
    指定されたサブディレクトリ内のCSVデータをプロットし、GIFを生成します。
    """
    data_dir = os.path.join(sub_dir, "csv")
    fig_and_log_dir = os.path.join(sub_dir, "fig_and_log")
    os.makedirs(fig_and_log_dir, exist_ok=True)
    gif_output_path = os.path.join(fig_and_log_dir, gif_output)

    data = load_data(data_dir)
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

    # ハイライト対象取得
    highlight_targets = get_highlight_targets(sub_dir)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.25)  # スライダー領域確保

    alpha_values = df['alpha']
    lines = []
    labels_handled = set()  # 重複ラベルを避けるためのセット

    for target_val in range(10):
        column_name = f'digit_probability_{target_val}'
        if column_name not in df.columns:
            print(f"Warning: Column '{column_name}' does not exist in the data. Skipping this target.")
            continue

        if target_val in highlight_targets:
            line_style = '-'
            label = f'digit_{target_val}'  # 修正: 凡例を 'digit_[label]' の形式に統一
        else:
            line_style = '--'
            label = f'digit_{target_val}'  # 修正: ハイライト以外も同様に 'digit_[label]' に統一

        # 同じラベルが複数回追加されないようにチェック
        if label in labels_handled:
            continue
        labels_handled.add(label)

        line, = ax.plot(alpha_values, df[column_name], line_style, label=label)
        lines.append(line)

    ax.set_xlabel('Alpha')
    ax.set_ylabel('Digit Probability')
    ax.set_title(f'Digit Probability for Epoch {initial_epoch}')
    ax.set_ylim(-0.1, 1.1)

    # 凡例をグラフの右側外部に配置
    legend = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    fig.subplots_adjust(right=0.75)  # 凡例が見切れないように右側に余白を確保

    def update(epoch):
        """
        プロットを指定されたエポックのデータに更新します。
        """
        if epoch not in data:
            print(f"Warning: Epoch {epoch} not found in data. Skipping update.")
            return
        df_epoch = data[epoch]
        ax.set_title(f'Digit Probability for Epoch {epoch}')
        for t, line in enumerate(lines):
            column_name = f'digit_probability_{t}'
            if column_name not in df_epoch.columns:
                print(f"Warning: Column '{column_name}' does not exist in the data for epoch {epoch}. Skipping this line.")
                continue
            line.set_ydata(df_epoch[column_name])
        fig.canvas.draw_idle()

    if gif:
        # gif=Trueの場合、インタラクティブな表示は行わず、即座にGIFを作成
        def animate(frame):
            epoch = epochs[frame]
            update(epoch)
            return lines

        anim = FuncAnimation(fig, animate, frames=len(epochs), interval=200, blit=False)
        anim.save(gif_output_path, writer=PillowWriter(fps=10))
        print(f"GIF saved as {gif_output_path}")
        plt.close(fig)
    else:
        # gif=Falseの場合のみスライダー表示とGIFボタン生成
        ax_epoch = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        epoch_slider = Slider(
            ax=ax_epoch,
            label='Epoch',
            valmin=epochs[0],
            valmax=epochs[-1],
            valinit=initial_epoch,
            valstep=epochs
        )

        ax_button = plt.axes([0.85, 0.025, 0.1, 0.04])
        button = Button(ax_button, 'Save GIF')

        def slider_update(val):
            epoch = int(epoch_slider.val)
            update(epoch)

        epoch_slider.on_changed(slider_update)

        def save_gif(event):
            def animate(frame):
                epoch = epochs[frame]
                update(epoch)
                return lines

            anim = FuncAnimation(fig, animate, frames=len(epochs), interval=200, blit=False)
            anim.save(gif_output_path, writer=PillowWriter(fps=10))
            print(f"GIF saved as {gif_output_path}")

        button.on_clicked(save_gif)

        plt.show()

def process_all_subdirs(root_dir, gif=False, selected_dirs=None, epoch_start=None, epoch_end=None, epoch_step=None):
    """
    ルートディレクトリ内のすべてのサブディレクトリを処理します。
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
            gif_output=f"d{name}.gif",  # GIF名にサブディレクトリ名を含める
            gif=gif,
            epoch_start=epoch_start,
            epoch_end=epoch_end,
            epoch_step=epoch_step
        )

# 使用例
if __name__ == "__main__":
    # ベースとなる実験ディレクトリを指定
    exp_dir = "alpha_test/EMNIST/cnn_5layers_emnist_digits_variance100_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.2_Optimsgd_Momentum0.0"

    # ルートディレクトリを指定（例として 'no_noise_no_noise/0' ディレクトリを指定）
    root_dir = os.path.join(exp_dir, "no_noise_no_noise", "0")

    # 必要なディレクトリが存在しない場合は作成
    os.makedirs(root_dir, exist_ok=True)

    # サブディレクトリ内の '2' を処理
    process_all_subdirs(
        root_dir=root_dir,
        gif=True,              # GIFを作成する場合は True に設定
        selected_dirs=["1","2","3","4","5","6","7","8","9"],   # 処理したいサブディレクトリ名をリストで指定
        epoch_start=0,
        epoch_end=2000,
        epoch_step=5
    )

    # 複数のサブディレクトリを処理する場合の例
    """
    list = ["1", "2", "3", "4"]
    for dir_name in list:
        sub_dir = os.path.join(root_dir, dir_name)
        process_all_subdirs(
            sub_dir,
            gif=True,
            selected_dirs=None,  # 全てのサブディレクトリを処理
            epoch_start=0,
            epoch_end=2000,
            epoch_step=10
        )
    """
