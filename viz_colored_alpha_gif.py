import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation, PillowWriter

# フォントサイズ、図のサイズ、解像度を指定
plt.rcParams["font.size"] = 14
plt.rcParams["figure.figsize"] = [12,8]
plt.rcParams["figure.dpi"] = 50

def load_data(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "alpha_log_epoch_*.csv")))
    data = {}
    for file in files:
        epoch = int(os.path.basename(file).split('_')[-1].split('.')[0])
        data[epoch] = pd.read_csv(file)
    return data

def extract_digit_for_digit_mode(name):
    # ディレクトリ名が "7" -> 10の位は0、"10" -> 1、"80" -> 8 など
    if len(name) == 1:
        tens_digit = 0
    else:
        tens_digit = int(name[0])
    return tens_digit

def extract_digit_for_color_mode(name):
    # 1の位: "7" -> 7, "10" -> 0, "80" -> 0, "90"->0
    ones_digit = int(name[-1])
    return ones_digit

def get_highlight_targets(data_dir, targets):
    current_dir_name = os.path.basename(data_dir)
    parent_dir_name = os.path.basename(os.path.dirname(data_dir))

    if targets == 'digit':
        parent_val = extract_digit_for_digit_mode(parent_dir_name)
        current_val = extract_digit_for_digit_mode(current_dir_name)

    elif targets == 'color':
        parent_val = extract_digit_for_color_mode(parent_dir_name)
        current_val = extract_digit_for_color_mode(current_dir_name)
    else:
        raise ValueError("targets must be 'digit' or 'color'.")

    return [parent_val, current_val]

def target_plot_probabilities(data_dir, targets='digit', gif_output="output.gif", gif=False, epoch_start=None, epoch_end=None, epoch_step=None):
    data = load_data(data_dir)
    epochs = sorted(data.keys())
    print("kkk")
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

    # fig_and_logディレクトリ作成
    fig_and_log_dir = os.path.join("alpha_test/EMNIST/cnn_5layers_emnist_digits_variance100_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.0_Optimsgd_Momentum0.0/no_noise_no_noise/0/2/", "fig_and_log")
    os.makedirs(fig_and_log_dir, exist_ok=True)
    gif_output_path = os.path.join(fig_and_log_dir, gif_output)

    # highlight対象取得
    highlight_targets = get_highlight_targets(data_dir, targets)

    fig = plt.figure()  # figsizeやdpiはrcParamsで指定済み
    ax = fig.add_subplot(111)
    # スライダー領域確保（gif=False時用）
    plt.subplots_adjust(bottom=0.25)

    alpha_values = df['alpha']
    lines = []
    for target_val in range(10):
        column_name = f'{targets}_probability_{target_val}'
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' does not exist in the data.")
        # highlight_targetsに含まれる場合は実線、それ以外は点線
        if target_val in highlight_targets:
            line_style = '-'
        else:
            line_style = '--'
        line, = ax.plot(alpha_values, df[column_name], line_style, label=f'{targets}:{target_val}')
        lines.append(line)

    ax.set_xlabel('Alpha')
    ax.set_ylabel(f'{targets} Probability')
    ax.set_title(f'{targets} Probability for Epoch {initial_epoch}')
    ax.set_ylim(-0.1, 1.1)

    # 凡例をグラフの右側外部に配置
    legend = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)

    # 凡例が見切れないように右側に余白を確保
    fig.subplots_adjust(right=0.8)

    def update(epoch):
        if epoch not in data:
            raise KeyError(f"Epoch {epoch} not found in data.")
        df_epoch = data[epoch]
        ax.set_title(f'{targets} Probability for Epoch {epoch}')
        for t, line in enumerate(lines):
            column_name = f'{targets}_probability_{t}'
            if column_name not in df_epoch.columns:
                raise KeyError(f"Column '{column_name}' does not exist in the data.")
            line.set_ydata(df_epoch[column_name])
        return fig,

    if gif:
        # gif=Trueの場合、インタラクティブな表示は行わず、即座にGIFを作成
        def animate(frame):
            epoch = epochs[frame]
            update(epoch)
            return fig,

        anim = FuncAnimation(fig, animate, frames=len(epochs), interval=200, blit=True)
        anim.save(gif_output_path, writer=PillowWriter(fps=10))
        print(f"GIF saved as {gif_output_path}")
        plt.close(fig)
    else:
        # gif=Falseの場合のみスライダー表示とGIFボタン生成
        ax_epoch = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        # Sliderに渡す valstep は離散値のリストでもよいが、epochsが飛び飛びの場合はvalstepにリストを渡します
        # この場合、valstep=epochs とすれば、スライダーのステップはepochsの値のみになります
        epoch_slider = Slider(ax_epoch, 'Epoch', epochs[0], epochs[-1], valinit=initial_epoch, valstep=epochs)

        ax_button = plt.axes([0.85, 0.025, 0.1, 0.04])
        button = Button(ax_button, 'Save GIF')

        def slider_update(val):
            epoch = int(epoch_slider.val)
            update(epoch)
            fig.canvas.draw_idle()

        epoch_slider.on_changed(slider_update)

        def save_gif(event):
            def animate(frame):
                epoch = epochs[frame]
                update(epoch)
                return fig,

            anim = FuncAnimation(fig, animate, frames=len(epochs), interval=200, blit=True)
            anim.save(gif_output_path, writer=PillowWriter(fps=10))
            print(f"GIF saved as {gif_output_path}")

        button.on_clicked(save_gif)

        plt.show()

def process_all_subdirs(root_dir, targets='digit', gif=False, selected_dirs=None, epoch_start=None, epoch_end=None, epoch_step=None):
    all_subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    if selected_dirs is not None:
        subdirs_to_process = [d for d in all_subdirs if d in selected_dirs]
    else:
        subdirs_to_process = all_subdirs

    for name in subdirs_to_process:
        sub_dir = os.path.join(root_dir, name)
        target_plot_probabilities(sub_dir, targets=targets, gif_output=f"{targets}.gif",gif=gif, epoch_start=epoch_start, epoch_end=epoch_end, epoch_step=epoch_step)

# 使用例
# epoch_start=0, epoch_end=2000の範囲で、epoch_step=10として10 epochごとに可視化やGIF作成する
root_dir = "alpha_test/EMNIST/cnn_5layers_emnist_digits_variance100_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.0_Optimsgd_Momentum0.0/no_noise_no_noise/0/2/csv"
#process_all_subdirs(root_dir, targets='color', gif=True, selected_dirs=["10","20","30","40","50","60","70","80","90"], epoch_start=0, epoch_end=2000, epoch_step=10)
process_all_subdirs(root_dir, targets='digit', gif=True, selected_dirs=["2"], epoch_start=0, epoch_end=2000, epoch_step=10)


# 使用例
# line_style='-'で実線、'--'で点線など
# target_plot_probabilities("/path/to/data_dir", targets='digit', epoch_start=0, epoch_end=2000, epoch_step=5, line_style='--')

# epoch_start=0, epoch_end=2000の範囲で、epoch_step=10として10 epochごとに可視化やGIF作成する
#root_dir = "/workspace/alpha_test/cnn_5layers_distribution_colored_emnist_variance10000_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.5_Optimsgd/no_noise_noise/0"
#process_all_subdirs(root_dir, targets='digit', gif=True, selected_dirs=["10","20","30","40","50","60","70","80","90"], epoch_start=0, epoch_end=2000, epoch_step=5)
#process_all_subdirs(root_dir, targets='digit', gif=True, selected_dirs=["20",], epoch_start=0, epoch_end=2000, epoch_step=10)

"""list =["10","20","30","40","50","60","70","80","90"]
list2 = ["55","9","27","89","67","82","58","91","71"]
#list =["0"]
#list2=["26"]
c = 0
for n1 in list:
    root_dir = f"/workspace/alpha_test/cnn_5layers_distribution_colored_emnist_variance10000_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.5_Optimsgd/no_noise_noise/{n1}"
    process_all_subdirs(root_dir, targets='digit', gif=True, selected_dirs=[list2[c]], epoch_start=0, epoch_end=2000, epoch_step=5)
    c+=1
"""
