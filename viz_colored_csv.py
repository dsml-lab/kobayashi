import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.lines import Line2D  # ダミー凡例作成用
import numpy as np
import matplotlib.ticker as ticker

# --- rcParams は固定値 (または不要ならコメントアウト) ---
plt.rcParams["font.size"] = 25
plt.rcParams["figure.figsize"] = (26, 18)  # ← ここを固定
plt.rcParams["figure.dpi"] = 100
#plt.rcParams["figure.dpi"] = 500
plt.rcParams['font.family'] = 'DejaVu Sans'

def load_data(data_dir, targets='color'):
    csv_dir = os.path.join(data_dir, "csv")
    data = {}
    
    if targets == 'combined':
        # combinedの場合はraw_probabilities_epoch_*.csvを読み込む
        files = sorted(glob.glob(os.path.join(csv_dir, "raw_probabilities_epoch_*.csv")))
        for file in files:
            epoch = int(os.path.basename(file).split('_')[-1].split('.')[0])
            data[epoch] = pd.read_csv(file)
    else:
        # color/digitの場合は従来通りalpha_log_epoch_*.csvを読み込む
        files = sorted(glob.glob(os.path.join(csv_dir, "alpha_log_epoch_*.csv")))
        for file in files:
            epoch = int(os.path.basename(file).split('_')[-1].split('.')[0])
            data[epoch] = pd.read_csv(file)
    
    return data

def extract_digit_for_digit_mode(name):
    if len(name) == 0:
        raise ValueError("Directory name is empty.")
    if len(name) == 1:
        tens_digit = 0
    else:
        tens_digit = int(name[0])
    return tens_digit

def extract_digit_for_color_mode(name):
    if len(name) == 0:
        raise ValueError("Directory name is empty.")
    last_char = name[-1]
    if not last_char.isdigit():
        raise ValueError(f"Directory name does not end with a digit: '{name}'")
    return int(last_char)

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

def determine_vertical_lines(data_dir):
    if "no_noise_no_noise" in data_dir:
        return {'alpha_0': 'black', 'alpha_1': 'black'}
    elif "noise_no_noise" in data_dir:
        return {'alpha_0': 'black', 'alpha_1': 'black'}
    else:
        return {'alpha_0': 'black', 'alpha_1': 'black'}

def target_plot_probabilities_single_epoch(
    data_dir, 
    epoch,
    targets='color',
    savefig=True,
    show_legend=True,
    show_ylabel=True,
    show_yticklabels=True,
    show_xlabel=True,
    show_xticks=True,
    ax=None
):
    data = load_data(data_dir, targets)
    if epoch not in data:
        print(f"Epoch {epoch} のデータが {data_dir} に存在しません。")
        return
    
    df = data[epoch]
    fig_and_log_dir = os.path.join(data_dir, "fig_and_log")
    os.makedirs(fig_and_log_dir, exist_ok=True)

    highlight_targets = get_highlight_targets(data_dir, targets)

    # --- Axes が指定されていない場合、新しい Figure を (12, 8) で作成 ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))  # ← 修正
    else:
        fig = ax.figure

    alpha_values = df['alpha']
    other_colors = ['green','orange','purple','brown','gray','pink','olive','cyan','lime','navy']
    dotted_color_cycle = cycle(other_colors)

    if targets == 'combined':
        # combinedモードの場合、probability_0からprobability_99までを描画
        for target_val in range(100):
            column_name = f'probability_{target_val}'
            if column_name not in df.columns:
                raise KeyError(f"Column '{column_name}' が存在しません。")

            if target_val == highlight_targets[0]:
                color = 'blue'
                line_style = '-'
                line_label = 'Original label'
                line_alpha = 1.0
                line_width = 2.0
            elif target_val == highlight_targets[1]:
                color = 'red'
                line_style = '-'
                line_label = 'Label after label noise'
                line_alpha = 1.0
                line_width = 2.0
            else:
                color = next(dotted_color_cycle)
                line_style = '--'
                line_label = None
                line_alpha = 1.0
                line_width = 1.0

            ax.plot(
                alpha_values, 
                df[column_name], 
                line_style, 
                color=color, 
                alpha=line_alpha,
                linewidth=line_width,  
                label=line_label
            )
    else:
        # 既存のcolor/digitモードの処理
        for target_val in range(10):
            column_name = f'{targets}_probability_{target_val}'
            if column_name not in df.columns:
                raise KeyError(f"Column '{column_name}' が存在しません。")

            if target_val == highlight_targets[0]:
                color = 'blue'
                line_style = '-'
                line_label = 'Original label'
                line_alpha = 1.0
                line_width = 2.0
            elif target_val == highlight_targets[1]:
                color = 'red'
                line_style = '-'
                line_label = 'Label after label noise'
                line_alpha = 1.0
                line_width = 2.0
            else:
                color = next(dotted_color_cycle)
                line_style = '--'
                line_label = None
                line_alpha = 1.0
                line_width = 1.0

            ax.plot(
                alpha_values, 
                df[column_name], 
                line_style, 
                color=color, 
                alpha=line_alpha,
                linewidth=line_width,  
                label=line_label
            )

    vline_colors = determine_vertical_lines(data_dir)
    ax.axvline(x=0.0, color=vline_colors['alpha_0'], linestyle='-', linewidth=1.0, label=None)
    ax.axvline(x=1.0, color=vline_colors['alpha_1'], linestyle='-', linewidth=1.0, label=None)

    if show_xlabel:
        ax.set_xticks([0.0, 1.0], [r'$x_0$', r'$x_1$'])

        # 目盛りに対応するラベルを設定
        ax.set_xticklabels([r'$x_0$', r'$x_1$'], fontsize=28)  # フォントサイズは必要に応じて調整

        
    else:
        ax.set_xlabel('')

    if show_xticks:
        ax.set_xticks([0.0, 1.0], [r'$x_0$', r'$x_1$'])

        # 目盛りに対応するラベルを設定
        ax.set_xticklabels([r'$x_0$', r'$x_1$'], fontsize=30) 
    else:
        ax.set_xticks([])

    if show_ylabel:
        ax.set_ylabel('probability', fontsize=28)
    if not show_yticklabels:
        ax.tick_params(axis='y', which='both', labelleft=False)

    ax.set_ylim(-0.1, 1.1)
    marker_size = 10
    if "no_noise_no_noise" in data_dir:
        ax.plot(0.0, 0.0, marker='o', color='blue', markersize=marker_size)
        ax.plot(1.0, 0.0, marker='o', color='blue', markersize=marker_size)
    elif "noise_no_noise" in data_dir:
        ax.plot(0.0, 0.0, marker='o', color='blue', markersize=marker_size)
        ax.plot(1.0, 0.0, marker='o', color='red', markersize=marker_size)

    if show_legend and ax is not None:
        handles, labels = ax.get_legend_handles_labels()
        others_line = Line2D([0],[0], color='black', linestyle='--', label='others', linewidth=1.0, alpha=1.0)
        handles.append(others_line)
        labels.append("others")
        unique = list(dict(zip(labels, handles)).items())
        if unique:
            unique_labels, unique_handles = zip(*unique)
            ax.legend(unique_handles, unique_labels, loc="upper right")

    if ax is None:
        if savefig:
            save_path = os.path.join(fig_and_log_dir, f"{targets}_epoch{epoch}.png")
            plt.savefig(save_path, save_path,format='svg', dpi=100, bbox_inches='tight', transparent=False)
            #plt.savefig(save_path, save_path,format='svg', dpi=300, bbox_inches='tight', transparent=False)
            print(f"画像を保存しました: {save_path}")
            plt.close(fig)
        else:
            plt.show()

def plot_multiple_epochs_grid(
    data_dir,
    epochs,
    targets='color',
    savefig=True,
    show_legend=True,
    show_ylabel=True,
    show_yticklabels=True,
    save_filename='combined_plot.png',
    columns=4
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import os

    num_epochs = len(epochs)
    rows = (num_epochs + columns - 1) // columns

    # --- ここを固定サイズにする ---
    # 修正前: fig_width = 12 * columns; fig_height = 8 * rows
    # 修正後: 例として一律 (12, 8)
    fig, axes = plt.subplots(rows, columns, figsize=(20, 8), dpi=100)
    #fig, axes = plt.subplots(rows, columns, figsize=(20, 8), dpi=400)
    if rows == 1 and columns == 1:
        axes = np.array([[axes]])
    elif rows == 1 or columns == 1:
        axes = np.reshape(axes, (rows, columns))
    else:
        axes = np.array(axes).reshape(rows, columns)

    last_row = rows - 1
    list = ["H","A","B","C","D","E","F","G"]
    i=0
    for idx, epoch in enumerate(epochs):
        row = idx // columns
        col = idx % columns
        ax = axes[row, col]

        current_show_xlabel = (row == last_row)
        current_show_xticks = (row == last_row)
        current_show_ylabel = (col == 0 and show_ylabel)
        current_show_yticklabels = (col == 0 and show_yticklabels)

        target_plot_probabilities_single_epoch(
            data_dir=data_dir,
            epoch=epoch,
            targets=targets,
            savefig=False,
            show_legend=False,
            show_ylabel=current_show_ylabel,
            show_yticklabels=current_show_yticklabels,
            show_xlabel=current_show_xlabel,
            show_xticks=current_show_xticks,
            ax=ax
        )
        ax.set_title(f'epoch{epoch} ({list[i]})')
        i=i+1
    total_subplots = rows * columns
    for idx in range(num_epochs, total_subplots):
        row = idx // columns
        col = idx % columns
        fig.delaxes(axes[row, col])

    if show_legend and axes.size > 0:
        handles, labels = ax.get_legend_handles_labels()
        others_line = Line2D([0],[0], color='black', linestyle='--', label='others', linewidth=1.0, alpha=1.0)
        handles.append(others_line)
        labels.append("others")
        unique = list(dict(zip(labels, handles)).items())
        if unique:
            unique_labels, unique_handles = zip(*unique)
            axes[0, 0].legend(unique_handles, unique_labels, loc="upper right")

    if savefig:
        save_path = f"/workspace/miru_vizualize/sigma_0/C_C/daihyou/{save_filename}"
        #後でsvgに変える
        plt.savefig(save_path,format='pdf', dpi=100, bbox_inches='tight', transparent=False)
        #plt.savefig(save_path,format='pdf', dpi=400, bbox_inches='tight', transparent=False)
        print(f"結合した画像を保存しました: {save_path}")
        plt.close(fig)
    else:
        plt.show()

def main():
    #48 30/65

    root_dir = f"/workspace/alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/62/37"
    #root_dir = f"/workspace/alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/no_noise_no_noise/94/94"

    combined_epochs = [1,5,30,50,60,100,140,1000]
    #[20, 70, 100, 120, 150, 300, 500, 2000]
    plot_multiple_epochs_grid(
        data_dir=root_dir,
        epochs=combined_epochs,
        targets="combined",
        savefig=True,
        show_legend=False,
        show_ylabel=True,
        show_yticklabels=True,
        save_filename=f'C_N_combined_plot.svg',
        columns=4
    )

def main_all_dirs():
    print("main_all_dirs")
    base_dir = "/workspace/alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/no_noise_no_noise"
    combined_epochs =[1,5,30,50,60,100,140,1000]
    # 第一階層のディレクトリを取得（1, 10, 11など）
    first_level_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for first_dir in first_level_dirs:
        first_path = os.path.join(base_dir, first_dir)
        # 第二階層のディレクトリを取得（38, 53など）
        second_level_dirs = [d for d in os.listdir(first_path) if os.path.isdir(os.path.join(first_path, d))]
        
        for second_dir in second_level_dirs:
            target_dir = os.path.join(first_path, second_dir)
            print(f"Processing directory: {first_dir}/{second_dir}")
            
            plot_multiple_epochs_grid(
                data_dir=target_dir,
                targets="combined",
                savefig=True,
                show_legend=False,
                show_ylabel=True,
                show_yticklabels=True,
                save_filename=f"output_{first_dir}_{second_dir}.pdf",
                epochs=combined_epochs,
                columns=4
            )

if __name__ == "__main__":
    #main_all_dirs()
    main()