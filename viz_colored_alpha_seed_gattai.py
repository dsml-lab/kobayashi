import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from smoothing import moving_average  # スムージング関数のインポート
from viz_colored_alpha_gif_csv_miseruyou import target_plot_probabilities

# グローバルなプロット設定
plt.rcParams["font.size"] = 23
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["figure.dpi"] = 400
plt.rcParams['font.family'] = 'DejaVu Sans'

def main_process_csvs(csv_directories, targets=["combined"]):
    """
    CSVファイルを処理する関数
    この関数は viz_colored_alpha_gif_csv_miseruyou.py から import する必要があります
    """
    for directory in csv_directories:
        try:
            target_plot_probabilities(
                data_dir=directory,
                targets="combined",
                gif=True,
                show_legend=False,
                gif_output="output.gif",
                epoch_start=0,
                epoch_end=150,
                epoch_step=1
            )
        except Exception as e:
            print(f"Error processing {directory}: {e}")

def evaluate_label_changes(
    directory, 
    mode='alpha', 
    target='digit', 
    y_lim=None,
    smoothing=False,           
    smoothing_window=5,        
    y_scale='ratio',
    plot_result=True,
    epoch_start=None,  # 追加：開始エポック
    epoch_end=None     # 追加：終了エポック
):
    """
    CSVファイルからラベル変更を評価し、結果をCSVとして保存します。
    さらに、オプションで作成したCSVファイルからプロットを作成し保存します。

    Parameters
    ----------
    directory : str
        データが格納されているディレクトリのパス。
    mode : str
        'alpha' または 'epoch' のモードを選択。
    target : str
        'digit', 'color', 'combined' のいずれか。
    y_lim : tuple, optional
        y軸の範囲を指定。
    smoothing : bool, optional
        スムージングを適用するかどうか。
    smoothing_window : int, optional
        スムージングのウィンドウサイズ。
    y_scale : str, optional
        'ratio' または 'percent' のスケールを選択。
    plot_result : bool, optional
        CSV保存後にプロットを作成し保存するかどうか。デフォルトはFalse。
    epoch_start : int, optional
        処理を開始するエポック番号
    epoch_end : int, optional
        処理を終了するエポック番号
    """
    if target not in ['digit', 'color', "combined"]:
        raise ValueError("Invalid target. Choose 'digit', 'color', or 'combined'.")
    if mode not in ['alpha', 'epoch']:
        raise ValueError("Invalid mode. Choose 'alpha' or 'epoch'.")
    if y_scale not in ['ratio', 'percent']:
        raise ValueError("Invalid y_scale. Choose 'ratio' or 'percent'.")

    csv_dir = os.path.join(directory, "csv")
    save_dir = os.path.join(directory, "fig_and_log")
    os.makedirs(save_dir, exist_ok=True)

    # CSVファイルの取得とフィルタリング
    all_files = sorted([
        f for f in os.listdir(csv_dir)
        if f.startswith("alpha_log_epoch_") and f.endswith(".csv")
    ])

    if not all_files:
        raise ValueError("No valid files found in the specified directory.")

    # エポック番号の抽出と範囲フィルタリング
    files = []
    for f in all_files:
        epoch_str = f.split('_')[-1].split('.')[0]
        try:
            epoch = int(epoch_str)
            if epoch_start is not None and epoch < epoch_start:
                continue
            if epoch_end is not None and epoch > epoch_end:
                continue
            files.append(os.path.join(csv_dir, f))
        except ValueError:
            print(f"Warning: Skipping file with invalid epoch number: {f}")

    if not files:
        raise ValueError("No files found in the specified epoch range.")

    # ファイル名の接尾辞を作成
    epoch_suffix = ""
    if epoch_start is not None or epoch_end is not None:
        start_str = str(epoch_start) if epoch_start is not None else "start"
        end_str = str(epoch_end) if epoch_end is not None else "end"
        epoch_suffix = f"_epoch_{start_str}_to_{end_str}"

    target_col = 'predicted_' + target
    scores = {}

    if mode == 'alpha':
        # alphaモード：各CSVファイルから最初の202行を使い、ラベル変更率を計算
        for file in files:
            df = pd.read_csv(file).iloc[:202]  # 最初の202行を使用
            n = len(df) - 1
            changes = ((df[target_col] != df[target_col].shift()).sum() - 1) / n
            if y_scale == 'percent':
                changes *= 100
            epoch_str = os.path.basename(file).split('_')[-1].split('.')[0]
            try:
                epoch = int(epoch_str)
            except ValueError:
                raise ValueError(f"Invalid epoch number in file name: {file}")
            scores[epoch] = changes

        # エポックで昇順にソート
        score_df = pd.DataFrame(list(scores.items()), columns=['epoch', 'label_change'])
        score_df = score_df.sort_values('epoch')  # エポックで昇順ソート
        csv_filename = os.path.join(save_dir, f"label_change_scores_alpha_{target}{epoch_suffix}.csv")
        score_df.to_csv(csv_filename, index=False)
        print(f"Scores saved as {csv_filename}")

        # オプションがTrueの場合、保存したCSVを用いてプロットを作成（複数シードではなく1ファイルのみ）
        if plot_result:
            # CSVからデータを読み込み
            score_df = pd.read_csv(csv_filename)
            epochs = score_df['epoch'].values
            label_changes = score_df['label_change'].values
            # 単一ファイルなので標準偏差はゼロとしてプロット（塗りつぶし部分は線と同一）
            std_scores = np.zeros_like(label_changes)
            
            fig, ax = plt.subplots(figsize=(12, 8), dpi=400)
            ax.plot(epochs, label_changes, label="$INST_s(\\chi,t)$", color="blue", linewidth=1.5, zorder=3)
            ax.fill_between(epochs, label_changes - std_scores, label_changes + std_scores,
                            color="blue", alpha=0.2, zorder=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Spatial Instability" if y_scale == 'ratio' else "Spatial Instability")
            if y_lim:
                ax.set_ylim(y_lim)
            ax.set_xscale('log')
            
            plot_filename = os.path.splitext(os.path.basename(csv_filename))[0] + "_plot.svg"
            plot_path = os.path.join(save_dir, plot_filename)
            fig.savefig(plot_path, bbox_inches='tight')
            plt.close(fig)
            print(f"Plot saved as {plot_path}")

    elif mode == 'epoch':
        if len(files) < 2:
            raise ValueError("At least two files are required for 'epoch' mode.")

        dataframes = [pd.read_csv(file, usecols=['alpha', target_col]).iloc[:202] for file in files]
        alpha_values = dataframes[0]['alpha'].tolist()
        unsmoothed_scores = [0] * 202  

        for i in range(len(files) - 1):
            df_current = dataframes[i]
            df_next = dataframes[i + 1]
            row_changes = (df_current[target_col] != df_next[target_col]).astype(int)
            unsmoothed_scores = [score + change for score, change in zip(unsmoothed_scores, row_changes)]
        
        # エポック範囲に基づいてスコアを正規化
        total_epochs = len(files) - 1
        unsmoothed_scores = [score / total_epochs for score in unsmoothed_scores]

        # ファイル名にエポック範囲を追加
        epoch_csv_path = os.path.join(save_dir, f"epoch_unsmoothed_scores_{target}{epoch_suffix}.csv")
        epoch_data = pd.DataFrame({'alpha': alpha_values, 'unsmoothed_scores': unsmoothed_scores})
        epoch_data.to_csv(epoch_csv_path, index=False)
        print(f"Epoch scores saved to {epoch_csv_path}")

        if plot_result:
            # epochモードの場合、'alpha'と'unsmoothed_scores'を用いてプロット
            df_epoch = pd.read_csv(epoch_csv_path)
            alphas = df_epoch['alpha'].values
            scores_epoch = df_epoch['unsmoothed_scores'].values
            std_scores = np.zeros_like(scores_epoch)
            
            fig, ax = plt.subplots(figsize=(12, 8), dpi=400)
            ax.plot(alphas, scores_epoch, label="$INST_s(\\chi,t)$", color="blue", linewidth=1.5, zorder=3)
            ax.fill_between(alphas, scores_epoch - std_scores, scores_epoch + std_scores,
                            color="blue", alpha=0.2, zorder=2)
            ax.set_xlabel("Alpha")
            ax.set_ylabel("Temporal Instability" if y_scale == 'ratio' else "Temporal Instability")
            if y_lim:
                ax.set_ylim(y_lim)
            
            # プロットファイル名にもエポック範囲を追加
            plot_filename = f"epoch_unsmoothed_scores_{target}{epoch_suffix}_plot.png"
            plot_path = os.path.join(save_dir, plot_filename)
            fig.savefig(plot_path, bbox_inches='tight')
            plt.close(fig)
            print(f"Plot saved as {plot_path}")

    return scores

def plot_label_changes_across_seeds(
    csv_paths,
    output_path,
    target="digit",
    y_scale='ratio',
    ylim=None,
    highlight_epochs=None
):
    """
    複数のシードによって生成された CSV ファイルを読み込み、
    epoch 毎の平均と標準偏差をプロットする。

    Parameters
    ----------
    csv_paths : list of str
        プロット対象のディレクトリまたはCSVファイルパスのリスト。
    output_path : str
        プロットを保存するファイルパス（保存先とファイル名）。例: "output/figure.png"
    target : str, optional
        ターゲット。デフォルトは "digit"。
    y_scale : str, optional
        'ratio' または 'percent' の指定。デフォルトは 'ratio'。
    ylim : tuple, optional
        y軸の範囲を指定。
    highlight_epochs : list, optional
        強調表示するエポックのリスト。
    """
    df_list = []
    for path in csv_paths:
        try:
            # パスがディレクトリの場合
            if os.path.isdir(path):
                csv_dir = os.path.join(path, "fig_and_log")
                csv_file = os.path.join(csv_dir, f"label_change_scores_alpha_{target}.csv")
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    df = df[df["epoch"] >= 1].sort_values(by="epoch")  # 1epoch以上を選択
                    df_list.append(df)
                else:
                    print(f"[Warning] File not found: {csv_file}")
            # パスが直接CSVファイルの場合
            elif os.path.isfile(path):
                df = pd.read_csv(path)
                df = df[df["epoch"] >= 1].sort_values(by="epoch")  # 1epoch以上を選択
                df_list.append(df)
            else:
                print(f"[Warning] Invalid path: {path}")
        except Exception as e:
            print(f"[Warning] Error processing {path}: {e}")

    if not df_list:
        print("No valid CSV found. Process aborted.")
        return

    # 1epoch以上のデータのみを対象
    epochs = df_list[0]["epoch"].values
    all_scores = np.array([df["label_change"].values for df in df_list])
    mean_scores = np.mean(all_scores, axis=0)
    std_scores = np.std(all_scores, axis=0)

    if y_scale == 'percent':
        mean_scores *= 100
        std_scores *= 100

    fig, ax = plt.subplots(figsize=(12, 10), dpi=400)
    if highlight_epochs:
        for epoch in highlight_epochs:
            if epoch >= 1:
                ax.axvline(x=epoch, color='black', linestyle='-', linewidth=1.5, zorder=1)
    
    ax.plot(epochs, mean_scores, label="$INST_s(\\chi,t)$", color="blue", linewidth=3, zorder=3)
    ax.fill_between(epochs, mean_scores - std_scores, mean_scores + std_scores,
                    color="blue", alpha=0.2, zorder=2)
    ax.plot([], [], label=r'$\mathcal{X}=[\mathit{x}(-\frac{1}{2}),\mathit{x}(\frac{3}{2})]$', linewidth=0,)
    
    legend = ax.legend(fontsize=32, loc='upper center')
    legend.get_frame().set_alpha(1.0)
    ax.set_xlabel("Epoch", fontsize=45)
    ax.set_ylabel("Spatial Instability" if y_scale == 'ratio' else "Spatial Instability", fontsize=45)
    
    if ylim:
        ymin, ymax = ylim
    else:
        ymin, ymax = ax.get_ylim()
    
    yticks = np.arange(np.floor(ymin * 100) / 100, np.ceil(ymax * 100) / 100 + 0.005, 0.01)
    ax.set_yticks(yticks)
    if ylim:
        ax.set_ylim(ylim)

    ax.set_xscale('log')
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=35)

    ax.grid(True)
    fig.tight_layout()

    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # ファイル名の拡張子がない場合、.pngを追加
    if not os.path.splitext(output_path)[1]:
        output_path = output_path + ".svg"
    
    fig.savefig(output_path, bbox_inches=None)
    plt.close(fig)
    print(f"[Info] Figure saved to: {output_path}")

def main3(csv_directories, base_output_dir, targets=["digit","color","combined"], output_name="a", epoch_start=None, epoch_end=None, plot_save_path=None):
    """
    CSVファイルを読み込み、各ターゲットごとにプロットを作成して保存します。
    α 毎の平均と標準偏差をプロットする。

    Parameters
    ----------
    csv_directories : list of str
        プロット対象の CSV ディレクトリパスのリスト。
    base_output_dir : str
        プロットを保存するベースの出力ディレクトリのパス。
    targets : list of str
        プロット対象のターゲットリスト（例: ["digit", "color", "combined"]）。
    output_name : str
        出力ファイルの基本名。
    epoch_start : int, optional
        開始エポック番号。
    epoch_end : int, optional
        終了エポック番号。
    plot_save_path : str, optional
        プロットを保存するファイル名の接頭辞。指定がある場合、ファイル名は
        plot_save_path + {target}{epoch_suffix}.svg となります。
    """
    os.makedirs(base_output_dir, exist_ok=True)

    # エポック範囲の文字列を作成
    epoch_suffix = ""
    if epoch_start is not None or epoch_end is not None:
        start_str = str(epoch_start) if epoch_start is not None else "start"
        end_str = str(epoch_end) if epoch_end is not None else "end"
        epoch_suffix = f"_epoch_{start_str}_to_{end_str}"

    for target in targets:
        data_list = []
        groups_present = set()

        # CSV読み込み
        for directory in csv_directories:
            # エポック範囲を含むファイル名を使用
            csv_filename = f"epoch_unsmoothed_scores_{target}{epoch_suffix}.csv"
            csv_path = os.path.join(directory, 'fig_and_log', csv_filename)

            if not os.path.exists(csv_path):
                print(f"[Warning] CSVファイルが存在しません: {csv_path}")
                continue

            # グループ分類
            if 'no_noise_no_noise' in directory:
                group = 'no_noise_no_noise'
            elif 'noise_no_noise' in directory:
                group = 'noise_no_noise'
            else:
                group = 'other'
            groups_present.add(group)

            try:
                df = pd.read_csv(csv_path)
                df = df[['alpha', 'unsmoothed_scores']]
                # シード情報を抽出
                seed_match = re.search(r'seed(\d+)', directory)
                seed = seed_match.group(1) if seed_match else "unknown"
                df['target'] = target
                df['seed'] = seed
                data_list.append(df)
            except Exception as e:
                print(f"CSVファイルの読み込みエラー ({csv_path}): {e}")

        if not data_list:
            print(f"[Warning] データが収集されませんでした: {target}")
            continue

        # データ集計
        combined_df = pd.concat(data_list, ignore_index=True)
        stats_df = combined_df.groupby('alpha')['unsmoothed_scores'].agg(['mean', 'std']).reset_index()

        # --- Figure作成 ---
        fig, ax = plt.subplots(figsize=(10, 6), dpi=400)

        marker_size = 20
        if "no_noise_no_noise" in directory:
            ax.plot(0.0, 0.0, marker='o', color='blue', markersize=marker_size, zorder=5)
            ax.plot(1.0, 0.0, marker='o', color='blue', markersize=marker_size)
        elif "noise_no_noise" in directory:
            ax.plot(0.0, 0.0, marker='o', color='blue', markersize=marker_size)
            ax.plot(1.0, 0.0, marker='o', color='red', markersize=marker_size)
        else:
            ax.plot(0.0, 0.0, marker='o', color='blue', markersize=marker_size, zorder=5)
            ax.plot(1.0, 0.0, marker='o', color='red', markersize=marker_size)
        # X 軸
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.set_yticks([0, 0.2, 0.4, 0.6])
        ax.set_yticklabels(["0", "0.2", "0.4", "0.6"], fontsize=30)  # フォントサイズを適用
        ax.set_ylim(-0.01, 0.6)

        if "no_noise_no_noise" in csv_directories:
            ax.plot(0.0, 0.0, marker='o', color='black',
                    markerfacecolor='none', markeredgecolor='black',
                    markersize=marker_size)
            ax.plot(1.0, 0.0, marker='o', color='black',
                    markerfacecolor='none', markeredgecolor='black',
                    markersize=marker_size)
        elif "noise_no_noise" in csv_directories:
            ax.plot(0.0, 0.0, marker='o', color='black',
                    markerfacecolor='none', markeredgecolor='black',
                    markersize=marker_size)
            ax.plot(1.0, 0.0, marker='x', color='black',
                    markersize=marker_size)

        # 平均＆標準偏差を描画
        ax.plot(stats_df['alpha'], stats_df['mean'],
                label=r'$INST_{\mathrm{T}}(\mathcal{T},x)$', color='blue', linewidth=3, zorder=2)
        ax.fill_between(stats_df['alpha'],
                        stats_df['mean'] - stats_df['std'],
                        stats_df['mean'] + stats_df['std'],
                        color='blue', alpha=0.15,
                        zorder=1)

        # ダミープロットを追加して 'T=(0,1000)' をレジェンドに含める
        ax.plot([], [], label=rf'$\mathcal{{T}}$=({epoch_start},{epoch_end})', linewidth=0)

        for group in groups_present:
            if group == 'noise_no_noise':
                ax.axvline(x=0.0, color='black', linestyle='-', linewidth=2, zorder=0)
                ax.axvline(x=1.0, color='black', linestyle='-', linewidth=2, zorder=0)
            elif group == 'noise_no_noise':  # （意図的に同条件が2回記述されています）
                ax.axvline(x=0.0, color='black', linestyle='-', linewidth=2, zorder=0)
                ax.axvline(x=1.0, color='red', linestyle='-', linewidth=2, zorder=0)
            else:
                ax.axvline(x=0.0, color='black', linestyle='-', linewidth=2, zorder=0)
                ax.axvline(x=1.0, color='black', linestyle='-', linewidth=2, zorder=0)
        ax.set_xticks([0.0, 1.0], [r'$x_2$', r'$x_3$'])
        ax.set_xticklabels([r'$x_2$', r'$x_3$'], fontsize=50)
        ax.set_ylabel("Temporal Instability", fontsize=30)
        ax.grid(True)

        legend = ax.legend(fontsize=30)
        legend.get_frame().set_alpha(1.0)  
        fig.tight_layout()

        # --- 保存処理 ---
        if plot_save_path is not None:
            # 指定された接頭辞に target と epoch_suffix を連結してファイル名とする
            plot_filename = f"{plot_save_path}{target}{epoch_suffix}.svg"
        else:
            plot_filename = f'width_4_C_N_output_{target}{epoch_suffix}.svg'
        plot_path = os.path.join(base_output_dir, plot_filename)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

        fig.savefig(plot_path, format='svg', bbox_inches=None)
        plt.close(fig)
        print(f"[Info] プロットを保存しました: {plot_path}")
def process_all_subdirectories():
    """
    二階層下のすべてのディレクトリに対してevaluate_label_changesを実行する関数
    
    ディレクトリ構造:
    base_dir/
        1/
            38/
        2/
            57/
        3/
            9/
        ...
    """    
    print("start_subrotin")
    base_dir = "/alpha_test/seed/width_4/lr_0.2/ori_closet_seed_{}width4_cnn_5layers_distribution_colored_emnist_variance1000_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise"
    
    # ディレクトリの存在確認
    if not os.path.exists(base_dir):
        print(f"Base directory does not exist: {base_dir}")
        return
    
    # 第一階層のディレクトリを取得
    try:
        first_level_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        print(f"Found first level directories: {first_level_dirs}")
    except Exception as e:
        print(f"Error listing directory {base_dir}: {e}")
        return
    
    for first_dir in sorted(first_level_dirs, key=lambda x: int(x)):  # 数値順にソート
        first_path = os.path.join(base_dir, first_dir)
        
        # 第二階層のディレクトリを取得
        try:
            second_level_dirs = [d for d in os.listdir(first_path) if os.path.isdir(os.path.join(first_path, d))]
            if second_level_dirs:  # 空でない場合
                second_level_path = os.path.join(first_path, second_level_dirs[0])  # 最初のディレクトリを使用
                print(f"Processing directory: {second_level_path}")
                try:
                    evaluate_label_changes(
                        directory=second_level_path,
                        target="combined",
                        mode="alpha",
                        y_lim=(0,0.03),
                    )
                    evaluate_label_changes(
                        directory=second_level_path,
                        target="combined",
                        mode="epoch",
                        y_lim=(0,0.4),
                    )
                    print(f"Successfully processed: {second_level_path}")
                except Exception as e:
                    print(f"Error processing directory {second_level_path}: {e}")
                    import traceback
                    print(traceback.format_exc())
        except Exception as e:
            print(f"Error accessing directory {first_path}: {e}")


def main_combine_seed_results():
    # 基本パスのパターン
    noise="62"
    no_noise="37"
    # noise="94"
    # no_noise="94"
    base_pattern = "/workspace/alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_{}width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/62/37"
    # 使用するシード値
    seeds = range(42, 52)  # 42から51まで
    
    # 有効なディレクトリを収集
    csv_directories = []
    for seed in seeds:
        dir_path = base_pattern.format(seed)
        if os.path.exists(dir_path):
            csv_directories.append(dir_path)
        else:
            print(f"Warning: Directory not found - {dir_path}")

    if not csv_directories:
        print("No valid directories found!")
        return

    # 出力ディレクトリの設定

    stability_dir = f"/workspace/miru_vizualize/sigma_0/C_C/stablity/daihyou{noise}_{no_noise}"
    os.makedirs(stability_dir, exist_ok=True)

    # 解析対象
    targets = ["combined"]

    try:
        
        # for directory in csv_directories:
        #     evaluate_label_changes(
        #         directory=directory,
        #         target="combined",
        #         mode="epoch",
        #         y_lim=(0,3),
        #         epoch_start=1,
        #         epoch_end=30,
        #         plot_result=False,
        #     )

        #     evaluate_label_changes(
        #         directory=directory,
        #         target="combined",
        #         mode="epoch",
        #         y_lim=(0,3),
        #         epoch_start=30,
        #         epoch_end=60,
        #         plot_result=False,
        #     )
        #     evaluate_label_changes(
        #         directory=directory,
        #         target="combined",
        #         mode="epoch",
        #         y_lim=(0,3),
        #         epoch_start=60,
        #         epoch_end=140,  
        #         plot_result=False,
        #      )
        #     evaluate_label_changes(
        #         directory=directory,
        #         target="combined",
        #         mode="epoch",
        #         y_lim=(0,3), 
        #         epoch_start=140,
        #         epoch_end=1000
        #         )




        plot_label_changes_across_seeds(
            csv_paths=csv_directories,

            output_path=os.path.join(stability_dir, f"spatial_instability_{noise}_{no_noise}"),
            target="combined",
            y_scale='ratio',
            ylim=(-0.0005, 0.02),
            highlight_epochs= [1,30,60,140,1000]
        )
        # main3(
        #     csv_directories=csv_directories,
        #     base_output_dir=stability_dir,
        #     targets=targets,
        #     plot_save_path=f"temporal_instability_{noise}_{no_noise}",
        #     epoch_start=1,
        #     epoch_end=30,
        # )
        # main3(
        #     csv_directories=csv_directories,
        #     base_output_dir=stability_dir,
        #     targets=targets,
        #     plot_save_path=f"temporal_instability_{noise}_{no_noise}",
        #     epoch_start=30,
        #     epoch_end=60,
        # )
        # main3(
        #     csv_directories=csv_directories,
        #     base_output_dir=stability_dir,
        #     targets=targets,
        #     plot_save_path=f"temporal_instability_{noise}_{no_noise}",
        #     epoch_start=60,
        #     epoch_end=140,
        # )
        # main3(
        #     csv_directories=csv_directories,
        #     base_output_dir=stability_dir,
        #     targets=targets,
        #     plot_save_path=f"temporal_instability_{noise}_{no_noise}",
        #     epoch_start=140,
        #     epoch_end=1000,
        # )

        print(f"Successfully processed {len(csv_directories)} directories")
        #print(csv_directories)
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        print("\nDebug information:")
        print("CSV directories:", csv_directories)



if __name__ == "__main__":
    #combine_plots(output_dir="/workspace/miru_vizualize/sigma_0/C_N/stablity/daihyou", file_format="svg")
    main_combine_seed_results()
    # 以下、必要に応じて関数呼び出しを追加
