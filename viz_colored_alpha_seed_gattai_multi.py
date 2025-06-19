import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
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

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
def main3_multi_epoch(csv_directories, base_output_dir, epoch_ranges, targets=["digit", "color", "combined"], 
                        output_name="a", plot_save_path=None, plot_legend=True, epoch_label_pairs=None):
    """
    複数の epoch 区間における図を横に並べた図を作成し、svg形式で保存するプログラムです。
    各サブプロットは、左端のものにのみ yticks と ylabel を表示し、他は y 軸の目盛りとラベルを非表示にしますが、
    横方向（水平）のグリッド線は残します。

    Parameters
    ----------
    csv_directories : list of str
        プロット対象の CSV ディレクトリパスのリスト。
    base_output_dir : str
        プロットを保存するベースの出力ディレクトリのパス。
    epoch_ranges : list of tuple
        (epoch_start, epoch_end) のタプルのリスト。例: [(0, 1000), (1001, 2000)]
    targets : list of str
        プロット対象のターゲットリスト（例: ["digit", "color", "combined"]）。
    output_name : str
        出力ファイルの基本名（現在は未使用）。
    plot_save_path : str, optional
        プロットを保存するファイル名の接頭辞。指定がある場合、ファイル名は
        plot_save_path + {target}_multiple_epochs.svg となります。
    plot_legend : bool, optional
        レジェンドをプロットに表示するかどうか（デフォルトは True）。
    epoch_label_pairs : list of tuple, optional
        各 epoch 範囲に対応する文字列ペアのリスト。例: [("A", "B"), ("C", "D"), …]
        指定された場合、レジェンドのラベルは
        rf'$\mathcal{{T}}$_{{<first>}}{{<second>}}=({epoch_start},{epoch_end})'
        の形式となります。
    """
    os.makedirs(base_output_dir, exist_ok=True)

    # 複数の epoch 範囲を横並びのサブプロットに配置
    num_epochs = len(epoch_ranges)
    fig, axes = plt.subplots(ncols=num_epochs, nrows=1, figsize=(12*num_epochs, 10), dpi=400, sharey=True)
    if num_epochs == 1:
        axes = [axes]

    for i, (epoch_start, epoch_end) in enumerate(epoch_ranges):
        # エポック範囲の文字列作成
        start_str = str(epoch_start) if epoch_start is not None else "start"
        end_str = str(epoch_end) if epoch_end is not None else "end"
        epoch_suffix = f"_epoch_{start_str}_to_{end_str}"
        
        ax = axes[i]
        data_list = []
        groups_present = set()

        # 各 CSV ディレクトリからデータ読み込み（ここでは targets[0] のファイル名を使用）
        for directory in csv_directories:
            csv_filename = f"epoch_unsmoothed_scores_{targets[0]}{epoch_suffix}.csv"
            csv_path = os.path.join(directory, 'fig_and_log', csv_filename)
            if not os.path.exists(csv_path):
                print(f"[Warning] CSVファイルが存在しません: {csv_path}")
                continue

            # グループ分類（ディレクトリ名により）
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
                seed_match = re.search(r'seed(\d+)', directory)
                seed = seed_match.group(1) if seed_match else "unknown"
                df['target'] = targets[0]
                df['seed'] = seed
                data_list.append(df)
            except Exception as e:
                print(f"CSVファイルの読み込みエラー ({csv_path}): {e}")
        
        if not data_list:
            print(f"[Warning] データが収集されませんでした: {targets[0]} {epoch_suffix}")
            continue
        
        combined_df = pd.concat(data_list, ignore_index=True)
        stats_df = combined_df.groupby('alpha')['unsmoothed_scores'].agg(['mean', 'std']).reset_index()

        # --- 以下、main3 のグラフ形式に合わせた描画設定 ---
        marker_size = 20
        
        # ① マーカー描画（ディレクトリのサンプルを用いて判定）
        sample_dir = csv_directories[0]
        if "no_noise_no_noise" in sample_dir:
            ax.plot(0.0, 0.0, marker='o', color='blue', markersize=marker_size, zorder=5)
            ax.plot(1.0, 0.0, marker='o', color='blue', markersize=marker_size)
        elif "noise_no_noise" in sample_dir:
            ax.plot(0.0, 0.0, marker='o', color='blue', markersize=marker_size)
            ax.plot(1.0, 0.0, marker='o', color='red', markersize=marker_size)
        else:
            ax.plot(0.0, 0.0, marker='o', color='blue', markersize=marker_size, zorder=5)
            ax.plot(1.0, 0.0, marker='o', color='red', markersize=marker_size)

        # ② X軸、Y軸のフォーマット設定
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.set_yticks([0, 0.2, 0.4, 0.6])
        ax.set_yticklabels(["0", "0.2", "0.4", "0.6"], fontsize=40)
        ax.set_ylim(-0.01, 0.6)

        # ③ 平均値と標準偏差のプロット
        ax.plot(stats_df['alpha'], stats_df['mean'],
                label=r'$INST_{\mathrm{T}}(\mathcal{T},x)$', color='blue', linewidth=3, zorder=2)
        ax.fill_between(stats_df['alpha'],
                        stats_df['mean'] - stats_df['std'],
                        stats_df['mean'] + stats_df['std'],
                        color='blue', alpha=0.15, zorder=1)

        # ④ ダミープロットでエポック範囲を legend に追加
        if epoch_label_pairs is not None:
            epoch_start_mozi, epoch_end_mozi = epoch_label_pairs[i]
            dummy_label = rf'$\mathcal{{T}}_{{{epoch_start_mozi}{epoch_end_mozi}}}=[{epoch_start},{epoch_end}]$'

        else:
            dummy_label = rf'$\mathcal{{T}}$=({epoch_start},{epoch_end})'
        ax.plot([], [], label=dummy_label, linewidth=0)

        # ⑤ 縦ラインの描画（main3 の形式に合わせる）
        for group in groups_present:
            if group == 'noise_no_noise':
                ax.axvline(x=0.0, color='black', linestyle='-', linewidth=2, zorder=0)
                ax.axvline(x=1.0, color='black', linestyle='-', linewidth=2, zorder=0)
            elif group == 'noise_no_noise':  # （重複条件ですがそのまま再現）
                ax.axvline(x=0.0, color='black', linestyle='-', linewidth=2, zorder=0)
                ax.axvline(x=1.0, color='red', linestyle='-', linewidth=2, zorder=0)
            else:
                ax.axvline(x=0.0, color='black', linestyle='-', linewidth=2, zorder=0)
                ax.axvline(x=1.0, color='black', linestyle='-', linewidth=2, zorder=0)

        # ⑥ X軸の目盛りとラベル設定
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels([r'$x_0$', r'$x_1$'], fontsize=60)

        # ⑦ 左端のサブプロットのみ y 軸ラベルを表示（その他は非表示）
        if i == 0:
            ax.set_ylabel("Temporal Instability", fontsize=45)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', labelleft=False,)

        # ⑧ グリッドを表示
        ax.grid(True)

        if plot_legend:
            legend = ax.legend(fontsize=30)
            legend.get_frame().set_alpha(1.0)
    
    fig.tight_layout()
    # 保存ファイル名の生成（ここでは targets[0] を使用）
    if plot_save_path is not None:
        plot_filename = f"{plot_save_path}{targets[0]}_multiple_epochs.svg"
    else:
        plot_filename = f'width_4_C_N_output_{targets[0]}_multiple_epochs.svg'
    plot_path = os.path.join(base_output_dir, plot_filename)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"[Info] プロットを保存しました: {plot_path}")
if __name__ == "__main__":
    seeds = range(42, 51)  # 42から51まで

    noise="94"
    no_noise="94"
    base_pattern = "/workspace/alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_{}width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/no_noise_no_noise/94/94"

    csv_directories = []
    for seed in seeds:
        dir_path = base_pattern.format(seed)
        if os.path.exists(dir_path):
            csv_directories.append(dir_path)
        else:
            print(f"Warning: Directory not found - {dir_path}")

    if not csv_directories:
        print("No valid directories found!")
    output_dir = f"/workspace/miru_vizualize/sigma_0/C_C/stablity/daihyou{noise}_{no_noise}"
    # 例: epoch 0〜1000 と 1001〜2000 の2区間でプロット
    epoch_intervals = [(1,30), (30, 60),(60,140),(140,1000)]
    epoch_labels = [("A", "C"), ("C", "E"), ("E", "G"), ("G", "H")]

    main3_multi_epoch(csv_directories, output_dir, epoch_intervals, targets=["combined"], plot_save_path=f"gattai{noise}_{no_noise}.svg",plot_legend=False,epoch_label_pairs=epoch_labels)
        
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