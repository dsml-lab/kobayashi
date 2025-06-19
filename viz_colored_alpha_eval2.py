import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker  # 追加
from smoothing import moving_average 
plt.rcParams["font.size"] = 14
plt.rcParams["figure.figsize"] = [8,5]
plt.rcParams["figure.dpi"] = 300


import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker  # 追加
from smoothing import moving_average 
plt.rcParams["font.size"] = 14
plt.rcParams["figure.figsize"] = [8,5]
plt.rcParams["figure.dpi"] = 300


def evaluate_label_changes(
    directory, 
    mode='alpha', 
    target='digit', 
    y_lim=None,
    smoothing=False,           # スムージングを行うかどうか
    smoothing_window=5,        # スムージングのウィンドウサイズ
    y_scale='ratio'            # 'ratio' または 'percent' で縦軸のスケールを選択
):
    """
    新ディレクトリ構成に対応:
      ./alpha_test/{experiment_name}/{mode1}_{mode2}/{n1}/{n2}/
        ├─ csv
        └─ fig_and_log

    スムージング機能をオプションで適用可能。
    縦軸のスケールを 'ratio' (0~1) または 'percent' (0~100) から選択可能。
    """

    if target not in ['digit', 'color',"combined"]:
        raise ValueError("Invalid target. Choose 'digit' or 'color'.")
    if mode not in ['alpha', 'epoch']:
        raise ValueError("Invalid mode. Choose 'alpha' or 'epoch'.")
    if y_scale not in ['ratio', 'percent']:
        raise ValueError("Invalid y_scale. Choose 'ratio' or 'percent'.")

    # csv ディレクトリ
    csv_dir = os.path.join(directory, "csv")
    # 保存先ディレクトリ
    save_dir = os.path.join(directory, "fig_and_log")
    os.makedirs(save_dir, exist_ok=True)

    # 対象CSVを取得
    files = sorted([
        os.path.join(csv_dir, f) for f in os.listdir(csv_dir)
        if f.startswith("alpha_log_epoch_") and f.endswith(".csv")
    ])

    if not files:
        raise ValueError("No valid files found in the specified directory.")

    target_col = 'predicted_' + target
    scores = {}

    if mode == 'alpha':
        for file in files:
            df = pd.read_csv(file).iloc[:202]  # Use only the first 202 rows
            n = len(df) - 1  # 比較するため1を引く
            changes = ((df[target_col] != df[target_col].shift()).sum() -1)/ n  # 平均値
            if y_scale == 'percent':
                changes *= 100
            epoch_str = os.path.basename(file).split('_')[-1].split('.')[0]
            try:
                epoch = int(epoch_str)
            except ValueError:
                raise ValueError(f"Invalid epoch number in file name: {file}")
            scores[epoch] = changes

        # CSVファイルとして保存
        score_df = pd.DataFrame(list(scores.items()), columns=['epoch', 'label_change'])
        csv_filename = os.path.join(save_dir, f"label_change_scores_alpha_{target}.csv")
        score_df.to_csv(csv_filename, index=False)
        print(f"Scores saved as {csv_filename}")

        """        
        sorted_epochs = sorted(scores.keys())
        unsmoothed_scores = [scores[epoch] for epoch in sorted_epochs]

        if smoothing:
            smoothed_scores = moving_average(unsmoothed_scores, window_size=smoothing_window)
            save_filename = f"eval_alpha_{target}_smoothing.png"
        else:
            smoothed_scores = None
            save_filename = f"new_eval_alpha_{target}.png"

        if y_lim is None:
            max_score = max(unsmoothed_scores)
            if y_scale == 'ratio':
                current_y_lim = (0, max_score + 0.1)
            elif y_scale == 'percent':
                current_y_lim = (0, max_score + 5)
        else:
            current_y_lim = y_lim

        fig, ax = plt.subplots(figsize=(10, 6))

        if smoothing:
            ax.plot(sorted_epochs, unsmoothed_scores, color='lightblue', label='Unsmoothed')
            ax.plot(sorted_epochs, smoothed_scores, color='red', label='Smoothed')
            ax.legend()
        else:
            ax.plot(sorted_epochs, unsmoothed_scores, color='blue', label='Average Changes')
            ax.legend()

        ax.set_xlabel('Epoch')
        ylabel = 'Average Label Change across epochs'
        if y_scale == 'percent':
            ylabel += 'Spatial stability'
        else:
            ylabel += 'Spatial stability'
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        ax.set_ylim(current_y_lim)

        ax.set_xscale('log')
        log_ticks = [10**0, 10**1, 10**2, 10**3]
        log_ticks_filtered = [tick for tick in log_ticks if tick >= sorted_epochs[0] and tick <= sorted_epochs[-1]]
        ax.set_xticks(log_ticks_filtered)
        ax.set_xticklabels([f'$10^{{{int(np.log10(tick))}}}$' for tick in log_ticks_filtered])
        """
        #plt.tight_layout()
        #plt.savefig(os.path.join(save_dir, save_filename))
        #print(f"Figure saved as {save_filename}")
        #plt.close()

    elif mode == 'epoch':
        # ファイルが2つ以上ないと比較できない
        if len(files) < 2:
            raise ValueError("At least two files are required for 'epoch' mode.")

        # 必要な列のみ読み込んで保持
        dataframes = [pd.read_csv(file, usecols=['alpha', target_col]).iloc[:202] for file in files]

        # alpha 列は全ファイル同一想定
        alpha_values = dataframes[0]['alpha'].tolist()
        # スムージング前スコア (行数分の0配列で初期化)
        unsmoothed_scores = [0] * 202  

        # 隣接ファイルを比較して、行単位で変更をカウント
        for i in range(len(files) - 1):
            df_current = dataframes[i]
            df_next = dataframes[i + 1]
            row_changes = (df_current[target_col] != df_next[target_col]).astype(int)
            unsmoothed_scores = [score + change for score, change in zip(unsmoothed_scores, row_changes)]

        # CSVとして保存
        epoch_csv_path = os.path.join(save_dir, f"epoch_unsmoothed_scores_{target}.csv")
        epoch_data = pd.DataFrame({'alpha': alpha_values, 'unsmoothed_scores': unsmoothed_scores})
        epoch_data.to_csv(epoch_csv_path, index=False)
        print(f"Epoch scores saved to {epoch_csv_path}")

        # ---------- スムージング処理 ----------
        """if smoothing:
            smoothed_scores = moving_average(unsmoothed_scores, window_size=smoothing_window)
            save_filename = f"eval_epoch_{target}_smoothing.png"
        else:
            smoothed_scores = None
            save_filename = f"s_eval_epoch_{target}.png"""
        # ------------------------------------

        """  # Y軸の上限下限
        if y_lim is None:
            max_score = max(unsmoothed_scores)
            current_y_lim = (0, max_score + 1)
        else:
            current_y_lim = y_lim

        # 可視化
        fig, ax = plt.subplots(figsize=(10, 6))

        if smoothing:
            # スムージング前 (薄い青色)
            ax.plot(alpha_values, unsmoothed_scores, color='lightblue', label='Unsmoothed')
            # スムージング後 (赤色)
            ax.plot(alpha_values, smoothed_scores, color='red', label='Smoothed')
            ax.legend()
        else:
            ax.plot(alpha_values, unsmoothed_scores, color='blue')  # 色はお好みで

        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel('Temporal stability')
        #ax.set_title(f'Label Change Score across Alpha Values ({target})')
        ax.grid(True)
        ax.set_ylim(0.0,0.05)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # ---------- 縦線の追加 ----------
        # alpha_csv_dirに応じた縦線の色を決定
        if 'no_noise_no_noise' in directory:
            color_alpha0 = 'black'
            color_alpha1 = 'black'
        elif 'noise_no_noise' in directory:
            color_alpha0 = 'black'
            color_alpha1 = 'red'
        elif 'no_noise_noise' in directory:
            color_alpha0 = 'red'
            color_alpha1 = 'black'
        else:
            # デフォルトの色設定（必要に応じて変更可能）
            color_alpha0 = 'black'
            color_alpha1 = 'black'

        # 縦線を追加
        ax.axvline(x=0.0, color=color_alpha0, linestyle='--', linewidth=1.5, label='alpha=0.0')
        ax.axvline(x=1.0, color=color_alpha1, linestyle='--', linewidth=1.5, label='alpha=1.0')

        # 凡例に縦線のラベルを追加
        handles, labels = ax.get_legend_handles_labels()
        # 縦線のラベルを追加（重複を避けるため）
        if 'alpha=0.0' not in labels:
            handles += [plt.Line2D([], [], color=color_alpha0, linestyle='--', linewidth=1.5)]
            labels += ['alpha=0.0']
        if 'alpha=1.0' not in labels:
            handles += [plt.Line2D([], [], color=color_alpha1, linestyle='--', linewidth=1.5)]
            labels += ['alpha=1.0']
        ax.legend(handles, labels)"""

        # ------------------------------------

        #plt.tight_layout()
        #plt.savefig(os.path.join(save_dir, save_filename))
        #print(f"Figure saved as {save_filename}")
        #plt.close()

    return scores


# 使用例（適宜パスなどを修正してください）
def main():
    base_dir = f"/workspace/alpha_test/seed/ori_closet_seed42_cnn_5layers_distribution_colored_emnist_variance1000_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.5_Optimsgd_Momentum0.0"
    evaluate_label_changes(
                                directory=base_dir,
                                mode="alpha",
                                target="combined",
                                y_lim=(0,1600),  # y_limはスケールに応じて自動設定
                                smoothing=False,
                                y_scale='ratio'  # 'ratio' または 'percent' を選択
                            )

def main2():
    """
    メイン関数:
    指定されたベースディレクトリ以下のすべての 'csv' ディレクトリを探索し、
    各ディレクトリに対して評価を実行します。
    """
    # ベースディレクトリを設定してください
    seeds = {43,44,45,46,47,48,49,50,51}
    for s in seeds:
        base_dir = f"/workspace/alpha_test/seed/lr_0.2/ori_closet_seed{s}_cnn_5layers_distribution_colored_emnist_variance1000_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/no_noise_no_noise/30/30"

            # 評価するターゲット
        targets = ["digit", "color",]

            # 各 'csv' ディレクトリを探索
        for dirpath, dirnames, filenames in os.walk(base_dir):
            if 'csv' in dirnames:
                csv_dir = os.path.join(dirpath, 'csv')
                fig_and_log_dir = os.path.join(dirpath, 'fig_and_log')

                # 'fig_and_log' ディレクトリが存在しない場合は作成
                os.makedirs(fig_and_log_dir, exist_ok=True)

                # 現在のディレクトリパスを基に処理
                current_directory = dirpath

                for target in targets:
                    # mode='alpha' の評価
                    try:
                        evaluate_label_changes(
                            directory=current_directory,
                            mode="alpha",
                            target=target,
                            y_lim=None,  # y_limはスケールに応じて自動設定
                            smoothing=False,
                            y_scale='ratio'  # 'ratio' または 'percent' を選択
                        )
                    except Exception as e:
                        print(f"Error processing alpha mode for directory {current_directory}, target {target}: {e}")
                    

                    # mode='epoch' の評価
                    try:
                        evaluate_label_changes(
                            directory=current_directory,
                            mode="epoch",
                            target=target,
                            y_lim=(0, 1600),
                            smoothing=False,
                            y_scale='ratio'  # 'epoch' モードではスケール選択は不要だが、パラメータは無視されます
                        )
                    except Exception as e:
                        print(f"Error processing epoch mode for directory {current_directory}, target {target}: {e}")
                    
import os
import pandas as pd
import matplotlib.pyplot as plt
import re

def main3():
    # CSVファイルが格納されているベースディレクトリを指定
    base_dir = "/workspace/alpha_test/seed/lr_0.2"
    # ベースの出力ディレクトリ
    base_output_dir = "/workspace/cvpr_visualizes"

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(base_output_dir, exist_ok=True)

    # シードのリスト
    seeds = {43,44,45,46,47,48,49,50,51}

    # ターゲットのリスト
    targets = ["digit", "color"]

    # 各ターゲットごとにプロットを作成
    for target in targets:
        data_list = []
        groups_present = set()  # このターゲットで見つかったグループを記録

        # 最初のシードから保存先ディレクトリを決定するためのフラグ
        output_dir_set = False
        plot_dir = base_output_dir  # デフォルト

        # 各シードに対してCSVファイルを探索
        for seed in seeds:
            # 各シードのベースディレクトリ
            seed_dir = f"ori_closet_seed{seed}_cnn_5layers_distribution_colored_emnist_variance1000_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/no_noise_no_noise/2/2"
            full_seed_dir = os.path.join(base_dir, seed_dir)

            # 予想されるCSVファイルのパス
            csv_filename = f"epoch_unsmoothed_scores_{target}.csv"
            csv_path = os.path.join(full_seed_dir, 'fig_and_log', csv_filename)

            if not os.path.exists(csv_path):
                print(f"CSVファイルが存在しません: {csv_path}")
                continue

            # グループ判定
            if 'no_noise_no_noise' in seed_dir:
                group = 'no_noise_no_noise'
            elif 'noise_no_noise' in seed_dir:
                group = 'noise_no_noise'
            else:
                group = 'other'

            groups_present.add(group)

            # 最初のシードで保存先ディレクトリを設定
            if not output_dir_set:
                # seed_dir から "LabelNoiseRate0.2/noise_no_noise/40/86" を抽出
                match = re.search(r'LabelNoiseRate([0-9.]+)/noise_no_noise/(\d+)/(\d+)', seed_dir)
                if match:
                    label_noise_value = match.group(1)
                    param1 = match.group(2)
                    param2 = match.group(3)
                    # "LabelNoiseRate0.2" を "label_noise=0.2" に変換
                    label_noise_dir = f"label_noise={label_noise_value}"
                    # 新しい保存先ディレクトリを構築
                    plot_dir = os.path.join(base_output_dir, label_noise_dir, "noise_no_noise", param1, param2)
                    # ディレクトリを作成
                    os.makedirs(plot_dir, exist_ok=True)
                    output_dir_set = True
                else:
                    print(f"seed_dirから必要な情報を抽出できませんでした: {seed_dir}")

            try:
                df = pd.read_csv(csv_path)
                # 必要な列のみを抽出
                df = df[['alpha', 'unsmoothed_scores']]
                # ターゲット情報を追加（必要に応じて）
                df['target'] = target
                # シード情報を追加（必要に応じて）
                df['seed'] = seed
                data_list.append(df)
            except Exception as e:
                print(f"CSVファイルの読み込みエラー ({csv_path}): {e}")

        # データフレームの統合
        if not data_list:
            print(f"データが収集されませんでした: {target}")
            continue

        combined_df = pd.concat(data_list, ignore_index=True)

        # 各alphaごとの平均と標準偏差を計算
        stats_df = combined_df.groupby('alpha')['unsmoothed_scores'].agg(['mean', 'std']).reset_index()

        # プロットの作成
        plt.figure(figsize=(10, 6))
        plt.plot(stats_df['alpha'], stats_df['mean'], label='Mean Unsmoothed Scores', color='blue')
        plt.fill_between(stats_df['alpha'], 
                         stats_df['mean'] - stats_df['std'], 
                         stats_df['mean'] + stats_df['std'], 
                         color='blue', alpha=0.2, label='Standard Deviation (±1σ)')
        
        # グループに基づいて垂直線を追加
        for group in groups_present:
            if group == 'no_noise_no_noise':
                # α=0 と α=1 に黒い縦線
                plt.axvline(x=0, color='black', linestyle='--', linewidth=1, label='α=0 (no_noise_no_noise)' if 'α=0 (no_noise_no_noise)' not in plt.gca().get_legend_handles_labels()[1] else "")
                plt.axvline(x=1, color='black', linestyle='--', linewidth=1, label='α=1 (no_noise_no_noise)' if 'α=1 (no_noise_no_noise)' not in plt.gca().get_legend_handles_labels()[1] else "")
            elif group == 'noise_no_noise':
                # α=0 に黒い縦線、α=1 に赤い縦線
                plt.axvline(x=0, color='black', linestyle='--', linewidth=1, label='α=0 (noise_no_noise)' if 'α=0 (noise_no_noise)' not in plt.gca().get_legend_handles_labels()[1] else "")
                plt.axvline(x=1, color='red', linestyle='--', linewidth=1, label='α=1 (noise_no_noise)' if 'α=1 (noise_no_noise)' not in plt.gca().get_legend_handles_labels()[1] else "")
            else:
                # α=1 に黒い縦線、α=0 に赤い縦線
                plt.axvline(x=1, color='black', linestyle='--', linewidth=1, label='α=1 (other)' if 'α=1 (other)' not in plt.gca().get_legend_handles_labels()[1] else "")
                plt.axvline(x=0, color='red', linestyle='--', linewidth=1, label='α=0 (other)' if 'α=0 (other)' not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.xlabel(r'$\alpha$')
        plt.ylabel('Temporal stability')
        plt.grid(True)
        plt.tight_layout()

        # 保存ファイル名の指定
        plot_filename = f'output_{target}.png'  # ターゲットごとにファイル名を分ける場合

        # 保存先パスの設定
        plot_path = os.path.join(plot_dir, plot_filename)

        # グラフをPNGファイルとして保存
        plt.savefig(plot_path)
        plt.close()  # 表示せずに閉じる
        print(f"プロットを保存しました: {plot_path}")
        

if __name__ == "__main__":
    #main2()
    main3()

#main3がunsmoothedを使用するやつ
#evaluate_label_changesがunsmootedを保存するやつ


#main2がlabel_changesを保存するやつ
#eval_seedがlabel_changesを使用するやつ