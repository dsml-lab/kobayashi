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
    smoothing_window=5         # スムージングのウィンドウサイズ
):
    """
    新ディレクトリ構成に対応:
      ./alpha_test/{experiment_name}/{mode1}_{mode2}/{n1}/{n2}/
        ├─ csv
        └─ fig_and_log

    スムージング機能をオプションで適用可能。
    """

    if target not in ['digit', 'color']:
        raise ValueError("Invalid target. Choose 'digit' or 'color'.")
    if mode not in ['alpha', 'epoch']:
        raise ValueError("Invalid mode. Choose 'alpha' or 'epoch'.")

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

    # ---------------------------
    # mode='alpha' の処理
    # ---------------------------
    if mode == 'alpha':
        for file in files:
            df = pd.read_csv(file).iloc[:202]  # Use only the first 202 rows
            changes = (df[target_col] != df[target_col].shift()).sum()  # Count changes

            # Extract epoch number as integer
            epoch_str = os.path.basename(file).split('_')[-1].split('.')[0]
            try:
                epoch = int(epoch_str)
            except ValueError:
                raise ValueError(f"Invalid epoch number in file name: {file}")
            scores[epoch] = changes

        # 並べ替え
        sorted_epochs = sorted(scores.keys())
        # スムージング前の生データ
        unsmoothed_scores = [scores[epoch] for epoch in sorted_epochs]

        # ---------- スムージング処理 ----------
        if smoothing:
            smoothed_scores = moving_average(unsmoothed_scores, window_size=smoothing_window)
            save_filename = f"eval_alpha_{target}_smoothing.png"
        else:
            smoothed_scores = None  # スムージングしない場合はNone
            save_filename = f"s_eval_alpha_{target}.png"
        # ------------------------------------

        # Y軸の上限下限
        if y_lim is None:
            max_score = max(unsmoothed_scores)
            current_y_lim = (0, max_score + 1)  # Add 1 for better visualization
        else:
            current_y_lim = y_lim

        # 可視化
        fig, ax = plt.subplots(figsize=(10, 6))

        if smoothing:
            # スムージング前 (薄い青色)
            ax.plot(sorted_epochs, unsmoothed_scores, color='lightblue', label='Unsmoothed')
            # スムージング後 (赤色)
            ax.plot(sorted_epochs, smoothed_scores, color='red', label='Smoothed')
            ax.legend()
        else:
            # スムージングなし
            ax.plot(sorted_epochs, unsmoothed_scores, color='blue')  # 色はお好みで

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Label Change across epochs')
        #ax.set_title(f'Label Change Score across Epochs ({target})')
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        ax.set_ylim(current_y_lim)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # xをlogスケールに
        ax.set_xscale('log')
        log_ticks = [10**0, 10**1, 10**2, 10**3]
        log_ticks_filtered = [tick for tick in log_ticks if tick >= sorted_epochs[0] and tick <= sorted_epochs[-1]]
        ax.set_xticks(log_ticks_filtered)
        ax.set_xticklabels([f'$10^{{{int(np.log10(tick))}}}$' for tick in log_ticks_filtered])

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, save_filename))
        print(f"Figure saved as {save_filename}")
        plt.close()

    # ---------------------------
    # mode='epoch' の処理
    # ---------------------------
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
        if smoothing:
            smoothed_scores = moving_average(unsmoothed_scores, window_size=smoothing_window)
            save_filename = f"eval_epoch_{target}_smoothing.png"
        else:
            smoothed_scores = None
            save_filename = f"s_eval_epoch_{target}.png"
        # ------------------------------------

        # Y軸の上限下限
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

        ax.set_xlabel('Alpha')
        ax.set_ylabel('Label Change Score across alpha')
        #ax.set_title(f'Label Change Score across Alpha Values ({target})')
        ax.grid(True)
        ax.set_ylim(current_y_lim)
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
        ax.legend(handles, labels)

        # ------------------------------------

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, save_filename))
        print(f"Figure saved as {save_filename}")
        plt.close()

    return scores


# 使用例（適宜パスなどを修正してください）
def main():
    """
    メイン関数:
    指定されたベースディレクトリ以下のすべての 'csv' ディレクトリを探索し、
    各ディレクトリに対して評価を実行します。
    """

    # ベースディレクトリを設定してください
    var = {1000,3162}
    lr = {0.1,0.2}
    for v in var:
        for noise in lr:
            base_dir = f"alpha_test/closet_cnn_5layers_distribution_colored_emnist_variance{v}_combined_lr0.01_batch256_epoch2000_LabelNoiseRate{noise}_Optimsgd_Momentum0.0/no_noise_no_noise"

            # 評価するターゲット
            targets = ["digit", "color"]

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
                        """try:
                            evaluate_label_changes(
                                directory=current_directory,
                                mode="alpha",
                                target=target,
                                y_lim=(0, 25),
                                smoothing=False
                            )
                        except Exception as e:
                            print(f"Error processing alpha mode for directory {current_directory}, target {target}: {e}")
                        """
                        # mode='epoch' の評価
                        try:
                            evaluate_label_changes(
                                directory=current_directory,
                                mode="epoch",
                                target=target,
                                y_lim=(0, 1600),
                                smoothing=False
                            )
                        except Exception as e:
                            print(f"Error processing epoch mode for directory {current_directory}, target {target}: {e}")

    print("全ての評価が完了しました。")


if __name__ == "__main__":
    main()
