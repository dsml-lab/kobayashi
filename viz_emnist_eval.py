import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from smoothing import moving_average  # 別ファイル smoothing.py をimport

plt.rcParams["font.size"] = 14
plt.rcParams["figure.figsize"] = [15, 8]
plt.rcParams["figure.dpi"] = 300

def evaluate_label_changes(
    directory, 
    mode='alpha', 
    target='digit', 
    y_lim=None, 
    smooth_window=None,
    vertical_epochs=None
):
    """
    ラベル変更スコアを評価し、グラフを作成する関数。（digit, color両対応）
    CSVは `directory/csv` フォルダ内、図やログは `directory/fig_and_log` へ保存。
    
    Parameters
    ----------
    directory : str
        CSVファイルが格納されている「親ディレクトリ」のパス
        (例: /workspace/alpha_test/EMNIST/.../no_noise_no_noise/0/{i} 等)
    mode : {'alpha', 'epoch'}
        'alpha' または 'epoch' のどちらかを指定
    target : {'digit', 'color'}
        どちらのラベルで評価するか
    y_lim : tuple or None
        y軸の範囲 (例: (0, 100))。Noneの場合は自動決定
    smooth_window : int or None
        スムージングのウィンドウサイズ。Noneまたは1以下ならスムージングしない
    vertical_epochs : list or None
        mode='alpha' 時のみ有効。縦線を入れたい Epoch 番号のリスト (例: [50, 100, 500])

    Returns
    -------
    scores : dict or list
        mode='alpha' の場合 -> {エポック: ラベル変更数} の辞書
        mode='epoch' の場合 -> 各行に対するラベル変更数の合計リスト
    """

    # -------------------------------------------------
    # パラメータチェック
    # -------------------------------------------------
    if target not in ['digit', 'color']:
        raise ValueError("Invalid target. Choose 'digit' or 'color'.")
    if mode not in ['alpha', 'epoch']:
        raise ValueError("Invalid mode. Choose 'alpha' or 'epoch'.")

    # CSVフォルダを指定
    csv_dir = os.path.join(directory, "csv")
    if not os.path.exists(csv_dir):
        raise ValueError(f"CSV directory not found: {csv_dir}")

    # CSVファイルを検索
    files = sorted([
        os.path.join(csv_dir, f) for f in os.listdir(csv_dir)
        if f.startswith("alpha_log_epoch_") and f.endswith(".csv")
    ])
    if not files:
        raise ValueError(f"No valid CSV files found under: {csv_dir}")

    # ターゲット列名
    target_col = 'predicted_' + target
    # 出力先ディレクトリ
    save_dir = os.path.join(directory, "fig_and_log")
    os.makedirs(save_dir, exist_ok=True)

    # スムージングの有無をファイル名に反映
    if smooth_window is not None and smooth_window > 1:
        smoothing_suffix = "_smoothed"
    else:
        smoothing_suffix = "_original"

    # ---------------------------------------------------------
    # mode='alpha'
    # ---------------------------------------------------------
    if mode == 'alpha':
        scores = {}
        for file in files:
            df = pd.read_csv(file).iloc[:202]
            changes = (df[target_col] != df[target_col].shift()).sum()
            epoch_str = os.path.basename(file).split('_')[-1].split('.')[0]
            try:
                epoch = int(epoch_str)
            except ValueError:
                raise ValueError(f"Invalid epoch number in file name: {file}")
            scores[epoch] = changes

        sorted_epochs = sorted(scores.keys())
        sorted_scores = [scores[ep] for ep in sorted_epochs]

        # スムージング (長さは同じ)
        if smooth_window is not None and smooth_window > 1:
            sorted_scores_smoothed = moving_average(sorted_scores, smooth_window)
        else:
            sorted_scores_smoothed = None

        # y軸の範囲設定
        if y_lim is None:
            max_score = max(sorted_scores_smoothed) if sorted_scores_smoothed is not None else max(sorted_scores)
            current_y_lim = (0, max_score + 1)
        else:
            current_y_lim = y_lim

        # グラフ描画
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # オリジナル曲線
        if sorted_scores_smoothed is not None:
            ax.plot(sorted_epochs, sorted_scores, color='blue', alpha=0.3, label='Original')
            # スムージング後
            ax.plot(sorted_epochs, sorted_scores_smoothed, color='red', 
                    label=f'Smoothed (window={smooth_window})')
        else:
            ax.plot(sorted_epochs, sorted_scores, color='blue', label='Original')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Label Change Score')
        ax.set_title(f'Label Change Score across Epochs ({target})')
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        ax.set_ylim(current_y_lim)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # x軸をログスケール
        ax.set_xscale('log')
        log_ticks = [10**0, 10**1, 10**2, 10**3]
        log_ticks_filtered = [
            tick for tick in log_ticks 
            if sorted_epochs[0] <= tick <= sorted_epochs[-1]
        ]
        ax.set_xticks(log_ticks_filtered)
        ax.set_xticklabels([f'$10^{{{int(np.log10(t))}}}$' for t in log_ticks_filtered])

        # -------------------------
        # 縦線を引く (mode='alpha' のみ)
        # -------------------------
        if vertical_epochs is not None:
            for v_ep in vertical_epochs:
                # xscale='log' でも .axvline(...) はそのままの x=値を指定すればOK
                ax.axvline(x=v_ep, color='black', linestyle='--', alpha=0.8)

        ax.legend()
        plt.tight_layout()
        filename_suffix = f"eval_alpha{smoothing_suffix}.png"
        plt.savefig(os.path.join(save_dir, filename_suffix))
        print(f"Figure saved as {filename_suffix}")
        plt.show()
        plt.close()

        return scores

    # ---------------------------------------------------------
    # mode='epoch'
    # ---------------------------------------------------------
    else:
        if len(files) < 2:
            raise ValueError("At least two files are required for 'epoch' mode.")

        dataframes = [pd.read_csv(file, usecols=['alpha', target_col]).iloc[:202] for file in files]
        alpha_values = dataframes[0]['alpha'].tolist()
        scores_list = [0] * 202

        for i in range(len(dataframes) - 1):
            df_current = dataframes[i]
            df_next = dataframes[i + 1]
            row_changes = (df_current[target_col] != df_next[target_col]).astype(int)
            scores_list = [s + rc for s, rc in zip(scores_list, row_changes)]

        # スムージング (長さは同じ)
        if smooth_window is not None and smooth_window > 1:
            scores_smoothed = moving_average(scores_list, smooth_window)
        else:
            scores_smoothed = None

        if y_lim is None:
            max_score = max(scores_smoothed) if scores_smoothed is not None else max(scores_list)
            current_y_lim = (0, max_score + 1)
        else:
            current_y_lim = y_lim

        fig, ax = plt.subplots(figsize=(10, 6))
        
        if scores_smoothed is not None:
            ax.plot(alpha_values, scores_list, color='blue', alpha=0.3, label='Original')
            ax.plot(alpha_values, scores_smoothed, color='red', 
                    label=f'Smoothed (window={smooth_window})')
        else:
            ax.plot(alpha_values, scores_list, color='blue', label='Original')

        ax.set_xlabel('Alpha')
        ax.set_ylabel('Label Change Score')
        ax.set_title(f'Label Change Score across Alpha Values ({target})')
        ax.grid(True)
        ax.set_ylim(current_y_lim)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # epochモードでは縦線は挿入しない想定 (必要なら同様に ax.axvline で追加可)

        plt.tight_layout()
        filename_suffix = f"eval_epoch{smoothing_suffix}.png"
        plt.savefig(os.path.join(save_dir, filename_suffix))
        print(f"Figure saved as {filename_suffix}")
        plt.show()
        plt.close()

        return scores_list


# --------------------------------------------
# 使用例
# --------------------------------------------
if __name__ == "__main__":
    # 例: alphaモード + スムージングあり + 縦線を複数
    #label noise=0.2
    #line_0.2 = [70,145,259,350]
    line_half = [60,160,1000]
    label_noise = 0.5
    list = [1,2,3,4,5,6,7,8,9]
    for i in list:
        evaluate_label_changes(
            directory=f"/workspace/alpha_test/EMNIST/cnn_5layers_emnist_digits_variance100_combined_lr0.01_batch256_epoch2000_LabelNoiseRate{label_noise}_Optimsgd_Momentum0.0/no_noise_no_noise/0/{i}",        
            mode='alpha',
            target='digit',
            y_lim=(0, 20),
            smooth_window=5,
            vertical_epochs=line_half # ここで縦線を引きたいepochを指定
        )
        evaluate_label_changes(
            directory=f"/workspace/alpha_test/EMNIST/cnn_5layers_emnist_digits_variance100_combined_lr0.01_batch256_epoch2000_LabelNoiseRate{label_noise}_Optimsgd_Momentum0.0/no_noise_no_noise/0/{i}",        
            mode='alpha',
            target='digit',
            y_lim=(0, 20),
            smooth_window=None,
            vertical_epochs=line_half  # ここで縦線を引きたいepochを指定
        )


        # 例: epochモード + スムージングなし (縦線挿入なし)
        evaluate_label_changes(
            directory=f"/workspace/alpha_test/EMNIST/cnn_5layers_emnist_digits_variance100_combined_lr0.01_batch256_epoch2000_LabelNoiseRate{label_noise}_Optimsgd_Momentum0.0/no_noise_no_noise/0/{i}",        
            mode='epoch',
            target='digit',
            y_lim=(0, 2000),
            smooth_window=None
        )
