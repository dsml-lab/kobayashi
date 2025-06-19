import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 14
plt.rcParams["figure.figsize"] = [8,5]
plt.rcParams["figure.dpi"] = 300

def plot_label_changes_across_seeds(
    seeds = [43,44,45,46,47,48,49,50,51],
    base_dir_template = "/workspace/alpha_test/seed/ori_closet_seed{}_cnn_5layers_distribution_colored_emnist_variance1000_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.5_Optimsgd_Momentum0.0",
    target="digit",
    y_scale='ratio',
    ylim=None
):
    """
    複数のseed実行によって生成された epoch_scores.csv を読み込み、
    epoch毎の平均と標準偏差をプロットするコード。

    Parameters
    ----------
    seeds : list
        解析対象のseed一覧。
    base_dir_template : str
        seed値を{}で指定した文字列テンプレート。ここにseedを埋め込んでパスを生成します。
    target : str
        'digit', 'color', 'combined' など（プロットのファイル名やメッセージ等に使用する用途）
    y_scale : str
        'ratio' か 'percent'。平均スコアが割合の場合に0～1を想定するなら 'ratio',
        0～100の範囲を想定するなら 'percent'。
    ylim : tuple
        (ymin, ymax) のように、y軸の範囲を指定。
    """

    # seed毎の DataFrame を格納するリスト
    df_list = []

    for s in seeds:
        # seed毎のディレクトリ
        current_seed_dir = base_dir_template.format(s)
        # その中の fig_and_log/epoch_scores.csv を読み込む
        csv_path = os.path.join(current_seed_dir, "fig_and_log", f"label_change_scores_alpha_{target}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df_sorted = df.sort_values(by="epoch")
            df_list.append(df_sorted)
        else:
            print(f"[Warning] File not found for seed={s}: {csv_path}")

    # df_list が空の場合、処理終了
    if not df_list:
        print("No valid CSV found. Process aborted.")
        return

    # 全データの epoch が同じ前提で処理する
    # epoch のリストを取得（最初のDataFrameを使用）
    epochs = df_list[0]["epoch"].values

    # スコアを格納する配列 ( #seeds x #epoch )
    # 各seedの scores を取り出して 2次元配列化
    all_scores = np.array([df["label_change"].values for df in df_list])

    # 平均と標準偏差を計算
    mean_scores = np.mean(all_scores, axis=0)
    std_scores = np.std(all_scores, axis=0)

    # スケールが 'percent' の場合、スコアを100倍
    if y_scale == 'percent':
        mean_scores = mean_scores * 100
        std_scores = std_scores * 100

    # プロット開始
    fig, ax = plt.subplots()

    # 平均
    ax.plot(epochs, mean_scores, label="mean", color="blue")
    # 標準偏差
    ax.fill_between(epochs, mean_scores - std_scores, mean_scores + std_scores,
                    color="blue", alpha=0.2, label="std")

    ax.set_xlabel("Epoch")
    ax.set_ylim(0.0,0.05)
    if y_scale == 'percent':
        ax.set_ylabel("Spatial stability")
    else:
        ax.set_ylabel("Spatial stability")

    ax.grid(True)

    if ylim:
        ax.set_ylim(ylim)

    # x軸をログスケールに（不要であればコメントアウト）
    ax.set_xscale('log')

    # プロットを保存
    out_fig_path = os.path.join(
        os.path.dirname(base_dir_template.format(seeds[0])),
        f"fig_and_log",
        f"mean_std_across_seeds_{target}.png"
    )
    os.makedirs(os.path.dirname(out_fig_path), exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_fig_path)
    print(f"[Info] Figure saved to: {out_fig_path}")
    plt.close()

if __name__ == "__main__":
    # 使用例
    seed_list = [43, 44, 45, 46, 47, 48, 49, 50, 51]

    base = "/workspace/alpha_test/seed/lr_0.2/ori_closet_seed{}_cnn_5layers_distribution_colored_emnist_variance1000_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/40/86"
    # digit
    plot_label_changes_across_seeds(
        seeds=seed_list,
        base_dir_template=base,       
        #base_dir_template = "/workspace/alpha_test/seed/ori_closet_seed{}_cnn_5layers_distribution_colored_emnist_variance1000_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.5_Optimsgd_Momentum0.0/no_noise_no_noise/20/42",
        target="digit",
        y_scale='ratio',
        ylim=None  # (0, 1) など必要なら指定
    )

    # color
    plot_label_changes_across_seeds(
        seeds=seed_list,
        base_dir_template=base,   
        #base_dir_template = "/workspace/alpha_test/seed/ori_closet_seed{}_cnn_5layers_distribution_colored_emnist_variance1000_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.5_Optimsgd_Momentum0.0/noise_no_noise/20/42",
        target="color",
        y_scale='ratio',
        ylim=None
    )
