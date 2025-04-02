import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
def get_sample_dirs(base_dir):
    """
    base_dir 以下の2階層下のサンプルディレクトリをリストアップする。
    例: base_dir/数字/数字 の形状。
    各サンプルディレクトリ内に "fig_and_log" が存在することを前提とする。
    """
    sample_dirs = []
    for d in os.listdir(base_dir):
        d_path = os.path.join(base_dir, d)
        if os.path.isdir(d_path):
            for sub_d in os.listdir(d_path):
                sub_d_path = os.path.join(d_path, sub_d)
                if os.path.isdir(sub_d_path) and os.path.exists(os.path.join(sub_d_path, "fig_and_log")):
                    sample_dirs.append(sub_d_path)
    return sample_dirs

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
        CSV保存後にプロットを作成し保存するかどうか。デフォルトはTrue。
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

        if plot_result:
            # プロット作成
            score_df = pd.read_csv(csv_filename)
            epochs = score_df['epoch'].values
            label_changes = score_df['label_change'].values
            std_scores = np.zeros_like(label_changes)  # 単一ファイルなので標準偏差はゼロ

            fig, ax = plt.subplots(figsize=(12, 8), dpi=400)
            ax.plot(epochs, label_changes, label="$INST_s(\\chi,t)$", color="blue", linewidth=1.5, zorder=3)
            ax.fill_between(epochs, label_changes - std_scores, label_changes + std_scores,
                            color="blue", alpha=0.2, zorder=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Spatial Instability")
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
        
        total_epochs = len(files) - 1
        unsmoothed_scores = [score / total_epochs for score in unsmoothed_scores]

        epoch_csv_path = os.path.join(save_dir, f"epoch_unsmoothed_scores_{target}{epoch_suffix}.csv")
        epoch_data = pd.DataFrame({'alpha': alpha_values, 'unsmoothed_scores': unsmoothed_scores})
        epoch_data.to_csv(epoch_csv_path, index=False)
        print(f"Epoch scores saved to {epoch_csv_path}")

        if plot_result:
            df_epoch = pd.read_csv(epoch_csv_path)
            alphas = df_epoch['alpha'].values
            scores_epoch = df_epoch['unsmoothed_scores'].values
            std_scores = np.zeros_like(scores_epoch)
            
            fig, ax = plt.subplots(figsize=(12, 8), dpi=400)
            ax.plot(alphas, scores_epoch, label="$INST_s(\\chi,t)$", color="blue", linewidth=1.5, zorder=3)
            ax.fill_between(alphas, scores_epoch - std_scores, scores_epoch + std_scores,
                            color="blue", alpha=0.2, zorder=2)
            ax.set_xlabel("Alpha")
            ax.set_ylabel("Temporal Instability")
            if y_lim:
                ax.set_ylim(y_lim)
            
            plot_filename = f"epoch_unsmoothed_scores_{target}{epoch_suffix}_plot.png"
            plot_path = os.path.join(save_dir, plot_filename)
            fig.savefig(plot_path, bbox_inches='tight')
            plt.close(fig)
            print(f"Plot saved as {plot_path}")

    return scores

def evaluate_label_changes_all(sample_dirs, **kwargs):
    """
    get_sample_dirsで取得したすべてのサンプルディレクトリに対して
    evaluate_label_changes を実行します。

    Parameters
    ----------
    sample_dirs : list of str
        サンプルディレクトリのパスのリスト。
    kwargs :
        evaluate_label_changes に渡すその他の引数。

    Returns
    -------
    all_scores : dict
        各ディレクトリごとの評価結果の辞書。
        キーはディレクトリパス、値は evaluate_label_changes の戻り値（scores）。
    """
    all_scores = {}
    for sample_dir in sample_dirs:
        try:
            print(f"[Info] Processing directory: {sample_dir}")
            scores = evaluate_label_changes(sample_dir, **kwargs)
            all_scores[sample_dir] = scores
        except Exception as e:
            print(f"[Error] Processing {sample_dir} failed: {e}")
    return all_scores

def main3_samples(sample_dirs, base_output_dir, targets=["digit","color","combined"], output_name="a", epoch_start=None, epoch_end=None):
    """
    各サンプルディレクトリ内の CSV ファイルを読み込み、
    各ターゲットごとに、α ごとの unsmoothed_scores の平均と標準偏差をプロットして保存します。

    Parameters
    ----------
    sample_dirs : list of str
        各サンプルディレクトリのパスのリスト。各ディレクトリは内部に "fig_and_log" を持つ前提。
    base_output_dir : str
        プロット画像の保存先ディレクトリ。
    targets : list of str
        対象とするターゲット（例: ["digit", "color", "combined"]）。
    output_name : str
        出力ファイル名の基本名（今回の例では未使用）。
    epoch_start : int, optional
        エポック範囲の開始番号。ファイル名に反映されます。
    epoch_end : int, optional
        エポック範囲の終了番号。ファイル名に反映されます。
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
        # 各サンプルディレクトリから CSV ファイルを読み込む
        for sample_dir in sample_dirs:
            csv_filename = f"epoch_unsmoothed_scores_{target}{epoch_suffix}.csv"
            csv_path = os.path.join(sample_dir, "fig_and_log", csv_filename)
            if not os.path.exists(csv_path):
                print(f"[Warning] CSVファイルが存在しません: {csv_path}")
                continue

            try:
                df = pd.read_csv(csv_path)
                if 'alpha' not in df.columns or 'unsmoothed_scores' not in df.columns:
                    print(f"[Warning] 必要なカラムが存在しません: {csv_path}")
                    continue
                df = df[['alpha', 'unsmoothed_scores']]
                # サンプルIDはディレクトリ名（最下層の名前）から抽出
                sample_id = os.path.basename(sample_dir)
                df['sample'] = sample_id
                data_list.append(df)
            except Exception as e:
                print(f"[Warning] CSVファイルの読み込みエラー ({csv_path}): {e}")

        if not data_list:
            print(f"[Warning] 対象 {target} のデータが収集されませんでした。")
            continue

        # データを統合し、α ごとに平均・標準偏差を算出
        combined_df = pd.concat(data_list, ignore_index=True)
        stats_df = combined_df.groupby('alpha')['unsmoothed_scores'].agg(['mean', 'std']).reset_index()

        # プロット作成（サイズは seed wise のものと同じに設定）
        fig, ax = plt.subplots(figsize=(12, 8), dpi=400)
        marker_size = 20

        # サンプルの例として1件目のディレクトリの情報を用いて、グループに応じたマーカーを描画（必要に応じて調整）
        sample_example = sample_dirs[0]
        if "no_noise_no_noise" in sample_example:
            ax.plot(0.0, 0.0, marker='o', color='blue', markersize=marker_size, zorder=5)
            ax.plot(1.0, 0.0, marker='o', color='blue', markersize=marker_size)
        elif "noise_no_noise" in sample_example:
            ax.plot(0.0, 0.0, marker='o', color='blue', markersize=marker_size)
            ax.plot(1.0, 0.0, marker='o', color='red', markersize=marker_size)
        else:
            ax.plot(0.0, 0.0, marker='o', color='blue', markersize=marker_size, zorder=5)
            ax.plot(1.0, 0.0, marker='o', color='red', markersize=marker_size)

        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        # ax.set_yticks([0, 0.2, 0.4, 0.6])
        # ax.set_ylim(-0.01, 0.61)
        ax.set_yticks([0, 0.2,])
        ax.set_ylim(-0.01, 0.21)

        # 平均＆標準偏差のプロット
        ax.plot(stats_df['alpha'], stats_df['mean'],
                label=r'$INST_{\mathrm{T}}(\mathcal{T},x)$', color='blue', linewidth=3, zorder=2)
        ax.fill_between(stats_df['alpha'],
                        stats_df['mean'] - stats_df['std'],
                        stats_df['mean'] + stats_df['std'],
                        color='blue', alpha=0.15, zorder=1)
        # ax.set_yticks([0, 0.2, 0.4, 0.6])
        # ax.set_ylim(-0.01, 0.61)
        ax.set_yticks([0, 0.2,])
        ax.set_ylim(-0.01, 0.21)
        # ダミープロットでエポック範囲をレジェンドに追加

        if epoch_start is None and epoch_end is None:
                epoch_start, epoch_end = 0, 1000
            # ダミープロットでエポック範囲をレジェンドに追加
        ax.plot([], [], label=rf'$\mathcal{{T}}$=({epoch_start},{epoch_end})', linewidth=0)

        # 垂直線（例として x=0 と x=1 の位置に描画）
        ax.axvline(x=0.0, color='black', linestyle='-', linewidth=2, zorder=0)
        ax.axvline(x=1.0, color='black', linestyle='-', linewidth=2, zorder=0)

        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels([r'$x_2$', r'$x_3$'], fontsize=30)
        ax.set_ylabel("Temporal Instability", fontsize=30)
        ax.grid(True)

        legend = ax.legend(fontsize=30)
        legend.get_frame().set_alpha(1.0)
        fig.tight_layout()

        # SVG 形式で保存
        plot_filename = f'width_4_C_N_output_{target}{epoch_suffix}.svg'
        plot_path = os.path.join(base_output_dir, plot_filename)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig.savefig(plot_path, format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"[Info] プロットを保存しました: {plot_path}")
def plot_label_changes_across_samples(
    sample_dirs,
    output_path,
    target="digit",
    y_scale='ratio',
    ylim=None,
    highlight_epochs=None
):
    """
    複数のサンプルディレクトリから CSV ファイルを読み込み、
    epoch 毎の平均と標準偏差をプロットする。

    Parameters
    ----------
    sample_dirs : list of str
        プロット対象のサンプルディレクトリパスのリスト。
        各ディレクトリは内部に "fig_and_log" フォルダを持つことが前提。
    output_path : str
        プロット画像を保存するファイルパス。
    target : str
        CSV ファイル名の target 部分（例: "digit", "color" など）。
    y_scale : str
        Y軸のスケール指定（"ratio" または "percent"）。
    ylim : tuple, optional
        Y軸の表示範囲 (例: (0,1))。指定がなければ自動設定。
    highlight_epochs : list of int, optional
        強調表示するエポック番号のリスト。
    """

    df_list = []
    for sample_dir in sample_dirs:
        # sample_dir 内の fig_and_log フォルダを参照
        fig_and_log_path = os.path.join(sample_dir, "fig_and_log")
        csv_file = os.path.join(fig_and_log_path, f"label_change_scores_alpha_{target}.csv")
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file).sort_values(by="epoch")
                df = df[df["epoch"] >= 1].sort_values(by="epoch") 
                df_list.append(df)
            except Exception as e:
                print(f"[Warning] {csv_file} の読み込みに失敗しました: {e}")
        else:
            print(f"[Warning] CSVファイルが見つかりません: {csv_file}")
    
    if not df_list:
        print("有効な CSV ファイルが見つかりませんでした。処理を中断します。")
        return

    # 複数のサンプルの CSV が同一のエポック軸を持つ前提
    epochs = df_list[0]["epoch"].values
    all_scores = np.array([df["label_change"].values for df in df_list])
    mean_scores = np.mean(all_scores, axis=0)
    std_scores = np.std(all_scores, axis=0)

    if y_scale == 'percent':
        mean_scores *= 100
        std_scores *= 100

    fig, ax = plt.subplots(figsize=(12, 8), dpi=400)
    if highlight_epochs:
        for epoch in highlight_epochs:
            ax.axvline(x=epoch, color='black', linestyle='-', linewidth=1.5, zorder=1)
    
    ax.plot(epochs, mean_scores, label="$INST_s(\\chi,t)$", color="blue", linewidth=3, zorder=3)
    ax.fill_between(epochs, mean_scores - std_scores, mean_scores + std_scores,
                    color="blue", alpha=0.2, zorder=2)
    ax.plot([], [], label=r'$\mathcal{X}=[\mathit{x}(-\frac{1}{2}),\mathit{x}(\frac{3}{2})]$', linewidth=0,)
    legend = ax.legend(fontsize=32, loc='upper center')
    legend.get_frame().set_alpha(1.0)
    ax.set_xlabel("Epoch", fontsize=37)
    ax.set_ylabel("Spatial Instability" if y_scale == 'ratio' else "Spatial Instability", fontsize=37)
    
    if ylim:
        ax.set_ylim(ylim)
    else:
        
        ax.set_ylim(-0.0005, 0.02)
    
    
    ax.set_yticks(np.arange(0, 0.02 + 0.001, 0.01))
    
    # autoscale を無効にする（設定した ylim を維持
    ax.grid(True)
    ax.set_xscale('log')
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=35)
    fig.tight_layout()

    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 拡張子がない場合は .png を付加
    if not os.path.splitext(output_path)[1]:
        output_path = output_path + ".svg"
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[Info] Figure saved to: {output_path}")
if __name__ == "__main__":
    # base_dir はサンプルディレクトリを含む上位フォルダ（例： "noise_no_noise" のディレクトリ）
    base_dir = "/workspace/alpha_test/cifar10/0.2/64/noise"
    sample_dirs = get_sample_dirs(base_dir)
    print(f"見つかったサンプル数: {len(sample_dirs)}")
    


    # results = evaluate_label_changes_all(sample_dirs, mode='epoch', target='combined', epoch_start=1, epoch_end=30,plot_result=False)
    # results = evaluate_label_changes_all(sample_dirs, mode='epoch', target='combined', epoch_start=30, epoch_end=60,plot_result=False)
    # results = evaluate_label_changes_all(sample_dirs, mode='epoch', target='combined', epoch_start=60, epoch_end=140,plot_result=False)
    # results = evaluate_label_changes_all(sample_dirs, mode='epoch', target='combined', epoch_start=140, epoch_end=1000,plot_result=False)
    #results = evaluate_label_changes_all(sample_dirs, mode='epoch', target='combined', epoch_start=1, epoch_end=1000,plot_result=False)
    
    # base_output_dir = "/workspace/miru_vizualize/sigma_0/C_C/stablity/change_lim_sample_wise_plots"
    # # main3_samples(sample_dirs, base_output_dir, targets=["combined"], epoch_start=1, epoch_end=30)
    # # main3_samples(sample_dirs, base_output_dir, targets=["combined"], epoch_start=30, epoch_end=60)
    # # main3_samples(sample_dirs, base_output_dir, targets=["combined"], epoch_start=60, epoch_end=140)
    # #main3_samples(sample_dirs, base_output_dir, targets=["combined"], epoch_start=140, epoch_end=1000)

    output_dir = "/workspace/miru_vizualize/sigma_0/C_C/stablity"
    output_filename = "sample_wise_label_change.svg"
    output_path = os.path.join(output_dir, output_filename)

    # # 必要ならディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # 実行
    plot_label_changes_across_samples(sample_dirs, output_path, target="combined", y_scale="ratio",highlight_epochs=[1,30,60,140,1000])

