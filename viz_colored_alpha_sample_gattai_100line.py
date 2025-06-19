import os
import re
import fnmatch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

###############################################################################
# ユーティリティ関数
###############################################################################
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


def get_distance(sample_dir):
    """
    sample_dir/fig_and_log/distances.txt 内から、
    "Distance between x_noisy and y_clean:" または
    "Distance between x_clean and y_clean:" の値を取得して返す。
    値が取得できなかった場合は None を返す。
    """
    distances_path = os.path.join(sample_dir, "fig_and_log", "distances.txt")
    if not os.path.exists(distances_path):
        print(f"[Warning] distances.txtが存在しません: {distances_path}")
        return None
    try:
        with open(distances_path, "r") as f:
            for line in f:
                m = re.search(r"Distance between (x_noisy|x_clean) and y_clean:\s*([\d\.]+)", line)
                if m:
                    return float(m.group(2))
    except Exception as e:
        print(f"[Warning] {distances_path} の読み込みエラー: {e}")
    return None


def get_line_color(fig_and_log_dir):
    """
    fig_and_log_dir 内の distances.txt を読み込み、対象行から距離値を取得し、
    距離の大小に応じたカラーグラデーションを返す（例: 黄色〜緑）。
    何らかの理由で取得できなかった場合は "blue" を返す。

    ※ 実際の補間ロジックは簡略化しており、例示としての動作に留まります。
    """
    distances_file = os.path.join(fig_and_log_dir, "distances.txt")
    try:
        with open(distances_file, 'r') as f:
            for line in f:
                if ("Distance between x_noisy and y_clean:" in line) or ("Distance between x_clean and y_clean:" in line):
                    parts = line.split(':')
                    if len(parts) >= 2:
                        try:
                            distance = float(parts[1].strip())
                            # デモ用：distanceをもとに適当に R 成分を補正
                            # たとえば距離が大きいほど R->0 に近づくイメージ
                            val = max(0.0, min(1.0, 1.0 - distance*0.05))
                            return (val, 1.0, 0.0)
                        except ValueError:
                            pass
        return "blue"
    except Exception as e:
        print(f"[Warning] distances.txtの読み込みに失敗しました ({fig_and_log_dir}): {e}")
        return "blue"


###############################################################################
# ラベル変更の評価と可視化
###############################################################################
def evaluate_label_changes(
    directory,
    mode='alpha',
    target='digit',
    y_lim=None,
    smoothing=False,
    smoothing_window=5,
    y_scale='ratio',   # 'ratio', 'percent', または 'raw'
    plot_result=True,
    epoch_start=None,
    epoch_end=None
):
    """
    CSVファイルからラベル変更を評価し、結果をCSVとして保存。
    さらにオプションでプロットを作成。

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
        スムージングを適用するかどうか（今回は未使用、引数のみ残置）。
    smoothing_window : int, optional
        スムージングのウィンドウサイズ（同上）。
    y_scale : str, optional
        'ratio', 'percent', 'raw' のいずれか。
    plot_result : bool, optional
        True の場合は可視化プロットを保存。
    epoch_start : int, optional
    epoch_end : int, optional
    """
    csv_dir = os.path.join(directory, "csv")
    save_dir = os.path.join(directory, "fig_and_log")
    os.makedirs(save_dir, exist_ok=True)

    # 対象ファイルの取得
    all_files = sorted([
        f for f in os.listdir(csv_dir)
        if f.startswith("alpha_log_epoch_") and f.endswith(".csv")
    ])
    if not all_files:
        raise ValueError("No valid files found in the specified directory.")

    # epoch 範囲フィルタ
    files = []
    for f in all_files:
        epoch_str = f.split('_')[-1].split('.')[0]
        try:
            epoch_num = int(epoch_str)
            if epoch_start is not None and epoch_num < epoch_start:
                continue
            if epoch_end is not None and epoch_num > epoch_end:
                continue
            files.append(os.path.join(csv_dir, f))
        except ValueError:
            print(f"Warning: Skipping file with invalid epoch number: {f}")

    if not files:
        raise ValueError("No files found in the specified epoch range.")

    # ファイル名suffix
    epoch_suffix = ""
    if epoch_start is not None or epoch_end is not None:
        start_str = str(epoch_start) if epoch_start is not None else "start"
        end_str = str(epoch_end) if epoch_end is not None else "end"
        epoch_suffix = f"_epoch_{start_str}_to_{end_str}"
    file_suffix = epoch_suffix + ("_raw" if y_scale == "raw" else "")

    target_col = 'predicted_' + target
    scores = {}

    # =========================================================================
    # alpha モード
    # =========================================================================
    if mode == 'alpha':
        # 各CSVファイルから最初の202行を用い、隣接のラベル違いをカウント
        for file in files:
            df = pd.read_csv(file).iloc[:202]
            n = len(df) - 1
            raw_changes = (df[target_col] != df[target_col].shift()).sum() - 1

            if y_scale == 'raw':
                changes = raw_changes
            elif y_scale == 'ratio':
                changes = raw_changes / n
            elif y_scale == 'percent':
                changes = (raw_changes / n) * 100

            epoch_num = int(os.path.basename(file).split('_')[-1].split('.')[0])
            scores[epoch_num] = changes

        score_df = pd.DataFrame(list(scores.items()), columns=['epoch', 'label_change'])
        score_df.sort_values('epoch', inplace=True)

        csv_filename = os.path.join(save_dir, f"label_change_scores_alpha_{target}{file_suffix}.csv")
        score_df.to_csv(csv_filename, index=False)
        print(f"[Info] Scores saved as {csv_filename}")

        # 可視化
        if plot_result:
            epochs = score_df['epoch'].values
            label_changes = score_df['label_change'].values
            std_scores = np.zeros_like(label_changes)

            fig, ax = plt.subplots(figsize=(12, 8), dpi=400)
            ax.plot(epochs, label_changes, label="$INST_s(\\chi,t)$",
                    color="blue", linewidth=1.5, zorder=3)
            ax.fill_between(epochs, label_changes - std_scores,
                            label_changes + std_scores, color="blue", alpha=0.2, zorder=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Spatial Instability")
            if y_lim:
                ax.set_ylim(y_lim)
            ax.set_xscale('log')

            plot_filename = os.path.splitext(os.path.basename(csv_filename))[0] + "_plot.svg"
            plot_path = os.path.join(save_dir, plot_filename)
            fig.savefig(plot_path, bbox_inches='tight')
            plt.close(fig)
            print(f"[Info] Plot saved as {plot_path}")

    # =========================================================================
    # epoch モード
    # =========================================================================
    elif mode == 'epoch':
        # epochモード：連続するCSVファイル（エポック）間で行ごとのラベルが変わるかを集計
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
        if y_scale in ['ratio', 'percent']:
            unsmoothed_scores = [score / total_epochs for score in unsmoothed_scores]
        if y_scale == 'percent':
            unsmoothed_scores = [score * 100 for score in unsmoothed_scores]

        epoch_csv_path = os.path.join(save_dir, f"epoch_unsmoothed_scores_{target}{file_suffix}.csv")
        epoch_data = pd.DataFrame({'alpha': alpha_values, 'unsmoothed_scores': unsmoothed_scores})
        epoch_data.to_csv(epoch_csv_path, index=False)
        print(f"[Info] Epoch scores saved to {epoch_csv_path}")

        if plot_result:
            df_epoch = pd.read_csv(epoch_csv_path)
            alphas = df_epoch['alpha'].values
            scores_epoch = df_epoch['unsmoothed_scores'].values
            std_scores = np.zeros_like(scores_epoch)

            fig, ax = plt.subplots(figsize=(12, 8), dpi=400)
            ax.plot(alphas, scores_epoch, label="$INST_s(\\chi,t)$",
                    color="blue", linewidth=1.5, zorder=3)
            ax.fill_between(alphas, scores_epoch - std_scores,
                            scores_epoch + std_scores, color="blue", alpha=0.2, zorder=2)
            ax.set_xlabel("Alpha")
            ax.set_ylabel("Temporal Instability")
            if y_lim:
                ax.set_ylim(y_lim)

            plot_filename = f"epoch_unsmoothed_scores_{target}{file_suffix}_plot.png"
            plot_path = os.path.join(save_dir, plot_filename)
            fig.savefig(plot_path, bbox_inches='tight')
            plt.close(fig)
            print(f"[Info] Plot saved as {plot_path}")

    return scores


def evaluate_label_changes_all(sample_dirs, **kwargs):
    """
    get_sample_dirs で取得したすべてのサンプルディレクトリに対して
    evaluate_label_changes を実行するヘルパー関数。
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


###############################################################################
# プロット関連のメイン関数
###############################################################################
def main3_multi_epoch(
    csv_directories,
    base_output_dir,
    epoch_ranges,
    targets=["digit", "color", "combined"],
    output_name="a",
    plot_save_path=None,
    plot_legend=True,
    plot_individuals=False,
    use_color=False,
    filter_by_distance=False,
    filter_order="smallest",
    filter_n=10,
    epoch_label_pairs=None
):
    """
    複数の epoch 区間における図を横に並べた図を作成し、svg形式で保存する。
    （Code1より引用）

    - filter_by_distance=True の場合、get_distance() の結果でフィルタリングを行う。
    - epoch_ranges: [(start1, end1), (start2, end2), ...] の形で指定。
    """
    if filter_by_distance:
        filtered_dirs = []
        for d in csv_directories:
            dist = get_distance(d)
            if dist is not None:
                filtered_dirs.append((d, dist))
            else:
                print(f"[Warning] 距離が取得できませんでした: {d}")
        if filter_order == "smallest":
            sorted_dirs = sorted(filtered_dirs, key=lambda x: x[1])
        elif filter_order == "largest":
            sorted_dirs = sorted(filtered_dirs, key=lambda x: x[1], reverse=True)
        else:
            print("[Warning] filter_order は 'smallest' または 'largest' を指定してください。")
            sorted_dirs = filtered_dirs
        csv_directories = [d for d, _ in sorted_dirs[:filter_n]]
        print("[Info] 処理対象のディレクトリ（相対パス）:")
        for d in csv_directories:
            print(os.path.relpath(d))

    os.makedirs(base_output_dir, exist_ok=True)

    for target in targets:
        num_epochs = len(epoch_ranges)
        fig, axes = plt.subplots(
            ncols=num_epochs, nrows=1,
            figsize=(12 * num_epochs, 9), dpi=400, sharey=True
        )
        if num_epochs == 1:
            axes = [axes]

        for i, (epoch_start, epoch_end) in enumerate(epoch_ranges):
            start_str = str(epoch_start) if epoch_start is not None else "start"
            end_str = str(epoch_end) if epoch_end is not None else "end"
            epoch_suffix = f"_epoch_{start_str}_to_{end_str}"

            ax = axes[i]
            data_list = []
            groups_present = set()

            for directory in csv_directories:
                csv_filename = f"epoch_unsmoothed_scores_{target}{epoch_suffix}.csv"
                csv_path = os.path.join(directory, 'fig_and_log', csv_filename)
                if not os.path.exists(csv_path):
                    print(f"[Warning] CSVファイルが存在しません: {csv_path}")
                    continue

                # ディレクトリ名を見て適当なラベル
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
                    df['target'] = target
                    df['seed'] = seed
                    data_list.append({
                        'df': df,
                        'fig_and_log': os.path.join(directory, 'fig_and_log')
                    })
                except Exception as e:
                    print(f"CSVファイルの読み込みエラー ({csv_path}): {e}")

            if not data_list:
                print(f"[Warning] データが収集されませんでした: {target} {epoch_suffix}")
                continue

            combined_df = pd.concat([item['df'] for item in data_list], ignore_index=True)
            stats_df = combined_df.groupby('alpha')['unsmoothed_scores'].agg(['mean', 'std']).reset_index()

            marker_size = 20
            # グループによりマーカー（0.0, 1.0）の色を変える例
            if 'no_noise_no_noise' in groups_present:
                ax.plot(0.0, 0.0, marker='o', color='blue', markersize=marker_size, zorder=5)
                ax.plot(1.0, 0.0, marker='o', color='blue', markersize=marker_size)
            elif 'noise_no_noise' in groups_present:
                ax.plot(0.0, 0.0, marker='o', color='blue', markersize=marker_size)
                ax.plot(1.0, 0.0, marker='o', color='red', markersize=marker_size)
            else:
                ax.plot(0.0, 0.0, marker='o', color='blue', markersize=marker_size, zorder=5)
                ax.plot(1.0, 0.0, marker='o', color='red', markersize=marker_size)

            # 軸フォーマット
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            ax.set_ylim(-0.01, 0.6)
            if i == 0:
                ax.set_yticks([0, 0.2, 0.4, 0.6])
                ax.set_yticklabels(["0", "0.2", "0.4", "0.6"], fontsize=40)
                ax.set_ylabel("Temporal Instability", fontsize=45)
            else:
                ax.tick_params(axis='y', which='both', length=0, labelleft=False)

            # 凡例用ダミー
            ax.plot([], [], label=rf'$\mathcal{{T}}$=[{epoch_start},{epoch_end}]', linewidth=0)
            ax.axvline(x=0.0, color='black', linestyle='-', linewidth=2, zorder=0)
            ax.axvline(x=1.0, color='black', linestyle='-', linewidth=2, zorder=0)

            # 個別プロットか平均プロットか
            if plot_individuals:
                for item in data_list:
                    if use_color:
                        color = get_line_color(item['fig_and_log'])
                    else:
                        color = "red"
                    ax.plot(item['df']['alpha'], item['df']['unsmoothed_scores'],
                            color=color, linewidth=1, alpha=0.8, zorder=2)
                ax.plot(stats_df['alpha'], stats_df['mean'],
                        label=r'$INST_{\mathrm{T}}(\mathcal{T},x)$', color='blue', linewidth=3, zorder=3)
            else:
                ax.plot(stats_df['alpha'], stats_df['mean'],
                        label=r'$INST_{\mathrm{T}}(\mathcal{T},x)$', color='blue', linewidth=3, zorder=2)
                ax.fill_between(stats_df['alpha'],
                                stats_df['mean'] - stats_df['std'],
                                stats_df['mean'] + stats_df['std'],
                                color='blue', alpha=0.15, zorder=1)

            ax.set_xticks([0.0, 1.0])
            ax.set_xticklabels([r'$x_{0}$', r'$x_{1}$'], fontsize=60)
            ax.grid(True)
            if plot_legend:
                legend = ax.legend(fontsize=30)
                legend.get_frame().set_alpha(1.0)

        fig.tight_layout()
        if plot_save_path is not None:
            plot_filename = f"{plot_save_path}{target}_multiple_epochs.svg"
        else:
            plot_filename = f'width_4_C_N_output_{target}_multiple_epochs.svg'
        plot_path = os.path.join(base_output_dir, plot_filename)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig.savefig(plot_path, format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"[Info] プロットを保存しました: {plot_path}")


def main3_samples(
    sample_dirs,
    base_output_dir,
    targets=["digit", "color", "combined"],
    output_name="a",
    epoch_start=None,
    epoch_end=None,
    plot_individuals=False,
    use_color=False,
    y_scale="ratio"
):
    """
    各サンプルディレクトリ内の CSV ファイル（epoch_unsmoothed_scores_*）を読み込み、
    α ごとの unsmoothed_scores の平均と標準偏差を計算して可視化。

    - plot_individuals=True の場合、個別サンプルの曲線も描画。
    - use_color=True の場合、各サンプルの distances.txt に基づくカラーを適用。
    - y_scale=="raw" の場合、末尾に "_raw" が付与されたCSVファイルを読み込む。
    """
    os.makedirs(base_output_dir, exist_ok=True)
    epoch_suffix = ""
    if epoch_start is not None or epoch_end is not None:
        start_str = str(epoch_start) if epoch_start is not None else "start"
        end_str = str(epoch_end) if epoch_end is not None else "end"
        epoch_suffix = f"_epoch_{start_str}_to_{end_str}"

    for target in targets:
        data_info = []
        for sample_dir in sample_dirs:
            # ファイル名を決定
            if y_scale == "raw":
                csv_filename = f"epoch_unsmoothed_scores_{target}{epoch_suffix}_raw.csv"
            else:
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
                data_info.append({
                    'df': df,
                    'fig_and_log': os.path.join(sample_dir, "fig_and_log")
                })
            except Exception as e:
                print(f"[Warning] CSVファイルの読み込みエラー ({csv_path}): {e}")

        if not data_info:
            print(f"[Warning] 対象 {target} のデータが収集されませんでした。")
            continue

        combined_df = pd.concat([info['df'] for info in data_info], ignore_index=True)
        stats_df = combined_df.groupby('alpha')['unsmoothed_scores'].agg(['mean', 'std']).reset_index()

        fig, ax = plt.subplots(figsize=(12, 8), dpi=400)

        if plot_individuals:
            # 各サンプルを薄い線で描画
            for info in data_info:
                if use_color:
                    color = get_line_color(info['fig_and_log'])
                else:
                    color = "blue"
                ax.plot(info['df']['alpha'], info['df']['unsmoothed_scores'],
                        color=color, linewidth=1, alpha=0.3, zorder=2)
            # 平均＋標準偏差
            ax.plot(stats_df['alpha'], stats_df['mean'],
                    label=r'$INST_{\mathrm{T}}(\mathcal{T},x)$', color='blue', linewidth=3, zorder=3)
            ax.fill_between(stats_df['alpha'],
                            stats_df['mean'] - stats_df['std'],
                            stats_df['mean'] + stats_df['std'],
                            color='blue', alpha=0.15, zorder=1)
        else:
            # 平均＋標準偏差のみ
            ax.plot(stats_df['alpha'], stats_df['mean'],
                    label=r'$INST_{\mathrm{T}}(\mathcal{T},x)$', color='blue', linewidth=3, zorder=2)
            ax.fill_between(stats_df['alpha'],
                            stats_df['mean'] - stats_df['std'],
                            stats_df['mean'] + stats_df['std'],
                            color='blue', alpha=0.15, zorder=1)

        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.set_ylim(-0.01, 0.6)
        ax.set_ylabel("Temporal Instability", fontsize=30)
        ax.axvline(x=0.0, color='black', linestyle='-', linewidth=2, zorder=0)
        ax.axvline(x=1.0, color='black', linestyle='-', linewidth=2, zorder=0)
        ax.set_xticks([0.0, 1.0])
        ax.grid(True)
        legend = ax.legend(fontsize=20)
        legend.get_frame().set_alpha(1.0)
        fig.tight_layout()

        plot_filename = f'output_{target}{epoch_suffix}.svg'
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
    highlight_epochs=None,
    plot_individuals=False,
    use_color=False
):
    """
    複数のサンプルディレクトリから CSV ファイル (label_change_scores_alpha_*) を読み込み、
    エポック毎に平均・標準偏差を可視化。
    """
    df_info = []
    for sample_dir in sample_dirs:
        fig_and_log_path = os.path.join(sample_dir, "fig_and_log")
        csv_file = os.path.join(fig_and_log_path, f"label_change_scores_alpha_{target}.csv")
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file).sort_values(by="epoch")
                df_info.append({
                    'df': df,
                    'fig_and_log': fig_and_log_path
                })
            except Exception as e:
                print(f"[Warning] {csv_file} の読み込みに失敗しました: {e}")
        else:
            print(f"[Warning] CSVファイルが見つかりません: {csv_file}")

    if not df_info:
        print("[Warning] 有効な CSV ファイルが見つかりませんでした。処理を中断します。")
        return

    epochs = df_info[0]['df']["epoch"].values
    all_scores = np.array([info['df']["label_change"].values for info in df_info])
    mean_scores = np.mean(all_scores, axis=0)
    std_scores = np.std(all_scores, axis=0)

    if y_scale == 'percent':
        mean_scores *= 100
        std_scores *= 100

    fig, ax = plt.subplots(figsize=(12, 8), dpi=400)
    if highlight_epochs:
        for ep in highlight_epochs:
            ax.axvline(x=ep, color='black', linestyle='-', linewidth=1.5, zorder=1)

    if plot_individuals:
        for info in df_info:
            if use_color:
                color = get_line_color(info['fig_and_log'])
            else:
                color = "blue"
            ax.plot(info['df']['epoch'], info['df']['label_change'],
                    color=color, linewidth=1, alpha=0.2, zorder=2)
        ax.plot(epochs, mean_scores, label="$INST_s(\\chi,t)$",
                color="blue", linewidth=1.5, zorder=3)
    else:
        ax.plot(epochs, mean_scores, label="$INST_s(\\chi,t)$",
                color="blue", linewidth=1.5, zorder=3)
        ax.fill_between(epochs, mean_scores - std_scores,
                        mean_scores + std_scores, color="blue", alpha=0.2, zorder=2)

    ax.plot([], [], label=r'$\mathcal{X}=[\mathit{x}(-\frac{1}{2}),\mathit{x}(\frac{3}{2})]$', linewidth=0)
    legend = ax.legend(fontsize=12, loc='upper center')
    legend.get_frame().set_alpha(1.0)

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Spatial Instability", fontsize=14)
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, 0.03)

    ax.set_xscale('log')
    ax.grid(True)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not os.path.splitext(output_path)[1]:
        output_path = output_path + ".svg"
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[Info] Figure saved to: {output_path}")
if __name__ == "__main__":
    # base_dir はサンプルディレクトリを含む上位フォルダ（例： "noise_no_noise" のディレクトリ）

    noise= "0.2"
    seed = "42"
    pair = "noise_no_noise"
    var = "0"

    base_dir = f"/workspace/alpha_test/seed/width_4/lr_{noise}_sigma_{var}/ori_closet_seed_{seed}width4_cnn_5layers_distribution_colored_emnist_variance{var}_combined_lr0.01_batch256_epoch1000_LabelNoiseRate{noise}_Optimsgd_Momentum0.0/{pair}"
    
    sample_dirs = get_sample_dirs(base_dir)
    print(f"見つかったサンプル数: {len(sample_dirs)}")
    # for sd in sample_dirs:
    #     evaluate_label_changes(
    #         sd,
    #         mode='epoch',
    #         target='combined',
    #         epoch_start=1,
    #         epoch_end=1000,
    #         y_scale='ratio',
    #         plot_result=False
    #     )
    #     print(sd)
    # 例：距離が小さい順に上位 5 個を対象とする場合
    plot_individual = False
    filter="smallest"
    #smallest
    #largest
    f_n=10
    base_output_dir = f"/workspace/fig/seed_{seed}/noise_{noise}/var_{var}/{pair}"
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"[Info] Output directory created: {base_output_dir}")
    #[(1, 30), (30, 60), (60,140), (140,1000)]
    epoch_ranges =    [(1,1000)]
    epoch_labels = [("A", "H"),]
    # ("C", "E"), ("E", "G"), ("G", "H")
    main3_multi_epoch(sample_dirs, base_output_dir, epoch_ranges, targets=["combined"],
                     plot_save_path="a", plot_legend=True,
                     plot_individuals=plot_individual, use_color=False,
                     filter_by_distance=False, filter_order=filter, filter_n=f_n)
