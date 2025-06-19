import os
import re
from typing import List, Tuple, Dict, Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
import os, glob, concurrent.futures
def get_sample_dirs(base_dir: str) -> List[str]:
    """
    base_dir 以下の 2 階層下で "fig_and_log" を含むディレクトリを返す。
    """
    sample_dirs = []
    for d in os.listdir(base_dir):
        d_path = os.path.join(base_dir, d)
        if not os.path.isdir(d_path):
            continue
        for sub in os.listdir(d_path):
            sub_path = os.path.join(d_path, sub)
            if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "fig_and_log")):
                sample_dirs.append(sub_path)
    return sample_dirs

def get_sample_dirs_one_level(base_dir: str) -> List[str]:
    """
    base_dir 直下のディレクトリで、"fig_and_log" を含むものを返す。
    例: base_dir/sample1/fig_and_log が存在する場合、base_dir/sample1 を返す。
    """
    sample_dirs = []
    for d in os.listdir(base_dir):
        d_path = os.path.join(base_dir, d)
        if os.path.isdir(d_path) and os.path.exists(os.path.join(d_path, "fig_and_log")):
            sample_dirs.append(d_path)
    return sample_dirs

def list_epoch_files(
    csv_dir: str,
    start: Optional[int] = None,
    end:   Optional[int] = None
) -> List[Tuple[int, str]]:
    """
    csv_dir 内の epoch_{n}.csv を (epoch, filepath) のリストで返す。
    範囲指定 (start,end) があればフィルタ。
    """
    files = []
    for fname in os.listdir(csv_dir):
        match = re.match(r'epoch_(\d+)\.csv$', fname)
        if not match:
            continue
        epoch = int(match.group(1))
        if start is not None and epoch < start:
            continue
        if end   is not None and epoch > end:
            continue
        files.append((epoch, os.path.join(csv_dir, fname)))
    files.sort(key=lambda x: x[0])
    return files


def compute_spatial_instability(
    df: pd.DataFrame,
    y_scale: Literal['ratio', 'percent', 'raw'] = 'ratio'
) -> float:
    """
    各 epoch 内での predicted_label の変化回数を計算。
    'ratio'/'percent'/'raw' に対応。
    """
    preds = df['predicted_label']
    n = len(preds) - 1
    if n <= 0 or preds.nunique() <= 1:
        return 0.0

    changes = int((preds != preds.shift()).sum()) - 1
    if y_scale == 'ratio':
        return changes / n
    if y_scale == 'percent':
        return (changes / n) * 100
    return float(changes)


def compute_temporal_instability(
    dfs: List[pd.DataFrame],
    y_scale: Literal['ratio', 'percent', 'raw'] = 'ratio'
) -> Dict[float, float]:
    """
    複数 epoch の df リストを受け取り、各 alpha ごとに変化回数を計算。
    'ratio'/'percent'/'raw' に対応。

    Returns
    -------
    { alpha_value: score }
    """
    if len(dfs) < 2:
        return {}
    alphas = dfs[0]['alpha'].to_numpy()
    M = len(alphas)
    counts = np.zeros(M, dtype=int)
    # 各 epoch 間の変化を集計
    for i in range(len(dfs) - 1):
        cur = dfs[i]['predicted_label'].to_numpy()
        nxt = dfs[i+1]['predicted_label'].to_numpy()
        counts += (cur != nxt).astype(int)
    # スケール
    if y_scale == 'ratio':
        scores = counts / (len(dfs) - 1)
    elif y_scale == 'percent':
        scores = (counts / (len(dfs) - 1)) * 100
    else:
        scores = counts.astype(float)
    return {alpha: float(score) for alpha, score in zip(alphas, scores)}


def save_scores_csv(
    df: pd.DataFrame,
    path: str
) -> None:
    """
    DataFrame を CSV に保存し、ディレクトリも自動作成。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[✓] Scores saved to: {path}")


def load_score_csv(path: str) -> pd.DataFrame:
    """
    CSV を読み込んで DataFrame で返す。
    """
    return pd.read_csv(path)


def plot_instability_curve(
    x:       np.ndarray,
    y:       np.ndarray,
    std:     Optional[np.ndarray],
    xlabel:  str,
    ylabel:  str,
    save_path: str,
    log_scale_x: bool = False,
    y_lim:   Optional[Tuple[float,float]] = None
) -> None:
    """
    mean (y) ± std を帯域表示した折れ線図を保存。
    """
    fig, ax = plt.subplots(figsize=(8,5), dpi=300)
    ax.plot(x, y, linewidth=2, zorder=3)
    if std is not None:
        ax.fill_between(x, y - std, y + std, alpha=0.2, zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if log_scale_x:
        ax.set_xscale('log')
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[✓] Plot saved to: {save_path}")


def evaluate_label_changes(
    pair_csv_dir: str,
    output_dir:   str,
    mode:         Literal['alpha','epoch'] = 'alpha',
    y_scale:      Literal['ratio','percent','raw'] = 'ratio',
    epoch_start:  Optional[int] = None,
    epoch_end:    Optional[int] = None,
    plot:         bool = True
) -> Dict[float,float]:
    """
    単一サンプル (pair_csv_dir) に対して指定モードの不安定性を計算し、
    scores を CSV/プロットとして output_dir に保存。
    """
    files = list_epoch_files(pair_csv_dir, epoch_start, epoch_end)
    if not files:
        raise ValueError("No epoch CSV files found in directory.")
    suffix = ''
    if epoch_start is not None or epoch_end is not None:
        s = str(epoch_start) if epoch_start is not None else 'start'
        e = str(epoch_end)   if epoch_end   is not None else 'end'
        suffix = f"_epoch_{s}_to_{e}"

    if mode == 'alpha':
        scores = {}
        for ep, fp in files:
            df = pd.read_csv(fp)
            score = compute_spatial_instability(df, y_scale)
            scores[ep] = score
        df_out = pd.DataFrame({'epoch': list(scores.keys()), 'label_change': list(scores.values())})
        csv_path = os.path.join(output_dir, f'label_change_scores_alpha{suffix}.csv')
        save_scores_csv(df_out, csv_path)
        if plot:
            xs = df_out['epoch'].to_numpy()
            ys = df_out['label_change'].to_numpy()
            plot_instability_curve(
                xs, ys, None,
                xlabel='Epoch',
                ylabel=f"Spatial Instability ({y_scale})",
                save_path=os.path.join(output_dir, f'label_change_scores_alpha{suffix}.svg'),
                log_scale_x=True
            )
        return scores

    # mode == 'epoch'
    dfs = [pd.read_csv(fp) for _, fp in files]
    scores = compute_temporal_instability(dfs, y_scale)
    df_out = pd.DataFrame({'alpha': list(scores.keys()), 'unsmoothed_scores': list(scores.values())})
    csv_path = os.path.join(output_dir, f'epoch_unsmoothed_scores{suffix}.csv')
    save_scores_csv(df_out, csv_path)
    if plot:
        xs = np.array(list(scores.keys()))
        ys = np.array(list(scores.values()))
        plot_instability_curve(
            xs, ys, None,
            xlabel='Alpha',
            ylabel=f"Temporal Instability ({y_scale})",
            save_path=os.path.join(output_dir, f'epoch_unsmoothed_scores{suffix}.svg')
        )
    return scores


def evaluate_label_changes_all(
    sample_dirs: List[str],
    **kwargs
) -> Dict[str, Dict[float,float]]:
    """
    複数サンプルで evaluate_label_changes を一括実行。
    """
    all_scores = {}
    for d in sample_dirs:
        try:
            print(f"[Info] Processing {d}")
            sc = evaluate_label_changes(d, **kwargs)
            all_scores[d] = sc
        except Exception as ex:
            print(f"[Error] {d}: {ex}")
    return all_scores


def aggregate_instability_across_samples(
    sample_dirs:  List[str],
    target:       str,
    mode:         Literal['alpha','epoch'],
    y_scale:      Literal['ratio','percent','raw'],
    epoch_range:  Optional[Tuple[int,int]] = None
) -> pd.DataFrame:
    """
    各サンプルのスコア CSV を読み込み、x_value ごとの mean/std を返す DataFrame。
    """
    # suffix
    suffix = ''
    if epoch_range:
        s, e = epoch_range
        suffix = f"_epoch_{s}_to_{e}"

    rows = []
    for d in sample_dirs:
        base = os.path.join(d, 'fig_and_log')
        if mode == 'alpha':
            # fname = f'label_change_scores_alpha{suffix}.csv'
            fname = f'label_change_scores_alpha.csv'
            x_col, y_col = 'epoch', 'label_change'
        else:
            fname = f'epoch_unsmoothed_scores{suffix}.csv'
            x_col, y_col = 'alpha', 'unsmoothed_scores'
        fpath = os.path.join(base, fname)
        if not os.path.exists(fpath):
            print(f"[Warn] missing {fpath}")
            continue
        df = pd.read_csv(fpath)
        for _, row in df.iterrows():
            rows.append((row[x_col], row[y_col]))
    if not rows:
        return pd.DataFrame()
    df_all = pd.DataFrame(rows, columns=['x', 'score'])
    stats = df_all.groupby('x')['score'].agg(['mean','std']).reset_index()
    stats.rename(columns={'x':'x_value','mean':'mean_score','std':'std_score'}, inplace=True)
    return stats


def plot_aggregate_instability(
    stats_df: pd.DataFrame,
    xlabel:   str,
    ylabel:   str,
    save_path: str,
    highlight: Optional[List[float]] = None,
    log_scale_x: bool = False,
    y_lim:    Optional[Tuple[float,float]] = None
) -> None:
    """
    aggregate_instability の結果をプロット保存。
    """
    fig, ax = plt.subplots(figsize=(8,5), dpi=300)
    x = stats_df['x_value'].to_numpy()
    y = stats_df['mean_score'].to_numpy()
    std = stats_df['std_score'].to_numpy()
    if highlight:
        for v in highlight:
            ax.axvline(v, color='gray', linestyle='--')
    ax.plot(x, y, linewidth=2, zorder=3)
    ax.fill_between(x, y-std, y+std, alpha=0.2, zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if y_lim:
        ax.set_ylim(y_lim)
    if log_scale_x:
        ax.set_xscale('log')
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[✓] Aggregate plot saved to: {save_path}")

def append_cross_entropy_to_csv(
    csv_dir:      str,
    true_label:   int,
    epoch_start:  Optional[int] = None,
    epoch_end:    Optional[int] = None
) -> None:
    """
    各 epoch_{n}.csv に対して、行ごとの -log(prob_{true_label}) を計算し
    'cross_entropy' 列として追加して上書き保存する。
    """
    files = list_epoch_files(csv_dir, epoch_start, epoch_end)
    if not files:
        raise ValueError(f"No epoch CSV files found in {csv_dir}")

    prob_col = f'prob_{true_label}'
    for ep, fp in files:
        df = pd.read_csv(fp)
        if prob_col not in df.columns:
            raise ValueError(f"File {fp} does not contain column '{prob_col}'")
        probs = df[prob_col].astype(float).to_numpy()
        # log(0) 対策
        probs = np.clip(probs, 1e-12, 1.0)
        df['cross_entropy'] = -np.log(probs)
        # 上書き保存
        df.to_csv(fp, index=False)
        print(f"[✓] Appended 'cross_entropy' to: {fp}")

def save_label_change_to_csv_with_sample_dirs(base_root, widths, target_epoch, save_path):
    data = []

    for width in widths:
        for noise_type in ['noise', 'no_noise']:
            base_dir = os.path.join(base_root, str(width), noise_type)
            sample_dirs = get_sample_dirs(base_dir)  # get_sample_dirsを使用
            scores = []

            for sdir in sample_dirs:
                score_file = os.path.join(sdir, 'fig_and_log', 'label_change_scores_alpha.csv')
                if not os.path.exists(score_file):
                    print(f"[Warn] Missing score file: {score_file}")
                    continue
                try:
                    df = pd.read_csv(score_file)
                    row = df[df['epoch'] == target_epoch]
                    if not row.empty:
                        score = float(row['label_change'].values[0])
                        scores.append(score)
                    else:
                        print(f"[Warn] No data at epoch={target_epoch} in {score_file}")
                except Exception as e:
                    print(f"[Error] Failed reading {score_file}: {e}")

            if scores:
                mean_score = np.mean(scores)
                data.append({'width': width, 'noise': mean_score if noise_type == 'noise' else None, 'no_noise': mean_score if noise_type == 'no_noise' else None})

    # データフレームに変換してCSVに保存
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"[✓] CSV saved to: {save_path}")

def plot_mean_match_rates_per_epoch(base_dir, plot_save_path, csv_save_path):
    """
    指定フォルダ以下にある "epoch_*.csv" ファイルを探索し、
    predicted_label がフォルダ名に含まれる数値 (num1, num2) と一致した割合を
    各エポックごとに平均し、プロットおよびデータ保存を行う。

    Parameters
    ----------
    base_dir : str
        ペアごとのCSVファイルを格納しているベースディレクトリ
        例: base_dir/pairXXX/num1_num2/csv/epoch_0.csv ... のような構造
    
    plot_save_path : str
        プロット画像の保存先パス
    
    csv_save_path : str
        プロットに使用するデータを保存するCSVのパス
    """

    # 各epochごとに、「数字1との一致数」「数字2との一致数」「全サンプル数」を保持
    sum_matches_num1 = defaultdict(int)
    sum_matches_num2 = defaultdict(int)
    sum_samples = defaultdict(int)

    # base_dir配下を探索
    for root, dirs, files in os.walk(base_dir):
        # epoch_*.csv だけを抽出
        epoch_files = [f for f in files if f.startswith("epoch_") and f.endswith(".csv")]

        # パスから pairXXX/num1_num2/csv の部分を取り出す
        #   例: .../pair3/7_2/csv/ → num1=7, num2=2
        match = re.search(r'pair\d+/(\d+)_(\d+)/csv', root)
        if not match:
            continue
        num1 = int(match.group(1))
        num2 = int(match.group(2))

        # epoch_* のファイルを順番に処理
        for file in sorted(epoch_files):
            epoch_match = re.match(r'epoch_(\d+)\.csv', file)
            if not epoch_match:
                continue

            epoch = int(epoch_match.group(1))
            csv_path = os.path.join(root, file)

            # 「predicted_label」列のみ取得してメモリ節約
            df = pd.read_csv(csv_path, usecols=['predicted_label'])
            predicted = df['predicted_label'].values

            # num1, num2との一致数をそれぞれカウント
            matches_num1 = np.count_nonzero(predicted == num1)
            matches_num2 = np.count_nonzero(predicted == num2)

            # 同じ epoch に対して合計していく
            sum_matches_num1[epoch] += matches_num1
            sum_matches_num2[epoch] += matches_num2
            sum_samples[epoch]      += len(predicted)

    # 各epochごとに平均一致率を計算し、プロット用データを作成
    epochs = sorted(sum_matches_num1.keys())
    results = []
    for ep in epochs:
        avg_rate_num1 = sum_matches_num1[ep] / sum_samples[ep]
        avg_rate_num2 = sum_matches_num2[ep] / sum_samples[ep]
        results.append([ep, avg_rate_num1, avg_rate_num2])

    # DataFrame化して CSV 保存
    result_df = pd.DataFrame(results, columns=['epoch', 'number1', 'number2'])
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    result_df.to_csv(csv_save_path, index=False)
    print(f"✅ データをCSVに保存しました: {csv_save_path}")

    # プロット
    plt.figure(figsize=(8, 5))
    plt.plot(result_df['epoch'], result_df['number1'], label='数字1との平均一致率')
    plt.plot(result_df['epoch'], result_df['number2'], label='数字2との平均一致率')
    plt.xlabel('Epoch')
    plt.ylabel('一致率')
    plt.title('Epochごとの平均一致率（全pair平均）')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 画像を保存
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path)
    plt.close()
    print(f"✅ 図を保存しました: {plot_save_path}")

def analyze_all_temporal_instability(base_root, widths, output_dir, target_row=None):
    """
    すべてのalphaに対して temporal instability を計算し、
    width ごとの CSV ファイルを生成する。
    さらに、各 alpha に対して width vs instability のプロットを生成する。

    Parameters
    ----------
    base_root : str
        ベースディレクトリのパス
    widths : list
        調査対象の width のリスト
    output_dir : str
        結果を保存するディレクトリ
    target_row : int, optional
        特定の行のみを解析する場合の行番号（1-indexed）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 最初のファイルからalpha値のリストを取得
    first_sample = None
    for width in widths:
        base_dir = os.path.join(base_root, str(width), 'noise')
        samples = get_sample_dirs(base_dir)
        if samples:
            first_sample = samples[0]
            break
    
    if not first_sample:
        raise ValueError("No sample directories found")
    
    first_file = os.path.join(first_sample, 'fig_and_log', 'epoch_unsmoothed_scores.csv')
    df_first = pd.read_csv(first_file)
    alpha_values = df_first['alpha'].tolist()
    
    if target_row is not None:
        alpha_values = [alpha_values[target_row - 1]]

    # 各widthに対してデータを収集
    width_data = {}
    for width in widths:
        print(f"[Info] Processing width {width}...")
        results = []
        
        for alpha in alpha_values:
            scores_noise = []
            scores_no_noise = []
            
            # noise データの収集
            for noise_type in ['noise', 'no_noise']:
                base_dir = os.path.join(base_root, str(width), noise_type)
                sample_dirs = get_sample_dirs(base_dir)
                scores = []
                
                for sdir in sample_dirs:
                    score_file = os.path.join(sdir, 'fig_and_log', 'epoch_unsmoothed_scores.csv')
                    if not os.path.exists(score_file):
                        continue
                    try:
                        df = pd.read_csv(score_file)
                        closest_alpha_idx = (df['alpha'] - alpha).abs().idxmin()
                        score = float(df.iloc[closest_alpha_idx]['unsmoothed_scores'])
                        if noise_type == 'noise':
                            scores_noise.append(score)
                        else:
                            scores_no_noise.append(score)
                    except Exception as e:
                        print(f"[Error] Failed reading {score_file}: {e}")
            
            # 結果を記録
            results.append({
                'alpha': alpha,
                'noise_mean': np.mean(scores_noise) if scores_noise else np.nan,
                'noise_std': np.std(scores_noise) if scores_noise else np.nan,
                'no_noise_mean': np.mean(scores_no_noise) if scores_no_noise else np.nan,
                'no_noise_std': np.std(scores_no_noise) if scores_no_noise else np.nan,
                'n_samples_noise': len(scores_noise),
                'n_samples_no_noise': len(scores_no_noise)
            })
        
        # widthごとのCSVファイルを保存
        df_results = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, f'temporal_instability_width_{width}.csv')
        df_results.to_csv(csv_path, index=False)
        print(f"[✓] Saved results for width {width} to {csv_path}")
        
        width_data[width] = df_results

    # 各alphaに対してプロットを生成
    for i, alpha in enumerate(alpha_values):
        plt.figure(figsize=(10, 6))
        
        # noise と no_noise のデータを収集
        noise_means = []
        noise_stds = []
        no_noise_means = []
        no_noise_stds = []
        
        for width in widths:
            df = width_data[width]
            row = df[df['alpha'] == alpha].iloc[0]
            noise_means.append(row['noise_mean'])
            noise_stds.append(row['noise_std'])
            no_noise_means.append(row['no_noise_mean'])
            no_noise_stds.append(row['no_noise_std'])
        
        # プロット
        plt.errorbar(widths, noise_means, yerr=noise_stds, 
                    label='noise', marker='o', capsize=5)
        plt.errorbar(widths, no_noise_means, yerr=no_noise_stds, 
                    label='no_noise', marker='s', capsize=5)
        
        plt.xlabel('Width')
        plt.ylabel('Temporal Instability')
        plt.title(f'Temporal Instability vs Width (alpha={alpha:.3f})')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        
        # プロットを保存
        plot_path = os.path.join(output_dir, f'temporal_instability_alpha_{alpha:.3f}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"[✓] Saved plot for alpha={alpha:.3f}")
if __name__ == '__main__':
    print("[Info] Analyzing temporal instability for all alpha values...")
    try:
        analyze_all_temporal_instability(
            base_root='/workspace/alpha_test/cifar10/0.2',
            widths=[1, 2, 4, 8,10,12,16, 32, 64],
            output_dir='/workspace/alpha_test/cifar10/0.2/temporal_instability_analysis',
            target_row=None  # すべてのalphaを解析
        )
    except Exception as e:
        print(f"[Error] Failed to analyze temporal instability: {e}")
# if __name__ == '__main__':
#     print("[Info] Plotting mean match rates per epoch...")
#     try:
#         plot_mean_match_rates_per_epoch(
#             base_dir="/workspace/alpha_test/cifar10/0.2/64/noise",
#             plot_save_path="/workspace/alpha_test/cifar10/0.2/64/noise/fig/match.png",
#             csv_save_path="/workspace/alpha_test/cifar10/0.2/64/noise/fig/match.csv"
#         )
#     except Exception as e:
#         print(f"[Error] Failed to plot match rates: {e}")


# if __name__ == '__main__':
#     # 使用例
#     save_label_change_to_csv_with_sample_dirs(
#         base_root='alpha_test/cifar10/0.2',
#         widths=[1, 2, 4, 8, 10, 12, 16, 32, 64],
#         target_epoch=4000,  # 注目するエポックを指定
#         save_path='alpha_test/cifar10/0.2/label_change_results.csv'
#     )
# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser(
#         description='Instability Analysis: compute spatial/temporal instability per sample and aggregate.'
#     )
#     parser.add_argument('base_dir',help='Base directory containing sample subdirectories (two levels down with csv/ and fig_and_log/)'  )
#     parser.add_argument('--mode', choices=['alpha','epoch'], default='alpha',help="""'alpha': spatial instability per epoch (predict label changes along alpha axis)'epoch': temporal instability per alpha (changes across epochs)""")
#     parser.add_argument('--y_scale', choices=['ratio','percent','raw'], default='ratio',help='Scale for instability scores')
#     parser.add_argument('--epoch_start', type=int, default=None, help='Start epoch filter')
#     parser.add_argument('--epoch_end',   type=int, default=None, help='End epoch filter')
#     parser.add_argument('--aggregate', action='store_true', help='After per-sample eval, aggregate across samples')
#     parser.add_argument('--target', default='combined', help='Target name for aggregation CSV suffix')
#     parser.add_argument('--plot_save', default='aggregate_instability.svg', help='Save path for aggregated plot')

#     args = parser.parse_args()

#     # サンプルディレクトリ一覧取得
#     samples = get_sample_dirs(args.base_dir)
#     # if not samples:
#     #     print(f"[Error] No sample directories found under {args.base_dir}")
#     #     sys.exit(1)


#     # for sample in samples:
#     #     csv_dir = os.path.join(sample, 'csv')
#     #     out_dir = os.path.join(sample, 'fig_and_log')
#     #     print(f"[Info] Evaluating sample: {sample}")
#     #     try:
#     #         evaluate_label_changes(
#     #             pair_csv_dir=csv_dir,
#     #             output_dir=out_dir,
#     #             mode=args.mode,
#     #             y_scale=args.y_scale,
#     #             epoch_start=args.epoch_start,
#     #             epoch_end=args.epoch_end,
#     #             plot=False
#     #         )
#     #     except Exception as e:
#     #         print(f"[Error] Failed to evaluate {sample}: {e}")

#     if args.aggregate:
#         stats_df = aggregate_instability_across_samples(
#             sample_dirs=samples,
#             target=args.target,
#             mode=args.mode,
#             y_scale=args.y_scale,
#             epoch_range=(args.epoch_start, args.epoch_end) if args.epoch_start is not None or args.epoch_end is not None else None
#         )
#         plot_aggregate_instability(
#             stats_df=stats_df,
#             xlabel='Epoch' if args.mode == 'alpha' else 'Alpha',
#             ylabel=f"{'Spatial' if args.mode == 'alpha' else 'Temporal'} Instability ({args.y_scale})",
#             save_path=args.plot_save,
#             log_scale_x=(args.mode == 'alpha'),
#             y_lim=None  # 必要なら(0,1)など指定
#         )    # # 各サンプルごとの評価
#     # for s in samples:
#     #     csv_dir = os.path.join(s, 'csv')
#     #     out_dir = os.path.join(s, 'fig_and_log')
#     #     print(f"[Info] Evaluating sample: {s}")
#     #     try:
#     #         evaluate_label_changes(
#     #             pair_csv_dir=csv_dir,
#     #             output_dir=out_dir,
#     #             mode=args.mode,
#     #             y_scale=args.y_scale,
#     #             epoch_start=args.epoch_start,
#     #             epoch_end=args.epoch_end,
#     #             plot=True
#     #         )
#     #     except Exception as e:
#     #         print(f"[Warning] Failed sample {s}: {e}")

#     # # 集約処理
#     # if args.aggregate:
#     #     print("[Info] Aggregating across samples...")
#     #     erange = (args.epoch_start, args.epoch_end) if args.epoch_start is not None or args.epoch_end is not None else None
#     #     stats_df = aggregate_instability_across_samples(
#     #         sample_dirs=samples,
#     #         target=args.target,
#     #         mode=args.mode,
#     #         y_scale=args.y_scale,
#     #         epoch_range=erange
#     #     )
#     #     if stats_df.empty:
#     #         print("[Error] No data to aggregate.")
#     #         sys.exit(1)
#     #     xlabel = 'Epoch' if args.mode == 'alpha' else 'Alpha'
#     #     ylabel = ('Spatial Instability' if args.mode=='alpha' else 'Temporal Instability') + f" ({args.y_scale})"
#     #     plot_aggregate_instability(
#     #         stats_df,
#     #         xlabel=xlabel,
#     #         ylabel=ylabel,
#     #         save_path=args.plot_save,
#     #         highlight=None,
#     #         log_scale_x=(args.mode=='alpha')
#     #     )

# # python viz_colored_cifar_alpha_sample_gattai.py \
# #   alpha_test/cifar10/0.0/64/no_noise/ \
# #   --mode alpha \
# #   --y_scale raw \
# #   --epoch_start 0 \
# #   --epoch_end  4000\
# #   --aggregate \
# #   --target combined \
# #   --plot_save alpha_test/cifar10/0.0/64/no_noise/spatial_instability.png
