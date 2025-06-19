import os
import re
from typing import List, Tuple, Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from smoothing import moving_average
import csv
def get_sample_dirs(base_dir: str) -> List[str]:
    """
    base_dir 以下の 2 階層下で "csv" ディレクトリを含むサンプルディレクトリを返す。
    各サンプルディレクトリは「.../{x}_{y}/csv」を想定。
    """
    sample_dirs = []
    for d1 in os.listdir(base_dir):
        p1 = os.path.join(base_dir, d1)
        if not os.path.isdir(p1):
            continue
        for d2 in os.listdir(p1):
            p2 = os.path.join(p1, d2)
            csv_dir = os.path.join(p2, 'csv')
            if os.path.isdir(p2) and os.path.isdir(csv_dir):
                sample_dirs.append(p2)
    return sample_dirs


def list_epoch_files(
    csv_dir: str,
    start: Optional[int] = None,
    end:   Optional[int] = None
) -> List[Tuple[int, str]]:
    files = []
    for fname in os.listdir(csv_dir):
        m = re.match(r'epoch_(\d+)\.csv$', fname)
        if not m:
            continue
        ep = int(m.group(1))
        if start is not None and ep < start:
            continue
        if end   is not None and ep > end:
            continue
        files.append((ep, os.path.join(csv_dir, fname)))
    files.sort(key=lambda x: x[0])
    return files


def append_cross_entropy_to_csv(
    csv_dir:      str,
    true_label:   int,
    label_part:   str,
    epoch_start:  Optional[int] = None,
    epoch_end:    Optional[int] = None
) -> None:
    files = list_epoch_files(csv_dir, epoch_start, epoch_end)
    if not files:
        print(f"[Warn] No files in {csv_dir}")
        return

    prob_col = f'prob_{true_label}'
    for ep, fp in files:
        df = pd.read_csv(fp)
        if prob_col not in df.columns:
            raise ValueError(f"Missing column '{prob_col}' in {fp}")
        
        probs = df[prob_col].astype(np.float64).to_numpy()
        probs = np.clip(probs, 1e-12, 1.0)  # 安定化のためにクリップ
        col_name = 'clean_cross_entropy' if label_part == 'x' else 'noisy_cross_entropy'
        df[col_name] = -np.log(probs)
        df.to_csv(fp, index=False)
        print(f"[✓] {fp} updated with {col_name}")

def visualize_clean_cross_entropy(
    csv_dir:     str,
    fig_dir:     str,
    row_numbers: List[int] = [52, 77, 103, 127, 153],
    window_size: int = 5
):
    """
    各 epoch CSV から指定行（1-index）の 'clean_cross_entropy' を抽出し、
    移動平均スムージングした値を log–log プロットして保存する。
    """
    epoch_files = list_epoch_files(csv_dir)
    if not epoch_files:
        print(f"[WARN] No epoch files in {csv_dir}")
        return

    sel_idxs = [n - 1 for n in row_numbers]
    epochs, paths = zip(*epoch_files)  # tuple of epoch numbers, tuple of file paths

    # データ格納用
    data = {n: [] for n in row_numbers}
    for fp in paths:
        df = pd.read_csv(fp)
        for orig, idx in zip(row_numbers, sel_idxs):
            try:
                data[orig].append(df.loc[idx, 'clean_cross_entropy'])
            except Exception:
                data[orig].append(np.nan)

    plt.figure(figsize=(8, 5))
    for n in row_numbers:
        y = np.array(data[n], dtype=np.float64)
        y_smooth = moving_average(y, window_size)

        # ← ここで長さを epochs に合わせる
        if len(y_smooth) > len(epochs):
            y_smooth = y_smooth[:len(epochs)]
        elif len(y_smooth) < len(epochs):
            # 欠損部分は nan で埋める
            y_smooth = np.concatenate([
                y_smooth,
                np.full(len(epochs) - len(y_smooth), np.nan)
            ])

        plt.plot(
            epochs,
            y_smooth,
            label=f'row {n} (MA{window_size})'
        )

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('clean_cross_entropy (smoothed)')
    plt.title(f'Clean CE for rows {row_numbers} (MA window={window_size})')
    plt.legend()
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, 'clean_cross_entropy_selected_rows_smoothed.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Saved smoothed log–log plot to {out_path}")# def visualize_clean_cross_entropy(
#     csv_dir:     str,
#     fig_dir:     str,
#     row_numbers: List[int] = [52, 77, 103, 127, 153]
# ):
#     """
#     各 epoch CSV から指定行（1-index）の 'clean_cross_entropy' を抽出し、
#     Epoch を横軸、cross_entropy を縦軸に 1 枚のグラフにプロットして保存する。
#     """
#     epoch_files = list_epoch_files(csv_dir)
#     if not epoch_files:
#         print(f"[WARN] No epoch files in {csv_dir}")
#         return

#     # header 行を含めて数える「n行目」を 0-index に変換
#     sel_idxs = [n - 1 for n in row_numbers]

#     epochs, paths = zip(*epoch_files)
#     data = {n: [] for n in row_numbers}

#     for fp in paths:
#         df = pd.read_csv(fp)
#         for orig, idx in zip(row_numbers, sel_idxs):
#             try:
#                 data[orig].append(df.loc[idx, 'clean_cross_entropy'])
#             except Exception:
#                 # 行数不足や列名ミス時は NaN
#                 data[orig].append(np.nan)

#     # プロット
#     plt.figure(figsize=(8, 5))
#     for n in row_numbers:
#         plt.plot(epochs, data[n], label=f'row {n}')
#     plt.xlabel('Epoch')
#     plt.ylabel('clean_cross_entropy')
#     plt.title('Selected Rows Clean Cross-Entropy over Epochs')
#     plt.xscale('log')  # Set x-scale to log
#     plt.yscale('log')  # Set y-scale to linear
#     plt.legend()
#     os.makedirs(fig_dir, exist_ok=True)
#     out_path = os.path.join(fig_dir, 'clean_cross_entropy_selected_rows.png')
#     plt.savefig(out_path, dpi=300)
#     plt.close()
#     print(f"[OK] Saved plot to {out_path}")

def read_clean_ce(fp: str, row_idx: int, col_name: str = 'clean_cross_entropy') -> float:
    """CSV をストリーム処理し、指定行・列だけを取得。見つからなければ nan。"""
    try:
        with open(fp, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            c_idx = header.index(col_name)
            for i, row in enumerate(reader):
                if i == row_idx:
                    return float(row[c_idx])
    except Exception:
        pass
    return np.nan


def aggregate_clean_cross_entropy(
    base_dir:    str,
    fig_dir:     str,
    row_numbers: List[int],
    max_samples: int = None,
    max_epochs:  int = None,
    window_size: int = 5,
    mode:        str = 'combined'
):
    """
    base_dir 以下のすべての sample_dir から指定行の clean_cross_entropy を読み出し、
    'combined' モードで全 series を一つのグラフに、
    'separate' モードで各行番号ごとに個別グラフを作成して保存する。
    進捗を標準出力に表示します。
    """
    sample_dirs = get_sample_dirs(base_dir)
    if max_samples and max_samples < len(sample_dirs):
        sample_dirs = sample_dirs[:max_samples]
    n_samples = len(sample_dirs)
    if n_samples == 0:
        print(f"[WARN] no samples under {base_dir}")
        return

    # Epoch 一覧取得
    first_csv = os.path.join(sample_dirs[0], 'csv')
    full_epochs = [ep for ep, _ in list_epoch_files(first_csv)]
    if max_epochs and max_epochs < len(full_epochs):
        idxs = np.linspace(0, len(full_epochs)-1, max_epochs, dtype=int)
        epochs = [full_epochs[i] for i in idxs]
    else:
        epochs = full_epochs
    n_epochs = len(epochs)

    print(f"Total samples: {n_samples}, Total epochs: {n_epochs}")

    # データ領域確保
    data = {n: np.full((n_samples, n_epochs), np.nan, dtype=np.float64)
            for n in row_numbers}

    # データ読み込み（進捗表示付き）
    for i, sample in enumerate(sample_dirs, start=1):
        print(f"[Sample {i}/{n_samples}] Processing {sample}")
        csv_dir = os.path.join(sample, 'csv')
        for j, epoch in enumerate(epochs, start=1):
            print(f"  [Epoch {j}/{n_epochs}] epoch_{epoch}.csv", end='\r')
            fp = os.path.join(csv_dir, f'epoch_{epoch}.csv')
            for n in row_numbers:
                data[n][i-1, j-1] = read_clean_ce(fp, n-1)
        print()  # 各 sample ごとの改行

    os.makedirs(fig_dir, exist_ok=True)

    # プロット
    if mode == 'combined':
        plt.figure(figsize=(8,5))
        for n in row_numbers:
            arr = data[n]
            mean_raw = np.nanmean(arr, axis=0)
            std_raw  = np.nanstd(arr, axis=0)
            mean_s = moving_average(mean_raw, window_size)
            std_s  = moving_average(std_raw, window_size)
            if len(mean_s) < n_epochs:
                pad = n_epochs - len(mean_s)
                mean_s = np.concatenate([mean_s, np.full(pad, np.nan)])
                std_s  = np.concatenate([std_s,  np.full(pad, np.nan)])
            plt.plot(epochs, mean_s, label=f'row {n} mean (MA{window_size})')
            plt.fill_between(epochs, mean_s-std_s, mean_s+std_s, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('clean_cross_entropy')
        plt.title(f'Combined Mean±Std over {n_samples} samples')
        plt.legend()
        out_path = os.path.join(fig_dir, 'clean_ce_combined_mean_std.png')
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[OK] saved combined plot to {out_path}")
    else:  # separate
        for n in row_numbers:
            print(f"[Plot] Row {n}")
            arr = data[n]
            mean_raw = np.nanmean(arr, axis=0)
            std_raw  = np.nanstd(arr, axis=0)
            mean_s = moving_average(mean_raw, window_size)
            std_s  = moving_average(std_raw, window_size)
            if len(mean_s) < n_epochs:
                pad = n_epochs - len(mean_s)
                mean_s = np.concatenate([mean_s, np.full(pad, np.nan)])
                std_s  = np.concatenate([std_s,  np.full(pad, np.nan)])
            plt.figure(figsize=(8,5))
            plt.plot(epochs, mean_s, label=f'mean (MA{window_size})')
            plt.fill_between(epochs, mean_s-std_s, mean_s+std_s, alpha=0.3)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Epoch')
            plt.ylim(0, 30)
            plt.ylabel('clean_cross_entropy')
            plt.title(f'Row {n}: Mean±Std over {n_samples} samples')
            plt.legend()
            out_path = os.path.join(fig_dir, f'clean_ce_row{n}_separate_log.png')
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"[OK] saved separate plot to {out_path}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Efficient aggregate clean_cross_entropy with sampling & plot modes'
    )
    parser.add_argument('base_dir', help='ベースディレクトリ')
    parser.add_argument('--rows', type=int, nargs='+',
                        default=[52,77,103,127,153],
                        help='1-index 行番号リスト')
    parser.add_argument('--max-samples', type=int, default=10,
                        help='使用する最大サンプル数（先頭から）')
    parser.add_argument('--max-epochs', type=int, default=2000,
                        help='プロットに使用する最大エポック数（ダウンサンプリング）')
    parser.add_argument('--window-size', type=int, default=1,
                        help='移動平均ウィンドウ幅')
    parser.add_argument('--mode', choices=['combined','separate'], default='separate',
                        help='"combined": 一つの図にまとめる／"separate": 個別プロット')
    args = parser.parse_args()

    fig_dir = os.path.join(args.base_dir, 'fig')
    aggregate_clean_cross_entropy(
        base_dir=args.base_dir,
        fig_dir=fig_dir,
        row_numbers=args.rows,
        max_samples=args.max_samples,
        max_epochs=args.max_epochs,
        window_size=args.window_size,
        mode=args.mode
    )
# if __name__ == '__main__':
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="複数サンプルでCSVにcross_entropy列を追加"
#     )
#     parser.add_argument('base_dir',
#                         help="ベースディレクトリ (サンプルが二階層下に存在)")
#     parser.add_argument('--label-part',
#                         choices=['x', 'y'],
#                         required=True,
#                         help="親ディレクトリ名 '{x}_{y}' のどちらを true label に使うか")
#     parser.add_argument('--epoch-start', type=int, default=None,
#                         help="開始 epoch (inclusive)")
#     parser.add_argument('--epoch-end',   type=int, default=None,
#                         help="終了 epoch (inclusive)")
#     args = parser.parse_args()

#     samples = get_sample_dirs(args.base_dir)
#     if not samples:
#         print(f"[Error] No sample directories under {args.base_dir}")
#         exit(1)

# for sample in samples:
#     # ディレクトリ名解析
#     name = os.path.basename(sample)
#     parts = name.split('_')
#     if len(parts) != 2 or not all(p.isdigit() for p in parts):
#         print(f"[Skip] '{name}' is not in 'int_int' format")
#         continue
#     x_label, y_label = map(int, parts)
#     true_label = x_label if args.label_part == 'x' else y_label

#     csv_dir = os.path.join(sample, 'csv')
#     print(f"[Info] Processing {csv_dir}, true_label={true_label}")
#     try:
#         append_cross_entropy_to_csv(
#             csv_dir=csv_dir,
#             true_label=true_label,
#             label_part = args.label_part,
#             epoch_start=args.epoch_start,
#             epoch_end=args.epoch_end
#         )
#     except Exception as e:
#         print(f"[Error] {csv_dir}: {e}")

#     # テスト実行なので１件処理したらループを抜ける

# #python viz_colored_cifar_alpha_loss.py alpha_test/cifar10/0.2/64/noise --label-part x --epoch-start 0 --epoch-end 4000
