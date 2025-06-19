# main.py
import math
import pandas as pd
from pathlib import Path
import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from itertools import cycle
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import os
import torch
import torch.optim as optim
import warnings
import torch.nn as nn
import numpy as np
import csv
import argparse
from utils import set_seed, set_device, clear_memory
from torch.utils.data import TensorDataset
from datasets import load_datasets, NoisyDataset,load_or_create_noisy_dataset
from models import load_models
from viz_colored_alpha_gif_csv_miseruyou import target_plot_probabilities
from train import (
    compute_class_centroid_and_variance,
    find_closest_data_point_to_centroid,
    alpha_interpolation_test_save,
    select_n2,  # 既存の関数を使用
)
from logger import setup_wandb, setup_alpha_csv_logging, log_alpha_test_results, log_to_wandb, setup_alpha_csv_logging_save, log_alpha_test_results_save, setup_alpha_csv_logging_save2, setup_alpha_csv_logging_save_dir
import matplotlib.pyplot as plt
import wandb
def parse_args_save_clo():
    """
    コマンドライン引数のパース

    Returns:
        argparse.Namespace: パース済み引数
    """
    parser = argparse.ArgumentParser(description="PyTorch Training Script")

    # 固定シード
    parser.add_argument("-fix_seed", "--fix_seed", type=int, default=42,
                        help="再現性のためのランダムシード")
    
    # モデル設定
    parser.add_argument("--model", type=str, choices=[
        "cnn_2layers", "cnn_3layers", "cnn_4layers",
        "cnn_5layers", "cnn_8layers", "cnn_16layers", "resnet18", "cnn_5layers_cus", "resnet18k"
    ], required=True, help="使用するモデルアーキテクチャ")
    parser.add_argument("-model_width", "--model_width", type=int, default=1,
                        help="モデルの幅（マルチプライヤ）")
    parser.add_argument("-epoch", "--epoch", type=int, default=1000,
                        help="トレーニングエポック数")
    
    # データセット設定
    parser.add_argument("-datasets", "--dataset", type=str, choices=[
        "mnist", "emnist", "emnist_digits", "cifar10", "cifar100",
        "tinyImageNet", "colored_emnist", "distribution_colored_emnist"
    ], default="cifar10", help="使用するデータセット")
    parser.add_argument("-variance", "--variance", type=int, default=10000,
                        help="分布データセット用の分散パラメータ")
    parser.add_argument("-correlation", "--correlation", type=float, default=0.5,
                        help="分布データセット用の相関パラメータ")
    parser.add_argument("-label_noise_rate", "--label_noise_rate", type=float, default=0.0,
                        help="ラベルノイズ率")
    parser.add_argument("-gray_scale", "--gray_scale", action='store_true',
                        help="画像をグレースケールに変換")
    parser.add_argument("-batch_size", "--batch_size", type=int, default=128,
                        help="トレーニング時のバッチサイズ")
    parser.add_argument("-img_size", "--img_size", type=int, default=32,
                        help="画像サイズ")
    parser.add_argument("-target", "--target", type=str, choices=["color", "digit", "combined"],
                        default='color', help="colored EMNIST のターゲット")

    # オプティマイザ設定
    parser.add_argument("-lr", "--lr", type=float, default=0.1,
                        help="学習率")
    parser.add_argument("-optimizer", "--optimizer", type=str, choices=["sgd", "adam"],
                        default="adam", help="使用するオプティマイザ")
    parser.add_argument("-momentum", "--momentum", type=float, default=0.9,
                        help="SGD用のモーメンタム")
    
    # 損失関数設定
    parser.add_argument("-loss", "--loss", type=str, choices=["cross_entropy", "focal_loss"],
                        default="cross_entropy", help="損失関数")
    
    # デバイス設定
    parser.add_argument("-gpu", "--gpu", type=int, default=0,
                        help="使用するGPUのデバイスID")
    parser.add_argument("-num_workers", "--num_workers", type=int, default=4,
                        help="データローダのワーカー数")
    
    # WandB設定
    parser.add_argument("-wandb", "--wandb", action='store_true', default=True,
                        help="Weights & Biases を用いたログ")
    parser.add_argument("-wandb_project", "--wandb_project", type=str, default="dd_scratch_models",
                        help="WandBプロジェクト名")
    parser.add_argument("--wandb_entity", type=str, default="dsml-kernel24",
                        help="WandBのエンティティ名")
    
    # 損失の重み付け
    parser.add_argument("-weight_noisy", "--weight_noisy", type=float, default=1.0,
                        help="ノイズサンプルの重み")
    parser.add_argument("-weight_clean", "--weight_clean", type=float, default=1.0,
                        help="クリーンサンプルの重み")
    
    # ペア選択用追加引数
    parser.add_argument("--mode", type=str, choices=["noise", "no_noise"], required=True,
                        help="ペア選択モード")
    parser.add_argument("--distance_metric", type=str, choices=["euclidean", "cosine", "l1", "linf"],
                        default="cosine", help="距離計算のメトリック")
    parser.add_argument("--num_pairs", type=int, default=1,
                        help="補間実験で選択するペア数")

    return parser.parse_args()
def evaluate_label_changes(
    pair_csv_dir,
    output_dir,
    mode='alpha',
    y_lim=None,
    y_scale='ratio',  # 'ratio', 'percent', 'raw'
    plot_result=True,
    epoch_start=None,
    epoch_end=None
):
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    if mode not in ['alpha', 'epoch']:
        raise ValueError("Invalid mode. Choose 'alpha' or 'epoch'.")
    if y_scale not in ['ratio', 'percent', 'raw']:
        raise ValueError("Invalid y_scale. Choose 'ratio', 'percent', or 'raw'.")

    csv_dir = pair_csv_dir
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)

    all_files = sorted([
        f for f in os.listdir(csv_dir)
        if f.startswith("epoch_") and f.endswith(".csv")
    ])

    if not all_files:
        raise ValueError("No valid CSV files found in the directory.")

    files = []
    for f in all_files:
        try:
            epoch = int(f.split('_')[1].split('.')[0])
            if epoch_start is not None and epoch < epoch_start:
                continue
            if epoch_end is not None and epoch > epoch_end:
                continue
            files.append((epoch, os.path.join(csv_dir, f)))
        except ValueError:
            print(f"[!] Skipping invalid file: {f}")

    if not files:
        raise ValueError("No valid files in specified epoch range.")

    epoch_suffix = ""
    if epoch_start is not None or epoch_end is not None:
        start_str = str(epoch_start) if epoch_start is not None else "start"
        end_str = str(epoch_end) if epoch_end is not None else "end"
        epoch_suffix = f"_epoch_{start_str}_to_{end_str}"

    scores = {}

    if mode == 'alpha':
        for epoch, filepath in files:
            df = pd.read_csv(filepath).iloc[:202]
            n = len(df) - 1
            if n <= 0:
                print(f"[!] Skipping epoch {epoch} — insufficient rows.")
                continue

            if df['predicted_label'].nunique() <= 1:
                print(f"[!] Epoch {epoch} — predicted_label has no variation. Set label_change=0.")
                changes = 0.0
            else:
                change_count = (df['predicted_label'] != df['predicted_label'].shift()).sum()
                changes = change_count - 1  # 生の変化回数（raw）

                if y_scale == 'ratio':
                    changes /= n
                elif y_scale == 'percent':
                    changes = (changes / n) * 100

            scores[epoch] = float(changes)

        score_df = pd.DataFrame(list(scores.items()), columns=['epoch', 'label_change'])
        score_df = score_df.sort_values('epoch')
        csv_filename = os.path.join(save_dir, f"label_change_scores_alpha{epoch_suffix}.csv")
        score_df.to_csv(csv_filename, index=False)
        print(f"[✓] Alpha mode scores saved: {csv_filename}")

        if plot_result:
            epochs = score_df['epoch'].values
            label_changes = score_df['label_change'].values
            std_scores = np.zeros_like(label_changes)

            fig, ax = plt.subplots(figsize=(12, 8), dpi=400)
            ax.plot(epochs, label_changes, label="Spatial Instability", color="blue", linewidth=1.5, zorder=3)
            ax.fill_between(epochs, label_changes - std_scores, label_changes + std_scores,
                            color="blue", alpha=0.2, zorder=2)
            ax.set_xlabel("Epoch")
            if y_scale == 'percent':
                ax.set_ylabel("Spatial Instability (%)")
            elif y_scale == 'raw':
                ax.set_ylabel("Spatial Instability (Count)")
            else:
                ax.set_ylabel("Spatial Instability (Ratio)")
            if y_lim:
                ax.set_ylim(y_lim)
            ax.set_xscale('log')

            plot_path = os.path.join(save_dir, f"label_change_scores_alpha{epoch_suffix}_plot.svg")
            fig.savefig(plot_path, bbox_inches='tight')
            plt.close(fig)
            print(f"[✓] Alpha plot saved: {plot_path}")

    elif mode == 'epoch':
        if len(files) < 2:
            raise ValueError("At least two files required for 'epoch' mode.")

        files = sorted(files)
        dfs = [pd.read_csv(f[1]).iloc[:202] for f in files]
        alpha_values = dfs[0]['alpha'].tolist()
        unsmoothed_scores = [0] * len(alpha_values)

        for i in range(len(dfs) - 1):
            cur = dfs[i]['predicted_label']
            nxt = dfs[i + 1]['predicted_label']
            row_changes = (cur != nxt).astype(int)
            unsmoothed_scores = [u + r for u, r in zip(unsmoothed_scores, row_changes)]

        if y_scale == 'ratio':
            total_epochs = len(dfs) - 1
            unsmoothed_scores = [s / total_epochs for s in unsmoothed_scores]
        elif y_scale == 'percent':
            total_epochs = len(dfs) - 1
            unsmoothed_scores = [(s / total_epochs) * 100 for s in unsmoothed_scores]

        epoch_csv_path = os.path.join(save_dir, f"epoch_unsmoothed_scores{epoch_suffix}.csv")
        pd.DataFrame({'alpha': alpha_values, 'unsmoothed_scores': unsmoothed_scores}).to_csv(epoch_csv_path, index=False)
        print(f"[✓] Epoch mode scores saved: {epoch_csv_path}")

        if plot_result:
            fig, ax = plt.subplots(figsize=(12, 8), dpi=400)
            ax.plot(alpha_values, unsmoothed_scores, label="Temporal Instability", color="blue", linewidth=1.5, zorder=3)
            ax.fill_between(alpha_values, np.array(unsmoothed_scores), np.array(unsmoothed_scores),
                            color="blue", alpha=0.2, zorder=2)
            ax.set_xlabel("Alpha")
            if y_scale == 'percent':
                ax.set_ylabel("Temporal Instability (%)")
            elif y_scale == 'raw':
                ax.set_ylabel("Temporal Instability (Count)")
            else:
                ax.set_ylabel("Temporal Instability (Ratio)")
            if y_lim:
                ax.set_ylim(y_lim)

            plot_path = os.path.join(save_dir, f"epoch_unsmoothed_scores{epoch_suffix}_plot.svg")
            fig.savefig(plot_path, bbox_inches='tight')
            plt.close(fig)
            print(f"[✓] Epoch plot saved: {plot_path}")

    return scores

def get_highlight_labels_from_path(data_dir):
    """
    data_dir: 例 → alpha_test/cifar10/0.2/64/noise/pair0001/7_3/csv
    """
    label_dir = os.path.basename(os.path.dirname(data_dir))  # "7_3"
    label1, label2 = map(int, label_dir.split("_"))
    return label1, label2

def generate_alpha_probabilities_gif(data_dir, output_path, targets='combined', epoch_stride=1,
                                     start_epoch=1, end_epoch=300):
    """
    指定した data_dir 内の epoch_*.csv を読み込み、alpha と予測確率の推移を GIF 保存する。
    ハイライトはディレクトリ名にあるラベル2つ（青・赤）に基づく。
    
    Parameters:
    - data_dir: CSVが入っているディレクトリ
    - output_path: 出力するGIFのパス
    - targets: ラベル指定（未使用）
    - epoch_stride: 何エポックごとに描画するか（例：5なら5エポックごと）
    - start_epoch: 開始エポック（Noneなら最初のエポック）
    - end_epoch: 終了エポック（Noneなら最後のエポック）
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "epoch_*.csv")))
    if not csv_files:
        print(f"[!] No CSV files in {data_dir}")
        return

    data = {}
    for f in csv_files:
        epoch = int(os.path.basename(f).split("_")[1].split(".")[0])
        data[epoch] = pd.read_csv(f)

    all_epochs = sorted(data.keys())

    # 指定された範囲でエポックをフィルター
    filtered_epochs = [e for e in all_epochs
                       if (start_epoch is None or e >= start_epoch) and
                          (end_epoch is None or e <= end_epoch)]

    epochs = filtered_epochs[::epoch_stride]
    if not epochs:
        print("[!] No epochs match the given range and stride.")
        return

    alpha_values = data[epochs[0]]['alpha']
    label1, label2 = get_highlight_labels_from_path(data_dir)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    other_colors = cycle(['gray', 'purple', 'green', 'orange', 'cyan', 'brown'])
    lines = []

    df_first = data[epochs[0]]

    for t in range(100):
        col = f'prob_{t}'
        if col in df_first.columns:
            if t == label1:
                color = 'blue'
                lw = 2.5
                alpha = 1.0
            elif t == label2:
                color = 'red'
                lw = 2.5
                alpha = 1.0
            else:
                color = next(other_colors)
                lw = 0.8
                alpha = 0.5
            line, = ax.plot(alpha_values, df_first[col], color=color, linewidth=lw, alpha=alpha)
            lines.append((t, line))

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Probability')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(-0.5, 1.6, 0.5))
    ax.set_title(f"Alpha Interpolation (Epoch {epochs[0]})")

    def update(epoch):
        ax.set_title(f"Alpha Interpolation (Epoch {epoch})")
        df = data[epoch]
        for t, line in lines:
            col = f'prob_{t}'
            if col in df.columns:
                line.set_ydata(df[col])
        return [line for _, line in lines]

    anim = FuncAnimation(fig, update, frames=epochs, interval=200, blit=True)
    anim.save(output_path, writer=PillowWriter(fps=8))
    plt.close()
    print(f"[✓] Saved alpha probability GIF to {output_path}")

def get_noise_info(dataset):
    """
    TensorDatasetなどにラップされている場合からnoise_infoを推定する。

    条件:
    - (image, label, noise_flag) の形式で __getitem__ が返す場合 → noise_flag を収集して返す
    - (image, label) のみしか返さない → すべて0（clean）と仮定

    Parameters:
        dataset (torch.utils.data.Dataset): PyTorch Datasetオブジェクト

    Returns:
        torch.Tensor: noise_info（0: clean, 1: noisy）
    """
    import torch
    from torch.utils.data import Subset

    try:
        first_item = dataset[0]
        if isinstance(first_item, tuple) and len(first_item) == 3:
            noise_info = []
            for i in range(len(dataset)):
                _, _, noise_flag = dataset[i]
                noise_info.append(int(noise_flag))
            return torch.tensor(noise_info, dtype=torch.long)
        else:
            print("[!] Dataset does not contain noise flags — assuming all clean.")
            return torch.zeros(len(dataset), dtype=torch.long)
    except Exception as e:
        print(f"[!] Error reading noise info: {e}")
        return torch.zeros(len(dataset), dtype=torch.long)

def unwrap_to_tensor_dataset(ds):
    """
    TensorDataset になるまで再帰的に unwrap する
    """
    visited = set()
    while True:
        if isinstance(ds, TensorDataset):
            return ds
        elif hasattr(ds, "dataset"):
            ds = ds.dataset
        elif hasattr(ds, "base_dataset"):
            ds = ds.base_dataset
        else:
            raise TypeError(f"Unwrapping failed: {type(ds)} has no dataset/base_dataset attribute")
        if id(ds) in visited:
            raise RuntimeError("Circular reference detected while unwrapping dataset")
        visited.add(id(ds))

def get_tensor_dataset_components(dataset):
    ds = unwrap_to_tensor_dataset(dataset)
    return ds.tensors


def alpha_interpolation_test_save_combined_only(model, x_clean, x_noisy, label_x, label_y, num_classes, device):
    """
    線形補間を行い、補間結果に対するモデルの予測結果を取得する関数。
    α ∈ [-0.5, 1.5] で clean ↔ noisy を補間し、分類結果・確率を返す。
    """
    model = model.to(device)
    model.eval()

    alpha_values = torch.arange(-0.5, 1.51, 0.01, device=device)
    N = alpha_values.shape[0]

    x_clean = x_clean.to(device).unsqueeze(0)
    x_noisy = x_noisy.to(device).unsqueeze(0)

    x_clean_batch = x_clean.expand(N, *x_clean.shape[1:])
    x_noisy_batch = x_noisy.expand(N, *x_noisy.shape[1:])
    alphas = alpha_values.view(-1, 1, 1, 1)

    z = alphas * x_clean_batch + (1 - alphas) * x_noisy_batch

    with torch.no_grad():
        outputs = model(z)
        output_probs = torch.nn.functional.softmax(outputs, dim=1)
        predicted_labels = torch.argmax(output_probs, dim=1)

        label_matches = torch.where(
            predicted_labels == label_x,
            torch.tensor(1, device=device),
            torch.tensor(0, device=device)
        )
        label_matches = torch.where(
            predicted_labels == label_y,
            torch.tensor(-1, device=device),
            label_matches
        )

    return {
        'alpha_values': alpha_values.cpu().numpy().tolist(),
        'predicted_labels': predicted_labels.cpu().numpy().tolist(),
        'raw_probabilities': output_probs.cpu().numpy().tolist(),
        'label_matches': label_matches.cpu().numpy().tolist()
    }
def find_random_n_pairs_by_mode(x_train, noise_info, y_original_train, n,
                                 distance_metric='euclidean', mode='noise'):
    """
    ランダムに n 個のクエリを選び、それぞれに対して同じラベルの候補から最近傍を1つ探す。

    Parameters:
        x_train (Tensor): 学習サンプル画像 [N, C, H, W]
        noise_info (Tensor): ノイズフラグ [N]（0: clean, 1: noisy）
        y_original_train (Tensor): ノイズなしの正解ラベル [N]
        n (int): 選択するペア数
        distance_metric (str): 'euclidean', 'cosine', 'l1', 'linf'
        mode (str): 'noise' or 'no_noise'

    Returns:
        List[Dict]: 各ペアの dict（query_index, matched_index, label, distance）
    """
    import random

    device = x_train.device
    N = x_train.size(0)
    x_flat = x_train.view(N, -1)

    if mode == 'noise':
        query_indices_all = (noise_info == 1).nonzero(as_tuple=True)[0]
        candidate_indices_all = (noise_info == 0).nonzero(as_tuple=True)[0]
    elif mode == 'no_noise':
        query_indices_all = (noise_info == 0).nonzero(as_tuple=True)[0]
        candidate_indices_all = query_indices_all
    else:
        raise ValueError(f"Unknown mode: {mode}")

    query_indices_selected = random.sample(query_indices_all.tolist(), min(n, len(query_indices_all)))
    pairs = []

    for query_idx in query_indices_selected:
        label = y_original_train[query_idx].item()
        same_label_candidates = candidate_indices_all[y_original_train[candidate_indices_all] == label]

        if mode == 'no_noise':
            same_label_candidates = same_label_candidates[same_label_candidates != query_idx]

        if len(same_label_candidates) == 0:
            continue

        x_query = x_flat[query_idx].unsqueeze(0)
        x_candidates = x_flat[same_label_candidates]

        if distance_metric == 'euclidean':
            dists = torch.norm(x_query - x_candidates, dim=1)
        elif distance_metric == 'cosine':
            x_q = x_query / (x_query.norm(dim=1, keepdim=True) + 1e-8)
            x_c = x_candidates / (x_candidates.norm(dim=1, keepdim=True) + 1e-8)
            dists = 1 - torch.mm(x_q, x_c.T).squeeze()
        elif distance_metric == 'l1':
            dists = torch.sum(torch.abs(x_query - x_candidates), dim=1)
        elif distance_metric == 'linf':
            dists = torch.max(torch.abs(x_query - x_candidates), dim=1)[0]
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        min_idx = torch.argmin(dists)
        matched_idx = same_label_candidates[min_idx].item()
        distance = dists[min_idx].item()

        pairs.append({
            "query_index": query_idx,
            "matched_index": matched_idx,
            "label": label,
            "distance": distance
        })

    return pairs
def load_saved_pairs_by_mode(args, base_dir="alpha_test"):
    """
    保存された *_selected_pairs.csv を読み込み、
    find_random_n_pairs_by_mode() と同形式のペアリストを返す。
    
    Returns:
        pairs: list of dict with keys ['query_index', 'matched_index', 'label', 'distance']
    """
    save_dir = Path(base_dir) / args.dataset / str(args.label_noise_rate)
    csv_path = save_dir / f"{args.mode}_selected_pairs.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"[!] Saved pair file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    pairs = []
    for _, row in df.iterrows():
        pair = {
            "query_index": int(row["query_index"]),
            "matched_index": int(row["matched_index"]),
            "label": int(row["label"]),
            "distance": float(row["distance"])
        }
        pairs.append(pair)

    print(f"[✓] Loaded {len(pairs)} pairs from {csv_path}")
    return pairs
def save_pair_log(pairs, args):
    """
    選択したペア情報をCSVとして保存する。

    保存先:
        alpha_test/{dataset}/{label_noise_rate}/{mode}_selected_pairs.csv

    各行には：
        - query_index
        - matched_index
        - label（オリジナルのラベル）
        - distance（距離尺度でのスコア）

    Parameters:
        pairs (list of dict): find_random_n_pairs_by_modeで得たペアリスト
        args (Namespace): 実験引数（dataset, label_noise_rate, mode を使用）
    """
    import pandas as pd
    from pathlib import Path

    save_dir = Path("alpha_test") / args.dataset / str(args.label_noise_rate)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{args.mode}_selected_pairs.csv"

    rows = []
    for pair in pairs:
        rows.append({
            "query_index": pair["query_index"],
            "matched_index": pair["matched_index"],
            "label": pair["label"],
            "distance": pair["distance"]
        })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"[✓] Pair log saved to: {save_path}")
def get_pair_save_dir(base_dir, args, pair_id, matched_label, query_label):
    pair_str = f"pair{pair_id}"
    sub_dir = f"{matched_label}_{query_label}"
    csv_dir = Path(base_dir) / args.dataset / str(args.label_noise_rate) / str(args.model_width) / args.mode / pair_str / sub_dir / "csv"
    fig_and_log_dir = csv_dir.parent / "fig_and_log"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_and_log_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir, fig_and_log_dir

def save_tensor_image(tensor, save_path, dataset, clean_label, noisy_label):
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    if dataset.lower() == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        inv_transform = transforms.Normalize(
            mean=[-m/s for m, s in zip(mean, std)],
            std=[1/s for s in std]
        )
        tensor = inv_transform(tensor)

    np_img = tensor.detach().cpu().numpy()
    if np_img.shape[0] == 3:
        np_img = np.transpose(np_img, (1, 2, 0))

    plt.figure(figsize=(2, 2), dpi=150)
    plt.imshow(np.clip(np_img, 0, 1))
    plt.title(f"Clean: {clean_label} / Noisy: {noisy_label}")
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main(): 
    print('Start session')
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.benchmark = True
    try:
        args = parse_args_save_clo()
        set_seed(args.fix_seed)
        device = set_device(args.gpu)
        print(f'Using device: {device}')
        print('Loading datasets...')
        try:
            train_dataset_original, test_dataset, imagesize, num_classes, in_channels = load_datasets(
                args.dataset, args.target, args.gray_scale, args
            )
        except FileNotFoundError as e:
            print(f"Error loading dataset: {e}")
            return

        if not hasattr(train_dataset_original, 'tensors'):
            from torch.utils.data import TensorDataset
            if hasattr(train_dataset_original, 'data') and hasattr(train_dataset_original, 'targets'):
                x_data = torch.tensor(train_dataset_original.data)
                if len(x_data.shape) == 4:
                    x_data = x_data.permute(0, 3, 1, 2)
                y_data = torch.tensor(train_dataset_original.targets)
                train_dataset_original = TensorDataset(x_data, y_data)
            else:
                raise ValueError("Dataset cannot be converted to TensorDataset.")

        x_original_train, y_original_train = train_dataset_original.tensors
        experiment_name = f'seed_{args.fix_seed}width{args.model_width}_{args.model}_{args.dataset}_variance{args.variance}_{args.target}_lr{args.lr}_batch{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}_Optim{args.optimizer}_Momentum{args.momentum}'

        print('Creating noisy datasets...')
        train_dataset, test_dataset, meta = load_or_create_noisy_dataset(
            args.dataset, args.target, args.gray_scale, args, return_type="torch"
        )
        imagesize = meta["imagesize"]
        num_classes = meta["num_classes"]
        in_channels = meta["in_channels"]

        if isinstance(train_dataset, NoisyDataset):
            print("NoisyDataset detected.")
            x_train_noisy, y_train_noisy = get_tensor_dataset_components(train_dataset)
            noise_info = train_dataset.noise_info
        else:
            print("TensorDataset or other detected.")
            x_train_noisy, y_train_noisy = get_tensor_dataset_components(train_dataset)
            noise_info = get_noise_info(train_dataset)

        print('Selecting sample pairs...')
        n_pairs = 100
        # pairs = find_random_n_pairs_by_mode(
        #     x_train_noisy, noise_info, y_original_train,
        #     n=n_pairs,
        #     distance_metric=args.distance_metric,
        #     mode=args.mode
        # )
       
        pairs = load_saved_pairs_by_mode(args, base_dir="alpha_test")

        # save_pair_log(pairs, args)
        print(f"{len(pairs)} pairs selected.")
        model = load_models(in_channels, args, imagesize, num_classes).to(device)
        base_save_dir = os.path.join("save_model", args.dataset, f"noise_{args.label_noise_rate}", experiment_name)

        for i, pair in enumerate(pairs):
            query_idx = pair["query_index"]
            matched_idx = pair["matched_index"]
            label = pair["label"]
            distance = pair["distance"]
            print(f"\n[{i+1}/{len(pairs)}] Pair: query={query_idx}, matched={matched_idx}, label={label}, dist={distance:.4f}")

            x_query = x_train_noisy[query_idx]
            x_matched = x_train_noisy[matched_idx]
            y_query = y_original_train[query_idx]
            y_matched = y_original_train[matched_idx]
            csv_dir, fig_and_log_dir = get_pair_save_dir(
                base_dir="alpha_test",
                args=args,
                pair_id=i + 1,
                matched_label=y_train_noisy[matched_idx].item(),
                query_label=y_train_noisy[query_idx].item()
            )

            save_tensor_image(x_query, fig_and_log_dir / "query.png", dataset=args.dataset, clean_label=y_query.item(), noisy_label=y_train_noisy[query_idx].item())
            save_tensor_image(x_matched, fig_and_log_dir / "matched.png", dataset=args.dataset, clean_label=y_matched.item(), noisy_label=y_train_noisy[matched_idx].item())

            for epoch in range(args.epoch + 2):
                model_path = os.path.join(base_save_dir, f"model_epoch_{epoch}.pth")
                if not os.path.exists(model_path):
                    print(model_path)
                    print(f"  [!] Skip epoch {epoch} (no model found)")
                    continue

                checkpoint = torch.load(model_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                model.eval()

                result = alpha_interpolation_test_save_combined_only(
                    model=model,
                    x_clean=x_query,
                    x_noisy=x_matched,
                    label_x=y_query,
                    label_y=y_matched,
                    num_classes=num_classes,
                    device=device
                )

                df = pd.DataFrame({
                    'alpha': result['alpha_values'],
                    'predicted_label': result['predicted_labels'],
                    'label_match': result['label_matches'],
                })
                probs = pd.DataFrame(result['raw_probabilities'], columns=[f'prob_{i}' for i in range(num_classes)])
                df = pd.concat([df, probs], axis=1)

                csv_path = csv_dir / f"epoch_{epoch}.csv"
                df.to_csv(csv_path, index=False)

            gif_path = fig_and_log_dir / "alpha_plot.gif"
            generate_alpha_probabilities_gif(data_dir=csv_dir, output_path=str(gif_path))

            # 評価の実行（raw指定）
            evaluate_label_changes(pair_csv_dir=str(csv_dir), output_dir=str(fig_and_log_dir), mode='alpha', y_lim=None, y_scale='raw', plot_result=True)
            evaluate_label_changes(pair_csv_dir=str(csv_dir), output_dir=str(fig_and_log_dir), mode='epoch', y_lim=None, y_scale='raw', plot_result=True)

    except Exception as e:
        print(f"Error during processing: {e}")
if __name__ == "__main__":
    main()
