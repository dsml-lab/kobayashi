#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
from PIL import Image
import random
import pandas as pd
import contextlib

# models.py から load_models をインポート
from models import load_models

# -------------------------------------------------------
# 1. 引数のパース
# -------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate label-noise region in input space.")
    parser.add_argument('--folder', type=str, required=True,
                        help='Directory containing epoch_{n}.pth model files.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing EMNIST_{label_noise}/train_images.npy etc.')
    parser.add_argument('--label_noise', type=str, required=True,
                        help='Label noise rate (string to match folder name, e.g. "0.1").')
    parser.add_argument('--epochs', type=str, default=None,
                        help='Comma-separated list of epochs to evaluate (e.g. "100,500,999").')
    parser.add_argument('--auto', action='store_true',
                        help='If set, evaluate every 10 epochs automatically (e.g. 10,20,30,...).')
    parser.add_argument('--nsample', type=int, default=100,
                        help='Number of samples for epsilon-ball sampling around each data point.')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Epsilon for L-infinity norm sampling.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use.')
    parser.add_argument('--sample_type', type=str, default='noisy', choices=['noisy', 'clean', 'both'],
                        help='Sample type to evaluate: "noisy", "clean", or "both".')
    parser.add_argument('--npoints', type=int, default=5,
                        help='Number of data points to evaluate per sample type.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model architecture name (e.g. "cnn_5layers").')
    parser.add_argument('--model_width', type=int, default=1,
                        help='Width parameter for the CNN model.')
    parser.add_argument('--seed', type=int, default=43,
                        help='Random seed for reproducibility.')
    parser.add_argument('--compute_volume', action='store_true',
                        help='If set, compute volume along with sample rate.')
    return parser.parse_args()


# -------------------------------------------------------
# 2. データ読み込み
# -------------------------------------------------------
def load_data(args):
    """
    EMNIST_{label_noise}/train_images.npy
                   /train_labels.npy
                   /train_noise_info.npy
    これらを読み込む。
    画像は (N, 32, 32) で [0,1] スケール。
    """
    data_folder = os.path.join(args.data_dir, f"EMNIST_{args.label_noise}")
    train_images_path = os.path.join(data_folder, "train_images.npy")
    train_labels_path = os.path.join(data_folder, "train_labels.npy")
    train_noise_info_path = os.path.join(data_folder, "train_noise_info.npy")

    if not (os.path.exists(train_images_path) and os.path.exists(train_labels_path) and os.path.exists(train_noise_info_path)):
        raise FileNotFoundError(f"Required data files not found in {data_folder}.")

    x_train = np.load(train_images_path)  # shape: (N,32,32)
    y_train = np.load(train_labels_path)  # shape: (N,)
    noise_info = np.load(train_noise_info_path)  # shape: (N,)

    # チャンネル次元を削除（もし存在すれば）
    if x_train.ndim == 4 and x_train.shape[1] == 1:
        x_train = np.squeeze(x_train, axis=1)  # shape: (N,32,32)
        print(f"Squeezed x_train shape: {x_train.shape}")
    else:
        print(f"x_train shape: {x_train.shape}")

    return x_train, y_train, noise_info


# -------------------------------------------------------
# 3. エポックの取得 (手動 or 自動)
# -------------------------------------------------------
def get_target_epochs(args):
    # フォルダ内の "epoch_{n}.pth" を探す
    pattern = re.compile(r"epoch_(\d+)\.pth$")
    files = os.listdir(args.folder)
    all_epochs = []
    for f in files:
        m = pattern.match(f)
        if m:
            all_epochs.append(int(m.group(1)))
    all_epochs.sort()

    # --auto が指定されていれば 10epoch ごと
    if args.auto:
        target_epochs = [ep for ep in all_epochs if ep % 10 == 0]
    else:
        # 手動指定の場合
        if args.epochs is not None:
            # "100,500,999" をパース
            specified = [int(e.strip()) for e in args.epochs.split(',')]
            target_epochs = [ep for ep in all_epochs if ep in specified]
        else:
            # neither auto nor epochs specified → fallback to last epoch or an empty list
            if len(all_epochs) > 0:
                target_epochs = [all_epochs[-1]]  # デフォルトで最終epochだけ評価
            else:
                target_epochs = []
    return target_epochs


# -------------------------------------------------------
# 4. サンプルの選定
# -------------------------------------------------------
def select_data_points(x_train, y_train, noise_info, sample_type, npoints):
    """
    sample_type: 'noisy' → noise_info == 1 だけ
                 'clean' → noise_info == 0 だけ
                 'both'  → noisy と clean をそれぞれ半分ずつ抽出
    """
    n_total = len(x_train)
    indices_noisy = np.where(noise_info == 1)[0]
    indices_clean = np.where(noise_info == 0)[0]

    if sample_type == 'noisy':
        if len(indices_noisy) == 0:
            raise ValueError(f"No noisy samples found for sample_type={sample_type}.")
        indices_chosen = indices_noisy
    elif sample_type == 'clean':
        if len(indices_clean) == 0:
            raise ValueError(f"No clean samples found for sample_type={sample_type}.")
        indices_chosen = indices_clean
    else:
        # both
        half_1 = int(np.ceil(npoints / 2))
        half_2 = npoints - half_1
        if len(indices_noisy) < half_1 or len(indices_clean) < half_2:
            raise ValueError("Not enough samples for 'both' sample_type.")
        # ノイズ付きから half_1 個, ノイズなしから half_2 個
        idx_noisy_chosen = np.random.choice(indices_noisy, size=half_1, replace=False)
        idx_clean_chosen = np.random.choice(indices_clean, size=half_2, replace=False)
        return np.concatenate([idx_noisy_chosen, idx_clean_chosen])

    # npoints 個だけランダムに抽出
    npoints_eff = min(len(indices_chosen), npoints)
    selected = np.random.choice(indices_chosen, size=npoints_eff, replace=False)
    return selected


# -------------------------------------------------------
# 5. 画像を可視化して保存
# -------------------------------------------------------
def save_image_sample(img_array, save_path):
    """
    img_array: shape (32,32), [0,1]
    """
    if img_array.ndim == 3:
        # グレースケールの場合、squeezeして (32,32) にする
        img_array = img_array.squeeze()
    pil_img = Image.fromarray((img_array * 255).astype(np.uint8))
    #pil_img = pil_img.convert("L")  # グレースケール
    pil_img.save(save_path)


# -------------------------------------------------------
# 6. モデル読み込み (PyTorch想定)
# -------------------------------------------------------
def load_model(args, folder, epoch, device='cuda'):
    model_path = os.path.join(folder, f"epoch_{epoch}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")

    # モデルのインスタンスを作成
    # args.model と args.model_width を渡す
    model = load_models(in_channels=1, args=args, img_size=(32,32), num_classes=10)
    model.to(device)

    # state_dict をロードしてモデルに適用
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()  # 推論モードに設定
    return model


# -------------------------------------------------------
# 7. L^∞ノルムでのサンプリング
# -------------------------------------------------------
def sample_around_x(x, epsilon, nsample):
    """
    x: shape (32,32), [0,1]
    epsilon: float
    nsample: int
    return: np.array shape (nsample, 1, 32, 32)
    """
    samples = []
    for _ in range(nsample):
        # 一様乱数で摂動を生成
        perturb = np.random.uniform(low=-epsilon, high=epsilon, size=(32, 32))
        x_pert = x + perturb
        # ピクセル値を [0,1] にクリップ
        x_pert = np.clip(x_pert, 0.0, 1.0)
        samples.append(x_pert)
    samples = np.stack(samples, axis=0)  # (nsample, 32, 32)
    samples = samples[:, np.newaxis, :, :]  # (nsample,1,32,32)
    return samples


# -------------------------------------------------------
# 8. 推論して "ノイズ後ラベル" と一致する割合を計算
# -------------------------------------------------------
def compute_ratio_and_volume(model, x_base, y_label, epsilon, nsample, device='cuda', compute_volume=False):
    """
    x_base: shape (32,32), in [0,1]
    y_label: int
    - sample_around_x で近傍を作り推論
    - ratio = (#(pred == y_label)) / nsample
    - volume = (2 epsilon)^d * ratio, d=1024 (32*32) [オプション]
    """
    d = 32 * 32  # 次元数

    # 近傍サンプリング
    x_samples = sample_around_x(x_base, epsilon, nsample)  # shape (nsample,1,32,32)

    # モデル推論
    x_torch = torch.from_numpy(x_samples).float().to(device)

    # 入力テンソルの形状を確認・修正
    if x_torch.dim() == 5:
        # 余分な次元がある場合は削除
        x_torch = x_torch.squeeze(2)  # [N,1,1,32,32] -> [N,1,32,32]
        print(f"Squeezed x_torch shape: {x_torch.shape}")
    elif x_torch.dim() == 4:
        print(f"x_torch shape: {x_torch.shape}")
    else:
        raise ValueError(f"Unexpected x_torch dimensions: {x_torch.dim()}")

    with torch.no_grad():
        logits = model(x_torch)  # shape (nsample, num_classes)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    count = np.sum(preds == y_label)
    ratio = count / nsample

    if compute_volume:
        try:
            # 体積計算（対数変換を使用）
            log_volume = d * np.log(2.0 * epsilon) + np.log(ratio) if ratio > 0 else float('-inf')
            volume = np.exp(log_volume) if log_volume > float('-inf') else 0.0
        except OverflowError:
            volume = float('inf')  # オーバーフロー時は無限大とする
    else:
        volume = None  # 体積を計算しない場合

    return ratio, volume


# -------------------------------------------------------
# 9. グラフ化して保存
# -------------------------------------------------------
def plot_csv(csv_path, png_path, ylabel="Value"):
    """
    CSV: 1行目: epoch, idx_...
         2行目以降: epoch, val1, val2,...
    グラフ化して png_path に保存
    """
    df = pd.read_csv(csv_path)
    # epoch 列
    epochs = df['epoch'].values

    # idx_... 列をまとめてplot
    cols = [c for c in df.columns if c.startswith("idx_")]
    plt.figure(figsize=(10,6))
    for col in cols:
        plt.plot(epochs, df[col], label=col)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()


# -------------------------------------------------------
# 10. メイン評価ロジック
# -------------------------------------------------------
def evaluate_model(args):
    # シードの設定
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True  # 必要に応じて
    # torch.backends.cudnn.benchmark = False     # 必要に応じて

    # GPU 設定
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU: {args.gpu}")
    else:
        print("CUDA not available. Using CPU.")

    # データ読み込み
    x_train, y_train, noise_info = load_data(args)

    # 対象エポックの決定
    target_epochs = get_target_epochs(args)
    if len(target_epochs) == 0:
        print("No epochs to evaluate. Exiting.")
        return

    # 出力ディレクトリ
    experiment_name = f"model_{args.model}_npoint{args.npoints}_epsilon{args.epsilon}_{args.nsample}_seed{args.seed}"
    out_dir = os.path.join("region",experiment_name, f"EMNIST{args.label_noise}", args.sample_type)

    os.makedirs(out_dir, exist_ok=True)

    # サンプルの選定
    indices_selected = select_data_points(x_train, y_train, noise_info, args.sample_type, args.npoints)
    print(f"Selected data indices: {indices_selected}")

    # 選んだデータ点を可視化して保存
    for idx in indices_selected:
        img_path = os.path.join(out_dir, f"select_point_{idx}.png")
        save_image_sample(x_train[idx], img_path)
    print(f"Saved selected sample images to {out_dir}")

    # 結果保存用の CSV 下準備 (volume.csv と sample_rate.csv)
    if args.compute_volume:
        volume_csv_path = os.path.join(out_dir, "volume.csv")
    sample_rate_csv_path = os.path.join(out_dir, "sample_rate.csv")

    # `ExitStack` を使用して条件付きでファイルを開く
    with contextlib.ExitStack() as stack:
        # sample_rate.csv を開く
        fsr = stack.enter_context(open(sample_rate_csv_path, 'w', newline=''))
        sr_writer = csv.writer(fsr)
        sr_writer.writerow(["epoch"] + [f"idx_{idx}" for idx in indices_selected])

        # compute_volume が True の場合のみ volume.csv を開く
        if args.compute_volume:
            fvol = stack.enter_context(open(volume_csv_path, 'w', newline=''))
            vol_writer = csv.writer(fvol)
            vol_writer.writerow(["epoch"] + [f"idx_{idx}" for idx in indices_selected])
        else:
            vol_writer = None

        # 各 epoch で評価
        for ep in target_epochs:
            print(f"Evaluating epoch {ep} ...")
            try:
                model = load_model(args, args.folder, ep, device=device)
            except FileNotFoundError as e:
                print(e)
                continue  # 次の epoch に進む

            sample_rate_list = []
            volume_list = [] if args.compute_volume else None

            for idx in indices_selected:
                # サンプル x & ラベル y
                x_base = x_train[idx]  # shape: (32,32)
                y_label = y_train[idx]  # ノイズ後ラベル or cleanラベル
                ratio, volume = compute_ratio_and_volume(model, x_base, y_label,
                                                         args.epsilon, args.nsample,
                                                         device=device,
                                                         compute_volume=args.compute_volume)
                sample_rate_list.append(ratio)
                if args.compute_volume and volume is not None:
                    volume_list.append(volume)

            # CSV に書き込み
            sr_writer.writerow([ep] + sample_rate_list)
            if args.compute_volume and vol_writer is not None:
                vol_writer.writerow([ep] + volume_list)
            print(f"Epoch {ep}: Sample Rate{' and Volume' if args.compute_volume else ''} recorded.")

    # グラフ化
    if args.compute_volume:
        plot_csv(volume_csv_path, os.path.join(out_dir, "volume.png"), "Volume")
    plot_csv(sample_rate_csv_path, os.path.join(out_dir, "sample_rate.png"), "Sample Rate")
    print(f"Saved graphs to {out_dir}")


# -------------------------------------------------------
# main
# -------------------------------------------------------
def main():
    args = parse_args()

    # シード設定を早めに適用（再現性の確保）
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True  # 必要に応じて
    # torch.backends.cudnn.benchmark = False     # 必要に応じて

    print(f"Random seed set to: {args.seed}")

    evaluate_model(args)


if __name__ == "__main__":
    main()
