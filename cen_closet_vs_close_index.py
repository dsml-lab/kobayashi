# main.py
import math
import os
import torch
import torch.optim as optim
import warnings
import torch.nn as nn
import numpy as np
import csv
from config import parse_args, parse_args_save_clo
from utils import set_seed, set_device, clear_memory
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_datasets, BalancedBatchSampler, NoisyDataset
from models import load_models
from viz_colored_alpha_gif_csv import target_plot_probabilities
from viz_colored_alpha_eval import evaluate_label_changes
from train import (
    train_model,
    test_model,
    add_label_noise,
    select_data_points,
    alpha_interpolation_test,
    compute_class_centroid_and_variance,
    find_closest_data_point_to_centroid,
    alpha_interpolation_test_save,
    select_n2,  # 既存の関数を使用
)
from logger import setup_wandb, setup_alpha_csv_logging, log_alpha_test_results, log_to_wandb, setup_alpha_csv_logging_save, log_alpha_test_results_save, setup_alpha_csv_logging_save2, setup_alpha_csv_logging_save_dir
import matplotlib.pyplot as plt
import wandb

def get_unique_directory(base_path):
    """
    指定されたbase_pathが存在しない場合はそのまま返し、存在する場合は
    hをインクリメントした新しいディレクトリパスを返す。
    """
    if not os.path.exists(base_path):
        return base_path
    h = 1
    while True:
        new_path = f"{base_path}_{h}"
        if not os.path.exists(new_path):
            return new_path
        h += 1

def main():
    print('Start session')

    # Ignore warnings
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.benchmark = True

    try:
        # Parse arguments and set initial configurations
        args = parse_args_save_clo()
        set_seed(args.fix_seed)
        
        # Set device
        device = set_device(args.gpu)
        print(f'Using device: {device}')

        # Load original datasets for test set, image size, etc. (no noise added here)
        print('Loading datasets...')
        try:
            train_dataset_original, test_dataset, imagesize, num_classes, in_channels = load_datasets(
                args.dataset, args.target, args.gray_scale, args
            )
        except FileNotFoundError as e:
            print(f"Error loading dataset: {e}")
            return
        except Exception as e:
            print(f"Unexpected error loading dataset: {e}")
            return

        # Set number of digit and color classes
        if args.target == 'combined':
            num_digits = 10
            num_colors = 10
        else:
            num_digits = 10
            num_colors = 1

        # 下記のデータセットはnoiseのついていないきれいなデータセットです。
        x_original_train, y_original_train = train_dataset_original.tensors
        
        experiment_name = f'seed{args.fix_seed}_{args.model}_{args.dataset}_variance{args.variance}_{args.target}_lr{args.lr}_batch{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}_Optim{args.optimizer}_Momentum{args.momentum}'

        # きれいなデータセットのラベルをoriginal_targetsが持っている。
        original_targets = y_original_train  # Before label noise

        # label_noise_rate に応じて読み込むディレクトリを変更
        noise_rate_str = str(args.label_noise_rate)
        base_dir = f"/workspace/data/distribution_colored_EMNIST_Seed42_Var{args.variance}_Corr0.5/distribution_colored_EMNIST_Seed42_Var{args.variance}_Corr0.5"
        data_dir = f"{base_dir}_noise_{noise_rate_str}"

        print(f'Loading pre-saved noisy dataset from: {data_dir}')

        # 保存済みデータの読み込み
        x_train = torch.from_numpy(np.load(os.path.join(data_dir, f"noisy{args.label_noise_rate}_x_train_colored.npy")))
        y_train_noisy = torch.from_numpy(np.load(os.path.join(data_dir, f"noisy{args.label_noise_rate}_y_train_colored.npy")))
        noise_info = torch.from_numpy(np.load(os.path.join(data_dir, f"noisy{args.label_noise_rate}_info_y_train_colored.npy")))

        # NoisyDatasetの作成
        train_dataset = TensorDataset(x_train, y_train_noisy)
        train_dataset = NoisyDataset(train_dataset, noise_info)

        # Extract indices for clean and noisy samples (optional)
        clean_indices = [i for i, label in enumerate(noise_info) if label == 0]
        noisy_indices = [i for i, label in enumerate(noise_info) if label == 1]

        # Get noisy targets
        if hasattr(train_dataset.dataset, 'tensors'):
            noisy_targets = y_train_noisy
        else:
            noisy_targets = torch.tensor(train_dataset.dataset.targets)

        # ------------------- 変更後のコード -------------------
        # Requirement 2: Compute centroid for a single label
        n1 = args.n1  # クラスラベル

        print(f"Computing centroid and variance for class label {n1}...")
        try:
            centroid1, variance1 = compute_class_centroid_and_variance(
                train_dataset.dataset,
                original_targets,
                class_label=n1
            )
            print(f"Centroid computed for class label {n1}")
        except ValueError as e:
            print(e)
            return

        # Requirement 3: Find closest noisy data point to the centroid (x)
        mode1 = args.mode1  # ノイズ付きデータを選択
        mode2 = args.mode2  # ノイズなしデータを選択

        # n2 を n1 に統一（同じクラス内で選択）
        n2 = n1  # 【変更】

        print(f"Finding closest noisy data point to centroid for class label {n1} in mode '{mode1}'...")
        try:
            idx_noisy = 50361  # 適切なインデックスを指定
            idx_clean = 50171  # 適切なインデックスを指定

            # ノイズ付きデータ点 (x_noisy) を指定されたインデックスから取得
            x_noisy = train_dataset.dataset.tensors[0][idx_noisy]
            label_x_noisy = train_dataset.dataset.tensors[1][idx_noisy]

            # ノイズなしデータ点 (y_clean) を指定されたインデックスから取得
            y_clean = train_dataset.dataset.tensors[0][idx_clean]
            label_y_clean = train_dataset.dataset.tensors[1][idx_clean]

            # 距離を計算
            distance_x_y = torch.norm(x_noisy.view(-1) - y_clean.view(-1), p=2)

            print(f"Closest noisy data point index: {idx_noisy} (label: {label_x_noisy})")
            print(f"Closest clean data point index: {idx_clean} (label: {label_y_clean})")
            print(f"Distance between the two points: {distance_x_y:.4f}")


            distance_x_y = torch.norm(x_noisy.view(-1) - y_clean.view(-1), p=2)

            # ラベルの取得
            label_y_clean = noisy_targets[idx_clean].item()
            label_x_noisy = noisy_targets[idx_noisy].item()

            # ベースディレクトリを定義（hを追加する前）
            base_alpha_csv_dir = f"./alpha_test/seed/lr_{args.label_noise_rate}/ori_closet_{experiment_name}/{mode1}_{mode2}/{label_y_clean}/{label_x_noisy}"
            # ユニークなディレクトリパスを取得
            alpha_csv_dir = get_unique_directory(base_alpha_csv_dir)

            # ディレクトリを作成
            os.makedirs(alpha_csv_dir, exist_ok=True)

            print(f"Closest noisy data point index: {idx_noisy} (class {n1})")
            print(f"Closest clean data point index: {idx_clean} (class {label_y_clean})")
            print(f"Distance between noisy and clean points: {distance_x_y:.4f}")
        except ValueError as e:
            print(e)
            return

        fig_and_log_dir = os.path.join(alpha_csv_dir, "fig_and_log")
        os.makedirs(fig_and_log_dir, exist_ok=True)

        # x_noisy, y_cleanを画像としてplotし、保存
        x_noisy_np = x_noisy.cpu().numpy()  # (C, H, W)
        y_clean_np = y_clean.cpu().numpy()  # (C, H, W)

        x_noisy_np = np.transpose(x_noisy_np, (1, 2, 0))
        y_clean_np = np.transpose(y_clean_np, (1, 2, 0))

        # グレースケールかカラーかを判定
        if x_noisy_np.shape[2] == 1:
            cmap = 'gray'
        else:
            cmap = None  # RGBの場合

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # ノイズ付与前後のラベル取得
        orig_label_x_noisy = y_original_train[idx_noisy].item()
        noisy_label_x_noisy = y_train_noisy[idx_noisy].item()

        orig_label_y_clean = y_original_train[idx_clean].item()
        noisy_label_y_clean = y_train_noisy[idx_clean].item()

        # タイトルに元ラベルとノイズ後ラベルを組み込み
        axes[0].imshow(y_clean_np.squeeze(), cmap=cmap)
        axes[0].set_title(
            f"Clean Data (idx={idx_clean})\nOri: {orig_label_y_clean}, Noisy: {noisy_label_y_clean}"
        )
        axes[0].axis('off')
        
        axes[1].imshow(x_noisy_np.squeeze(), cmap=cmap)
        axes[1].set_title(
            f"Noisy Data (idx={idx_noisy})\nOri: {orig_label_x_noisy},\nDistance to x y: {distance_x_y:.2f}"
        )
        axes[1].axis('off')

        fig_path = os.path.join(fig_and_log_dir, f"selected_data_points_{idx_noisy}_{idx_clean}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        # データ点をnpy形式で保存
        x_noisy_npy_path = os.path.join(fig_and_log_dir, f"x_noisy_idx{idx_noisy}.npy")
        y_clean_npy_path = os.path.join(fig_and_log_dir, f"y_clean_idx{idx_clean}.npy")
        np.save(x_noisy_npy_path, x_noisy_np)
        np.save(y_clean_npy_path, y_clean_np)

        distance_log_path = os.path.join(fig_and_log_dir, "distances.txt")

        # 距離情報をテキストファイルに保存
        with open(distance_log_path, 'w') as distance_file:
            #distance_file.write(f"Distance to centroid (Class {n1}): {distance_noisy:.6f}\n")
            distance_file.write(f"Distance between x_noisy and y_clean: {distance_x_y:.6f}\n")

        # Requirement 4: Run alpha_interpolation_test using the two data points
        # Get labels for the data points
        
        # Prepare 2-point experiment's labels
        label_x_noisy = noisy_targets[idx_noisy].item()
        label_y_clean = noisy_targets[idx_clean].item()

        digit_label_x_noisy = label_x_noisy // num_colors
        color_label_x_noisy = label_x_noisy % num_colors
        digit_label_y_clean = label_y_clean // num_colors
        color_label_y_clean = label_y_clean % num_colors

        combined_label_x_noisy = label_x_noisy
        combined_label_y_clean = label_y_clean

        model = load_models(in_channels, args, imagesize, num_classes).to(device)
        print("Loading model checkpoints...")
        for epoch in range(0, args.epoch + 1):
            checkpoint_path = f"save_model/{experiment_name}/model_epoch_{epoch}.pth"
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint not found: {checkpoint_path}")
                continue

            checkpoint = torch.load(checkpoint_path, map_location=device)

            # state_dictをロード
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()  # 評価モードに切り替え

            # x_noisyとy_cleanを使用したアルファ補間テストの実行
            alpha_logs = alpha_interpolation_test_save(
                model,
                x_noisy,
                y_clean,
                digit_label_x_noisy,
                digit_label_y_clean,
                color_label_x_noisy,
                color_label_y_clean,
                combined_label_x_noisy,
                combined_label_y_clean,
                num_digits,
                num_colors,
                device,
            )
            save_dir = os.path.join(alpha_csv_dir, "csv")
            alpha_csv_path = setup_alpha_csv_logging_save_dir(save_dir, epoch, n1, n2, mode1, mode2)  # n2 を n1 に統一
            log_alpha_test_results_save(alpha_logs, alpha_csv_path)
    except Exception as e:
        print(f"Error in main: {e}")
        return 
        
    target = ["digit","color"]
    for tg in target:
        evaluate_label_changes(directory=alpha_csv_dir, mode="alpha", target=tg, y_lim=(0,25), smoothing=True, smoothing_window=5)
        evaluate_label_changes(directory=alpha_csv_dir, mode="alpha", target=tg, y_lim=(0,25), smoothing=False)
        evaluate_label_changes(directory=alpha_csv_dir, mode="epoch", target=tg, y_lim=(0,1600), smoothing=False)
    target_plot_probabilities(alpha_csv_dir, targets="color", gif=True, gif_output=f"color_gif.gif", epoch_start=0, epoch_end=2000, epoch_step=10)
    target_plot_probabilities(alpha_csv_dir, targets="digit", gif=True, gif_output=f"digit_gif.gif", epoch_start=0, epoch_end=2000, epoch_step=10)
    clear_memory()

if __name__ == '__main__':
    main()
