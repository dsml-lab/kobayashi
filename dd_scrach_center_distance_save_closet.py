# main.py
import math
import os
import torch
import torch.optim as optim
import warnings
import torch.nn as nn
import numpy as np
import csv
from config import parse_args, parse_args_save
from utils import set_seed, set_device, clear_memory
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_datasets, BalancedBatchSampler, NoisyDataset
from models import load_models
from viz_colored_alpha_gif_csv import target_plot_probabilities
from viz_colored_alpha_eval import evaluate_label_changes
from train2 import (
    train_model,
    test_model,
    add_label_noise,
    select_data_points,
    alpha_interpolation_test,
    compute_class_centroid_and_variance,
    find_closest_data_point_to_centroid,
    alpha_interpolation_test_save,
    select_n2,
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
        args = parse_args_save()
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
        
        experiment_name = f'{args.model}_{args.dataset}_variance{args.variance}_{args.target}_lr{args.lr}_batch{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}_Optim{args.optimizer}_Momentum{args.momentum}'

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

        # ------------------- New Code Starts Here -------------------
        # Requirement 2: Compute centroids and variances for two labels
        n1 = args.n1  # First class label
        n2 = args.n2  # Second class label

        # n2 = args.n1#クラス内を使用する場合

        print(f"Computing centroids and variances for class labels {n1} and {n2}...")
        try:
            centroid1, variance1 = compute_class_centroid_and_variance(
                train_dataset.dataset,
                original_targets,
                class_label=n1
            )
            print(f"Centroids computed for class labels {n1} and {n2}")
            #print(f"{n1}_{variance1} |||||{n2}_{variance2}")
        except ValueError as e:
            print(e)
            return

        # Requirement 3: Find closest data points to the centroids
        #mode1 = 'no_noise'  # 'no_noise' or 'noise'
        mode1 = "no_noise"
        # mode2 = 'no_noise'
        mode2 = 'noise'

        """if mode2 == 'noise':
            n2 = n1"""
        n2 = args.n2
        print(f"Finding closest data points to centroids for class labels {n1} and {n2} in mode '{mode1} and {mode2}'...")
        try:
            x1, idx1, distance1 = find_closest_data_point_to_centroid(
                centroid1,
                train_dataset.dataset,
                noisy_targets,
                class_label=n1,
                mode=mode1,
                noise_info=noise_info
            )
            x2, idx2, distance2 = select_n2(
                n1=x1,
                idx1=idx1,
                target="combined",
                mode=mode2,
                original_targets=original_targets,
                train_dataset=train_dataset.dataset,
                y_train_noisy=noisy_targets,
                noise_info=noise_info,
            )
            distance_x1_x2 = torch.norm(x1.view(-1) - x2.view(-1), p=2)

            # x2のノイズ後ラベルを取得
            label_x2 = noisy_targets[idx2].item()
            label_x1 = noisy_targets[idx1].item()

            # ベースディレクトリを定義（hを追加する前）
            #base_alpha_csv_dir = f"./alpha_test/{experiment_name}/{mode1}_{mode2}/bupunai/{n1}/{label_x2}"
            base_alpha_csv_dir = f"./alpha_test/{experiment_name}/{mode1}_{mode2}/{label_x2}/{label_x1}"
            # ユニークなディレクトリパスを取得
            alpha_csv_dir = get_unique_directory(base_alpha_csv_dir)

            # ディレクトリを作成
            os.makedirs(alpha_csv_dir, exist_ok=True)

            print(f"Closest data point indices: {idx1} (class {n1}), {idx2} (class {label_x2})")
            print(f"Distances to centroids: {distance1}, {distance2}")
            print(distance_x1_x2)
        except ValueError as e:
            print(e)
            return

        fig_and_log_dir = os.path.join(alpha_csv_dir, "fig_and_log")
        os.makedirs(fig_and_log_dir, exist_ok=True)

        # x1, x2を画像としてplotし、保存
        x1_np = x1.cpu().numpy()  # (C, H, W)
        x2_np = x2.cpu().numpy()  # (C, H, W)

        x1_np = np.transpose(x1_np, (1, 2, 0))
        x2_np = np.transpose(x2_np, (1, 2, 0))

        # グレースケールかカラーかを判定
        if x1_np.shape[2] == 1:
            cmap = 'gray'
        else:
            cmap = None  # RGBの場合

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # ノイズ付与前後のラベル取得
        orig_label_x1 = y_original_train[idx1].item()
        noisy_label_x1 = y_train_noisy[idx1].item()

        orig_label_x2 = y_original_train[idx2].item()
        noisy_label_x2 = y_train_noisy[idx2].item()

        # タイトルに元ラベルとノイズ後ラベルを組み込み
        axes[0].imshow(x2_np.squeeze(), cmap=cmap)
        axes[0].set_title(
            f"Class {n1} Data (idx={idx2})\nOri: {orig_label_x2},Noisy: {noisy_label_x2},"
        )
        axes[0].axis('off')
        
        axes[1].imshow(x1_np.squeeze(), cmap=cmap)
        axes[1].set_title(
            f"Class {n1} Data (idx={idx1})\nOri: {orig_label_x1}, Noisy: {noisy_label_x1}, C_Dis={distance1:.2f},Dis_x1_x2={distance_x1_x2:.2f}"
        )
        axes[1].axis('off')

        fig_path = os.path.join(fig_and_log_dir, f"selected_data_points_{idx1}_{idx2}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        # データ点をnpy形式で保存
        x1_npy_path = os.path.join(fig_and_log_dir, f"x1_idx{idx1}.npy")
        x2_npy_path = os.path.join(fig_and_log_dir, f"x2_idx{idx2}.npy")
        np.save(x1_npy_path, x1_np)
        np.save(x2_npy_path, x2_np)

        distance_log_path = os.path.join(fig_and_log_dir, "distances.txt")

        # 距離情報をテキストファイルに保存
        with open(distance_log_path, 'w') as distance_file:
            distance_file.write(f"Distance to centroid (Class {n1}): {distance1:.6f}\n")
            distance_file.write(f"Distance to centroid (Class {n2}): {distance2:.6f}\n")
            distance_file.write(f"Distance between x1 and x2: {distance_x1_x2:.6f}\n")

        # Requirement 4: Run alpha_interpolation_test using the two data points
        # Get labels for the data points
        
        # Prepare 2-point experiment's labels
        label_x1 = noisy_targets[idx1].item()
        label_x2 = noisy_targets[idx2].item()

        digit_label_x1 = label_x1 // num_colors
        color_label_x1 = label_x1 % num_colors
        digit_label_x2 = label_x2 // num_colors
        color_label_x2 = label_x2 % num_colors

        combined_label_x1 = label_x1
        combined_label_x2 = label_x2

        model = load_models(in_channels, args, imagesize, num_classes).to(device)
        print("loading_model")
        for epoch in range(0, args.epoch + 1):
            checkpoint_path = f"save_model/{experiment_name}/model_epoch_{epoch}.pth"
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint not found: {checkpoint_path}")
                continue

            checkpoint = torch.load(checkpoint_path, map_location=device)

            # state_dictをロード
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()  # 評価モードに切り替え

            # x1とx2を使用したアルファ補間テストの実行
            alpha_logs = alpha_interpolation_test_save(
                model,
                x1,
                x2,
                digit_label_x1,
                digit_label_x2,
                color_label_x1,
                color_label_x2,
                combined_label_x1,
                combined_label_x2,
                num_digits,
                num_colors,
                device,
            )
            save_dir = os.path.join(alpha_csv_dir,"csv")
            alpha_csv_path=setup_alpha_csv_logging_save_dir(save_dir,epoch,n1,n2,mode1,mode2)
            log_alpha_test_results_save(alpha_logs, alpha_csv_path)
    except Exception as e:
        print(f"Error in main: {e}")
        return 
        
    target = ["digit","color"]
    for tg in target:
        evaluate_label_changes(directory=alpha_csv_dir,mode="alpha",target=tg,y_lim=(0,25), smoothing=True,smoothing_window=5)
        evaluate_label_changes(directory=alpha_csv_dir,mode="alpha",target=tg,y_lim=(0,25), smoothing=False)
        evaluate_label_changes(directory=alpha_csv_dir,mode="epoch",target=tg,y_lim=(0,1600), smoothing=False)
    target_plot_probabilities(alpha_csv_dir,targets="color",gif=True,gif_output=f"color_gif.gif",epoch_start=0,epoch_end=2000,epoch_step=10)
    target_plot_probabilities(alpha_csv_dir,targets="digit",gif=True,gif_output=f"digit_gif.gif",epoch_start=0,epoch_end=2000,epoch_step=10)
    clear_memory()

if __name__ == '__main__':
    main()
