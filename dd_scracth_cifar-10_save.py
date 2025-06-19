import math
import os
import torch
import torch.optim as optim
import warnings
import torch.nn as nn
import numpy as np
import csv
from config import parse_args,parse_args_model_save
from utils import set_seed, set_device, clear_memory
from torch.utils.data import  DataLoader, TensorDataset
from datasets import load_datasets, BalancedBatchSampler, NoisyDataset
from models import load_models
from train import (
    train_model,
    test_model,
    add_label_noise,
    select_data_points,
    alpha_interpolation_test,
    compute_class_centroid_and_variance,
    find_closest_data_point_to_centroid
)
from logger import setup_wandb, setup_alpha_csv_logging, log_alpha_test_results, log_to_wandb
import matplotlib.pyplot as plt
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import csv
from sklearn.metrics import classification_report, accuracy_score
from utils import clear_memory
from datasets import compute_distances_between_indices  # Assuming it's moved here
import math



def log_metrics_to_csv(csv_file_path, epoch, train_metrics, test_metrics):
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        # ヘッダー行が必要な場合は書き込み
        if not file_exists:
            writer.writerow([
                'epoch',
                'train_loss',
                'train_loss_variance',
                'train_accuracy_total',
                'train_accuracy_noisy',
                'train_accuracy_clean',
                'train_accuracy_digit_total',
                'train_accuracy_color_total',
                'train_accuracy_digit_noisy',
                'train_accuracy_color_noisy',
                'train_accuracy_digit_clean',
                'train_accuracy_color_clean',
                'test_loss',
                'test_accuracy_total',
                'test_accuracy_digit_total',
                'test_accuracy_color_total'
            ])
        writer.writerow([
            epoch,
            train_metrics['avg_loss'],
            train_metrics['var_loss'],
            train_metrics['accuracy_total'],
            train_metrics['accuracy_noisy'],
            train_metrics['accuracy_clean'],
            test_metrics['avg_loss'],
            test_metrics['accuracy_total'],
            test_metrics['accuracy_digit_total'],
            test_metrics['accuracy_color_total']
        ])

def train_model(model, train_loader, optimizer, criterion, weight_noisy, weight_clean, device):
    """
    Training function for CIFAR-10 with noise/clean sample weighting.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    correct_total = 0

    # Initialize counters for noisy and clean samples
    correct_noisy = 0
    total_noisy = 0
    correct_clean = 0
    total_clean = 0

    # Lists for loss tracking
    loss_values = []
    loss_values_noisy = []
    loss_values_clean = []

    # Ensure criterion returns per-sample losses
    criterion.reduction = 'none'

    for inputs, labels, noise_labels in train_loader:
        try:
            # Move data to device
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            noise_labels = noise_labels.to(device, non_blocking=True)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Calculate batch size and update total samples
            batch_size = labels.size(0)
            total_samples += batch_size

            # Get indices for noisy and clean samples
            idx_noisy = (noise_labels == 1)
            idx_clean = (noise_labels == 0)

            # Count noisy and clean samples in batch
            num_noisy = idx_noisy.sum().item()
            num_clean = idx_clean.sum().item()

            # Compute per-sample losses
            per_sample_loss = criterion(outputs, labels)

            # Calculate weights for the batch
            total_weight = weight_clean + weight_noisy
            weights = torch.zeros_like(per_sample_loss, device=device)

            if num_noisy == 0:  # All clean samples
                weights = torch.ones_like(per_sample_loss, device=device) * (weight_clean / total_weight) * 2
            elif num_clean == 0:  # All noisy samples
                weights = torch.ones_like(per_sample_loss, device=device) * (weight_noisy / total_weight) * 2
            else:  # Mixed batch
                weights[idx_noisy] = (weight_noisy / total_weight) * 2
                weights[idx_clean] = (weight_clean / total_weight) * 2

            # Apply weights to losses
            per_sample_loss_weighted = per_sample_loss * weights

            # Compute mean loss and backpropagate
            loss = per_sample_loss_weighted.mean()
            loss.backward()
            optimizer.step()

            # Update running loss and total accuracy
            running_loss += loss.item() * batch_size
            correct_total += (predicted == labels).sum().item()

            # Process noisy samples
            if num_noisy > 0:
                labels_noisy = labels[idx_noisy]
                predicted_noisy = predicted[idx_noisy]
                correct_noisy += (predicted_noisy == labels_noisy).sum().item()
                total_noisy += num_noisy

                # Store noisy sample losses
                loss_values_noisy.extend(per_sample_loss_weighted[idx_noisy].detach().cpu().numpy())

            # Process clean samples
            if num_clean > 0:
                labels_clean = labels[idx_clean]
                predicted_clean = predicted[idx_clean]
                correct_clean += (predicted_clean == labels_clean).sum().item()
                total_clean += num_clean

                # Store clean sample losses
                loss_values_clean.extend(per_sample_loss_weighted[idx_clean].detach().cpu().numpy())

            # Store all losses
            loss_values.extend(per_sample_loss_weighted.detach().cpu().numpy())

        except Exception as e:
            print(f"Error in training batch: {str(e)}")
            continue

    # Calculate loss statistics
    avg_loss = np.mean(loss_values) if loss_values else float('nan')
    var_loss = np.var(loss_values) if loss_values else float('nan')

    avg_loss_noisy = np.mean(loss_values_noisy) if loss_values_noisy else float('nan')
    var_loss_noisy = np.var(loss_values_noisy) if loss_values_noisy else float('nan')

    avg_loss_clean = np.mean(loss_values_clean) if loss_values_clean else float('nan')
    var_loss_clean = np.var(loss_values_clean) if loss_values_clean else float('nan')

    # Calculate accuracies
    metrics = {
        'avg_loss': avg_loss,
        'var_loss': var_loss,
        'accuracy_total': 100. * correct_total / total_samples if total_samples > 0 else float('nan'),
        'accuracy_noisy': 100. * correct_noisy / total_noisy if total_noisy > 0 else float('nan'),
        'accuracy_clean': 100. * correct_clean / total_clean if total_clean > 0 else float('nan'),
        'avg_loss_noisy': avg_loss_noisy,
        'var_loss_noisy': var_loss_noisy,
        'avg_loss_clean': avg_loss_clean,
        'var_loss_clean': var_loss_clean,
        'total_samples': total_samples,
        'total_noisy': total_noisy,
        'total_clean': total_clean,
        'correct_total': correct_total,
        'correct_noisy': correct_noisy,
        'correct_clean': correct_clean
    }

    return metrics

def test_model(model, test_loader, device):
    """
    Evaluation function to compute loss and accuracy on the test set for CIFAR-10.

    Args:
        model: The neural network model.
        test_loader: DataLoader for test data.
        device: Device to run the evaluation on.

    Returns:
        dict: Dictionary containing test metrics.
    """
    model.eval()
    test_loss = 0
    correct_total = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()  # Default reduction='mean'

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_total += (predicted == labels).sum().item()

    avg_loss = test_loss / total_samples if total_samples > 0 else float('nan')
    accuracy_total = 100. * correct_total / total_samples if total_samples > 0 else float('nan')

    return {
        'avg_loss': avg_loss,
        'accuracy_total': accuracy_total,
        'total_samples': total_samples,
        'correct_total': correct_total
    }

def train_model_0_epoch_no_update(model, train_loader, optimizer, criterion, weight_noisy, weight_clean, device):
    """
    モデルのパラメータを更新せずに、入力に対する出力と各種指標（noise別の損失・精度）を計算する関数です。
    （CIFAR-10用に、digit/colorに関する処理は削除済み）
    """
    model.eval()  # 評価モード
    running_loss = 0.0
    total_samples = 0
    correct_total = 0

    # Initialize counters for noisy and clean samples
    correct_noisy = 0
    total_noisy = 0
    correct_clean = 0
    total_clean = 0

    # Lists for loss tracking
    loss_values = []
    loss_values_noisy = []
    loss_values_clean = []

    # Ensure criterion returns per-sample losses
    criterion.reduction = 'none'

    # 勾配計算を行わないために no_grad() コンテキストを使用
    with torch.no_grad():
        for inputs, labels, noise_labels in train_loader:
            try:
                # Move data to device
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                noise_labels = noise_labels.to(device, non_blocking=True)

                # Forward pass
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # Calculate batch size and update total samples
                batch_size = labels.size(0)
                total_samples += batch_size

                # Get indices for noisy and clean samples
                idx_noisy = (noise_labels == 1)
                idx_clean = (noise_labels == 0)

                # Count noisy and clean samples in batch
                num_noisy = idx_noisy.sum().item()
                num_clean = idx_clean.sum().item()

                # Compute per-sample losses
                per_sample_loss = criterion(outputs, labels)

                # Calculate weights for the batch
                total_weight = weight_clean + weight_noisy
                weights = torch.zeros_like(per_sample_loss, device=device)

                if num_noisy == 0:  # All clean samples
                    weights = torch.ones_like(per_sample_loss, device=device) * (weight_clean / total_weight) * 2
                elif num_clean == 0:  # All noisy samples
                    weights = torch.ones_like(per_sample_loss, device=device) * (weight_noisy / total_weight) * 2
                else:  # Mixed batch
                    weights[idx_noisy] = (weight_noisy / total_weight) * 2
                    weights[idx_clean] = (weight_clean / total_weight) * 2

                # Apply weights to losses
                per_sample_loss_weighted = per_sample_loss * weights

                # パラメータ更新は行わず、損失計算のみ実施
                loss = per_sample_loss_weighted.mean()
                running_loss += loss.item() * batch_size
                correct_total += (predicted == labels).sum().item()

                # Process noisy samples
                if num_noisy > 0:
                    labels_noisy = labels[idx_noisy]
                    predicted_noisy = predicted[idx_noisy]
                    correct_noisy += (predicted_noisy == labels_noisy).sum().item()
                    total_noisy += num_noisy
                    loss_values_noisy.extend(per_sample_loss_weighted[idx_noisy].detach().cpu().numpy())

                # Process clean samples
                if num_clean > 0:
                    labels_clean = labels[idx_clean]
                    predicted_clean = predicted[idx_clean]
                    correct_clean += (predicted_clean == labels_clean).sum().item()
                    total_clean += num_clean
                    loss_values_clean.extend(per_sample_loss_weighted[idx_clean].detach().cpu().numpy())

                # Store all losses
                loss_values.extend(per_sample_loss_weighted.detach().cpu().numpy())

            except Exception as e:
                print(f"Error in training batch: {str(e)}")
                continue

    # Calculate loss statistics
    avg_loss = np.mean(loss_values) if loss_values else float('nan')
    var_loss = np.var(loss_values) if loss_values else float('nan')

    avg_loss_noisy = np.mean(loss_values_noisy) if loss_values_noisy else float('nan')
    var_loss_noisy = np.var(loss_values_noisy) if loss_values_noisy else float('nan')

    avg_loss_clean = np.mean(loss_values_clean) if loss_values_clean else float('nan')
    var_loss_clean = np.var(loss_values_clean) if loss_values_clean else float('nan')

    # Calculate accuracies
    metrics = {
        'avg_loss': avg_loss,
        'var_loss': var_loss,
        'accuracy_total': 100. * correct_total / total_samples if total_samples > 0 else float('nan'),
        'accuracy_noisy': 100. * correct_noisy / total_noisy if total_noisy > 0 else float('nan'),
        'accuracy_clean': 100. * correct_clean / total_clean if total_clean > 0 else float('nan'),
        'avg_loss_noisy': avg_loss_noisy,
        'var_loss_noisy': var_loss_noisy,
        'avg_loss_clean': avg_loss_clean,
        'var_loss_clean': var_loss_clean,
        'total_samples': total_samples,
        'total_noisy': total_noisy,
        'total_clean': total_clean,
        'correct_total': correct_total,
        'correct_noisy': correct_noisy,
        'correct_clean': correct_clean
    }

    return metrics

def add_label_noise(targets, label_noise_rate, num_digits, num_colors):
    """
    Add label noise to the targets.

    Args:
        targets (torch.Tensor): Original labels.
        label_noise_rate (float): Fraction of labels to corrupt.
        num_digits (int): Number of digit classes.
        num_colors (int): Number of color classes.

    Returns:
        tuple: Noisy targets and noise information tensor.
    """
    noisy_targets = targets.clone()
    num_noisy = int(label_noise_rate * len(targets))
    noisy_indices = torch.randperm(len(targets))[:num_noisy]
    noise_info = torch.zeros(len(targets), dtype=torch.int)  # Initialize as clean

    if num_digits == 10 and num_colors == 1:
        for idx in noisy_indices:
            original_label = targets[idx].item()
            new_label = random.randint(0, num_digits - 1)
            while new_label == original_label:
                new_label = random.randint(0, num_digits - 1)
            noisy_targets[idx] = new_label
            noise_info[idx] = 1  # Mark as noisy

    elif num_digits == 10 and num_colors == 10:
        for idx in noisy_indices:
            original_label = targets[idx].item()
            original_digit = original_label // num_colors
            original_color = original_label % num_colors

            new_digit = random.randint(0, num_digits - 1)
            new_color = random.randint(0, num_colors - 1)
            new_label = new_digit * num_colors + new_color
            while new_label == original_label:
                new_digit = random.randint(0, num_digits - 1)
                new_color = random.randint(0, num_colors - 1)
                new_label = new_digit * num_colors + new_color

            noisy_targets[idx] = new_label
            noise_info[idx] = 1  # Mark as noisy

    return noisy_targets, noise_info


def main():
    """
    Main training loop with comprehensive error handling and logging for CIFAR-10.
    """
    print('Start session')
    wandb_run = None

    # Ignore warnings
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.benchmark = True

    # CIFAR-10 parameters
    imagesize = 32
    num_classes = 10
    in_channels = 3

    try:
        # Parse arguments and set initial configurations
        args = parse_args_model_save()
        set_seed(args.fix_seed)
        
        # Set device
        device = set_device(args.gpu)
        print(f'Using device: {device}')

        # --------------------
        # データのロード部分の修正 (CIFAR-10)
        # --------------------
        print(args.use_saved_data)
        if args.use_saved_data == True:
            print("using_presaved")
            # CIFAR-10用のデータセットを読み込む（load_datasets内でCIFAR-10用に実装済みとする）
            train_dataset, test_dataset, imagesize, num_classes, in_channels = load_datasets(
                    args.dataset, args.target, args.gray_scale, args
                )
            # label_noise_rate に応じたディレクトリ
            noise_rate_str = str(args.label_noise_rate)
            base_dir = "data/cifar-10/cifar-10"
            data_dir = f"{base_dir}_noise_{noise_rate_str}"
            
            print(f'Loading pre-saved noisy dataset from: {data_dir}')

            # 保存済みデータの読み込み
            x_train = torch.from_numpy(np.load(os.path.join(data_dir, f"cifar-10_noise_{args.label_noise_rate}_x_train.npy")))
            y_train_noisy = torch.from_numpy(np.load(os.path.join(data_dir, f"cifar-10_noise_{args.label_noise_rate}_y_train.npy")))
            noise_info = torch.from_numpy(np.load(os.path.join(data_dir, f"cifar-10_noise_{args.label_noise_rate}_info_y_train.npy")))
            
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
        
        else:
            # 既存のデータロードと処理（CIFAR-10用）
            print('ori_Loading datasets')
            try:
                train_dataset, test_dataset, imagesize, num_classes, in_channels = load_datasets(
                    args.dataset, args.target, args.gray_scale, args
                )
            except FileNotFoundError as e:
                print(f"Error loading dataset: {e}")
                return
            except Exception as e:
                print(f"Unexpected error loading dataset: {e}")
                return

            # CIFAR-10 の場合、num_classes は 10
            num_classes = 10

            # Add label noise and create NoisyDataset
            print(f'Preparing dataset with label noise rate: {args.label_noise_rate}')
            if args.label_noise_rate > 0.0:
                if hasattr(train_dataset, 'tensors'):
                    x_train, y_train = train_dataset.tensors
                    y_train_noisy, noise_info = add_label_noise(y_train, args.label_noise_rate, num_classes,1)
                    train_dataset = TensorDataset(x_train, y_train_noisy)
                    train_dataset = NoisyDataset(train_dataset, noise_info)
                else:
                    y_train = torch.tensor(train_dataset.targets)
                    y_train_noisy, noise_info = add_label_noise(y_train, args.label_noise_rate, num_classes,1)
                    train_dataset.targets = y_train_noisy.tolist()
                    train_dataset = NoisyDataset(train_dataset, noise_info)
            else:
                if hasattr(train_dataset, 'tensors'):
                    x_train, y_train = train_dataset.tensors
                    y_train_noisy = y_train.clone()
                    noise_info = torch.zeros(len(train_dataset), dtype=torch.int)
                    train_dataset = TensorDataset(x_train, y_train_noisy)
                    train_dataset = NoisyDataset(train_dataset, noise_info)
                else:
                    y_train = torch.tensor(train_dataset.targets)
                    y_train_noisy = y_train.clone()
                    noise_info = torch.zeros(len(train_dataset), dtype=torch.int)
                    train_dataset.targets = y_train_noisy.tolist()

            # Extract indices for clean and noisy samples (optional)
            clean_indices = [i for i, label in enumerate(noise_info) if label == 0]
            noisy_indices = [i for i, label in enumerate(noise_info) if label == 1]
            
            # 保存先ディレクトリを CIFAR-10 用に変更
            path = f"data/cifar-10/cifar-10_noise_{args.label_noise_rate}"
            os.makedirs(path, exist_ok=True)
            if hasattr(train_dataset.dataset, 'tensors'):
                np.save(f"{path}/cifar-10_noise_{args.label_noise_rate}_x_train.npy", train_dataset.dataset.tensors[0].cpu().numpy())
                np.save(f"{path}/cifar-10_noise_{args.label_noise_rate}_y_train.npy", train_dataset.dataset.tensors[1].cpu().numpy())
                np.save(f"{path}/cifar-10_noise_{args.label_noise_rate}_info_y_train.npy", train_dataset.noise_info.cpu().numpy())
            else:
                original_targets = torch.tensor(y_train)  # Before label noise

            # Get noisy targets
            if hasattr(train_dataset.dataset, 'tensors'):
                noisy_targets = y_train_noisy
            else:
                noisy_targets = torch.tensor(train_dataset.dataset.targets)
        
        # --------------------
        # モデル、最適化手法、損失関数の初期化
        # --------------------
        print('Initializing model...')
        model = load_models(in_channels, args, imagesize, num_classes)
        model = model.to(device)

        print('Setting up optimizer...')
        if args.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Set loss function (per-sample loss)
        criterion = nn.CrossEntropyLoss(reduction='none')

        # Initialize wandb
        if args.wandb:
            print('Initializing wandb...')
            experiment_name = f'width{args.model_width}_{args.model}_{args.dataset}_lr{args.lr}_batch{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}_Optim{args.optimizer}_Momentum{args.momentum}'
            wandb_run = setup_wandb(args, experiment_name)

        # --------------------
        # DataLoader の準備
        # --------------------
        if args.batch_size % 2 != 0:
            raise ValueError("Batch size must be even for balanced batches")

        if args.label_noise_rate == 0.0 or args.label_noise_rate == 1.0:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True
            )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        print('Starting training...')
        best_test_accuracy = 0.0
        
        # --------------------
        # 学習前 (epoch 0) の指標計算
        # --------------------
        save_dict = {
            "args": vars(args),
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "train_accuracy": None,  # トレーニング前なのでNone
            "test_accuracy": None    # トレーニング前なのでNone
        }
        train_metrics0 = train_model_0_epoch_no_update(
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    weight_noisy=args.weight_noisy,
                    weight_clean=args.weight_clean,
                    device=device
                )
        test_metrics0 = test_model(
                    model=model,
                    test_loader=test_loader,
                    device=device
                )
        if args.wandb:
            log_data = {
                'epoch': 0,
                'train_loss': train_metrics0['avg_loss'],
                'train_loss_variance': train_metrics0['var_loss'],
                'avg_loss_noisy': train_metrics0['avg_loss_noisy'],
                'avg_loss_clean': train_metrics0['avg_loss_clean'],
                'train_accuracy': train_metrics0['accuracy_total'],
                'accuracy_noisy': train_metrics0['accuracy_noisy'],
                'accuracy_clean': train_metrics0['accuracy_clean'],
                'test_loss': test_metrics0['avg_loss'],
                'test_accuracy': test_metrics0['accuracy_total']
            }
            log_to_wandb(wandb_run, log_data)

        # --------------------
        # モデル保存用ディレクトリの作成 (CIFAR-10用)
        # --------------------
        tyukan_dir = "noise"
        csv_path = f"save_model/CIFAR10/{tyukan_dir}/{experiment_name}/csv/training_metrics.csv"
        os.makedirs(f"save_model/CIFAR10/{tyukan_dir}/{experiment_name}/csv", exist_ok=True)
        log_metrics_to_csv(csv_path, 0, train_metrics0, test_metrics0)

        os.makedirs(f"save_model/CIFAR10/{tyukan_dir}/{experiment_name}", exist_ok=True)
        torch.save(save_dict, f"save_model/CIFAR10/{tyukan_dir}/{experiment_name}/model_epoch_0.pth")

        # --------------------
        # エポックごとの学習・評価ループ
        # --------------------
        for epoch in range(1, args.epoch + 1):
            try:
                # Training phase
                train_metrics = train_model(
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    weight_noisy=args.weight_noisy,
                    weight_clean=args.weight_clean,
                    device=device
                )

                # Testing phase
                test_metrics = test_model(
                    model=model,
                    test_loader=test_loader,
                    device=device
                )

                # モデル保存用辞書の作成
                save_dict = {
                    "args": vars(args),
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "train_accuracy": train_metrics['accuracy_total'],
                    "test_accuracy": test_metrics['accuracy_total']
                }

                os.makedirs(f"save_model/CIFAR10/{tyukan_dir}/{experiment_name}", exist_ok=True)
                torch.save(save_dict, f"save_model/CIFAR10/{tyukan_dir}/{experiment_name}/model_epoch_{epoch}.pth")
                
                # WandB logging
                if args.wandb:
                    log_data = {
                        'epoch': epoch,
                        'train_loss': train_metrics['avg_loss'],
                        'train_loss_variance': train_metrics['var_loss'],
                        'avg_loss_noisy': train_metrics['avg_loss_noisy'],
                        'avg_loss_clean': train_metrics['avg_loss_clean'],
                        'train_accuracy': train_metrics['accuracy_total'],
                        'accuracy_noisy': train_metrics['accuracy_noisy'],
                        'accuracy_clean': train_metrics['accuracy_clean'],
                        'test_loss': test_metrics['avg_loss'],
                        'test_accuracy': test_metrics['accuracy_total']
                    }
                    log_to_wandb(wandb_run, log_data)
                log_metrics_to_csv(csv_path, epoch, train_metrics, test_metrics)

            except Exception as e:
                print(f"Error in epoch {epoch}: {str(e)}")
                continue

    except Exception as e:
        print(f"Error in main: {e}")
        return 
    finally:
        # Cleanup
        if wandb_run:
            wandb_run.finish()
        print('Training completed')
if __name__ =='__main__':
    main()
