# main.py
import math
import os
import torch
import torch.optim as optim
import warnings
import torch.nn as nn
import numpy as np
import csv
from config import parse_args,parse_args_model_save
from utils import set_seed, set_device, clear_memory, seed_worker
from torch.utils.data import  DataLoader, TensorDataset
from datasets import load_datasets, BalancedBatchSampler, NoisyDataset
from models import load_models
from train import (
    train_model_0_epoch_no_update,
    train_model,
    test_model,
    add_label_noise,

)
from logger import setup_wandb, log_to_wandb
import matplotlib.pyplot as plt

import wandb

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
                'avg_loss_noisy',
                'avg_loss_clean',
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
            train_metrics['avg_loss_noisy'],
            train_metrics['avg_loss_clean'],
            train_metrics['accuracy_total'],
            train_metrics['accuracy_noisy'],
            train_metrics['accuracy_clean'],
            train_metrics['accuracy_digit_total'],
            train_metrics['accuracy_color_total'],
            train_metrics['accuracy_digit_noisy'],
            train_metrics['accuracy_color_noisy'],
            train_metrics['accuracy_digit_clean'],
            train_metrics['accuracy_color_clean'],
            test_metrics['avg_loss'],
            test_metrics['accuracy_total'],
            test_metrics['accuracy_digit_total'],
            test_metrics['accuracy_color_total']
        ])


def main():
    """
    Main training loop with comprehensive error handling and logging
    """
    print('Start session')
    wandb_run = None

    # Ignore warnings
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.benchmark = True
    imagesize=32
    num_classes=100
    in_channels=3
    print("debag")
    try:
        # Parse arguments and set initial configurations
        args = parse_args_model_save()
        set_seed(args.fix_seed)
        
        # Set device
        device = set_device(args.gpu)
        print(f'Using device: {device}')

        # --------------------
        # データのロード部分を修正
        # --------------------
        print(args.use_saved_data)
        if args.use_saved_data==True:
            print("using_presaved")
            if args.target == 'combined':
                num_digits = 10
                num_colors = 10
            else:
                num_digits = 10
                num_colors = 1
            train_dataset, test_dataset, imagesize, num_classes, in_channels = load_datasets(
                    args.dataset, args.target, args.gray_scale, args
                )
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
            
        
        else:
            # 既存のデータロードと処理
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

            # Set number of digit and color classes
            if args.target == 'combined':
                num_digits = 10
                num_colors = 10
            else:
                num_digits = 10
                num_colors = 1

            # Add label noise and create NoisyDataset
            print(f'Preparing dataset with label noise rate: {args.label_noise_rate}')
            if args.label_noise_rate > 0.0:
                if hasattr(train_dataset, 'tensors'):
                    x_train, y_train = train_dataset.tensors
                    y_train_noisy, noise_info = add_label_noise(y_train, args.label_noise_rate, num_digits, num_colors)
                    train_dataset = TensorDataset(x_train, y_train_noisy)
                    train_dataset = NoisyDataset(train_dataset, noise_info)
                else:
                    y_train = torch.tensor(train_dataset.targets)
                    y_train_noisy, noise_info = add_label_noise(y_train, args.label_noise_rate, num_digits, num_colors)
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

            # Extract indices for clean and noisy samples
            clean_indices = [i for i, label in enumerate(noise_info) if label == 0]
            noisy_indices = [i for i, label in enumerate(noise_info) if label == 1]
            path = f"data/distribution_colored_EMNIST_Seed42_Var{args.variance}_Corr0.5/distribution_colored_EMNIST_Seed42_Var{args.variance}_Corr0.5_noise_{args.label_noise_rate}"
            os.makedirs(path, exist_ok=True)
            if hasattr(train_dataset.dataset, 'tensors'):
                np.save(f"{path}/noisy{args.label_noise_rate}_x_train_colored.npy", train_dataset.dataset.tensors[0].cpu().numpy())
                np.save(f"{path}/noisy{args.label_noise_rate}_y_train_colored.npy", train_dataset.dataset.tensors[1].cpu().numpy())
                np.save(f"{path}/noisy{args.label_noise_rate}_info_y_train_colored.npy", train_dataset.noise_info.cpu().numpy())
            else:
                original_targets = torch.tensor(y_train)  # Before label noise

            # Get noisy targets
            if hasattr(train_dataset.dataset, 'tensors'):
                noisy_targets = y_train_noisy
            else:
                noisy_targets = torch.tensor(train_dataset.dataset.targets)
        # Initialize model
        print('Initializing model...')
        model = load_models(in_channels, args, imagesize, num_classes)
        model = model.to(device)

        # Set optimizer
        print('Setting up optimizer...')
        if args.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Set loss function
        criterion = nn.CrossEntropyLoss(reduction='none')

        # Initialize wandb
        if args.wandb:
            print('Initializing wandb...')
            experiment_name = f'seed_{args.fix_seed}width{args.model_width}_{args.model}_{args.dataset}_variance{args.variance}_{args.target}_lr{args.lr}_batch{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}_Optim{args.optimizer}_Momentum{args.momentum}'
            wandb_run = setup_wandb(args, experiment_name)

        # Prepare data loaders
        if args.batch_size % 2 != 0:
            raise ValueError("Batch size must be even for balanced batches")
        #g = torch.Generator()
        #g.manual_seed(args.fix_seed)
        if args.label_noise_rate == 0.0 or args.label_noise_rate == 1.0:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                #generator=g,
                #worker_init_fn=seed_worker
            )
        else:
            # batch_sampler = BalancedBatchSampler(
            #     clean_indices,
            #     noisy_indices,
            #     args.batch_size,
            #     drop_last=False
            # )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                # batch_sampler=batch_sampler,
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
        
        save_dict = {
            "args": vars(args),
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "train_accuracy": None,  # トレーニング前なのでNone
            "test_accuracy": None    # トレーニング前なのでNone
        }
        train_metrics0=train_model_0_epoch_no_update(model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    weight_noisy=args.weight_noisy,
                    weight_clean=args.weight_clean,
                    device=device,
                    num_colors=num_colors,
                    num_digits=num_digits
                )
        test_metrics0 = test_model(
                    model=model,
                    test_loader=test_loader,
                    device=device,
                    num_colors=num_colors,
                    num_digits=num_digits
                )
        if args.wandb:
                    log_data = {
                        'epoch': 0,
                        'train_loss': train_metrics0['avg_loss'],
                        'train_loss_variance': train_metrics0['var_loss'],
                        'avg_loss_noisy': train_metrics0['avg_loss_noisy'],
                        'avg_loss_clean' : train_metrics0['avg_loss_clean'],
                        'train_accuracy': train_metrics0['accuracy_total'],
                        'train_accuracy_noisy': train_metrics0['accuracy_noisy'],
                        'train_accuracy_clean': train_metrics0['accuracy_clean'],
                        'train_accuracy_digit_total': train_metrics0['accuracy_digit_total'],
                        'train_accuracy_color_total': train_metrics0['accuracy_color_total'],
                        'train_accuracy_digit_noisy': train_metrics0['accuracy_digit_noisy'],
                        'train_accuracy_color_noisy': train_metrics0['accuracy_color_noisy'],
                        'train_accuracy_digit_clean': train_metrics0['accuracy_digit_clean'],
                        'train_accuracy_color_clean': train_metrics0['accuracy_color_clean'],
                        'test_loss': test_metrics0['avg_loss'],
                        'test_accuracy': test_metrics0['accuracy_total'],
                        'test_accuracy_digit_total': test_metrics0['accuracy_digit_total'],
                        'test_accuracy_color_total': test_metrics0['accuracy_color_total']
                    }
                    log_to_wandb(wandb_run, log_data)


        # ディレクトリがなければ作成
        tyukan_dir="noise_rate=0_sigma=1000"
        csv_path = f"save_model/Colored_EMSNIT/{tyukan_dir}/{experiment_name}/csv/training_metrics.csv"
        os.makedirs(f"save_model/Colored_EMSNIT/{tyukan_dir}/{experiment_name}/csv", exist_ok=True)
        log_metrics_to_csv(csv_path, 0, train_metrics0, test_metrics0)

        os.makedirs(f"save_model/Colored_EMSNIT/{tyukan_dir}/{experiment_name}", exist_ok=True)
        torch.save(save_dict, f"save_model/Colored_EMSNIT/{tyukan_dir}/{experiment_name}/model_epoch_0.pth")
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
                    device=device,
                    num_colors=num_colors,
                    num_digits=num_digits
                )

                # Testing phase
                test_metrics = test_model(
                    model=model,
                    test_loader=test_loader,
                    device=device,
                    num_colors=num_colors,
                    num_digits=num_digits
                )

                # --------------------
                # モデルを辞書形式で保存する処理
                # argsを辞書に変換: vars(args)
                # train_accuracy, test_accuracyを含める
                # --------------------
                save_dict = {
                    "args": vars(args),
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "train_accuracy": train_metrics['accuracy_total'],
                    "test_accuracy": test_metrics['accuracy_total']
                }

                # ディレクトリがなければ作成
                os.makedirs(f"save_model/Colored_EMSNIT/{tyukan_dir}/{experiment_name}", exist_ok=True)
                torch.save(save_dict, f"save_model/Colored_EMSNIT/{tyukan_dir}/{experiment_name}/model_epoch_{epoch}.pth")
                # --------------------

                # WandB logging
                if args.wandb:
                    log_data = {
                        'epoch': epoch,
                        'train_loss': train_metrics['avg_loss'],
                        'train_loss_variance': train_metrics['var_loss'],
                        'avg_loss_noisy': train_metrics['avg_loss_noisy'],
                        'avg_loss_clean' : train_metrics['avg_loss_clean'],
                        'train_accuracy': train_metrics['accuracy_total'],
                        'train_accuracy_noisy': train_metrics['accuracy_noisy'],
                        'train_accuracy_clean': train_metrics['accuracy_clean'],
                        'train_accuracy_digit_total': train_metrics['accuracy_digit_total'],
                        'train_accuracy_color_total': train_metrics['accuracy_color_total'],
                        'train_accuracy_digit_noisy': train_metrics['accuracy_digit_noisy'],
                        'train_accuracy_color_noisy': train_metrics['accuracy_color_noisy'],
                        'train_accuracy_digit_clean': train_metrics['accuracy_digit_clean'],
                        'train_accuracy_color_clean': train_metrics['accuracy_color_clean'],
                        'test_loss': test_metrics['avg_loss'],
                        'test_accuracy': test_metrics['accuracy_total'],
                        'test_accuracy_digit_total': test_metrics['accuracy_digit_total'],
                        'test_accuracy_color_total': test_metrics['accuracy_color_total']
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

if __name__ == '__main__':
    main()
