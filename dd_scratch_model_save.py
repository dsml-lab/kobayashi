from torch.utils.data import DataLoader, TensorDataset

import wandb


# --- main() 関数 ---
import os
import csv
import math
import torch
import warnings
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import traceback

from config import parse_args_model_save
from utils import set_seed, set_device,seed_worker,compute_fraction_of_loss_reduction_from_batch
from datasets import load_or_create_noisy_dataset, NoisyDataset
from models import load_models
from logger import setup_wandb, log_to_wandb
from dispatch_func import train_model_dispatch,train_model_0_epoch_dispatch,test_model_dispatch



from torch.utils.data import DataLoader, TensorDataset

import wandb



def save_metrics(epoch, train_metrics, test_metrics, args, csv_path, wandb_run=None):
    if args.dataset == "distribution_colored_emnist":
        _save_metrics_distr_colored(epoch, train_metrics, test_metrics, csv_path, wandb_run)
    else:
        _save_metrics_standard(epoch, train_metrics, test_metrics, csv_path, wandb_run)


def _save_metrics_distr_colored(epoch, train_metrics, test_metrics, csv_path, wandb_run):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "epoch", "train_loss", "train_accuracy", "train_accuracy_noisy", "train_accuracy_clean",
                "avg_loss_noisy", "avg_loss_clean", "total_samples", "total_noisy", "total_clean",
                "correct_total", "correct_noisy", "correct_clean", "train_error_total",
                "train_error_noisy", "train_error_clean", "test_loss", "test_accuracy",
                "test_total_samples", "test_correct", "test_error"
            ])

        writer.writerow([
            epoch,
            train_metrics["avg_loss"],
            train_metrics["accuracy_total"],
            train_metrics["accuracy_noisy"],
            train_metrics["accuracy_clean"],
            train_metrics["avg_loss_noisy"],
            train_metrics["avg_loss_clean"],
            train_metrics["total_samples"],
            train_metrics["total_noisy"],
            train_metrics["total_clean"],
            train_metrics["correct_total"],
            train_metrics["correct_noisy"],
            train_metrics["correct_clean"],
            100. - train_metrics["accuracy_total"],
            100. - train_metrics["accuracy_noisy"],
            100. - train_metrics["accuracy_clean"],
            test_metrics["avg_loss"],
            test_metrics["accuracy_total"],
            test_metrics["total_samples"],
            test_metrics["correct"],
            test_metrics["error"]
        ])

    if wandb_run:
        log_data = {
            "epoch": epoch,
            **train_metrics,
            **test_metrics
        }
        wandb_run.log(log_data)

def _save_metrics_standard(epoch, train_metrics, test_metrics, csv_path, wandb_run):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "epoch", "train_loss", "train_accuracy", "train_accuracy_noisy", "train_accuracy_clean",
                "avg_loss_noisy", "avg_loss_clean",
                "total_samples", "total_noisy", "total_clean",
                "correct_total", "correct_noisy", "correct_clean", "train_error_total",
                "train_error_noisy", "train_error_clean", "test_loss", "test_accuracy",
                "test_total_samples", "test_correct", "test_error"
            ])

        writer.writerow([
            epoch,
            train_metrics["avg_loss"],
            train_metrics["train_accuracy"],
            train_metrics["train_accuracy_noisy"],
            train_metrics["train_accuracy_clean"],
            train_metrics["avg_loss_noisy"],
            train_metrics["avg_loss_clean"],
            train_metrics["total_samples"],
            train_metrics["total_noisy"],
            train_metrics["total_clean"],
            train_metrics["correct_total"],
            train_metrics["correct_noisy"],
            train_metrics["correct_clean"],
            train_metrics["train_error_total"],
            train_metrics["train_error_noisy"],
            train_metrics["train_error_clean"],
            test_metrics["avg_loss"],
            test_metrics["test_accuracy"],
            test_metrics["test_total_samples"],
            test_metrics["test_correct"],
            test_metrics["test_error"]
        ])
    # ここでwithを抜けるとファイルが閉じる

    # wandbへのログ送信はここでOK
    if wandb_run:
        log_data = {
            "epoch": epoch,
            **train_metrics,
            **test_metrics
        }
        wandb_run.log(log_data)
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


def main():
    print('Start session')
    wandb_run = None

    warnings.filterwarnings("ignore")

    try:
        args = parse_args_model_save()
        set_seed(args.fix_seed)

        device = set_device(args.gpu)
        print(f'Using device: {device}')
        print(f'Using target: {args.target}')
        print('Creating noisy datasets...')
        train_dataset, test_dataset, meta = load_or_create_noisy_dataset(
            args.dataset, args.target, args.gray_scale, args, return_type="torch"
        )

        imagesize = meta["imagesize"]
        num_classes = meta["num_classes"]
        in_channels = meta["in_channels"]
        if isinstance(train_dataset, NoisyDataset):
            print('NoisyDataset detected')
            x_train_noisy, y_train_noisy = get_tensor_dataset_components(train_dataset)
            noise_info = train_dataset.noise_info
        else:
            print('TensorDataset detected')
            x_train_noisy, y_train_noisy = get_tensor_dataset_components(train_dataset)
            noise_info = None
        print(noise_info)
        if args.target == 'combined':
            num_digits = 10
            num_colors = 10
        else:
            num_digits = 10
            num_colors = 1

        if noise_info is not None:
            clean_indices = [i for i, label in enumerate(noise_info) if label == 0]
            noisy_indices = [i for i, label in enumerate(noise_info) if label == 1]
        else:
            clean_indices = None
            noisy_indices = None

        print('Initializing model...')
        model = load_models(in_channels, args, imagesize, num_classes).to(device)

        print('Setting up optimizer...')
        if args.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

        criterion_noisy = nn.CrossEntropyLoss(reduction='none')
        criterion_clean = nn.CrossEntropyLoss(reduction='none')
        if args.wandb:
            print('Initializing wandb...')
            experiment_name = f'test_seed_{args.fix_seed}width{args.model_width}_{args.model}_{args.dataset}_variance{args.variance}_{args.target}_lr{args.lr}_batch{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}_Optim{args.optimizer}_Momentum{args.momentum}'
            wandb_run = setup_wandb(args, experiment_name)
        else:
            experiment_name = f'seed_{args.fix_seed}width{args.model_width}_{args.model}_{args.dataset}_variance{args.variance}_{args.target}_lr{args.lr}_batch{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}_Optim{args.optimizer}_Momentum{args.momentum}'

        base_save_dir = f"save_model/{args.dataset}/noise_{args.label_noise_rate}/{experiment_name}"
        csv_path = os.path.join(base_save_dir, "csv", "training_metrics.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        g = torch.Generator()
        g.manual_seed(args.fix_seed)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

        test_loader = DataLoader(
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
            "train_accuracy": None,
            "test_accuracy": None
        }

        train_metrics0 = train_model_0_epoch_dispatch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion_noisy=criterion_noisy,
            criterion_clean=criterion_clean,
            args=args,
            weight_noisy=args.weight_noisy,
            weight_clean=args.weight_clean,
            device=device,
            num_colors=num_colors,
            num_digits=num_digits
        )

        test_metrics0 = test_model_dispatch(
            model=model,
            test_loader=test_loader,
            args=args,
            device=device,
            num_colors=num_colors,
            num_digits=num_digits
        )

        save_metrics(0, train_metrics0, test_metrics0, args, csv_path, wandb_run)
        torch.save(save_dict, os.path.join(base_save_dir, "model_epoch_0.pth"))

        for epoch in range(1, args.epoch + 1):
            try:
                train_metrics = train_model_dispatch(
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    criterion_noisy=criterion_noisy,
                    criterion_clean=criterion_clean,
                    weight_noisy=args.weight_noisy,
                    weight_clean=args.weight_clean,
                    args=args,
                    device=device,
                    num_colors=num_colors,
                    num_digits=num_digits,
                    epoch_batch=3,                         # ← 追加！
                    experiment_name=experiment_name )

                test_metrics = test_model_dispatch(
                    model=model,
                    test_loader=test_loader,
                    args=args,
                    device=device,
                    num_colors=num_colors,
                    num_digits=num_digits
                )

                save_dict = {
                    "args": vars(args),
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "train_accuracy": train_metrics["train_accuracy"] if "train_accuracy" in train_metrics else train_metrics.get("accuracy_total"),
                    "test_accuracy": test_metrics["test_accuracy"] if "test_accuracy" in test_metrics else test_metrics.get("accuracy_total")
                }

                torch.save(save_dict, os.path.join(base_save_dir, f"model_epoch_{epoch}.pth"))
                save_metrics(epoch, train_metrics, test_metrics, args, csv_path, wandb_run)

            except Exception as e:
                print(f"Error in epoch {epoch}: {str(e)}")
                continue

    except Exception as e:

        traceback.print_exc()
        print(f"Error in main: {e}")
        return
    finally:
        if wandb_run:
            wandb_run.finish()
        print('Training completed')

if __name__ == '__main__':
    main()
