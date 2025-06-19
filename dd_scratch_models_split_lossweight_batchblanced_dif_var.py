# combineのラベルノイズに関して、色・数字のラベル両方異なるラベルにする
# accuracyに関して、combineの正解率だけでなく、色・数字の正解率も出力する
# ラベルノイズの正解率とラベルノイズでない正解率を出力する
# 平均・分散も出力する
# 重みの加え方が均等

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import math
import torchvision.models as models
import numpy as np
import time
import wandb
import argparse
import random
import matplotlib.pyplot as plt
import csv 
import warnings
import gc
import gzip
# import models written by scratch
from model.cnn_2layers import CNN2Layer
from model.cnn_3layers import CNN3Layer
from model.cnn_4layers import CNN4Layer
from model.cnn_5layers import CNN5Layer
from model.cnn_8layers import CNN8Layer
from model.cnn_16layers import CNN16Layer
from model.resnet18 import ResNet18
# download datasets from pytorch
from torchvision import datasets

# Ignore warnings
warnings.filterwarnings("ignore")
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.backends.cudnn.benchmark = True

# settings
def parse_args():
    """
    Parse the command-line arguments.
    
    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    arg_parser = argparse.ArgumentParser()
    # set seed
    arg_parser.add_argument("-seed", "--fix_seed", type=int, default=42)
    
    # set model settings
    arg_parser.add_argument("--model", type=str, choices=["cnn_2layers", "cnn_3layers", "cnn_4layers", "cnn_5layers", "cnn_8layers", "cnn_16layers", "resnet18"], help="モデルアーキテクチャの選択")
    arg_parser.add_argument("-model_width", "--model_width", type=int, default=1)
    arg_parser.add_argument("-epoch", "--epoch", type=int, default=1000)
    
    # set dataset setting
    arg_parser.add_argument("-datasets", "--dataset", type=str, choices=["mnist", "emnist", "emnist_digits", "cifar10", "cifar100", "tinyImageNet", 
                                                                         "colored_emnist", "distribution_colored_emnist"],
                            default="cifar10")
    # --------------------------------------------------------------------------------
    # ここを変更: train と test で別々に variance を指定できる引数を追加
    arg_parser.add_argument("-train_variance", "--train_variance", type=int, default=0,
                            help="Train データで使用する分散(0なら colored_EMNIST)")
    arg_parser.add_argument("-test_variance", "--test_variance", type=int, default=0,
                            help="Test データで使用する分散(0なら colored_EMNIST)")
    # --------------------------------------------------------------------------------

    arg_parser.add_argument("-correlation", "--correlation", type=float, default=0.5)
    arg_parser.add_argument("-label_noise_rate", "--label_noise_rate", type=float, default=0.0) 
    arg_parser.add_argument("-gray_scale", "--gray_scale", action='store_true', help="グレースケールに変換するかどうか")
    arg_parser.add_argument("-batch_size", "--batch_size", type=int, default=128, help="バッチサイズ")
    arg_parser.add_argument("-img_size", "--img_size", type=int, default=32, help="画像サイズ")
    arg_parser.add_argument("-target", "--target", type=str, choices=["color", "digit", "combined"], default='color', help="colored EMNISTのターゲットの指定:color or digit or combined")
    
    # set optimizer setting
    arg_parser.add_argument("-lr", "--lr", type=float, default=0.1, help="学習率")
    arg_parser.add_argument("-optimizer", "--optimizer", type=str, choices=["sgd", "adam"], default="adam", help="最適化手法.adam were used in Nakkiran et al. (2019)")
    arg_parser.add_argument("-momentum", "--momentum", type=float, default=0.9, help="モーメンタム")
    
    # set loss function setting
    arg_parser.add_argument("-loss", "--loss", type=str, choices=["cross_entropy", "focal_loss"], default="cross_entropy", help="損失関数")
    
    # set device setting
    arg_parser.add_argument("-gpu", "--gpu", type=int, default=0, help="GPU device ID")
    arg_parser.add_argument("-num_workers", "--num_workers", type=int, default=4, help="データローダーの並列数")
    
    # wandb setting
    arg_parser.add_argument("-wandb", "--wandb", action='store_true', default=True ,help="wandbを使用するかどうか")
    arg_parser.add_argument("-wandb_project", "--wandb_project", type=str, default="dd_scratch_models", help="wandbのプロジェクト名")
    arg_parser.add_argument("--wandb_entity", type=str, default="dsml-kernel24", help="wandbのエンティティ名")
    
    arg_parser.add_argument("-weight_noisy", "--weight_noisy", type=float, default=1.0, help="Weight for the loss of noisy samples")
    arg_parser.add_argument("-weight_clean", "--weight_clean", type=float, default=1.0, help="Weight for the loss of clean samples")

    return arg_parser.parse_args()

# Define a custom dataset that includes noise_info
class NoisyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, noise_info):
        self.dataset = dataset
        self.noise_info = noise_info

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input, label = self.dataset[idx]
        noise_label = self.noise_info[idx]
        return input, label, noise_label

# Custom sampler to create balanced batches
class BalancedBatchSampler(Sampler):
    def __init__(self, clean_indices, noisy_indices, batch_size, drop_last):
        self.clean_indices = clean_indices
        self.noisy_indices = noisy_indices
        self.batch_size = batch_size
        self.drop_last = drop_last

        assert batch_size % 2 == 0, "Batch size must be even for balanced batches"
        self.num_samples_per_class = batch_size // 2

    def __iter__(self):
        # Shuffle the indices
        random.shuffle(self.clean_indices)
        random.shuffle(self.noisy_indices)

        # Calculate the number of batches
        min_len = min(len(self.clean_indices), len(self.noisy_indices))
        num_batches = min_len // self.num_samples_per_class

        for i in range(num_batches):
            clean_batch = self.clean_indices[i * self.num_samples_per_class: (i + 1) * self.num_samples_per_class]
            noisy_batch = self.noisy_indices[i * self.num_samples_per_class: (i + 1) * self.num_samples_per_class]
            batch = clean_batch + noisy_batch
            random.shuffle(batch)
            yield batch

        if not self.drop_last:
            # Handle remaining samples
            remaining_clean = self.clean_indices[num_batches * self.num_samples_per_class:]
            remaining_noisy = self.noisy_indices[num_batches * self.num_samples_per_class:]

            if len(remaining_clean) >= self.num_samples_per_class and len(remaining_noisy) >= self.num_samples_per_class:
                batch = remaining_clean[:self.num_samples_per_class] + remaining_noisy[:self.num_samples_per_class]
                random.shuffle(batch)
                yield batch

    def __len__(self):
        return len(self.clean_indices) // self.num_samples_per_class

# Set seeds
def set_seed(seed):
    """
    Set the seed for reproducibility.

    Args:
        seed (int): The seed value to set.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For GPU determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set device
def set_device(gpu_id):
    """
    Sets the device for computation.

    Args:
        gpu_id (int): The ID of the GPU to use.

    Returns:
        torch.device: The selected device (GPU or CPU).
    """
    # Choose the GPU device if available, otherwise use CPU
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    return device

def clear_memory():
    torch.cuda.empty_cache()  # Clear CUDA cache
    gc.collect()  # Force garbage collection

def apply_transform(x, transform):
    transformed_x = []
    for img in x:
        img = transform(img)
        transformed_x.append(img)
    return torch.stack(transformed_x)

def load_datasets(dataset, target, gray_scale, args):
    """
    Load the specified dataset and apply transformations based on the dataset type and grayscale option.
    
    Args:
        dataset (str): The name of the dataset to load. Supported options are:
                       ["mnist", "emnist", "emnist_digits", "cifar10", "cifar100", "tinyImageNet",
                        "colored_emnist", "distribution_colored_emnist", "distribution_to_normal"] など
        target (str): "color", "digit", または "combined" (colored EMNIST のターゲット)
        gray_scale (bool): True の場合、グレースケールに変換
        args: コマンドライン引数などを含むパラメータ (train_variance, test_variance, correlation, fix_seed などを含む)
        
    Returns:
        tuple: (train_dataset, test_dataset, imagesize, num_classes, in_channels)
    """
    
    # ------------------------------------------------------------
    # 追加したパラメータをここで取得 (train 用と test 用で別々の variance)
    # ------------------------------------------------------------
    train_variance = args.train_variance
    test_variance = args.test_variance
    correlation = args.correlation
    seed = args.fix_seed
    
    # 変換共通部 (例: 32x32, Normalizeなど)
    default_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # ----------------------------------------------
    # MNIST
    # ----------------------------------------------
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        imagesize = (32, 32)
        num_classes = 10
        in_channels = 1

    # ----------------------------------------------
    # EMNIST
    # ----------------------------------------------
    elif dataset == "emnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
        test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)
        imagesize = (32, 32)
        num_classes = 47
        in_channels = 1

    # ----------------------------------------------
    # EMNIST (digits split)
    # ----------------------------------------------
    elif dataset == "emnist_digits":
        emnist_path = './data/EMNIST'
        
        def load_gz_file(file_path, is_image=True):
            with gzip.open(file_path, 'rb') as f:
                if is_image:
                    return np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
                else:
                    return np.frombuffer(f.read(), dtype=np.uint8, offset=8)

        x_train = load_gz_file(os.path.join(emnist_path, 'emnist-digits-train-images-idx3-ubyte.gz'))
        y_train = load_gz_file(os.path.join(emnist_path, 'emnist-digits-train-labels-idx1-ubyte.gz'), is_image=False)
        x_test = load_gz_file(os.path.join(emnist_path, 'emnist-digits-test-images-idx3-ubyte.gz'))
        y_test = load_gz_file(os.path.join(emnist_path, 'emnist-digits-test-labels-idx1-ubyte.gz'), is_image=False)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        x_train_tensor = apply_transform(x_train, transform)
        x_test_tensor = apply_transform(x_test, transform)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

        num_classes = 10
        in_channels = 1
        imagesize = (32, 32)

    # ----------------------------------------------
    # colored_EMNIST (通常版)
    # ----------------------------------------------
    elif dataset == "colored_emnist":
        # target: color or digit or combined
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        if target == 'color':
            x_train = np.load('data/colored_EMNIST/x_train_colored.npy')
            y_train_colors = np.load('data/colored_EMNIST/y_train_colors.npy')
            x_test = np.load('data/colored_EMNIST/x_test_colored.npy')
            y_test_colors = np.load('data/colored_EMNIST/y_test_colors.npy')

            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            y_train_tensor = torch.tensor(y_train_colors, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_colors, dtype=torch.long)

            train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

        elif target == 'digit':
            x_train = np.load('data/colored_EMNIST/x_train_colored.npy')
            y_train_digits = np.load('data/colored_EMNIST/y_train_digits.npy')
            x_test = np.load('data/colored_EMNIST/x_test_colored.npy')
            y_test_digits = np.load('data/colored_EMNIST/y_test_digits.npy')

            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            y_train_tensor = torch.tensor(y_train_digits, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_digits, dtype=torch.long)

            train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

        elif target == 'combined':
            x_train = np.load('data/colored_EMNIST/x_train_colored.npy')
            y_train_combined = np.load('data/colored_EMNIST/y_train_combined.npy')
            x_test = np.load('data/colored_EMNIST/x_test_colored.npy')
            y_test_combined = np.load('data/colored_EMNIST/y_test_combined.npy')

            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            y_train_tensor = torch.tensor(y_train_combined, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_combined, dtype=torch.long)

            train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

        num_classes = 10 if target in ['color', 'digit'] else 100
        in_channels = 3
        imagesize = (32, 32)

    # ----------------------------------------------
    # distribution_colored_emnist (分散を指定)
    # ----------------------------------------------
    elif dataset == "distribution_colored_emnist":
        # 追加のパラメータを使用:
        # train_variance / test_variance (0なら通常colored_EMNIST)
        # correlation, seed など
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # ターゲットに応じてクラス数を設定
        num_classes = 10 if target in ['color', 'digit'] else 100
        in_channels = 3
        imagesize = (32, 32)

        # まず train データ
        if train_variance == 0:
            # variance=0 → 通常の colored_EMNIST を読み込む
            x_train = np.load('data/colored_EMNIST/x_train_colored.npy')
            if target == 'color':
                y_train = np.load('data/colored_EMNIST/y_train_colors.npy')
            elif target == 'digit':
                y_train = np.load('data/colored_EMNIST/y_train_digits.npy')
            else:  # combined
                y_train = np.load('data/colored_EMNIST/y_train_combined.npy')
        else:
            # variance≠0 → distribution_colored_EMNIST から
            x_train = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{train_variance}_Corr{correlation}/x_train_colored.npy')
            if target == 'color':
                y_train = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{train_variance}_Corr{correlation}/y_train_colors.npy')
            elif target == 'digit':
                y_train = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{train_variance}_Corr{correlation}/y_train_digits.npy')
            else:  # combined
                y_train = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{train_variance}_Corr{correlation}/y_train_combined.npy')

        # 次に test データ
        if test_variance == 0:
            # variance=0 → 通常の colored_EMNIST を読み込む
            x_test = np.load('data/colored_EMNIST/x_test_colored.npy')
            if target == 'color':
                y_test = np.load('data/colored_EMNIST/y_test_colors.npy')
            elif target == 'digit':
                y_test = np.load('data/colored_EMNIST/y_test_digits.npy')
            else:  # combined
                y_test = np.load('data/colored_EMNIST/y_test_combined.npy')
        else:
            # variance≠0 → distribution_colored_EMNIST から
            x_test = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{test_variance}_Corr{correlation}/x_test_colored.npy')
            if target == 'color':
                y_test = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{test_variance}_Corr{correlation}/y_test_colors.npy')
            elif target == 'digit':
                y_test = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{test_variance}_Corr{correlation}/y_test_digits.npy')
            else:  # combined
                y_test = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{test_variance}_Corr{correlation}/y_test_combined.npy')

        # 変換を適用
        x_train_tensor = apply_transform(x_train, transform)
        x_test_tensor = apply_transform(x_test, transform)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

    # ----------------------------------------------
    # CIFAR-10
    # ----------------------------------------------
    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        imagesize = (32, 32)
        num_classes = 10
        in_channels = 3

    # ----------------------------------------------
    # CIFAR-100
    # ----------------------------------------------
    elif dataset == "cifar100":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        imagesize = (32, 32)
        num_classes = 100
        in_channels = 3

    # ----------------------------------------------
    # tinyImageNet
    # ----------------------------------------------
    elif dataset == "tinyImageNet":
        transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
        ])
        train_dataset = datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform)
        test_dataset = datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=transform)
        imagesize = (64, 64)
        num_classes = 200
        in_channels = 3

    # ----------------------------------------------
    # distribution_to_normal (一部の追加データセット例)
    # ----------------------------------------------
    elif dataset == "distribution_to_normal":
        # target: color or digit or combined (以下はオリジナルのサンプル実装)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if target == 'combined':
            x_train = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{args.variance}_Corr{correlation}/x_train_colored.npy')
            y_train_combined = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{args.variance}_Corr{correlation}/y_train_combined.npy')
            x_test = np.load('data/colored_EMNIST/x_test_colored.npy')
            y_test_combined = np.load('data/colored_EMNIST/y_test_combined.npy')

            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            y_train_tensor = torch.tensor(y_train_combined, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_combined, dtype=torch.long)

            train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

        num_classes = 10 if target in ['color', 'digit'] else 100
        in_channels = 3
        imagesize = (32, 32)

    else:
        raise ValueError("Invalid dataset name")

    # ----------------------------------------------
    # gray_scale が True の場合、グレースケール化のために transform を上書き
    # ----------------------------------------------
    if gray_scale:
        transform_gray = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # PyTorch datasets の場合、.transform に再代入
        # TensorDataset の場合は都度変換する必要があるが、ここでは簡易に対応
        if hasattr(train_dataset, 'transform'):
            train_dataset.transform = transform_gray
        if hasattr(test_dataset, 'transform'):
            test_dataset.transform = transform_gray

    return train_dataset, test_dataset, imagesize, num_classes, in_channels

def load_models(in_channels, args, img_size, num_classes):
    if args.model == "cnn_2layers":
        model = CNN2Layer(in_channels, num_classes, args.model_width, img_size)
    elif args.model == "cnn_3layers":
        model = CNN3Layer(in_channels, num_classes, args.model_width, img_size)
    elif args.model == "cnn_4layers":
        model = CNN4Layer(in_channels, num_classes, args.model_width, img_size)
    elif args.model == "cnn_5layers":
        model = CNN5Layer(in_channels, num_classes, args.model_width, img_size)
    elif args.model == "cnn_8layers":
        model = CNN8Layer(in_channels, num_classes, args.model_width, img_size)
    elif args.model == "cnn_16layers":
        model = CNN16Layer(in_channels, num_classes, args.model_width, img_size)
    elif args.model == "resnet18":
        model = models.resnet18(num_classes=num_classes)
    else:
        raise ValueError("Invalid model name.")
    return model

# Modify add_label_noise to return indices of noisy samples
def add_label_noise(targets, label_noise_rate, num_digits, num_colors):
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

def train_model(model, train_loader, optimizer, criterion, weight_noisy, weight_clean, device, num_colors, num_digits):
    """
    Training function with comprehensive metrics tracking.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        optimizer: The optimizer
        criterion: Loss function (expected to support reduction='none')
        weight_noisy: Weight for noisy samples
        weight_clean: Weight for clean samples
        device: Device to run the training on
        num_colors: Number of color classes
        num_digits: Number of digit classes
        
    Returns:
        dict: Dictionary containing all training metrics
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

    # Initialize counters for digits and colors
    correct_digit_total = 0
    correct_color_total = 0

    correct_digit_noisy = 0
    correct_color_noisy = 0
    correct_digit_clean = 0
    correct_color_clean = 0

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
                        # Obtain digit and color labels and predictions
            digit_labels = labels // num_colors
            color_labels = labels % num_colors
            digit_predictions = predicted // num_colors
            color_predictions = predicted % num_colors

            # Total correct counts for digits and colors
            correct_digit_total += (digit_predictions == digit_labels).sum().item()
            correct_color_total += (color_predictions == color_labels).sum().item()


            # Process noisy samples
            if num_noisy > 0:
                labels_noisy = labels[idx_noisy]
                predicted_noisy = predicted[idx_noisy]
                correct_noisy += (predicted_noisy == labels_noisy).sum().item()
                total_noisy += num_noisy

                # Update digit and color accuracy for noisy samples
                digit_labels_noisy = labels_noisy // num_colors
                color_labels_noisy = labels_noisy % num_colors
                digit_predictions_noisy = predicted_noisy // num_colors
                color_predictions_noisy = predicted_noisy % num_colors

                correct_digit_noisy += (digit_predictions_noisy == digit_labels_noisy).sum().item()
                correct_color_noisy += (color_predictions_noisy == color_labels_noisy).sum().item()

                # Store noisy sample losses
                loss_values_noisy.extend(per_sample_loss_weighted[idx_noisy].detach().cpu().numpy())

            # Process clean samples
            if num_clean > 0:
                labels_clean = labels[idx_clean]
                predicted_clean = predicted[idx_clean]
                correct_clean += (predicted_clean == labels_clean).sum().item()
                total_clean += num_clean

                # Update digit and color accuracy for clean samples
                digit_labels_clean = labels_clean // num_colors
                color_labels_clean = labels_clean % num_colors
                digit_predictions_clean = predicted_clean // num_colors
                color_predictions_clean = predicted_clean % num_colors

                correct_digit_clean += (digit_predictions_clean == digit_labels_clean).sum().item()
                correct_color_clean += (color_predictions_clean == color_labels_clean).sum().item()

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
        'accuracy_digit_total': 100. * correct_digit_total / total_samples if total_samples > 0 else float('nan'),
        'accuracy_color_total': 100. * correct_color_total / total_samples if total_samples > 0 else float('nan'),
        'accuracy_digit_noisy': 100. * correct_digit_noisy / total_noisy if total_noisy > 0 else float('nan'),
        'accuracy_color_noisy': 100. * correct_color_noisy / total_noisy if total_noisy > 0 else float('nan'),
        'accuracy_digit_clean': 100. * correct_digit_clean / total_clean if total_clean > 0 else float('nan'),
        'accuracy_color_clean': 100. * correct_color_clean / total_clean if total_clean > 0 else float('nan'),
        'total_samples': total_samples,
        'total_noisy': total_noisy,
        'total_clean': total_clean,
        'correct_total': correct_total,
        'correct_noisy': correct_noisy,
        'correct_clean': correct_clean
    }

    return metrics

def test_model(model, test_loader, device, num_colors, num_digits):
    model.eval()
    test_loss = 0
    correct_total = 0
    total_samples = 0

    correct_digit_total = 0
    correct_color_total = 0

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

            # Calculate correct counts for digits and colors
            digit_labels = labels // num_colors
            color_labels = labels % num_colors
            digit_predictions = predicted // num_colors
            color_predictions = predicted % num_colors

            correct_digit_total += (digit_predictions == digit_labels).sum().item()
            correct_color_total += (color_predictions == color_labels).sum().item()

    avg_loss = test_loss / total_samples if total_samples > 0 else float('nan')
    accuracy_total = 100. * correct_total / total_samples if total_samples > 0 else float('nan')
    accuracy_digit_total = 100. * correct_digit_total / total_samples if total_samples > 0 else float('nan')
    accuracy_color_total = 100. * correct_color_total / total_samples if total_samples > 0 else float('nan')

    return {
        'avg_loss': avg_loss,
        'accuracy_total': accuracy_total,
        'accuracy_digit_total': accuracy_digit_total,
        'accuracy_color_total': accuracy_color_total,
        'total_samples': total_samples,
        'correct_total': correct_total
    }
def main():
    """
    Main training loop with comprehensive error handling and logging
    """
    print('Start session')
    wandb_run = None
    
    try:
        # Parse arguments and set initial configurations
        args = parse_args()
        set_seed(args.fix_seed)
        
        # Set device
        device = set_device(args.gpu)
        print(f'Using device: {device}')

        # Load datasets with error handling
        print('Loading datasets...')
        try:
            train_dataset, test_dataset, imagesize, num_classes, in_channels = load_datasets(
                args.dataset, args.target, args.gray_scale, args)
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
                train_dataset = torch.utils.data.TensorDataset(x_train, y_train_noisy)
                train_dataset = NoisyDataset(train_dataset, noise_info)
            else:
                y_train = torch.tensor(train_dataset.targets)
                y_train_noisy, noise_info = add_label_noise(y_train, args.label_noise_rate, num_digits, num_colors)
                train_dataset.targets = y_train_noisy.tolist()
                train_dataset = NoisyDataset(train_dataset, noise_info)
        else:
            noise_info = torch.zeros(len(train_dataset), dtype=torch.int)
            train_dataset = NoisyDataset(train_dataset, noise_info)

        # Extract indices for clean and noisy samples
        clean_indices = [i for i, label in enumerate(noise_info) if label == 0]
        noisy_indices = [i for i, label in enumerate(noise_info) if label == 1]

        # Validate batch size
        if args.batch_size % 2 != 0:
            raise ValueError("Batch size must be even for balanced batches")

        # Create data loaders
        print('Setting up data loaders...')
        if args.label_noise_rate == 0.0 or args.label_noise_rate == 1.0:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True
            )
        else:
            batch_sampler = BalancedBatchSampler(
                clean_indices,
                noisy_indices,
                args.batch_size,
                drop_last=False
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
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

        # Set experiment name
        experiment_name = (
            f'Optim{args.optimizer}_momentum{args.momentum}{args.model}_{args.dataset}_{args.target}_'
            f'lr{args.lr}_batch{args.batch_size}_epoch{args.epoch}_'
            f'LabelNoiseRate{args.label_noise_rate}_Optim{args.optimizer}_'
            f'cleanw{args.weight_clean}_noisew{args.weight_noisy}_train_variance{args.train_variance}_test_varianc{args.test_variance}'
        )
        print(f'Experiment name: {experiment_name}')

        # Initialize wandb
        if args.wandb:
            print('Initializing wandb...')
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=experiment_name,
                entity=args.wandb_entity,
                config=args
            )

        # Set up CSV logging
        csv_dir = f"./csv/combine/split_noise/{experiment_name}"
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, 'log.csv')
        
        if not os.path.isfile(csv_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "train_loss", "train_loss_variance",
                    "train_accuracy", "train_accuracy_noisy", "train_accuracy_clean",
                    "train_accuracy_digit_total", "train_accuracy_color_total",
                    "train_accuracy_digit_noisy", "train_accuracy_color_noisy",
                    "train_accuracy_digit_clean", "train_accuracy_color_clean",
                    "test_loss", "test_accuracy",
                    "test_accuracy_digit_total", "test_accuracy_color_total"
                ])

        # Training loop
        print('Starting training...')
        best_test_accuracy = 0.0
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

                # Print progress
                print(f"\nEpoch: {epoch}/{args.epoch}")
                print(f"Train Loss: {train_metrics['avg_loss']:.4f}, "
                      f"Train Loss Variance: {train_metrics['var_loss']:.4f}")
                print(f"Train Accuracy: {train_metrics['accuracy_total']:.2f}%, "
                      f"Test Accuracy: {test_metrics['accuracy_total']:.2f}%")
                print(f"Train Digit/Color Accuracy: {train_metrics['accuracy_digit_total']:.2f}%/"
                      f"{train_metrics['accuracy_color_total']:.2f}%")
                print(f"Test Digit/Color Accuracy: {test_metrics['accuracy_digit_total']:.2f}%/"
                      f"{test_metrics['accuracy_color_total']:.2f}%")

                # Save to CSV
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch,
                        train_metrics['avg_loss'], train_metrics['var_loss'],
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

                # Log to wandb
                if args.wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_metrics['avg_loss'],
                        'train_loss_variance': train_metrics['var_loss'],
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
                    })

                # Memory management
                if epoch % 10 == 0:
                    clear_memory()

                # Save best model
                if test_metrics['accuracy_total'] > best_test_accuracy:
                    best_test_accuracy = test_metrics['accuracy_total']
                    torch.save(model.state_dict(), 
                             os.path.join(csv_dir, 'best_model.pth'))

            except Exception as e:
                print(f"Error in epoch {epoch}: {str(e)}")
                continue

    except Exception as e:
        print(f"Fatal error in training: {str(e)}")
        raise
    
    finally:
        # Cleanup
        if wandb_run is not None:
            wandb_run.finish()
        print('Training completed')

if __name__ == '__main__':
    main()