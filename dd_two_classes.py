# combineのラベルノイズに関して、色・数字のラベル両方異なるラベルにする
# accuracyに関して、combineの正解率だけでなく、色・数字の正解率も出力する
# ラベルノイズの正解率とラベルノイズでない正解率を出力する
# 平均・分散も出力する

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
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
from model.cnn_5layers import CNN5Layer
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
    #set seed
    arg_parser.add_argument("-seed", "--fix_seed", type=int, default=42)
    
    #set model settings
    arg_parser.add_argument("--model", type=str, choices=["cnn_2layers", "cnn_5layers", "resnet18"], help="モデルアーキテクチャの選択")
    arg_parser.add_argument("-model_width", "--model_width", type=int, default=1)
    arg_parser.add_argument("-epoch", "--epoch", type=int, default=1000)
    
    #set dataset setting
    arg_parser.add_argument("-datasets", "--dataset", type=str, choices=["mnist", "emnist", "emnist_digits", "cifar10", "cifar100", "tinyImageNet", "colored_emnist", "distribution_colored_emnist"], default="cifar10")
    arg_parser.add_argument("-variance", "--variance", type=int, default=10000)
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
    
    #set loss function setting
    arg_parser.add_argument("-loss", "--loss", type=str, choices=["cross_entropy", "focal_loss"], default="cross_entropy", help="損失関数")
    
    # set device setting
    arg_parser.add_argument("-gpu", "--gpu", type=int, default=0, help="GPU device ID")
    arg_parser.add_argument("-num_workers", "--num_workers", type=int, default=4, help="データローダーの並列数")
    
    # wandb setting
    arg_parser.add_argument("-wandb", "--wandb", action='store_true', help="wandbを使用するかどうか")
    arg_parser.add_argument("-wandb_project", "--wandb_project", type=str, default="dd_scratch_models", help="wandbのプロジェクト名")
    arg_parser.add_argument("--wandb_entity", type=str, default="dsml-kernel24", help="wandbのエンティティ名")
    
    
    return arg_parser.parse_args()

# set seeds
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# set device
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
    torch.cuda.empty_cache()  # CUDAキャッシュのクリア
    gc.collect()  # ガベージコレクションの強制実行

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
        dataset (str): The name of the dataset to load. Supported options are "mnist", "emnist", "cifar10", "cifar100", and "tinyImageNet".
        gray_scale (bool): Flag indicating whether to convert the images to grayscale.
        
    Returns:
        tuple: A tuple containing the train dataset, test dataset, image size, and number of classes.
    """
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.resize((32, 32)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        imagesize = (32, 32)
        num_classes = 10
        in_channels = 1
    elif dataset == "emnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.resize((32, 32)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
        test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)
        imagesize = (32, 32)
        num_classes = 47
        in_channels = 1
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
        # 変換関数が必要な場合はここで定義
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((32, 32)),  # Same size as original, adjust if needed
            transforms.ToTensor()
        ])
    elif dataset == "emnist_digits_two_classes":
        emnist_path = './data/EMNIST'
        
        x_train = load_gz_file(os.path.join(emnist_path, 'emnist-digits-train-images-idx3-ubyte.gz'))
        y_train = load_gz_file(os.path.join(emnist_path, 'emnist-digits-train-labels-idx1-ubyte.gz'), is_image=False)
        x_test = load_gz_file(os.path.join(emnist_path, 'emnist-digits-test-images-idx3-ubyte.gz'))
        y_test = load_gz_file(os.path.join(emnist_path, 'emnist-digits-test-labels-idx1-ubyte.gz'), is_image=False)

        # ランダムに2つのクラスを選択
        classes = list(range(10))
        selected_classes = random.sample(classes, 2)
        print(f"Selected classes: {selected_classes}")

        # 選択したクラスのデータのみをフィルタリング
        train_mask = np.isin(y_train, selected_classes)
        test_mask = np.isin(y_test, selected_classes)

        x_train = x_train[train_mask]
        y_train = y_train[train_mask]
        x_test = x_test[test_mask]
        y_test = y_test[test_mask]

        # ラベルを0と1に変換
        y_train = np.where(y_train == selected_classes[0], 0, 1)
        y_test = np.where(y_test == selected_classes[0], 0, 1)

        # 変換関数
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((32, 32)),  # Resize to 32x32
            transforms.ToTensor()
        ])

        # Apply transformation
        x_train_tensor = apply_transform(x_train, transform)
        x_test_tensor = apply_transform(x_test, transform)

        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

        num_classes = 2  # Only two classes for binary classification
        in_channels = 1  # Grayscale images
        imagesize = (32, 32)  # Image size after transformation

        # Apply transformation
        x_train_tensor = apply_transform(x_train, transform)
        x_test_tensor = apply_transform(x_test, transform)

        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

    elif dataset == "colored_emnist":
        # target: color or digit or combined
        
        # Data augmentation
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        if target == 'color':
            x_train = np.load('data/colored_EMNIST/x_train_colored.npy')
            y_train_colors = np.load('data/colored_EMNIST/y_train_colors.npy')
            x_test = np.load('data/colored_EMNIST/x_test_colored.npy')
            y_test_colors = np.load('data/colored_EMNIST/y_test_colors.npy')
            
            # Apply transformation
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
            
            # Apply transformation
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
            
            # Apply transformation
            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            
            y_train_tensor = torch.tensor(y_train_combined, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_combined, dtype=torch.long)
            
            train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
            
        num_classes = 10 if target in ['color', 'digit'] else 100
        in_channels = 3
        imagesize = (32, 32)
    elif dataset == "distribution_colored_emnist":
        # target: color or digit or combined
        seed = args.fix_seed
        variance = args.variance
        correlation = args.correlation
        # Data augmentation
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        if target == 'color':
            x_train = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/x_train_colored.npy')
            y_train_colors = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/y_train_colors.npy')
            x_test = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/x_test_colored.npy')
            y_test_colors = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/y_test_colors.npy')
            
            # Apply transformation
            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            
            y_train_tensor = torch.tensor(y_train_colors, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_colors, dtype=torch.long)
            
            train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
        
        elif target == 'digit':
            x_train = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/x_train_colored.npy')
            y_train_digits = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/y_train_digits.npy')
            x_test = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/x_test_colored.npy')
            y_test_digits = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/y_test_digits.npy')
            
            # Apply transformation
            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            
            y_train_tensor = torch.tensor(y_train_digits, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_digits, dtype=torch.long)
            
            train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
            
        elif target == 'combined':
            x_train = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/x_train_colored.npy')
            y_train_combined = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/y_train_combined.npy')
            x_test = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/x_test_colored.npy')
            y_test_combined = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/y_test_combined.npy')
            
            # Apply transformation
            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            
            y_train_tensor = torch.tensor(y_train_combined, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_combined, dtype=torch.long)
            
            train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
    
        num_classes = 10 if target in ['color', 'digit'] else 100
        in_channels = 3
        imagesize = (32, 32)
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
    else:
        raise ValueError("Invalid dataset name")

    if gray_scale:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset.transform = transform
        test_dataset.transform = transform  

    return train_dataset, test_dataset, imagesize, num_classes, in_channels

def load_models(in_channels, args, img_size, num_classes):
    if args.model == "cnn_2layers":
        model = CNN2Layer(in_channels, num_classes, args.model_width, img_size)
    elif args.model == "cnn_5layers":
        model = CNN5Layer(in_channels, num_classes, args.model_width, img_size)
    elif args.model == "resnet18":
        model = models.resnet18(num_classes=num_classes)
    else:
        raise ValueError("Invalid model name.")
    
    return model

def add_label_noise(targets, label_noise_rate, num_digits, num_colors):
    noisy_targets = targets.clone()
    num_noisy = int(label_noise_rate * len(targets))
    noisy_indices = torch.randperm(len(targets))[:num_noisy]
    noise_info = torch.zeros(len(targets), dtype=torch.int)  # ノイズ情報を保存する列
    
    if num_digits == 10 and num_colors == 1:
        for idx in noisy_indices:
            original_label = targets[idx].item()
            new_label = random.randint(0, num_digits - 1)
            while new_label == original_label:
                new_label = random.randint(0, num_digits - 1)
            noisy_targets[idx] = new_label
            noise_info[idx] = 1
    
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
            noise_info[idx] = 1  # ノイズがついたことを示す

    return noisy_targets, noise_info

def train_model(model, train_loader, optimizer, criterion, device, noise_info):
    model.train()
    running_loss = 0.0
    total_samples = 0
    correct_total = 0

    correct_noisy = 0
    total_noisy = 0
    correct_clean = 0
    total_clean = 0

    running_loss_noisy = 0.0
    running_loss_clean = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_total += (predicted == labels).sum().item()

        for idx, pred in enumerate(predicted):
            if noise_info[labels[idx]] == 1:  # ノイズあり
                total_noisy += 1
                running_loss_noisy += criterion(outputs[idx].unsqueeze(0), labels[idx].unsqueeze(0)).item()
                if pred == labels[idx]:
                    correct_noisy += 1
            else:  # ノイズなし
                total_clean += 1
                running_loss_clean += criterion(outputs[idx].unsqueeze(0), labels[idx].unsqueeze(0)).item()
                if pred == labels[idx]:
                    correct_clean += 1

    avg_loss = running_loss / len(train_loader)
    avg_loss_noisy = running_loss_noisy / total_noisy if total_noisy > 0 else float('nan')
    avg_loss_clean = running_loss_clean / total_clean if total_clean > 0 else float('nan')
    accuracy_total = 100. * correct_total / total_samples
    accuracy_noisy = 100. * correct_noisy / total_noisy if total_noisy > 0 else float('nan')
    accuracy_clean = 100. * correct_clean / total_clean if total_clean > 0 else float('nan')

    return avg_loss, accuracy_total, accuracy_noisy, accuracy_clean, avg_loss_noisy, avg_loss_clean, \
        total_samples, total_noisy, total_clean, correct_total, correct_noisy, correct_clean
        
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    # output_means = []
    # output_variances = []
    # softmax_output_means = []
    # softmax_output_variances = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Compute means and variances
            # output_means.append(outputs.mean(dim=0))
            # output_variances.append(outputs.var(dim=0))
            # softmax_outputs = f.softmax(outputs, dim=1)
            # softmax_output_means.append(softmax_outputs.mean(dim=0))
            # softmax_output_variances.append(softmax_outputs.var(dim=0))

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total

    # avg_output_mean = torch.stack(output_means).mean(0)
    # avg_output_variance = torch.stack(output_variances).mean(0)
    # avg_softmax_output_mean = torch.stack(softmax_output_means).mean(0)
    # avg_softmax_output_variance = torch.stack(softmax_output_variances).mean(0)

    return \
        avg_loss, accuracy, total, correct
            # avg_output_mean, avg_output_variance, avg_softmax_output_mean, avg_softmax_output_variance

def main():
    print('start session')
    
    args = parse_args()
    # set seed
    set_seed(args.fix_seed)
    
    # set device
    num_gpus = torch.cuda.device_count()

    # GPUが利用可能かどうかをチェック
    if torch.cuda.is_available() and num_gpus > 0:
        device = torch.device("cuda")
        device_ids = list(range(num_gpus))  # 全GPUのIDリスト
    else:
        device = torch.device("cpu")
        device_ids = []
    
    # load datasets
    print('loading datasets')
    train_dataset, test_dataset, imagesize, num_classes, in_channels = load_datasets(args.dataset, args.target, args.gray_scale, args)
    
    # Number of digit and color classes
    num_digits = 10
    num_colors = 1
    
    # ラベルノイズの設定
    if args.label_noise_rate > 0.0:
        if args.dataset in ["colored_emnist", "distribution_colored_emnist", "emnist_digits"]:
            print('adding label noise')
            x_train, y_train = train_dataset.tensors
            y_train_noisy, noise_info = add_label_noise(y_train, args.label_noise_rate, num_digits, num_colors)
            train_dataset = torch.utils.data.TensorDataset(x_train, y_train_noisy)
        else:
            print('adding label noise')
            y_train_noisy, noise_info = add_label_noise(torch.tensor(train_dataset.targets), args.label_noise_rate, num_digits, num_colors)
            train_dataset.targets = y_train_noisy.numpy()  # 更新されたターゲットを設定

    # データローダーの設定
    print('setting data loader')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # load model
    model = load_models(in_channels, args, imagesize, num_classes)
    model = model.to(device)  
    # if device_ids:
    #     # 複数のGPUを使用してモデルをラップ
    #     model = nn.DataParallel(model, device_ids=device_ids).to(device)
    # else:
    #     model = model.to(device)
    
    # set optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) if args.optimizer == "sgd" else optim.Adam(model.parameters(), lr=args.lr)
    
    # set loss function
    criterion = nn.CrossEntropyLoss()
    
    # set experiment name
    experiment_name = f'{args.model}_{args.dataset}_{args.target}_lr{args.lr}_batch{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}_Optim{args.optimizer}'
    print(f'Experiment name: {experiment_name}')
    # set wandb
    if args.wandb:
        wandb.init(project=args.wandb_project, name=experiment_name, entity=args.wandb_entity)
        wandb.config.update(args)
    
    # CSVファイルの保存先を指定
    csv_dir = f"./csv/combine/split_noise/{experiment_name}"  # experiment_nameに適切な名前を入れてください
    os.makedirs(csv_dir, exist_ok=True)  # ディレクトリが存在しない場合は作成
    csv_path = os.path.join(csv_dir, 'log.csv')
    # 初回の場合はヘッダーを書き込み
    if not os.path.isfile(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_accuracy", "train_accuracy_noisy", "train_accuracy_clean", "test_loss", "test_accuracy"])    
            # "train_output_mean", "train_output_variance", "train_softmax_output_mean", "train_softmax_output_variance", \
            #             "test_output_mean", "test_output_variance", "test_softmax_output_mean", "test_softmax_output_variance"
 
    # train and test the model
    print('start training and testing')
    for epoch in range(1, args.epoch + 1):
        avg_loss, train_accuracy, train_accuracy_noisy, train_accuracy_clean, avg_loss_noisy, avg_loss_clean, \
                total_samples, total_noisy, total_clean, correct_total, correct_noisy, correct_clean = train_model(model, train_loader, optimizer, criterion, device, noise_info) 
        # train_mean, train_variance, train_softmax_mean, train_softmax_variance
        test_loss, test_accuracy, \
            test_total_samples, test_correct = test_model(model, test_loader, criterion, device) #test_mean, test_variance, test_softmax_mean, test_softmax_variance 

        print(f"Epoch: {epoch}, Train Loss: {avg_loss:.4f}, Overall Accuracy: {train_accuracy:.2f}%, Noisy Accuracy: {train_accuracy_noisy:.2f}%, Clean Accuracy: {train_accuracy_clean:.2f}%")
        print(f"Total Samples: {total_samples}, Total Noisy: {total_noisy}, Total Clean: {total_clean}")
        # print(f"Train Output Mean: {train_mean}, Train Output Variance: {train_variance}, Train Softmax Output Mean: {train_softmax_mean}, Train Softmax Output Variance: {train_softmax_variance}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        print(f"Total: {test_total_samples}, Correct: {test_correct}")
        # print(f"Test Output Mean: {test_mean}, Test Output Variance: {test_variance}, Test Softmax Output Mean: {test_softmax_mean}, Test Softmax Output Variance: {test_softmax_variance}")

        # CSVファイルにログを追加
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, train_accuracy, train_accuracy_noisy, train_accuracy_clean, avg_loss_noisy, avg_loss_clean, \
                test_loss, test_accuracy])
                    # train_mean, train_variance, train_softmax_mean, train_softmax_variance, \
                    #     test_mean, test_variance, test_softmax_mean, test_softmax_variance])

        # wandbにログを追加
        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_loss,
                'train_accuracy': train_accuracy,
                'train_accuracy_noisy': train_accuracy_noisy,
                'train_accuracy_clean': train_accuracy_clean,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'total_samples': total_samples,
                'total_noisy': total_noisy,
                'total_clean': total_clean,
                'correct_total': correct_total,
                'correct_noisy': correct_noisy,
                'correct_clean': correct_clean,
                'test_total_samples': test_total_samples,
                'test_correct': test_correct,
                'avg_loss_noisy': avg_loss_noisy,
                'avg_loss_clean': avg_loss_clean
                # 'train_output_mean': train_mean,
                # 'train_output_variance': train_variance,
                # 'train_softmax_output_mean': train_softmax_mean,
                # 'train_softmax_output_variance': train_softmax_variance,
                # 'test_output_mean': test_mean,
                # 'test_output_variance': test_variance,
                # 'test_softmax_output_mean': test_softmax_mean,
                # 'test_softmax_output_variance': test_softmax_variance
            })
        # clear_memory()
        # if args.save_model:
        #     if epoch & (epoch - 1) == 0:
        #         os.makedirs(f"./model_weights/split_noise/{experiment_name}", exist_ok=True)
        #         torch.save(model.state_dict(), f"./model_weights/split_noise/{experiment_name}/{epoch}.pth")

    wandb.finish()

main()
