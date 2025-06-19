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
    arg_parser.add_argument("-wandb", "--wandb", action='store_true',default=True ,help="wandbを使用するかどうか")
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

        # Apply transformation
        x_train_tensor = apply_transform(x_train, transform)
        x_test_tensor = apply_transform(x_test, transform)

        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

        num_classes = 10  # Digits from 0 to 9
        in_channels = 1  # Grayscale images
        imagesize = (32, 32)  # Original image size
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

def train_model(model, train_loader, optimizer, criterion_noisy, criterion_clean, weight_noisy, weight_clean, device):
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

    # 損失関数を各サンプルの損失を返すように設定
    criterion_noisy.reduction = 'none'
    criterion_clean.reduction = 'none'

    for inputs, labels, noise_labels in train_loader:
        inputs, labels, noise_labels = inputs.to(device), labels.to(device), noise_labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_total += (predicted == labels).sum().item()

        # ノイズのあるサンプルとないサンプルのインデックスを取得
        idx_noisy = (noise_labels == 1)
        idx_clean = (noise_labels == 0)

        # 各サンプルの損失を初期化
        loss_per_sample = torch.zeros_like(labels, dtype=torch.float, device=device)

        # ノイズのあるサンプルの損失計算
        if idx_noisy.sum() > 0:
            outputs_noisy = outputs[idx_noisy]
            labels_noisy = labels[idx_noisy]
            per_sample_loss_noisy = criterion_noisy(outputs_noisy, labels_noisy)  # shape: [num_noisy_samples]
            # 重みを適用
            per_sample_loss_noisy_weighted = per_sample_loss_noisy * weight_noisy
            # 全体の損失に加算
            loss_per_sample[idx_noisy] = per_sample_loss_noisy_weighted

            running_loss_noisy += per_sample_loss_noisy_weighted.sum().item()
            correct_noisy += (predicted[idx_noisy] == labels_noisy).sum().item()
            total_noisy += idx_noisy.sum().item()

        # ノイズのないサンプルの損失計算
        if idx_clean.sum() > 0:
            outputs_clean = outputs[idx_clean]
            labels_clean = labels[idx_clean]
            per_sample_loss_clean = criterion_clean(outputs_clean, labels_clean)  # shape: [num_clean_samples]
            # 重みを適用
            per_sample_loss_clean_weighted = per_sample_loss_clean * weight_clean
            # 全体の損失に加算
            loss_per_sample[idx_clean] = per_sample_loss_clean_weighted

            running_loss_clean += per_sample_loss_clean_weighted.sum().item()
            correct_clean += (predicted[idx_clean] == labels_clean).sum().item()
            total_clean += idx_clean.sum().item()

        # バッチ内の全サンプルの平均損失を計算
        loss = loss_per_sample.mean()

        loss.backward()
        optimizer.step()

        # バッチの総損失を累積
        running_loss += loss.item() * inputs.size(0)

    # エポック全体の平均損失と正解率を計算
    avg_loss = running_loss / total_samples
    avg_loss_noisy = running_loss_noisy / total_noisy if total_noisy > 0 else float('nan')
    avg_loss_clean = running_loss_clean / total_clean if total_clean > 0 else float('nan')
    accuracy_total = 100. * correct_total / total_samples
    accuracy_noisy = 100. * correct_noisy / total_noisy if total_noisy > 0 else float('nan')
    accuracy_clean = 100. * correct_clean / total_clean if total_clean > 0 else float('nan')

    return avg_loss, accuracy_total, accuracy_noisy, accuracy_clean, avg_loss_noisy, avg_loss_clean, \
        total_samples, total_noisy, total_clean, correct_total, correct_noisy, correct_clean


# Update test_model function to accept the criterion_clean
def test_model(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    # テスト用の損失関数を定義
    criterion = nn.CrossEntropyLoss()  # reduction='mean' がデフォルト

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / total
    accuracy = 100. * correct / total

    return avg_loss, accuracy, total, correct


# Update main function
def main():
    print('start session')
    
    args = parse_args()
    set_seed(args.fix_seed)
    
    # Set device
    device = set_device(args.gpu)
    
    # Load datasets
    print('loading datasets')
    train_dataset, test_dataset, imagesize, num_classes, in_channels = load_datasets(args.dataset, args.target, args.gray_scale, args)
    
    # Number of digit and color classes (modify as needed)
    num_digits = 10
    num_colors = 1
    
    # Add label noise and create a NoisyDataset
    if args.label_noise_rate > 0.0:
        print('adding label noise')
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
        # If no label noise, create a noise_info tensor of zeros
        noise_info = torch.zeros(len(train_dataset), dtype=torch.int)
        if hasattr(train_dataset, 'tensors'):
            train_dataset = NoisyDataset(train_dataset, noise_info)
        else:
            train_dataset = NoisyDataset(train_dataset, noise_info)
    
    # Set up data loaders
    print('setting data loader')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Load model
    model = load_models(in_channels, args, imagesize, num_classes)
    model = model.to(device)

    # Set optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) if args.optimizer == "sgd" else optim.Adam(model.parameters(), lr=args.lr)
    
    # Set loss functions and weights
    criterion_noisy = nn.CrossEntropyLoss(reduction='none')
    criterion_clean = nn.CrossEntropyLoss(reduction='none')
    weight_noisy = args.weight_noisy
    weight_clean = args.weight_clean

    # Set experiment name
    experiment_name = f'{args.model}_{args.dataset}_{args.target}_lr{args.lr}_batch{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}_Optim{args.optimizer}_cleanw{args.weight_clean}_noisew{args.weight_noisy}'
    print(f'Experiment name: {experiment_name}')
    
    # Set up wandb
    if args.wandb:
        wandb.init(project=args.wandb_project, name=experiment_name, entity=args.wandb_entity)
        wandb.config.update(args)
    
    # CSV logging setup
    csv_dir = f"./csv/combine/split_noise/{experiment_name}"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, 'log.csv')
    if not os.path.isfile(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_accuracy", "train_accuracy_noisy", "train_accuracy_clean", "test_loss", "test_accuracy"])

    # Start training and testing
    print('start training and testing')
    for epoch in range(1, args.epoch + 1):
        avg_loss, train_accuracy, train_accuracy_noisy, train_accuracy_clean, avg_loss_noisy, avg_loss_clean, \
            total_samples, total_noisy, total_clean, correct_total, correct_noisy, correct_clean = \
            train_model(model, train_loader, optimizer, criterion_noisy, criterion_clean, weight_noisy, weight_clean, device)
        
        test_loss, test_accuracy, test_total_samples, test_correct = test_model(model, test_loader, device)

        print(f"Epoch: {epoch}, Train Loss: {avg_loss:.4f}, Overall Accuracy: {train_accuracy:.2f}%, Noisy Accuracy: {train_accuracy_noisy:.2f}%, Clean Accuracy: {train_accuracy_clean:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        # CSV logging
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, train_accuracy, train_accuracy_noisy, train_accuracy_clean, avg_loss_noisy, avg_loss_clean, test_loss, test_accuracy])

        # wandb logging
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
            })
    if args.wandb:
        wandb.finish()

main()