# datasets.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import gzip
import random
from utils import apply_transform

def compute_distances_between_indices(train_dataset, clean_indices_groups, noisy_indices, mode=0, batch_size=1000):
    """
    Compute pairwise distances between clean and noisy indices using GPU acceleration.

    Args:
        train_dataset (Dataset): The dataset containing the images.
        clean_indices_groups (list of list): Groups of clean indices to compare against noisy indices.
        noisy_indices (list): List of noisy indices to compare.
        mode (int): Mode to determine the type of distance to find. 0 for closest, 1 for farthest.
        batch_size (int): Batch size for distance calculation to optimize GPU memory usage.

    Returns:
        dict: Dictionary containing distances for each clean index group.
    """
    if mode not in (0, 1):
        raise ValueError("mode must be 0 (closest) or 1 (farthest)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    distances_results = {}

    # Moving noisy images to GPU in batches for efficiency
    noisy_imgs_batches = []
    for i in range(0, len(noisy_indices), batch_size):
        batch_indices = noisy_indices[i:i+batch_size]
        noisy_imgs_batches.append(torch.stack([train_dataset[int(idx)][0] for idx in batch_indices]).to(device))

    # Calculating distances for each group of clean indices
    for clean_indices in clean_indices_groups:
        clean_imgs = torch.stack([train_dataset[int(idx)][0] for idx in clean_indices]).to(device)
        group_distances = []

        # Calculating distances against each noisy batch
        for noisy_imgs in noisy_imgs_batches:
            distances = torch.cdist(clean_imgs.view(len(clean_indices), -1), noisy_imgs.view(len(noisy_imgs), -1))
            group_distances.append(distances.cpu())

        # Concatenate all distances along the noisy dimension
        concatenated_distances = torch.cat(group_distances, dim=1).numpy()

        # Find closest or farthest distances based on mode
        if mode == 0:
            min_distance = concatenated_distances.min()
            min_index = concatenated_distances.argmin()
            noisy_index_min = noisy_indices[min_index % len(noisy_indices)]
            distances_results[tuple(clean_indices)] = ("closest", noisy_index_min, min_distance)
        else:
            max_distance = concatenated_distances.max()
            max_index = concatenated_distances.argmax()
            noisy_index_max = noisy_indices[max_index % len(noisy_indices)]
            distances_results[tuple(clean_indices)] = ("farthest", noisy_index_max, max_distance)

    return distances_results
class NoisyDataset(Dataset):
    """
    Custom dataset that includes noise information.
    """
    def __init__(self, dataset, noise_info):
        self.dataset = dataset
        self.noise_info = noise_info

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input, label = self.dataset[idx]
        noise_label = self.noise_info[idx]
        return input, label, noise_label

class BalancedBatchSampler(Sampler):
    """
    Custom sampler to create balanced batches of clean and noisy samples.
    """
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

def load_datasets(dataset, target, gray_scale, args):
    """
    Load the specified dataset and apply transformations based on the dataset type and grayscale option.
    
    Args:
        dataset (str): The name of the dataset to load.
        target (str): Target type for certain datasets.
        gray_scale (bool): Flag indicating whether to convert the images to grayscale.
        args (argparse.Namespace): Parsed command-line arguments.
        
    Returns:
        tuple: A tuple containing the train dataset, test dataset, image size, number of classes, and number of input channels.
    """
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
        
        # Transformation
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        # Apply transformation
        x_train_tensor = apply_transform(x_train, transform)
        x_test_tensor = apply_transform(x_test, transform)

        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

        num_classes = 10
        in_channels = 1
        imagesize = (32, 32)
    elif dataset == "colored_emnist":
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
            
            train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        
        elif target == 'digit':
            x_train = np.load('data/colored_EMNIST/x_train_colored.npy')
            y_train_digits = np.load('data/colored_EMNIST/y_train_digits.npy')
            x_test = np.load('data/colored_EMNIST/x_test_colored.npy')
            y_test_digits = np.load('data/colored_EMNIST/y_test_digits.npy')
            
            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            
            y_train_tensor = torch.tensor(y_train_digits, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_digits, dtype=torch.long)
            
            train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        
        elif target == 'combined':
            x_train = np.load('data/colored_EMNIST/x_train_colored.npy')
            y_train_combined = np.load('data/colored_EMNIST/y_train_combined.npy')
            x_test = np.load('data/colored_EMNIST/x_test_colored.npy')
            y_test_combined = np.load('data/colored_EMNIST/y_test_combined.npy')
            
            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            
            y_train_tensor = torch.tensor(y_train_combined, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_combined, dtype=torch.long)
            
            train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        
        num_classes = 10 if target in ['color', 'digit'] else 100
        in_channels = 3
        imagesize = (32, 32)
    elif dataset == "distribution_colored_emnist":
        seed = args.fix_seed
        variance = args.variance
        correlation = args.correlation
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        if target == 'color':
            base_path = f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}'
            x_train = np.load(os.path.join(base_path, 'x_train_colored.npy'))
            y_train_colors = np.load(os.path.join(base_path, 'y_train_colors.npy'))
            x_test = np.load(os.path.join(base_path, 'x_test_colored.npy'))
            y_test_colors = np.load(os.path.join(base_path, 'y_test_colors.npy'))
            
            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            
            y_train_tensor = torch.tensor(y_train_colors, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_colors, dtype=torch.long)
            
            train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        
        elif target == 'digit':
            base_path = f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}'
            x_train = np.load(os.path.join(base_path, 'x_train_colored.npy'))
            y_train_digits = np.load(os.path.join(base_path, 'y_train_digits.npy'))
            x_test = np.load(os.path.join(base_path, 'x_test_colored.npy'))
            y_test_digits = np.load(os.path.join(base_path, 'y_test_digits.npy'))
            
            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            
            y_train_tensor = torch.tensor(y_train_digits, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_digits, dtype=torch.long)
            
            train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        
        elif target == 'combined':
            base_path = f'data/distribution_colored_EMNIST_Seed42_Var{variance}_Corr{correlation}'
            x_train = np.load(os.path.join(base_path, 'x_train_colored.npy'))
            y_train_combined = np.load(os.path.join(base_path, 'y_train_combined.npy'))
            x_test = np.load(os.path.join(base_path, 'x_test_colored.npy'))
            y_test_combined = np.load(os.path.join(base_path, 'y_test_combined.npy'))
            
            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            
            y_train_tensor = torch.tensor(y_train_combined, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_combined, dtype=torch.long)
            
            train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
            print(base_path)
    
        num_classes = 10 if target in ['color', 'digit'] else 100
        in_channels = 3
        imagesize = (32, 32)
    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                 (0.2023, 0.1994, 0.2010))
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
            transforms.Normalize((0.5071, 0.4867, 0.4408), 
                                 (0.2675, 0.2565, 0.2761))
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
            transforms.Normalize((0.4802, 0.4481, 0.3975), 
                                 (0.2302, 0.2265, 0.2262))
        ])
        train_dataset = datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform)
        test_dataset = datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=transform)
        imagesize = (64, 64)
        num_classes = 200
        in_channels = 3
    elif dataset == "distribution_to_normal":
        # Similar to 'distribution_colored_emnist' but with different paths
        seed = args.fix_seed
        variance = args.variance
        correlation = args.correlation
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        if target == 'combined':
            base_path = f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}'
            x_train = np.load(os.path.join(base_path, 'x_train_colored.npy'))
            y_train_combined = np.load(os.path.join(base_path, 'y_train_combined.npy'))
            x_test = np.load('data/colored_EMNIST/x_test_colored.npy')
            y_test_combined = np.load('data/colored_EMNIST/y_test_combined.npy')
            
            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            
            y_train_tensor = torch.tensor(y_train_combined, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_combined, dtype=torch.long)
            
            train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    
        num_classes = 10 if target in ['color', 'digit'] else 100
        in_channels = 3
        imagesize = (32, 32)
    else:
        raise ValueError("Invalid dataset name")
        
    if gray_scale:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if isinstance(train_dataset, TensorDataset):
            train_dataset = TensorDataset(*[
                transform(img) for img in train_dataset.tensors[0]
            ], train_dataset.tensors[1])
        if isinstance(test_dataset, TensorDataset):
            test_dataset = TensorDataset(*[
                transform(img) for img in test_dataset.tensors[0]
            ], test_dataset.tensors[1])
    
    return train_dataset, test_dataset, imagesize, num_classes, in_channels


import os
import torch
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, Dataset

# --- 既存の関数群 ---
def add_label_noise(targets, label_noise_rate, num_digits, num_colors):
    """
    Add label noise to the targets.
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


def apply_label_noise_to_dataset(dataset, noise_rate, num_digits=10, num_colors=1):
    """
    指定されたデータセットにラベルノイズを加える。
    ノイズは一様にランダムなクラスに置き換える方式。
    - 入力: PyTorch Dataset (image, label) or (image, label, ...)
    - 出力: NoisyDataset (image, noisy_label, noise_flag)

    Args:
        dataset (Dataset): 元データセット（PyTorch Dataset）
        noise_rate (float): ノイズ率（0〜1）
        num_digits (int): 数字クラスの数（combined ターゲットに使う）
        num_colors (int): 色クラスの数（combined ターゲットに使う）

    Returns:
        noisy_dataset (NoisyDataset)
    """
    x_list, y_list, noise_info_list = [], [], []
    num_classes = num_digits * num_colors

    for i in range(len(dataset)):
        sample = dataset[i]
        if isinstance(sample, tuple) and len(sample) == 2:
            x, y = sample
        elif isinstance(sample, tuple) and len(sample) >= 3:
            x, y = sample[0], sample[1]
        else:
            raise ValueError("Unsupported dataset format")

        if np.random.rand() < noise_rate:
            y_noisy = np.random.randint(num_classes)
            while y_noisy == y:
                y_noisy = np.random.randint(num_classes)
            y_list.append(y_noisy)
            noise_info_list.append(1)
        else:
            y_list.append(y)
            noise_info_list.append(0)

        x_list.append(x)

    # Tensor へ変換
    x_tensor = torch.stack(x_list) if isinstance(x_list[0], torch.Tensor) else torch.tensor(np.stack(x_list))
    y_tensor = torch.tensor(y_list, dtype=torch.long)
    noise_tensor = torch.tensor(noise_info_list, dtype=torch.long)

    base_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    return NoisyDataset(base_dataset, noise_tensor)


# load_datasetsは、各データセット固有のデータ拡張（transform）を適用して読み込む既存の関数とする
# （ここでは既に定義済みのものとして利用）

# --- 新規関数群 ---
class NoisyDataset(Dataset):
    """
    Custom dataset that includes noise information.
    """
    def __init__(self, dataset, noise_info):
        self.dataset = dataset
        self.noise_info = noise_info

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input, label = self.dataset[idx]
        noise_label = self.noise_info[idx]
        return input, label, noise_label
import json
from torchvision import transforms
from torch.utils.data import Dataset

# -----------------------------------------
# ① transform.json を読み込み、transforms.Compose を返す関数
# -----------------------------------------
def load_transform_from_json(json_path):
    """
    transform.json を読み込み、transforms.Compose を返す。
    ToTensor, Normalize, Resize, Grayscale などは除外推奨。
    """
    # どの transform を実際にインスタンス化できるか管理する辞書
    TRANSFORM_REGISTRY = {
        "RandomCrop": transforms.RandomCrop,
        "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
        "RandomRotation": transforms.RandomRotation,
        "ColorJitter": transforms.ColorJitter,
        "GaussianBlur": transforms.GaussianBlur
        # 必要に応じて増やせます
    }

    with open(json_path, "r") as f:
        meta = json.load(f)

    transform_list = []
    for t in meta["transforms"]:
        name = t["name"]
        params = t.get("params", {})

        if name in TRANSFORM_REGISTRY:
            transform_cls = TRANSFORM_REGISTRY[name]
            transform_list.append(transform_cls(**params))
        else:
            print(f"Warning: transform '{name}' は未登録です。スキップします。")

    return transforms.Compose(transform_list)

# -----------------------------------------
# ② 既存Datasetに transform をかけ直すためのラッパー
# -----------------------------------------
class TransformedDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        # (x, y) or (x, y, noise_flag) を考慮
        x, y = data[:2]
        # x に transform を適用 (Tensor -> PIL に変換不要な augment のみ推奨)
        x = self.transform(x)
        # 3つ目の要素 (noise_flag) がある場合もそのまま返却
        return (x, y) if len(data) == 2 else (x, y, data[2])

    def __len__(self):
        return len(self.base_dataset)

def load_or_create_noisy_dataset(dataset, target, gray_scale, args, return_type="torch"):
    """
    指定されたデータセットについてラベルノイズを付与した学習データを保存／読み込みし、
    テストデータ・meta情報も取得する。

    Parameters:
    ----------
    dataset : str
        データセット名（"cifar10"、"emnist"など）。
    target : str
        EMNISTなどで使用するターゲット種別や、coloredの際の判定用。
    gray_scale : bool
        入力データをグレイスケール化するかどうか。
    args : argparse.Namespace
        label_noise_rateやfix_seedなどの実行時引数。
    return_type : {"torch", "npy"}
        返り値の型を指定。"torch"の場合はDataset形式、"npy"の場合はnumpy配列。
    """
    label_noise_rate = args.label_noise_rate

    # データセットごとに、_handle_*関数に分割
    if dataset.lower() in ["cifar10"]:
        return _handle_cifar10(args, return_type)
    elif dataset.lower() in ["emnist", "emnist_digits"]:
        return _handle_emnist_digits(target, gray_scale, args, return_type)
    elif "distribution_colored_emnist" in dataset.lower():
        print("distribution_colored_emnist")
        print(target)
        return _handle_distribution_colored_emnist(dataset, target, gray_scale, args, return_type)
    else:
        raise ValueError("Unsupported dataset type for noisy dataset loader")


def _handle_cifar10(args, return_type):
    """
    CIFAR-10用の保存／読み込みの設定を行い、_load_or_save_datasetに渡す。
    """
    base_dir = os.path.join("/workspace/data", "cifar10")
    train_dir = os.path.join(base_dir, f"cifar10_noise_{args.label_noise_rate}")
    test_dir = base_dir
    noise_num_colors = 1
    return _load_or_save_dataset(
        dataset="cifar10",
        train_dir=train_dir,
        test_dir=test_dir,
        noise_num_colors=noise_num_colors,
        gray_scale=False,
        args=args,
        return_type=return_type
    )


def _handle_emnist_digits(target, gray_scale, args, return_type):
    """
    EMNIST(またはEMNISTの数字部分)用の保存／読み込み設定を行い、_load_or_save_datasetに渡す。
    """
    base_dir = os.path.join("/workspace/data", "EMNIST")
    train_dir = os.path.join(base_dir, f"emnist_noise_{args.label_noise_rate}")
    test_dir = base_dir
    noise_num_colors = 1
    return _load_or_save_dataset(
        dataset="emnist",
        train_dir=train_dir,
        test_dir=test_dir,
        noise_num_colors=noise_num_colors,
        gray_scale=gray_scale,
        args=args,
        return_type=return_type
    )


def _handle_distribution_colored_emnist(dataset, target, gray_scale, args, return_type):
    """
    Distribution Colored EMNIST, あるいは通常のColored EMNIST用の設定を行い、
    _load_or_save_datasetに渡す。
    """
    if dataset.lower() == "distribution_colored_emnist":
        base_dir = os.path.join(
            "/workspace/data",
            f"distribution_colored_EMNIST_Seed{args.fix_seed}_Var{args.variance}_Corr{args.correlation}"
        )
        train_dir = os.path.join(base_dir, f"{os.path.basename(base_dir)}_noise_{args.label_noise_rate}")
    else:
        base_dir = os.path.join("/workspace/data", "colored_EMNIST")
        train_dir = base_dir + f"_noise_{args.label_noise_rate}"

    test_dir = base_dir
    noise_num_colors = 10 if target == "combined" else 1
    return _load_or_save_dataset(
        dataset=dataset,
        train_dir=train_dir,
        test_dir=test_dir,
        noise_num_colors=noise_num_colors,
        gray_scale=gray_scale,
        args=args,
        return_type=return_type
    )


def _check_files_exist(directory, file_list):
    """
    指定したディレクトリ内に、file_listで与えられた全てのファイルが存在するかを確認する。

    Parameters:
    ----------
    directory : str
        チェック対象のディレクトリ。
    file_list : list of str
        存在確認を行うファイル名のリスト。

    Returns:
    -------
    bool
        全てのファイルが存在すればTrue、それ以外はFalse。
    """
    if not os.path.exists(directory):
        return False
    return all(os.path.exists(os.path.join(directory, f)) for f in file_list)


def _load_or_save_dataset(dataset,train_dir, test_dir, noise_num_colors, gray_scale, args, return_type):
    from torch.utils.data import TensorDataset
    label_noise_rate = args.label_noise_rate

    train_file = os.path.join(train_dir, "train_data.pt")
    test_file = os.path.join(test_dir, "test_data.pt")
    meta_file = os.path.join(test_dir, "meta.pt")

    train_files_exist = os.path.exists(train_file)
    test_files_exist = os.path.exists(test_file) and os.path.exists(meta_file)

    if not (train_files_exist and test_files_exist):
        missing_files = []
        if not os.path.exists(train_file):
            missing_files.append(train_file)
        if not os.path.exists(test_file):
            missing_files.append(test_file)
        if not os.path.exists(meta_file):
            missing_files.append(meta_file)
        print("以下のファイルが存在しないため新規生成します：")
        for f in missing_files:
            print(f"  - {f}")
        
        full_train_dataset, full_test_dataset, imagesize, num_classes, in_channels = load_datasets(
            dataset, "combined", gray_scale, args
        )

        # 学習データの準備
        x_list, y_list = zip(*[(x, y) for x, y in full_train_dataset])
        x_train = torch.stack(x_list)
        y_train = torch.tensor(y_list)

        # ラベルノイズの付加
        print(f"Adding label noise with rate: {label_noise_rate}")
        y_train_noisy, noise_info = add_label_noise(
            y_train, label_noise_rate, num_digits=10, num_colors=noise_num_colors
        )

        train_dataset = TensorDataset(x_train, y_train_noisy)
        train_dataset = NoisyDataset(train_dataset, noise_info)

        meta_local = {
            "imagesize": imagesize,
            "num_classes": num_classes,
            "in_channels": in_channels
        }

        # 保存（.pt 形式でまとめて保存）
        if not train_files_exist:
            os.makedirs(train_dir, exist_ok=True)
            torch.save({
                "x_train": x_train,
                "y_train": y_train_noisy,
                "noise_info": noise_info
            }, train_file)

        if not test_files_exist:
            x_test_list, y_test_list = zip(*[(x, y) for x, y in full_test_dataset])
            x_test = torch.stack([x for x in x_test_list])
            y_test = torch.tensor(y_test_list)

            os.makedirs(test_dir, exist_ok=True)
            torch.save({
                "x_test": x_test,
                "y_test": y_test
            }, test_file)
            torch.save(meta_local, meta_file)

        meta = meta_local
    else:
        # 読み込み（.pt形式から）
        train_data = torch.load(train_file)
        x_train = train_data["x_train"]
        y_train = train_data["y_train"]
        noise_info = train_data["noise_info"]

        test_data = torch.load(test_file)
        x_test = test_data["x_test"]
        y_test = test_data["y_test"]

        meta = torch.load(meta_file)

    if return_type == "npy":
        # NumPy配列として返却
        return (
            x_train.numpy(), y_train.numpy(), noise_info.numpy(),
            x_test.numpy(), y_test.numpy(), meta
        )

    elif return_type == "torch":
        train_dataset = TensorDataset(x_train, y_train)
        
        # ラベルノイズがなくても NoisyDataset を使う（noise_info は全て0）
        train_dataset = NoisyDataset(train_dataset, noise_info)

        test_dataset = TensorDataset(x_test, y_test)

        json_path = os.path.join(test_dir, "transform.json")
        if os.path.exists(json_path):
            print("transform.json が存在するため、augment を適用します。")
            augment_transform = load_transform_from_json(json_path)
            train_dataset = TransformedDataset(train_dataset, augment_transform)

        return train_dataset, test_dataset, meta

    else:
        raise ValueError("return_type must be either 'npy' or 'torch'")

# その他の関数 (add_label_noise, NoisyDataset, load_datasets) は元のまま保持