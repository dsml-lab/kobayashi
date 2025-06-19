
#made by kobyahsi! 一号!



import torch.nn.functional as f
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
from model import resnet18
from model import resnet18k_v2
# download datasets from pytorch
from torchvision import datasets

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-gpu", "--gpu", type=int, default=0, help="GPU device ID")
    arg_parser.add_argument("-seed", "--fix_seed", type=int, default=42)
    
    arg_parser.add_argument("--model", type=str, choices=["cnn_2layers", "cnn_5layers", "resnet18_lib","resnet18_scr","resnet_v2",'resnet18k','resnet34k'], default='resnet18',help="モデルアーキテクチャの選択")
    arg_parser.add_argument("-model_width", "--model_width", type=int, default=1)
    arg_parser.add_argument("-epoch", "--epoch", type=int, default=1000)


    arg_parser.add_argument("-datasets", "--dataset", type=str, choices=["cifar10","cifar100"], default="cifar10")
    arg_parser.add_argument("-label_noise_rate", "--label_noise_rate", type=float, default=0.1) 
    arg_parser.add_argument("-batch_size", "--batch_size", type=int, default=128, help="バッチサイズ")
    
     # set optimizer setting
    arg_parser.add_argument("-lr", "--lr", type=float, default=0.1, help="学習率")
    arg_parser.add_argument("-optimizer", "--optimizer", type=str, choices=["sgd", "adam"], default="adam", help="最適化手法.adam were used in Nakkiran et al. (2019)")
    arg_parser.add_argument("-momentum", "--momentum", type=float, default=0.9, help="モーメンタム")
    
    #set loss function setting
    arg_parser.add_argument("-loss", "--loss", type=str, choices=["cross_entropy", "focal_loss"], default="cross_entropy", help="損失関数")
    


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
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
        print("GPU not recognized, using CPU.")
    
    return device
def save_random_state():
    """
    Save the current random states for reproducibility.
    
    Returns:
        dict: A dictionary containing the states of Python's `random`, NumPy, and PyTorch random generators.
    """
    state = {
        'random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return state

class WeightedSoftCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, reduction='mean'):
        """
        WeightedSoftCrossEntropyLossの初期化。

        Args:
            class_weights (Tensor, optional): 各クラスに割り当てる重みの1Dテンソル。
            reduction (str, optional): 出力に適用する縮約方法。
                                       'none' | 'mean' | 'sum'。デフォルトは'mean'
        """
        super(WeightedSoftCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, logits, target):
        """
        損失のフォワードパス。

        Args:
            logits (Tensor): モデルの出力（ソフトマックス適用前）のテンソル。形状は(batch_size, num_classes)。
            target (Tensor): ソフトラベルのテンソル。形状は(batch_size, num_classes)。

        Returns:
            Tensor: 計算された損失。
        """
        # 数値的安定性のためにlog_softmaxを適用
        log_probs = F.log_softmax(logits, dim=1)  # 形状: (batch_size, num_classes)

        # ターゲットとの要素ごとの積
        loss = -target * log_probs  # 形状: (batch_size, num_classes)

        if self.class_weights is not None:
            # class_weightsがブロードキャスト可能であることを確認
            loss = loss * self.class_weights

        # クラスごとに損失を合計
        loss = loss.sum(dim=1)  # 形状: (batch_size,)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # 'none'

# ラベルをソフトラベルに変換する関数
def convert_to_soft_labels(labels, num_classes, smoothing=0.1):
    """
    ラベルをソフトラベルに変換します。

    Args:
        labels (Tensor): バッチのラベル。形状は(batch_size,)。
        num_classes (int): クラス数。
        smoothing (float): ラベルスムージングの割合。

    Returns:
        Tensor: ソフトラベル。形状は(batch_size, num_classes)。
    """
    with torch.no_grad():
        soft_labels = torch.full((labels.size(0), num_classes), smoothing / (num_classes - 1)).to(labels.device)
        soft_labels.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
    return soft_labels



def load_random_state(state):
    """
    Load saved random states for reproducibility.
    
    Args:
        state (dict): A dictionary containing the states of Python's `random`, NumPy, and PyTorch random generators.
    
    Returns:
        None
    """
    random.setstate(state['random'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def load_datasets(dataset,args):
    if dataset == "cifar10":
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
    else:
        raise ValueError("Invalid dataset name")
    return train_dataset, test_dataset, imagesize, num_classes, in_channels

def load_models(in_channels, args, img_size, num_classes):
    if args.model == "cnn_2layers":
        model = CNN2Layer(in_channels, num_classes, args.model_width, img_size)
    elif args.model == "cnn_5layers":
        model = CNN5Layer(in_channels, num_classes, args.model_width, img_size)
    elif args.model == "resnet18_lib":
        model = models.resnet18(num_classes=num_classes)
    elif args.model == "resnet18_scr":
        model = resnet18.make_resnet18(in_channels,num_classes)
        print(model)
    elif args.model == "resnet18k":
        k1 = 64 *args.model_width
        model = resnet18k_v2.make_resnet18k(k=k1, num_classes=num_classes,)
        print('make resnet18k')
    elif args.model == "resnet34k":
        k1 = 64 *args.model_width
        model = resnet18k_v2.make_resnet34k(k=k1, num_classes=num_classes,)
        print('make resnet34k')
    else:
        raise ValueError("Invalid model name.")

    return model
    

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


def train_model(model, train_loader, optimizer, criterion, device):
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

    for inputs, labels, noise_flags in train_loader:
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

        for idx in range(len(labels)):
            if noise_flags[idx] == 1:  # ノイズあり
                total_noisy += 1
                running_loss_noisy += criterion(outputs[idx].unsqueeze(0), labels[idx].unsqueeze(0)).item()
                if predicted[idx] == labels[idx]:
                    correct_noisy += 1
            else:  # ノイズなし
                total_clean += 1
                running_loss_clean += criterion(outputs[idx].unsqueeze(0), labels[idx].unsqueeze(0)).item()
                if predicted[idx] == labels[idx]:
                    correct_clean += 1

    avg_loss = running_loss / len(train_loader)
    avg_loss_noisy = running_loss_noisy / total_noisy if total_noisy > 0 else float('nan')#ノイズデータに対するloss
    avg_loss_clean = running_loss_clean / total_clean if total_clean > 0 else float('nan')#ノイズのないデータにに対するloss
    accuracy_total = 100. * correct_total / total_samples
    accuracy_noisy = 100. * correct_noisy / total_noisy if total_noisy > 0 else float('nan')#ノイズデータに対するloss
    accuracy_clean = 100. * correct_clean / total_clean if total_clean > 0 else float('nan')
    error_total = 100. - accuracy_total
    error_noisy = 100. - accuracy_noisy if total_noisy > 0 else float('nan')
    error_clean = 100. - accuracy_clean if total_clean > 0 else float('nan')
    return avg_loss, accuracy_total, accuracy_noisy, accuracy_clean, avg_loss_noisy, avg_loss_clean, \
        total_samples, total_noisy, total_clean, correct_total, correct_noisy, correct_clean,error_total,error_noisy,error_clean
        

def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    test_error = 100. - accuracy

    return avg_loss, accuracy, total, correct,test_error
 

def add_label_noise(data,label_noise_rate,num_classes):
    noisy_data = data.clone()
    num_noisy = int(label_noise_rate * len(data))
    noisy_indices = torch.randperm(len(data))[:num_noisy]
    noise_info = torch.zeros(len(data), dtype=torch.int)  # ノイズ情報を保存する列

    # ランダムなラベルに変更
    for idx in noisy_indices:
        original_label = data[idx].item()
        new_label = random.randint(0, num_classes - 1)
        noise_info[idx] = 1 
        # 元のラベルと同じ場合は変更し続ける
        while new_label == original_label:
            new_label = random.randint(0, num_classes - 1)
        noisy_data[idx] = new_label

    return noisy_data,noise_info


def compute_distances_between_indices(train_dataset, clean_indices_groups, noisy_indices, mode=0, batch_size=1000):
    """
    Compute pairwise distances between clean and noisy indices using GPU acceleration.

    Args:
        train_dataset (list): List of tuples where each tuple contains image tensor and label.
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
            sorted_distances = sorted(concatenated_distances.flatten())
            min_distance = sorted_distances[1] if len(sorted_distances) > 1 else concatenated_distances.min()
            min_index = concatenated_distances.argmin()
            noisy_index_min = noisy_indices[min_index % len(noisy_indices)]
            distances_results[tuple(clean_indices)] = ("closest", noisy_index_min, min_distance)
        else:
            sorted_distances = sorted(concatenated_distances.flatten(), reverse=True)
            max_distance = sorted_distances[1] if len(sorted_distances) > 1 else concatenated_distances.max()
            max_index = concatenated_distances.argmax()
            noisy_index_max = noisy_indices[max_index % len(noisy_indices)]
            distances_results[tuple(clean_indices)] = ("farthest", noisy_index_max, max_distance)

    return distances_results


def select_data_points(original_labels, noisy_labels, noise_info, train_dataset, max_points=20):
    """
    ラベルノイズ付与前はクラスラベルが同じで、付与後は異なるクラスラベルとなるデータポイント (x, y) を選択し、距離を表示して保存します。

    Returns:
        tuple: 条件を満たすデータポイントのインデックスのペア (idx_clean, idx_noisy) と距離。
    """
    original_labels = original_labels
    noisy_labels = noisy_labels
    n = 5  # 任意のクラスラベルを指定（0〜9の範囲で）
    mode = 0  # 最近または最遠を指定

    indices_with_label_n = torch.where(original_labels == n)[0]
    clean_indices = indices_with_label_n[noise_info[indices_with_label_n] == 0]  # クラスラベルがnでかつクリーンインデックスを選択
    noisy_indices = indices_with_label_n[noise_info[indices_with_label_n] == 1]

    distance_pair = []
    for i in range(min(len(clean_indices), max_points)):
        clean_index = clean_indices[i:i+1]
        if len(clean_index) > 0 and len(noisy_indices) > 0:
            # compute_distances_between_indices 関数は、データセット内の指定されたインデックス間の距離を計算する関数です。
            distances_results = compute_distances_between_indices(train_dataset, [clean_index.tolist()], noisy_indices.tolist(), mode)
            # クリーンデータとノイズデータのインデックスが異なることを確認
            clean_index = [idx for idx in clean_index if idx not in noisy_indices]
            if not clean_index:
                print("クリーンデータとノイズデータのインデックスが同じであるため、再選択が必要です。")
                continue
            # デバッグ用：距離の結果を表示
            print(f"distances_results: {distances_results}")
            clean_group, result_tuple = next(iter(distances_results.items()))
            label, idx_noisy, distance = result_tuple if len(result_tuple) == 3 else (None, None, None)
            idx_clean = clean_group[0]
            print(f"選択したペアのインデックス (clean: {idx_clean}, noisy: {idx_noisy}), 距離: {distance}")
            # ペアを保存するディレクトリを作成
            pair_dir = f"pair_{n}"
            os.makedirs(pair_dir, exist_ok=True)
            pair_subdir = os.path.join(pair_dir, f"pair_{i}_distance_{distance:.3f}_clean_{idx_clean}_noisy_{idx_noisy}")
            os.makedirs(pair_subdir, exist_ok=True)
        
            # 選択したクリーンとノイズの画像を .npy ファイルとして保存
            clean_image = train_dataset[idx_clean][0].cpu().numpy()
            noisy_image = train_dataset[idx_noisy][0].cpu().numpy()
            #np.save(os.path.join(pair_subdir, f"{idx_clean}clean_image.npy"), clean_image)
            #np.save(os.path.join(pair_subdir, f"{idx_noisy}noisy_image.npy"), noisy_image)
            distance_pair.append((idx_clean, idx_noisy, distance))

    # 最も距離が小さいペアを返し、保存
    if distance_pair:
        min_pair = min(distance_pair, key=lambda x: x[2])
        idx_clean, idx_noisy, distance = min_pair
        # ペアを保存するディレクトリを作成
        pair_dir = f"cifar_pair_{n}"
        os.makedirs(pair_dir, exist_ok=True)
        pair_subdir = os.path.join(pair_dir, f"pair_best_distance_{distance:.3f}")
        os.makedirs(pair_subdir, exist_ok=True)

        # 選択したクリーンとノイズの画像を .npy ファイルとして保存
        clean_image = train_dataset[idx_clean][0].cpu().numpy()
        noisy_image = train_dataset[idx_noisy][0].cpu().numpy()
        np.save(os.path.join(pair_subdir, f"{idx_clean}clean_image.npy"), clean_image)
        np.save(os.path.join(pair_subdir, f"{idx_noisy}noisy_image.npy"), noisy_image)
        
        print("---best---", distance)
        return idx_clean, idx_noisy, distance
    else:
        print("条件を満たすデータポイントが見つかりませんでした。")
        return (None, None, None)

import torch.nn.functional as F
def alpha_interpolation_test(
    model, x_clean, x_noisy,
    label_x, label_y,
    num_classes, device
):
    alpha_values = np.arange(-0.5, 1.6, 0.01)
    model.eval()

    x_clean = x_clean.to(device)
    x_noisy = x_noisy.to(device)

    losses = []
    predicted_labels = []
    probabilities = []
    label_matches = []

    for alpha in alpha_values:
        # 補間されたデータを生成
        z = alpha * x_clean + (1 - alpha) * x_noisy
        z = z.unsqueeze(0)  # バッチ次元を追加

        # モデルの出力（ロジット）
        outputs = model(z)  # 形状: [1, num_classes]

        # softmaxを計算
        output_probs = F.softmax(outputs, dim=1)  # 形状: [1, num_classes]

        # ソフトラベルを作成
        soft_label = torch.zeros(num_classes, device=device)
        soft_label[label_x] = alpha
        soft_label[label_y] = 1 - alpha

        # 損失を計算
        loss = -torch.sum(soft_label * torch.log(output_probs.squeeze(0) + 1e-8))

        # 予測ラベルを取得
        predicted_label = torch.argmax(output_probs, dim=1).item()
        predicted_labels.append(predicted_label)

        # ラベルの一致結果を判定
        if predicted_label == label_x:
            label_matches.append(1)
        elif predicted_label == label_y:
            label_matches.append(-1)
        else:
            label_matches.append(0)

        # ログを保存
        losses.append(loss.item())
        probabilities.append(output_probs.detach().cpu().numpy())

    return {
        'alpha_values': alpha_values.tolist(),
        'losses': losses,
        'predicted_labels': predicted_labels,
        'probabilities': probabilities,
        'label_matches': label_matches
    }


def setup_alpha_csv_logging(experiment_name, epoch, num_classes):
    # Set up directory and path for alpha test CSV logging
    alpha_csv_dir = f"./csv/combine/alpha_test/{experiment_name}"
    os.makedirs(alpha_csv_dir, exist_ok=True)
    alpha_csv_path = os.path.join(alpha_csv_dir, f'alpha_log_epoch_{epoch}.csv')
    if not os.path.isfile(alpha_csv_path):
        with open(alpha_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header for alpha_logs
            header = [
                "alpha",
                "loss",
                "predicted_label",
                "label_match"
            ]
            # Add class probability headers
            for i in range(num_classes):
                header.append(f"class_probability_{i}")
            writer.writerow(header)
    return alpha_csv_path

def log_alpha_test_results(alpha_logs, alpha_csv_path):
    # Log alpha test results to CSV
    with open(alpha_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for i, alpha in enumerate(alpha_logs['alpha_values']):
            writer.writerow([
                alpha,
                alpha_logs['losses'][i],
                alpha_logs['predicted_labels'][i],
                alpha_logs['label_matches'][i],
                *alpha_logs['probabilities'][i][0]  # Log each class probability
            ])

def main():
    print("start session")
    wandb_run = None
    
    args = parse_args()

    set_seed(args.fix_seed)
    
    device = set_device(args.gpu)

     # load datasets
    train_dataset, test_dataset, imagesize, num_classes, in_channels = load_datasets(args.dataset, args)
    # ラベルノイズの設定


    # CIFAR-10 なら num_classes = 10、CIFAR-100 なら num_classes = 100
    print(f'ラベルノイズ率: {args.label_noise_rate}')

    # ラベルをテンソルに変換
    #y_train = torch.tensor(train_dataset.targets)

    # ラベルノイズを追加
    print(f'Preparing dataset with label noise rate: {args.label_noise_rate}')
    if args.label_noise_rate > 0.0:
        if hasattr(train_dataset, 'tensors'):
            print("===========================")
            x_train, y_train = train_dataset.tensors
            y_train_noisy, noise_info = add_label_noise(y_train, args.label_noise_rate,num_classes)
            train_dataset = torch.utils.data.TensorDataset(x_train, y_train_noisy)
            train_dataset = NoisyDataset(train_dataset, noise_info)
        else:
            print("----------------------")
            y_train = torch.tensor(train_dataset.targets)
            y_train_noisy, noise_info = add_label_noise(y_train, args.label_noise_rate,num_classes)
            train_dataset.targets = y_train_noisy.numpy()
            train_dataset = NoisyDataset(train_dataset, noise_info)
    else:
        if hasattr(train_dataset, 'tensors'):
            x_train, y_train = train_dataset.tensors
            y_train_noisy = y_train.clone()
            noise_info = torch.zeros(len(train_dataset), dtype=torch.int)
            train_dataset = torch.utils.data.TensorDataset(x_train, y_train_noisy)
            train_dataset = NoisyDataset(train_dataset, noise_info)
        else:
            y_train = torch.tensor(train_dataset.targets)
            y_train_noisy = y_train.clone()
            noise_info = torch.zeros(len(train_dataset), dtype=torch.int)
            train_dataset.targets = y_train_noisy.tolist()

    # ノイズ情報を持つデータセットを作成

    # クリーンおよびノイズサンプルのインデックスを取得
    clean_indices = torch.where(noise_info == 0)[0].tolist()
    noisy_indices = torch.where(noise_info == 1)[0].tolist()
    if hasattr(train_dataset.dataset, 'tensors'):
        original_labels = y_train  # ラベルノイズ付与前のラベル
    else:
        original_labels = torch.tensor(y_train)  # ラベルノイズ付与前のラベル

    # ノイズが付与されたラベルを取得（ラベルノイズ付与後のラベル）
    if hasattr(train_dataset.dataset, 'tensors'):
        noisy_labels = y_train_noisy
    else:
        noisy_labels = torch.tensor(train_dataset.dataset.targets)
    # ラベルノイズ付与前後のラベルを取得
    print("距離を計算中です...")
    random_state_before = None

    # 乱数状態を保存
    random_state_before = save_random_state()

    # 関数呼び出し
    idx_clean, idx_noisy, distance = select_data_points(
    original_labels, noisy_labels, noise_info, train_dataset
    )

    # 乱数状態を回復
    load_random_state(random_state_before)

    print("distance")
    x_clean = train_dataset.dataset[idx_clean][0]
    y_noisy = train_dataset.dataset[idx_noisy][0]
    x_label = original_labels[idx_clean]
    y_label = noisy_labels[idx_noisy]

    print(f"Clean Label: {x_label}")
    print(f"Noisy Label: {y_label}")
    # データセットにノイズ情報を含める
    # データローダーの設定
    print('setting data loader')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # load model
    print('load model')
    model = load_models(in_channels, args, imagesize, num_classes)
    model = model.to(device)  

    #print(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) if args.optimizer == "sgd" else optim.Adam(model.parameters(), lr=args.lr)
    #print(optimizer)
    criterion = nn.CrossEntropyLoss()

    experiment_name = f'{args.model}_{args.dataset}_lr{args.lr}_batch{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}_Optim{args.optimizer}_distance{distance:.4f}_xid{idx_clean}_yid{idx_noisy}'
        
    print(f'Experiment name: {experiment_name}')
    # wandbの初期化
    if args.wandb:
        print('Initializing wandb...')
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=experiment_name,
            entity=args.wandb_entity,
            config=args
        )

    # Set up CSV logging
    csv_dir = f"./csv/combine/alpha_test/{experiment_name}"
    os.makedirs(csv_dir, exist_ok=True)
    """img_clean_np,x_la = train_dataset[idx_clean][0].numpy()
    img_noisy_np,y_la = train_dataset[idx_noisy][0].numpy()
    np.save(os.path.join(csv_dir, "selected_clean_image.npy"), img_clean_np)
    np.save(os.path.join(csv_dir, "selected_noisy_image.npy"), img_noisy_np)"""
    #csv_dir = f"./csv/combine/split_noise/{experiment_name}"
    #os.makedirs(csv_dir, exist_ok=True)
    #csv_path = os.path.join(csv_dir, 'log.csv')


    print("finish data loading")
    for epoch in range(args.epoch):
        epoch = epoch + 1 # start from 1
        avg_loss, train_accuracy, train_accuracy_noisy, train_accuracy_clean, avg_loss_noisy, avg_loss_clean, \
        total_samples, total_noisy, total_clean, correct_total, correct_noisy, correct_clean,error_total,error_noisy,error_clean = train_model(model, train_loader, optimizer, criterion, device) 
        alpha_logs=alpha_interpolation_test(model, x_clean, y_noisy,x_label, y_label,num_classes, device)
        test_loss, test_accuracy,test_total_samples, test_correct,test_error = test_model(model, test_loader, criterion, device)
        print(f"epoch: {epoch}, train_loss: {avg_loss:.4f}, train_accuracy_noisy: {train_accuracy_noisy:.4f}, train_accuracy_clean: {train_accuracy_clean:.4f}, test_accuracy: {test_accuracy:.4f}, test_loss: {test_loss:.4f}")
        if args.wandb:
            log_data={
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
                'test_error':test_error,
                'avg_loss_noisy': avg_loss_noisy,
                'avg_loss_clean': avg_loss_clean,
                "tranin_error_total" : error_total,
                "train_error_noisy" : error_noisy,
                "train_error_clean" : error_clean,
            }
        valid_alpha_values = np.arange(-0.50, 1.60, 0.1).tolist()
        tolerance = 1e-5 
        if alpha_logs is not None:
            for i, alpha in enumerate(alpha_logs['alpha_values']):
                # alpha が valid_alpha_values のいずれかに近いかを確認
                if any(math.isclose(alpha, valid_alpha, abs_tol=tolerance) for valid_alpha in valid_alpha_values):
                    alpha_str = f"{alpha:.2f}"
                    log_data[f'{alpha_str}/losses'] = alpha_logs['losses'][i]
                    log_data[f'{alpha_str}/predicted_labels'] = alpha_logs['predicted_labels'][i]

                    # 各クラスの確率を個別にログに記録
                    class_probs = alpha_logs['probabilities'][i][0]  # バッチ次元を除去
                    for j, prob in enumerate(class_probs):
                        log_data[f'{alpha_str}/class_probability_{j}'] = prob
            if epoch % 10 == 0:
                alpha_csv_path = setup_alpha_csv_logging(experiment_name, epoch,num_classes)
                log_alpha_test_results(alpha_logs, alpha_csv_path)
        wandb.log(log_data)
    if args.wandb:
        wandb.finish()

if __name__ == '__main__':    
    main()
