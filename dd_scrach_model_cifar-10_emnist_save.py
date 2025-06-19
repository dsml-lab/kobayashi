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

# import models
from model.cnn_2layers import CNN2Layer
from model.cnn_3layers import CNN3Layer
from model.cnn_4layers import CNN4Layer
from model.cnn_5layers import CNN5Layer
from model import resnet18
from model import resnet18k_v2
# download datasets from pytorch
from torchvision import datasets
from datasets import load_datasets, BalancedBatchSampler, NoisyDataset


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-gpu", "--gpu", type=int, default=0, help="GPU device ID")
    arg_parser.add_argument("-seed", "--fix_seed", type=int, default=42)
    
    arg_parser.add_argument("--model", type=str, choices=["cnn_2layers", "cnn_3layers","cnn_4layers","cnn_5layers", "resnet18_lib","resnet18_scr","resnet_v2",'resnet18k'], default='resnet18',help="モデルアーキテクチャの選択")
    arg_parser.add_argument("-model_width", "--model_width", type=int, default=1)
    arg_parser.add_argument("-epoch", "--epoch", type=int, default=1000)

    arg_parser.add_argument("-datasets", "--dataset", type=str, choices=["cifar10","cifar100","emnist_digits"], default="cifar10")
    arg_parser.add_argument("-label_noise_rate", "--label_noise_rate", type=float, default=0.1) 
    arg_parser.add_argument("-batch_size", "--batch_size", type=int, default=128, help="バッチサイズ")
    
    # set optimizer setting
    arg_parser.add_argument("-lr", "--lr", type=float, default=0.1, help="学習率")
    arg_parser.add_argument("-optimizer", "--optimizer", type=str, choices=["sgd", "adam"], default="adam", help="最適化手法.adam were used in Nakkiran et al. (2019)")
    arg_parser.add_argument("-momentum", "--momentum", type=float, default=0.9, help="モーメンタム")
    
    # set loss function setting
    arg_parser.add_argument("-loss", "--loss", type=str, choices=["cross_entropy", "focal_loss",'soft_cross_entropy'], default="cross_entropy", help="損失関数")

    arg_parser.add_argument("-num_workers", "--num_workers", type=int, default=4, help="データローダーの並列数")

    # wandb setting
    arg_parser.add_argument("-wandb", "--wandb", action='store_true', default=True, help="wandbを使用するかどうか")
    arg_parser.add_argument("-wandb_project", "--wandb_project", type=str, default="dd_scratch_cifar-10_model", help="wandbのプロジェクト名")
    arg_parser.add_argument("--wandb_entity", type=str, default="dsml-kernel24", help="wandbのエンティティ名")
    
    return arg_parser.parse_args()


# set seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# set device
def set_device(gpu_id):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
        print("GPU not recognized, using CPU.")
    return device


def setup_wandb(args, experiment_name):
    if args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=experiment_name,
            entity=args.wandb_entity,
            config=args
        )
        return run
    return None


def apply_transform(x, transform):
    return torch.stack([transform(img) for img in x])


def load_datasets(dataset, args):
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
        noise_info = torch.zeros(len(train_dataset), dtype=torch.int)  # ノイズ情報（CIFARはデフォルトで扱わない）
    
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
        noise_info = torch.zeros(len(train_dataset), dtype=torch.int)
    
    elif dataset == "emnist_digits":
        emnist_path = './data/EMNIST'
        label_noise_rate = args.label_noise_rate

        # ノイズ付きデータセットのディレクトリ
        noisy_dataset_dir = os.path.join(emnist_path, f'EMNIST_{label_noise_rate}')
        os.makedirs(noisy_dataset_dir, exist_ok=True)

        # 保存済みデータセットのファイルパス
        noisy_train_images_path = os.path.join(noisy_dataset_dir, 'train_images.npy')
        noisy_train_labels_path = os.path.join(noisy_dataset_dir, 'train_labels.npy')
        noisy_train_noise_info_path = os.path.join(noisy_dataset_dir, 'train_noise_info.npy')
        noisy_test_images_path = os.path.join(noisy_dataset_dir, 'test_images.npy')
        noisy_test_labels_path = os.path.join(noisy_dataset_dir, 'test_labels.npy')

        if (
            os.path.exists(noisy_train_images_path) and 
            os.path.exists(noisy_train_labels_path) and 
            os.path.exists(noisy_train_noise_info_path) and 
            os.path.exists(noisy_test_images_path) and 
            os.path.exists(noisy_test_labels_path)
        ):
            print(f"Loading saved noisy EMNIST dataset from {noisy_dataset_dir}...")
            x_train = np.load(noisy_train_images_path)
            y_train = np.load(noisy_train_labels_path)
            noise_info = np.load(noisy_train_noise_info_path)
            x_test = np.load(noisy_test_images_path)
            y_test = np.load(noisy_test_labels_path)
        else:
            print(f"No saved noisy dataset found for label_noise_rate={label_noise_rate}. Creating and saving new noisy dataset...")
            def load_gz_file(file_path, is_image=True):
                with gzip.open(file_path, 'rb') as f:
                    if is_image:
                        return np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
                    else:
                        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)
            
            # ファイルパスの設定
            train_images_path = os.path.join(emnist_path, 'emnist-digits-train-images-idx3-ubyte.gz')
            train_labels_path = os.path.join(emnist_path, 'emnist-digits-train-labels-idx1-ubyte.gz')
            test_images_path = os.path.join(emnist_path, 'emnist-digits-test-images-idx3-ubyte.gz')
            test_labels_path = os.path.join(emnist_path, 'emnist-digits-test-labels-idx1-ubyte.gz')
            
            # データの読み込み
            x_train = load_gz_file(train_images_path)
            y_train = load_gz_file(train_labels_path, is_image=False)
            x_test = load_gz_file(test_images_path)
            y_test = load_gz_file(test_labels_path, is_image=False)
            
            # ラベルにノイズを追加
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            noisy_y_train, noise_info_tensor = add_label_noise(y_train_tensor, label_noise_rate, num_digits=10, num_colors=1)
            y_train = noisy_y_train.numpy()
            noise_info = noise_info_tensor.numpy()

            # 変換関数の定義
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),  
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            # 変換の適用
            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            
            # ラベルをテンソルに変換
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)
            
            # ノイズ情報をテンソルからNumPy配列に変換
            noise_info = noise_info_tensor.numpy()
            
            # NumPy形式で保存
            np.save(noisy_train_images_path, x_train_tensor.numpy())
            np.save(noisy_train_labels_path, y_train_tensor.numpy())
            np.save(noisy_train_noise_info_path, noise_info)  
            np.save(noisy_test_images_path, x_test_tensor.numpy())
            np.save(noisy_test_labels_path, y_test_tensor.numpy())
            print(f"Noisy dataset and noise_info saved to {noisy_dataset_dir}")

        # テンソルに変換
        x_train_tensor = torch.tensor(x_train, dtype=torch.float)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        noise_info = torch.tensor(noise_info, dtype=torch.int)

        # TensorDatasetの作成
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor, noise_info)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        
        num_classes = 10
        in_channels = 1
        imagesize = (32, 32)

    else:
        raise ValueError("Invalid dataset name")
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
    elif args.model == "resnet18_lib":
        if in_channels != 3:
            # torchvisionのResNet18は3チャネルが前提なので、最初の層を差し替え
            model = models.resnet18(num_classes=num_classes)
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            model = models.resnet18(num_classes=num_classes)
    elif args.model == "resnet18_scr":
        model = resnet18.make_resnet18(in_channels, num_classes)
        print(model)
    elif args.model == "resnet18k":
        k1 = 64 * args.model_width
        model = resnet18k_v2.make_resnet18k(k=k1, num_classes=num_classes)
        print('make resnet18k')
    elif args.model == "resnet_v2":
        k1 = 64 * args.model_width
        # ここで定義済みのResNet18_v2を呼び出す場合は、自前実装が必要
        # model = ResNet18_v2(num_classes=num_classes, k=k1)
        raise NotImplementedError("If you have a custom ResNet18_v2, please import it or implement it here.")
    else:
        raise ValueError("Invalid model name.")
    return model


class NoisyDataset(torch.utils.data.Dataset):
    """
    通常のDatasetにノイズフラグを付与するためのラッパー
    """
    def __init__(self, dataset, noise_info):
        self.dataset = dataset
        self.noise_info = noise_info

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # CIFARの場合 dataset[idx] -> (img, target)
        img, target = self.dataset[idx]
        noise_flag = self.noise_info[idx]
        return img, target, noise_flag


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

        # ノイズあり・なしを分けて処理
        noisy_indices = (noise_flags == 1).nonzero(as_tuple=True)[0]
        clean_indices = (noise_flags == 0).nonzero(as_tuple=True)[0]

        if len(noisy_indices) > 0:
            total_noisy += len(noisy_indices)
            loss_noisy = criterion(outputs[noisy_indices], labels[noisy_indices]).item()
            running_loss_noisy += loss_noisy
            correct_noisy += (predicted[noisy_indices] == labels[noisy_indices]).sum().item()

        if len(clean_indices) > 0:
            total_clean += len(clean_indices)
            loss_clean = criterion(outputs[clean_indices], labels[clean_indices]).item()
            running_loss_clean += loss_clean
            correct_clean += (predicted[clean_indices] == labels[clean_indices]).sum().item()

    avg_loss = running_loss / len(train_loader)
    avg_loss_noisy = running_loss_noisy / total_noisy if total_noisy > 0 else float('nan')
    avg_loss_clean = running_loss_clean / total_clean if total_clean > 0 else float('nan')
    accuracy_total = 100. * correct_total / total_samples
    accuracy_noisy = 100. * correct_noisy / total_noisy if total_noisy > 0 else float('nan')
    accuracy_clean = 100. * correct_clean / total_clean if total_clean > 0 else float('nan')
    error_total = 100. - accuracy_total
    error_noisy = 100. - accuracy_noisy if total_noisy > 0 else float('nan')
    error_clean = 100. - accuracy_clean if total_clean > 0 else float('nan')
    
    return (
        avg_loss, 
        accuracy_total, accuracy_noisy, accuracy_clean,
        avg_loss_noisy, avg_loss_clean,
        total_samples, total_noisy, total_clean,
        correct_total, correct_noisy, correct_clean,
        error_total, error_noisy, error_clean
    )


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

    return avg_loss, accuracy, total, correct, test_error


def add_label_noise(targets, label_noise_rate, num_digits=10, num_colors=1):
    noisy_targets = targets.clone()
    num_noisy = int(label_noise_rate * len(targets))
    noisy_indices = torch.randperm(len(targets))[:num_noisy]
    noise_info = torch.zeros(len(targets), dtype=torch.int)  # 0で初期化(=clean)

    if num_digits == 10 and num_colors == 1:
        for idx in noisy_indices:
            original_label = targets[idx].item()
            new_label = random.randint(0, num_digits - 1)
            while new_label == original_label:
                new_label = random.randint(0, num_digits - 1)
            noisy_targets[idx] = new_label
            noise_info[idx] = 1  # noisyにする

    elif num_digits == 10 and num_colors == 10:
        # カラー込みのケースがある場合はここに追加実装
        pass

    return noisy_targets, noise_info


def save_model(model, epoch, experiment_name):
    save_dir = f'./save_model/Cifar-10/{experiment_name}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'epoch_{epoch}.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def evaluate_initial_model(model, train_loader, test_loader, criterion, device, wandb_run, csv_path):
    """
    モデルの更新を一切行わず、初期状態（0epoch目）のモデルを用いて
    train, testに対する各種メトリクスを計算し、wandbおよびCSVに保存します。
    
    引数:
        model: 既に初期化済みのモデル
        train_loader: 学習用DataLoader（ノイズフラグ付きの場合もある）
        test_loader: テスト用DataLoader
        criterion: 損失関数（例: nn.CrossEntropyLoss()）
        device: 使用デバイス（CPUまたはGPU）
        wandb_run: wandbのrunオブジェクト（wandbを利用しない場合はNone）
        csv_path: ログ保存先のCSVファイルパス
    """
    import csv
    model.eval()  # 更新なしなのでevalモードでforward計算のみ
    
    # ---------------------------
    # Trainデータに対する評価
    # ---------------------------
    running_loss = 0.0
    total_samples = 0
    correct_total = 0

    running_loss_noisy = 0.0
    running_loss_clean = 0.0
    total_noisy = 0
    total_clean = 0
    correct_noisy = 0
    correct_clean = 0

    with torch.no_grad():
        for batch in train_loader:
            # ノイズフラグ付きの場合とそうでない場合で分岐
            if len(batch) == 3:
                inputs, labels, noise_flags = batch
                noise_flags = noise_flags.to(device)
            else:
                inputs, labels = batch
                noise_flags = None

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # バッチ内のサンプル数に合わせてlossを加重平均
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            _, predicted = torch.max(outputs, 1)
            correct_total += (predicted == labels).sum().item()

            # ノイズあり・なしがある場合は個別にloss/accuracyを計算
            if noise_flags is not None:
                noisy_indices = (noise_flags == 1).nonzero(as_tuple=True)[0]
                clean_indices = (noise_flags == 0).nonzero(as_tuple=True)[0]

                if len(noisy_indices) > 0:
                    total_noisy += len(noisy_indices)
                    loss_noisy = criterion(outputs[noisy_indices], labels[noisy_indices])
                    running_loss_noisy += loss_noisy.item() * len(noisy_indices)
                    correct_noisy += (predicted[noisy_indices] == labels[noisy_indices]).sum().item()

                if len(clean_indices) > 0:
                    total_clean += len(clean_indices)
                    loss_clean = criterion(outputs[clean_indices], labels[clean_indices])
                    running_loss_clean += loss_clean.item() * len(clean_indices)
                    correct_clean += (predicted[clean_indices] == labels[clean_indices]).sum().item()

    avg_loss = running_loss / total_samples
    accuracy_total = 100. * correct_total / total_samples
    error_total = 100. - accuracy_total

    if total_noisy > 0:
        avg_loss_noisy = running_loss_noisy / total_noisy
        accuracy_noisy = 100. * correct_noisy / total_noisy
        error_noisy = 100. - accuracy_noisy
    else:
        avg_loss_noisy = float('nan')
        accuracy_noisy = float('nan')
        error_noisy = float('nan')

    if total_clean > 0:
        avg_loss_clean = running_loss_clean / total_clean
        accuracy_clean = 100. * correct_clean / total_clean
        error_clean = 100. - accuracy_clean
    else:
        avg_loss_clean = float('nan')
        accuracy_clean = float('nan')
        error_clean = float('nan')

    # ---------------------------
    # Testデータに対する評価（既存のtest_model関数を利用）
    # ---------------------------
    test_loss, test_accuracy, test_total, test_correct, test_error = test_model(model, test_loader, criterion, device)

    # ---------------------------
    # ログ用ディクショナリの作成
    # ---------------------------
    metrics = {
        "epoch": 0,
        "train_loss": avg_loss,
        "train_accuracy": accuracy_total,
        "train_accuracy_noisy": accuracy_noisy,
        "train_accuracy_clean": accuracy_clean,
        "avg_loss_noisy": avg_loss_noisy,
        "avg_loss_clean": avg_loss_clean,
        "total_samples": total_samples,
        "total_noisy": total_noisy,
        "total_clean": total_clean,
        "correct_total": correct_total,
        "correct_noisy": correct_noisy,
        "correct_clean": correct_clean,
        "train_error_total": error_total,
        "train_error_noisy": error_noisy,
        "train_error_clean": error_clean,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_total_samples": test_total,
        "test_correct": test_correct,
        "test_error": test_error,
    }

    # ---------------------------
    # wandbへログ
    # ---------------------------
    if wandb_run is not None:
        wandb.log(metrics)

    # ---------------------------
    # CSVにログ（ファイルが存在しなければヘッダを書き込み）
    # ---------------------------
    header = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_accuracy_noisy",
        "train_accuracy_clean",
        "avg_loss_noisy",
        "avg_loss_clean",
        "total_samples",
        "total_noisy",
        "total_clean",
        "correct_total",
        "correct_noisy",
        "correct_clean",
        "train_error_total",
        "train_error_noisy",
        "train_error_clean",
        "test_loss",
        "test_accuracy",
        "test_total_samples",
        "test_correct",
        "test_error"
    ]
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

    print("Initial (0epoch) evaluation metrics saved to wandb and CSV.")


def main():
    print("start session")
    args = parse_args()

    set_seed(args.fix_seed)
    device = set_device(args.gpu)
    
    # データセット読み込み
    train_dataset, test_dataset, imagesize, num_classes, in_channels = load_datasets(args.dataset, args)

    # ラベルノイズを追加（EMNISTの場合はload時に処理）
    if args.dataset == "emnist_digits":
        if args.label_noise_rate > 0.0:
            print('Label noise has already been added and saved during dataset loading.')
            noise_info = train_dataset.tensors[2]  # すでにTensorDataset内にノイズ情報を格納済み
        else:
            print('No label noise applied to EMNIST dataset.')
            noise_info = torch.zeros(len(train_dataset), dtype=torch.int)
    else:
        # CIFARなどの場合
        if args.label_noise_rate > 0.0:
            print('Adding label noise to the dataset...')
            if args.dataset == "emnist_digits":
                # ここは本来呼ばれない想定だが残しておく
                y_train_noisy, noise_info = add_label_noise(train_dataset.tensors[1], args.label_noise_rate, num_digits=num_classes)
                train_dataset = TensorDataset(train_dataset.tensors[0], y_train_noisy, noise_info)
            else:
                # torchvision.datasets形式の場合
                y_train_noisy, noise_info = add_label_noise(torch.tensor(train_dataset.targets), args.label_noise_rate, num_digits=num_classes)
                train_dataset.targets = y_train_noisy.numpy()

            num_noisy_labels = torch.sum(noise_info).item()
            print('added_label_noise_dataset_size', num_noisy_labels)
        else:
            noise_info = torch.zeros(len(train_dataset), dtype=torch.int)

    # データセットをNoisyDatasetでラップ（EMNISTはTensorDatasetにすでにノイズフラグあり）
    if args.dataset != "emnist_digits":
        train_dataset = NoisyDataset(train_dataset, noise_info)
    else:
        train_dataset = TensorDataset(train_dataset.tensors[0], train_dataset.tensors[1], train_dataset.tensors[2])

    # DataLoaderの作成
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

    # モデル読み込み
    model = load_models(in_channels, args, imagesize, num_classes)
    model = model.to(device)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    # 実験名
    experiment_name = (
        f'{args.model}_width{args.model_width}_{args.dataset}_'
        f'lr{args.lr}_batch_size{args.batch_size}_epoch{args.epoch}_'
        f'LabelNoiseRate{args.label_noise_rate}_Optim{args.optimizer}_momentum{args.momentum}'
    )
    print(f'Experiment name: {experiment_name}')

    # wandb初期化
    wandb_run = None
    if args.wandb:
        wandb_run = setup_wandb(args, experiment_name)

    # CSVログ用ディレクトリ/ファイルの準備
    csv_dir = f"./save_model/Cifar-10/{experiment_name}/csv"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"training_metrics.csv")
    # CSVヘッダを書き込み（上書きモード'w'）
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "train_accuracy_noisy",
            "train_accuracy_clean",
            "avg_loss_noisy",
            "avg_loss_clean",
            "total_samples",
            "total_noisy",
            "total_clean",
            "correct_total",
            "correct_noisy",
            "correct_clean",
            "train_error_total",
            "train_error_noisy",
            "train_error_clean",
            "test_loss",
            "test_accuracy",
            "test_total_samples",
            "test_correct",
            "test_error"
        ])
    evaluate_initial_model(model, train_loader, test_loader, criterion, device, wandb_run, csv_path)
    print("Finished data loading.")
    # 開始時点のモデルを保存
    save_model(model, 0, experiment_name)

    for epoch in range(1, args.epoch + 1):
        (avg_loss,
         train_accuracy, train_accuracy_noisy, train_accuracy_clean,
         avg_loss_noisy, avg_loss_clean,
         total_samples, total_noisy, total_clean,
         correct_total, correct_noisy, correct_clean,
         error_total, error_noisy, error_clean) = train_model(model, train_loader, optimizer, criterion, device)

        (test_loss,
         test_accuracy,
         test_total_samples,
         test_correct,
         test_error) = test_model(model, test_loader, criterion, device)

        print(
            f"epoch: {epoch}, "
            f"train_loss: {avg_loss:.4f}, "
            f"train_accuracy_noisy: {train_accuracy_noisy:.4f}, "
            f"train_accuracy_clean: {train_accuracy_clean:.4f}, "
            f"test_accuracy: {test_accuracy:.4f}, "
            f"test_loss: {test_loss:.4f}"
        )

        # wandbログ
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
                'test_error': test_error,
                'avg_loss_noisy': avg_loss_noisy,
                'avg_loss_clean': avg_loss_clean,
                "train_error_total": error_total,
                "train_error_noisy": error_noisy,
                "train_error_clean": error_clean,
            })

        # CSVログにも同様の内容を追記（追加モード'a'）
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                avg_loss,
                train_accuracy,
                train_accuracy_noisy,
                train_accuracy_clean,
                avg_loss_noisy,
                avg_loss_clean,
                total_samples,
                total_noisy,
                total_clean,
                correct_total,
                correct_noisy,
                correct_clean,
                error_total,
                error_noisy,
                error_clean,
                test_loss,
                test_accuracy,
                test_total_samples,
                test_correct,
                test_error
            ])

        # 各エポック後にモデルを保存
        save_model(model, epoch, experiment_name)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
