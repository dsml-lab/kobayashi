#made by kobyahsi! 一号!

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torchvision.models as models
import numpy as np
import wandb
import argparse
import random
import csv
from torchvision import datasets

# import models written by scratch
from model.cnn_2layers import CNN2Layer
from model.cnn_5layers import CNN5Layer
from model import resnet18
from model import resnet18k_v2

# --------------------------
# CombinedDataset：trainとtestの画像データ結合用
class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data            # numpy array
        self.targets = targets      # list or numpy array
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]
        from PIL import Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

# --------------------------
# 各クラスごとに均等な割合でラベルノイズを付与する関数
def add_label_noise_balanced(data, label_noise_rate, num_classes):
    noisy_data = data.clone()
    noise_info = torch.zeros(len(data), dtype=torch.int)
    for c in range(num_classes):
        indices = (data == c).nonzero(as_tuple=True)[0]
        num_samples = len(indices)
        num_noisy = int(label_noise_rate * num_samples)
        if num_samples > 0 and num_noisy > 0:
            perm = torch.randperm(num_samples)[:num_noisy]
            noisy_indices = indices[perm]
            for idx in noisy_indices:
                original_label = data[idx].item()
                new_label = random.randint(0, num_classes - 1)
                while new_label == original_label:
                    new_label = random.randint(0, num_classes - 1)
                noisy_data[idx] = new_label
                noise_info[idx] = 1
    return noisy_data, noise_info

# --------------------------
def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-seed", "--fix_seed", type=int, default=42)
    
    arg_parser.add_argument("--model", type=str, choices=["cnn_2layers", "cnn_5layers", "resnet18_lib", "resnet18_scr", "resnet_v2", "resnet18k"], default="resnet18", help="モデルアーキテクチャの選択")
    arg_parser.add_argument("-model_width", "--model_width", type=int, default=1)
    arg_parser.add_argument("-epoch", "--epoch", type=int, default=10)  # demo用：少なめ
    
    arg_parser.add_argument("-datasets", "--dataset", type=str, choices=["cifar10", "cifar100"], default="cifar10")
    arg_parser.add_argument("-label_noise_rate", "--label_noise_rate", type=float, default=0.1)
    arg_parser.add_argument("-batch_size", "--batch_size", type=int, default=128, help="バッチサイズ")
    
    arg_parser.add_argument("-lr", "--lr", type=float, default=0.1, help="学習率")
    arg_parser.add_argument("-optimizer", "--optimizer", type=str, choices=["sgd", "adam"], default="adam", help="最適化手法")
    arg_parser.add_argument("-momentum", "--momentum", type=float, default=0.9, help="モーメンタム")
    
    arg_parser.add_argument("-loss", "--loss", type=str, choices=["cross_entropy", "focal_loss", "soft_cross_entropy"], default="cross_entropy", help="損失関数")
    arg_parser.add_argument("-num_workers", "--num_workers", type=int, default=4, help="データローダーの並列数")
    
    # wandb setting
    arg_parser.add_argument("-wandb", "--wandb", action='store_true', default=False, help="wandbを使用するかどうか")
    arg_parser.add_argument("-wandb_project", "--wandb_project", type=str, default="dd_scratch_cifar-10_model", help="wandbのプロジェクト名")
    arg_parser.add_argument("--wandb_entity", type=str, default="dsml-kernel24", help="wandbのエンティティ名")
    
    # GPUの選択：カンマ区切りで指定（例："0,1"）
    arg_parser.add_argument("--gpus", type=str, default="0,1", help="使用するGPUのインデックス（カンマ区切り）")
    
    return arg_parser.parse_args()

# --------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------------
def load_datasets(dataset, args):
    if dataset == "cifar10":
        # 学習時はデータ拡張も行う transform
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        imagesize = (32, 32)
        num_classes = 10
        in_channels = 3
    
    elif dataset == "cifar100":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=train_transform)
        imagesize = (32, 32)
        num_classes = 100
        in_channels = 3
    else:
        raise ValueError("Invalid dataset name")
    return train_dataset, test_dataset, imagesize, num_classes, in_channels

# --------------------------
class NoisyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, noise_info):
        self.dataset = dataset
        self.noise_info = noise_info
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        noise_flag = self.noise_info[idx]
        return img, target, noise_flag

# --------------------------
def load_models(in_channels, args, img_size, num_classes):
    if args.model == "cnn_2layers":
        model = CNN2Layer(in_channels, num_classes, args.model_width, img_size)
    elif args.model == "cnn_5layers":
        model = CNN5Layer(in_channels, num_classes, args.model_width, img_size)
    elif args.model == "resnet18_lib":
        model = models.resnet18(num_classes=num_classes)
    elif args.model == "resnet18_scr":
        model = resnet18.make_resnet18(in_channels, num_classes)
    elif args.model == "resnet18k":
        k1 = 64 * args.model_width
        model = resnet18k_v2.make_resnet18k(k=k1, num_classes=num_classes)
    elif args.model == "resnet18_v2":
        k1 = 64 * args.model_width
        model = ResNet18_v2(num_classes=num_classes, k=k1)
    else:
        raise ValueError("Invalid model name.")
    return model

# --------------------------
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
            if noise_flags[idx] == 1:
                total_noisy += 1
                running_loss_noisy += criterion(outputs[idx].unsqueeze(0), labels[idx].unsqueeze(0)).item()
                if predicted[idx] == labels[idx]:
                    correct_noisy += 1
            else:
                total_clean += 1
                running_loss_clean += criterion(outputs[idx].unsqueeze(0), labels[idx].unsqueeze(0)).item()
                if predicted[idx] == labels[idx]:
                    correct_clean += 1

    avg_loss = running_loss / len(train_loader)
    avg_loss_noisy = running_loss_noisy / total_noisy if total_noisy > 0 else float('nan')
    avg_loss_clean = running_loss_clean / total_clean if total_clean > 0 else float('nan')
    error_total = 100. - (100. * correct_total / total_samples)
    error_noisy = 100. - (100. * correct_noisy / total_noisy) if total_noisy > 0 else float('nan')
    error_clean = 100. - (100. * correct_clean / total_clean) if total_clean > 0 else float('nan')
    return avg_loss, total_samples, total_noisy, total_clean, avg_loss_noisy, avg_loss_clean, error_total, error_noisy, error_clean

# --------------------------
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss_sum = 0.0
    total_samples = 0
    correct_total = 0

    total_noisy = 0
    correct_noisy = 0
    loss_noisy = 0.0

    total_clean = 0
    correct_clean = 0
    loss_clean = 0.0

    with torch.no_grad():
        for inputs, labels, noise_flags in test_loader:
            inputs, labels, noise_flags = inputs.to(device), labels.to(device), noise_flags.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            test_loss_sum += batch_loss.item() * labels.size(0)
            total_samples += labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_total += (predicted == labels).sum().item()

            for idx in range(len(labels)):
                if noise_flags[idx] == 1:
                    total_noisy += 1
                    loss_noisy += criterion(outputs[idx].unsqueeze(0), labels[idx].unsqueeze(0)).item()
                    if predicted[idx] == labels[idx]:
                        correct_noisy += 1
                else:
                    total_clean += 1
                    loss_clean += criterion(outputs[idx].unsqueeze(0), labels[idx].unsqueeze(0)).item()
                    if predicted[idx] == labels[idx]:
                        correct_clean += 1

    avg_loss_overall = test_loss_sum / total_samples
    avg_loss_noisy = loss_noisy / total_noisy if total_noisy > 0 else float('nan')
    avg_loss_clean = loss_clean / total_clean if total_clean > 0 else float('nan')
    error_overall = 100. - (100. * correct_total / total_samples)
    error_noisy = 100. - (100. * correct_noisy / total_noisy) if total_noisy > 0 else float('nan')
    error_clean = 100. - (100. * correct_clean / total_clean) if total_clean > 0 else float('nan')
    return avg_loss_overall, total_samples, error_overall, avg_loss_noisy, avg_loss_clean, total_noisy, total_clean, error_noisy, error_clean

# --------------------------
def main():
    print("start session")
    args = parse_args()
    set_seed(args.fix_seed)

    # GPUのリストは --gpus で指定（例："0,1"）
    gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip().isdigit()]
    if not gpu_list:
        gpu_list = [0]
    print(f"Using GPUs: {gpu_list}")

    # データセットの読み込み（train, test を個別に取得）
    train_dataset, test_dataset, imagesize, num_classes, in_channels = load_datasets(args.dataset, args)
    
    # train と test のデータを結合
    combined_data = np.concatenate([train_dataset.data, test_dataset.data], axis=0)
    combined_targets = np.concatenate([np.array(train_dataset.targets), np.array(test_dataset.targets)], axis=0)
    # ここでは学習用のtransform（データ拡張付き）を使用
    combined_dataset = CombinedDataset(combined_data, combined_targets, transform=train_dataset.transform)

    # ラベルノイズの付与
    combined_targets_tensor = torch.tensor(combined_dataset.targets)
    noisy_targets, noise_info = add_label_noise_balanced(combined_targets_tensor, args.label_noise_rate, num_classes)
    combined_dataset.targets = noisy_targets.numpy().tolist()

    # ノイズ情報を含むデータセット
    combined_noisy_dataset = NoisyDataset(combined_dataset, noise_info)

    # クロスバリデーション用の層別キー作成（ラベルと noise_flag の組み合わせ）
    targets_arr = np.array(combined_dataset.targets)
    noise_info_arr = noise_info.numpy()
    stratify_keys = targets_arr * 2 + noise_info_arr

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.fix_seed)

    # ディレクトリ構造の作成
    parent_dir = "cifar-10_5fold"
    os.makedirs(parent_dir, exist_ok=True)
    validation_dir = os.path.join(parent_dir, "validation")
    os.makedirs(validation_dir, exist_ok=True)

    csv_records = []  # validation分布の記録用
    fold_idx = 0

    # 親実験グループ名
    parent_experiment = f'{args.model}_width{args.model_width*64}_{args.dataset}_lr{args.lr}_batch_size{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}_Optim{args.optimizer}_momentum{args.momentum}'

    for train_idx, val_idx in skf.split(np.zeros(len(stratify_keys)), stratify_keys):
        # 各foldごとに GPU を割り当て（gpu_list からラウンドロビン）
        assigned_gpu = gpu_list[fold_idx % len(gpu_list)]
        fold_device = torch.device(f"cuda:{assigned_gpu}") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Fold {fold_idx} assigned to GPU: {fold_device}")

        # 各fold用ディレクトリ作成（例: fold1～fold5, その中に metrics フォルダ）
        fold_dir = os.path.join(parent_dir, f"fold{fold_idx+1}")
        os.makedirs(fold_dir, exist_ok=True)
        metrics_dir = os.path.join(fold_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        # トレーニング用は、元の combined_noisy_dataset を Subset として使用（データ拡張付き）
        train_subset = torch.utils.data.Subset(combined_noisy_dataset, train_idx)
        
        # バリデーション用は、データ拡張を行わない正規化のみの transform を使用して新たに作成
        # ここで combined_dataset.data と combined_dataset.targets を利用して新たな CombinedDataset を作成
        val_data = combined_dataset.data[val_idx]
        val_targets = np.array(combined_dataset.targets)[val_idx]
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        val_dataset_new = CombinedDataset(val_data, val_targets, transform=val_transform)
        # noise_info[val_idx] をそのまま利用
        val_dataset_new = NoisyDataset(val_dataset_new, noise_info[val_idx])
        
        # DataLoaderの設定
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset_new, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # バリデーション集合の各ラベル分布の記録
        val_labels = []
        val_noise_flags = []
        for i in val_idx:
            label = combined_noisy_dataset.dataset.targets[i]
            noise_flag = noise_info_arr[i]
            val_labels.append(label)
            val_noise_flags.append(noise_flag)
        val_labels = np.array(val_labels)
        val_noise_flags = np.array(val_noise_flags)
        for label in range(num_classes):
            total_count = int(np.sum(val_labels == label))
            noisy_count = int(np.sum((val_labels == label) & (val_noise_flags == 1)))
            record = {"fold": fold_idx, "label": label, "total": total_count, "noisy": noisy_count}
            csv_records.append(record)

        # モデル初期化
        model = load_models(in_channels, args, imagesize, num_classes)
        model = model.to(fold_device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) if args.optimizer == "sgd" else optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        # 各foldの wandb 初期化（親グループの子ランとして）
        if args.wandb:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"fold_{fold_idx}",
                group=parent_experiment,
                entity=args.wandb_entity,
                config=args
            )

        fold_logs = []  # 各foldのログ記録用
        for epoch in range(args.epoch):
            epoch_num = epoch + 1
            # 学習
            train_results = train_model(model, train_loader, optimizer, criterion, fold_device)
            # train_results の順番:
            # avg_loss, train_total_samples, train_total_noisy, train_total_clean, train_avg_loss_noisy, train_avg_loss_clean, train_error_total, train_error_noisy, train_error_clean
            (avg_loss, train_total_samples, train_total_noisy, train_total_clean,
             train_avg_loss_noisy, train_avg_loss_clean, train_error_total, train_error_noisy, train_error_clean) = train_results

            # 検証
            test_results = test_model(model, val_loader, criterion, fold_device)
            # test_results の順番:
            # test_loss, test_total_samples, test_error, test_avg_loss_noisy, test_avg_loss_clean, test_total_noisy, test_total_clean, test_error_noisy, test_error_clean
            (test_loss, test_total_samples, test_error, test_avg_loss_noisy, test_avg_loss_clean,
             test_total_noisy, test_total_clean, test_error_noisy, test_error_clean) = test_results

            print(f"Fold: {fold_idx}, Epoch: {epoch_num}, Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}, Test Error: {test_error:.2f}%")
            
            log_record = {
                'fold': fold_idx,
                'epoch': epoch_num,
                'train_loss': avg_loss,
                'train_total_samples': train_total_samples,
                'train_total_noisy': train_total_noisy,
                'train_total_clean': train_total_clean,
                'test_total_samples': test_total_samples,
                'test_error': test_error,
                'test_error_noisy': test_error_noisy,
                'test_error_clean': test_error_clean,
                'test_loss': test_loss,
                'test_avg_loss_clean': test_avg_loss_clean,
                'test_avg_loss_noisy': test_avg_loss_noisy,
                'train_avg_loss_noisy': train_avg_loss_noisy,
                'train_avg_loss_clean': train_avg_loss_clean,
                'train_error_total': train_error_total,
                'train_error_noisy': train_error_noisy,
                'train_error_clean': train_error_clean,
            }
            fold_logs.append(log_record)
            if args.wandb:
                wandb.log(log_record)

        # 各foldのログを metrics フォルダ直下に CSV として保存
        csv_filename = os.path.join(metrics_dir, "log.csv")
        fieldnames = ['epoch', 'train_loss', 'train_total_samples', 'train_total_noisy', 'train_total_clean',
                      'test_total_samples', 'test_error', 'test_error_noisy', 'test_error_clean',
                      'test_loss', 'test_avg_loss_clean', 'test_avg_loss_noisy',
                      'train_avg_loss_noisy', 'train_avg_loss_clean', 'train_error_total', 'train_error_noisy', 'train_error_clean']
        with open(csv_filename, mode="w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for record in fold_logs:
                writer.writerow(record)
        print(f"Fold {fold_idx} logs saved to {csv_filename}")

        if args.wandb:
            wandb.finish()
        fold_idx += 1

    # validation分布のCSVを validation フォルダに保存
    csv_val_filename = os.path.join(validation_dir, "validation_distribution.csv")
    with open(csv_val_filename, mode="w", newline="") as csv_file:
        fieldnames = ["fold", "label", "total", "noisy"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in csv_records:
            writer.writerow(record)
    print(f"Validation distribution saved to {csv_val_filename}")

if __name__ == "__main__":
    main()
