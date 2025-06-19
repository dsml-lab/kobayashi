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

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-seed", "--fix_seed", type=int, default=42)
    
    arg_parser.add_argument("--model", type=str, 
                            choices=["cnn_2layers", "cnn_3layers", "cnn_4layers", "cnn_5layers", "cnn_8layers", "cnn_16layers", "resnet18"], 
                            help="モデルアーキテクチャの選択")
    arg_parser.add_argument("-model_width", "--model_width", type=int, default=1)
    arg_parser.add_argument("-epoch", "--epoch", type=int, default=1000)
    
    arg_parser.add_argument("-datasets", "--dataset", type=str, 
                            choices=["mnist", "emnist", "emnist_digits", 
                                     "cifar10", "cifar100", "tinyImageNet", 
                                     "colored_emnist", "distribution_colored_emnist"], 
                            default="cifar10")
    arg_parser.add_argument("-variance", "--variance", type=int, default=10000)
    arg_parser.add_argument("-correlation", "--correlation", type=float, default=0.5)
    arg_parser.add_argument("-label_noise_rate", "--label_noise_rate", type=float, default=0.0)
    arg_parser.add_argument("-gray_scale", "--gray_scale", action='store_true')
    arg_parser.add_argument("-batch_size", "--batch_size", type=int, default=128)
    arg_parser.add_argument("-img_size", "--img_size", type=int, default=32)

    # ※ combined を削除し、digit / color のみ受け付ける
    arg_parser.add_argument("-target", "--target", type=str, 
                            choices=["digit", "color"],
                            default='digit', 
                            help="colored EMNISTのターゲット指定: digit or color")
    
    arg_parser.add_argument("-lr", "--lr", type=float, default=0.1)
    arg_parser.add_argument("-optimizer", "--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    arg_parser.add_argument("-momentum", "--momentum", type=float, default=0.9)
    
    arg_parser.add_argument("-loss", "--loss", type=str, choices=["cross_entropy", "focal_loss"], default="cross_entropy")
    
    arg_parser.add_argument("-gpu", "--gpu", type=int, default=0)
    arg_parser.add_argument("-num_workers", "--num_workers", type=int, default=4)
    
    arg_parser.add_argument("-wandb", "--wandb", action='store_true', default=True)
    arg_parser.add_argument("-wandb_project", "--wandb_project", type=str, default="dd_scratch_models")
    arg_parser.add_argument("--wandb_entity", type=str, default="dsml-kernel24")
    
    arg_parser.add_argument("-weight_noisy", "--weight_noisy", type=float, default=1.0)
    arg_parser.add_argument("-weight_clean", "--weight_clean", type=float, default=1.0)

    return arg_parser.parse_args()


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


class BalancedBatchSampler(Sampler):
    def __init__(self, clean_indices, noisy_indices, batch_size, drop_last):
        self.clean_indices = clean_indices
        self.noisy_indices = noisy_indices
        self.batch_size = batch_size
        self.drop_last = drop_last

        assert batch_size % 2 == 0, "Batch size must be even for balanced batches"
        self.num_samples_per_class = batch_size // 2

    def __iter__(self):
        random.shuffle(self.clean_indices)
        random.shuffle(self.noisy_indices)

        min_len = min(len(self.clean_indices), len(self.noisy_indices))
        num_batches = min_len // self.num_samples_per_class

        for i in range(num_batches):
            clean_batch = self.clean_indices[i*self.num_samples_per_class:(i+1)*self.num_samples_per_class]
            noisy_batch = self.noisy_indices[i*self.num_samples_per_class:(i+1)*self.num_samples_per_class]
            batch = clean_batch + noisy_batch
            random.shuffle(batch)
            yield batch

        if not self.drop_last:
            remaining_clean = self.clean_indices[num_batches*self.num_samples_per_class:]
            remaining_noisy = self.noisy_indices[num_batches*self.num_samples_per_class:]
            if len(remaining_clean) >= self.num_samples_per_class and len(remaining_noisy) >= self.num_samples_per_class:
                batch = remaining_clean[:self.num_samples_per_class] + remaining_noisy[:self.num_samples_per_class]
                random.shuffle(batch)
                yield batch

    def __len__(self):
        return len(self.clean_indices) // self.num_samples_per_class


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(gpu_id):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    return device


def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()


def apply_transform(x, transform):
    transformed_x = []
    for img in x:
        img = transform(img)
        transformed_x.append(img)
    return torch.stack(transformed_x)


def load_datasets(dataset, target, gray_scale, args):
    """
    データセットを読み込み、target='digit'/'color' に応じて
    ラベルを10クラスに変換して返す。
    """
    # 例: colored_emnist / distribution_colored_emnist など
    #    もともと 0~99 の combined ラベル -> digit=(y//10), color=(y%10)
    # ※ combined関連は削除

    if dataset == "colored_emnist":
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        x_train = np.load('data/colored_EMNIST/x_train_colored.npy')
        y_train_combined = np.load('data/colored_EMNIST/y_train_combined.npy')  # 0~99
        x_test = np.load('data/colored_EMNIST/x_test_colored.npy')
        y_test_combined = np.load('data/colored_EMNIST/y_test_combined.npy')    # 0~99

        # ラベル変換: digit=10クラス, color=10クラス
        if target == 'digit':
            # 上位 10 の位だけ使う => y // 10
            y_train = y_train_combined // 10
            y_test = y_test_combined // 10
            num_classes = 10
        else:  # target == 'color'
            # 下位 1 の位だけ使う => y % 10
            y_train = y_train_combined % 10
            y_test = y_test_combined % 10
            num_classes = 10
        
        x_train_tensor = apply_transform(x_train, transform)
        x_test_tensor = apply_transform(x_test, transform)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

        imagesize = (32, 32)
        in_channels = 3

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

        x_train = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/x_train_colored.npy')
        y_train_combined = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/y_train_combined.npy')
        x_test = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/x_test_colored.npy')
        y_test_combined = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/y_test_combined.npy')

        if target == 'digit':
            y_train = y_train_combined // 10
            y_test = y_test_combined // 10
            num_classes = 10
        else:  # target == 'color'
            y_train = y_train_combined % 10
            y_test = y_test_combined % 10
            num_classes = 10

        x_train_tensor = apply_transform(x_train, transform)
        x_test_tensor = apply_transform(x_test, transform)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

        imagesize = (32, 32)
        in_channels = 3
    else:
        # CIFARやMNIST等、他のデータセットは従来通りの処理へ (必要に応じて実装)
        # ここでは省略例
        train_dataset, test_dataset = None, None
        imagesize = (32, 32)
        in_channels = 3
        num_classes = 10
        # ↑ 適宜修正してください

    if gray_scale:
        transform_gray = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset.transform = transform_gray
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


def add_label_noise(targets, label_noise_rate, num_classes):
    """
    digit(10クラス)またはcolor(10クラス)のラベルノイズ付与
    """
    noisy_targets = targets.clone()
    num_noisy = int(label_noise_rate * len(targets))
    noisy_indices = torch.randperm(len(targets))[:num_noisy]
    noise_info = torch.zeros(len(targets), dtype=torch.int)

    for idx in noisy_indices:
        original_label = targets[idx].item()
        # 範囲は 0 ~ (num_classes - 1) => 10クラス
        new_label = random.randint(0, num_classes - 1)
        while new_label == original_label:
            new_label = random.randint(0, num_classes - 1)
        noisy_targets[idx] = new_label
        noise_info[idx] = 1

    return noisy_targets, noise_info


def train_model(model, train_loader, optimizer, criterion, 
                weight_noisy, weight_clean, device):
    """
    digit or color いずれも単一ラベル10クラスなので、単純に1つの精度を計算。
    noisy/clean の区別は従来通り。
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    correct_total = 0

    correct_noisy = 0
    total_noisy = 0
    correct_clean = 0
    total_clean = 0

    # 損失をサンプル単位で計算
    loss_values = []
    loss_values_noisy = []
    loss_values_clean = []

    criterion.reduction = 'none'

    # noisy / clean の重み正規化
    total_weight = weight_clean + weight_noisy
    normalized_weight_clean = (weight_clean / total_weight) * 2
    normalized_weight_noisy = (weight_noisy / total_weight) * 2

    for inputs, labels, noise_labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        noise_labels = noise_labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)
        batch_size = labels.size(0)
        total_samples += batch_size
        correct_batch = (predicted == labels).sum().item()
        correct_total += correct_batch

        # noisy/clean のインデックス
        idx_noisy = (noise_labels == 1)
        idx_clean = (noise_labels == 0)

        per_sample_loss = criterion(outputs, labels)  # shape: [batch_size]

        weights = torch.zeros_like(per_sample_loss)
        weights[idx_noisy] = normalized_weight_noisy
        weights[idx_clean] = normalized_weight_clean

        loss = (per_sample_loss * weights).mean()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size

        # noisy サンプル
        if idx_noisy.sum() > 0:
            labels_noisy = labels[idx_noisy]
            predicted_noisy = predicted[idx_noisy]
            correct_noisy += (predicted_noisy == labels_noisy).sum().item()
            total_noisy += idx_noisy.sum().item()
            loss_values_noisy.extend((per_sample_loss[idx_noisy] * weights[idx_noisy]).detach().cpu().numpy())

        # clean サンプル
        if idx_clean.sum() > 0:
            labels_clean = labels[idx_clean]
            predicted_clean = predicted[idx_clean]
            correct_clean += (predicted_clean == labels_clean).sum().item()
            total_clean += idx_clean.sum().item()
            loss_values_clean.extend((per_sample_loss[idx_clean] * weights[idx_clean]).detach().cpu().numpy())

        loss_values.extend((per_sample_loss * weights).detach().cpu().numpy())

    # 平均・分散loss
    avg_loss = np.mean(loss_values) if loss_values else float('nan')
    var_loss = np.var(loss_values) if loss_values else float('nan')
    avg_loss_noisy = np.mean(loss_values_noisy) if loss_values_noisy else float('nan')
    var_loss_noisy = np.var(loss_values_noisy) if loss_values_noisy else float('nan')
    avg_loss_clean = np.mean(loss_values_clean) if loss_values_clean else float('nan')
    var_loss_clean = np.var(loss_values_clean) if loss_values_clean else float('nan')

    accuracy_total = 100. * correct_total / total_samples if total_samples > 0 else float('nan')
    accuracy_noisy = 100. * correct_noisy / total_noisy if total_noisy > 0 else float('nan')
    accuracy_clean = 100. * correct_clean / total_clean if total_clean > 0 else float('nan')

    return {
        'avg_loss': avg_loss,
        'var_loss': var_loss,
        'accuracy_total': accuracy_total,
        'accuracy_noisy': accuracy_noisy,
        'accuracy_clean': accuracy_clean,
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


def test_model(model, test_loader, device):
    """
    digit or color 単一ラベル(10クラス)のテスト。
    """
    model.eval()
    test_loss = 0.0
    correct_total = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

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


def main():
    print('start session')
    args = parse_args()
    set_seed(args.fix_seed)
    device = set_device(args.gpu)

    print(f"{args.target}")
    # データセット読み込み
    print('loading datasets')
    train_dataset, test_dataset, imagesize, num_classes, in_channels = load_datasets(
        args.dataset, args.target, args.gray_scale, args
    )

    # digit => 10クラス, color => 10クラス
    # ※ combined 機能は削除

    # ラベルノイズを付与
    if args.label_noise_rate > 0.0:
        print('adding label noise')
        if hasattr(train_dataset, 'tensors'):
            x_train, y_train = train_dataset.tensors
            y_train_noisy, noise_info = add_label_noise(y_train, args.label_noise_rate, num_classes)
            new_train_dataset = TensorDataset(x_train, y_train_noisy)
            train_dataset = NoisyDataset(new_train_dataset, noise_info)
        else:
            # datasetsのtargets属性を持つ場合
            y_train = torch.tensor(train_dataset.targets)
            y_train_noisy, noise_info = add_label_noise(y_train, args.label_noise_rate, num_classes)
            train_dataset.targets = y_train_noisy.tolist()
            train_dataset = NoisyDataset(train_dataset, noise_info)
    else:
        noise_info = torch.zeros(len(train_dataset), dtype=torch.int)
        if hasattr(train_dataset, 'tensors'):
            x_train, y_train = train_dataset.tensors
            new_train_dataset = TensorDataset(x_train, y_train)
            train_dataset = NoisyDataset(new_train_dataset, noise_info)
        else:
            train_dataset = NoisyDataset(train_dataset, noise_info)
    clean_indices = [i for i, label in enumerate(noise_info) if label == 0]
    noisy_indices = [i for i, label in enumerate(noise_info) if label == 1]


    
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

    # モデルロード
    model = load_models(in_channels, args, imagesize, num_classes)
    model.to(device)

    # Optimizer
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss(reduction='none')
    weight_noisy = args.weight_noisy
    weight_clean = args.weight_clean

    experiment_name = (
        f'{args.target}_{args.variance}_{args.model}_{args.dataset}_lr{args.lr}_'
        f'batch{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}_'
        f'Optim{args.optimizer}_cleanw{args.weight_clean}_noisew{args.weight_noisy}'
    )
    print(f'Experiment name: {experiment_name}')

    if args.wandb:
        wandb.init(project=args.wandb_project, name=experiment_name, entity=args.wandb_entity)
        wandb.config.update(args)

    # CSV設定 (combined関連の列は削除)
    csv_dir = f"./csv/no_combined/{args.target}/{experiment_name}"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, 'log.csv')
    if not os.path.isfile(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss", "train_loss_variance",
                "train_accuracy", "train_accuracy_noisy", "train_accuracy_clean",
                "test_loss", "test_accuracy"
            ])

    print('start training and testing')
    for epoch in range(1, args.epoch + 1):
        train_metrics = train_model(
            model, train_loader, optimizer, criterion, 
            weight_noisy, weight_clean, device
        )
        test_metrics = test_model(model, test_loader, device)

        # 標準出力
        print(f"Epoch: {epoch} "
              f"| Train Loss: {train_metrics['avg_loss']:.4f}, Var: {train_metrics['var_loss']:.4f} "
              f"| Train Acc: {train_metrics['accuracy_total']:.2f}% "
              f"| Test Loss: {test_metrics['avg_loss']:.4f} "
              f"| Test Acc: {test_metrics['accuracy_total']:.2f}%")

        # CSV保存
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics['avg_loss'], train_metrics['var_loss'],
                train_metrics['accuracy_total'], train_metrics['accuracy_noisy'], train_metrics['accuracy_clean'],
                test_metrics['avg_loss'], test_metrics['accuracy_total']
            ])

        # WandBログ
        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['avg_loss'],
                'train_loss_variance': train_metrics['var_loss'],
                'train_accuracy': train_metrics['accuracy_total'],
                'train_accuracy_noisy': train_metrics['accuracy_noisy'],
                'train_accuracy_clean': train_metrics['accuracy_clean'],
                'test_loss': test_metrics['avg_loss'],
                'test_accuracy': test_metrics['accuracy_total'],
                'total_samples': train_metrics['total_samples'],
                'total_noisy': train_metrics['total_noisy'],
                'total_clean': train_metrics['total_clean'],
                'correct_total': train_metrics['correct_total'],
                'correct_noisy': train_metrics['correct_noisy'],
                'correct_clean': train_metrics['correct_clean'],
                'test_total_samples': test_metrics['total_samples'],
                'test_correct': test_metrics['correct_total'],
                'avg_loss_noisy': train_metrics['avg_loss_noisy'],
                'avg_loss_clean': train_metrics['avg_loss_clean'],
                'var_loss_noisy': train_metrics['var_loss_noisy'],
                'var_loss_clean': train_metrics['var_loss_clean']
            })

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
