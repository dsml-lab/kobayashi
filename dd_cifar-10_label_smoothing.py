# made by kobyahsi! 一号!

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
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

    arg_parser.add_argument("--model", type=str, choices=["cnn_2layers", "cnn_5layers", "resnet18_lib","resnet18_scr","resnet_v2",'resnet18k'], default='resnet18',help="モデルアーキテクチャの選択")
    arg_parser.add_argument("-model_width", "--model_width", type=int, default=1)
    arg_parser.add_argument("-epoch", "--epoch", type=int, default=1000)


    arg_parser.add_argument("-datasets", "--dataset", type=str, choices=["cifar10","cifar100"], default="cifar10")
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
    arg_parser.add_argument("-wandb", "--wandb", action='store_true', help="wandbを使用するかどうか")
    arg_parser.add_argument("-wandb_project", "--wandb_project", type=str, default="dd_scratch_models", help="wandbのプロジェクト名")
    arg_parser.add_argument("--wandb_entity", type=str, default="dsml-kernel24", help="wandbのエンティティ名")
    
    # 追加: クラス重みとスムージングの割合
    arg_parser.add_argument("--class_weights", type=float, nargs='+', default=None, help="各クラスの重みをスペース区切りで指定")
    arg_parser.add_argument("--smoothing", type=float, default=0.0, help="ラベルスムージングの割合")

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
    elif args.model == "resnet18_v2":
        k1 = 64 * args.model_width
        model = ResNet18_v2(num_classes=num_classes,k = k1)
        print(model)
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

def add_label_noise(data, label_noise_rate, num_classes):
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

    return noisy_data, noise_info

def main():
    print("start session")
    
    args = parse_args()

    set_seed(args.fix_seed)
    
    device = set_device(args.gpu)

    # load datasets
    train_dataset, test_dataset, imagesize, num_classes, in_channels = load_datasets(args.dataset, args)
    # ラベルノイズの設定

    if args.label_noise_rate > 0.0:
        print('adding label noise')
        y_train_noisy, noise_info = add_label_noise(torch.tensor(train_dataset.targets), args.label_noise_rate, num_classes)
        train_dataset.targets = y_train_noisy.numpy()  # 更新されたターゲットを設定
        print('train_dataset_size', len(train_dataset))

        # 確認
        num_noisy_labels = torch.sum(noise_info).item()  # 1の個数をカウント
        print('added_label_noise_dataset_size', num_noisy_labels)
    else:
        noise_info = torch.zeros(len(train_dataset), dtype=torch.int)  # ノイズ情報を保存する列

    # データセットにノイズ情報を含める
    train_dataset = NoisyDataset(train_dataset, noise_info)
    # test_dataset はそのまま使用します
    # test_dataset = NoisyDataset(test_dataset, torch.zeros(len(test_dataset), dtype=torch.int))

    # データローダーの設定
    print('setting data loader')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # load model
    print('load model')
    model = load_models(in_channels, args, imagesize, num_classes)
    model = model.to(device)  

    print(model)

    # クラス重みの定義
    if args.class_weights is not None:
        if len(args.class_weights) != num_classes:
            raise ValueError(f"クラス重みの数がクラス数({num_classes})と一致しません。")
        class_weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    else:
        class_weights = torch.ones(num_classes, dtype=torch.float32).to(device)  # デフォルトは全て1.0

    # 損失関数の初期化
    if args.loss == "cross_entropy":
        train_criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean', label_smoothing=0.0)
        test_criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean', label_smoothing=0.0)
    elif args.loss == "focal_loss":
        # FocalLoss の実装が必要です。ここでは例としてCrossEntropyLossを使用
        # 実際にはFocalLossの実装を追加してください
        train_criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean', label_smoothing=0.0)
        test_criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean', label_smoothing=0.0)
    elif args.loss == "soft_cross_entropy":
        train_criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean', label_smoothing=args.smoothing)
        test_criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean', label_smoothing=0.0)  # テスト時はスムージングなし
    else:
        raise ValueError("Invalid loss type specified.")

    # オプティマイザの設定
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) if args.optimizer == "sgd" else optim.Adam(model.parameters(), lr=args.lr)
    print(optimizer)

    experiment_name = f'label_smoothing_rate{args.smoothing}_{args.model}_width{args.model_width * 64}_{args.dataset}_lr{args.lr}_batch_size{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}_Optim{args.optimizer}_Loss{args.loss}'
        
    print(f'Experiment name: {experiment_name}')

    # wandbの初期化
    if args.wandb:
        wandb.init(project=args.wandb_project, name=experiment_name, entity=args.wandb_entity)
        wandb.config.update(args)

    print("finish data loading")
    for epoch in range(args.epoch):
        epoch = epoch + 1  # start from 1
        avg_loss, train_accuracy, train_accuracy_noisy, train_accuracy_clean, avg_loss_noisy, avg_loss_clean, \
        total_samples, total_noisy, total_clean, correct_total, correct_noisy, correct_clean, error_total, error_noisy, error_clean = train_model(
            model, train_loader, optimizer, train_criterion, device)

        test_loss, test_accuracy, test_total_samples, test_correct, test_error = test_model(
            model, test_loader, test_criterion, device)  # 修正後の呼び出し
    
        print(f"epoch: {epoch}, train_loss: {avg_loss:.4f}, train_accuracy_noisy: {train_accuracy_noisy:.2f}, train_accuracy_clean: {train_accuracy_clean:.2f}, test_accuracy: {test_accuracy:.2f}, test_loss: {test_loss:.4f}")
        
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

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
