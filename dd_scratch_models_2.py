bbimport os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torch.backends.cudnn as cudnn
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
# import models written by scratch
from model.cnn_2layers import CNN2Layer
from model.cnn_5layers import CNN5Layer
from model.resnet18 import ResNet18
# download datasets from pytorch
from torchvision import datasets

# Ignore warnings
warnings.filterwarnings("ignore")

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
    arg_parser.add_argument("-datasets", "--dataset", type=str, choices=["mnist", "emnist", "cifar10", "cifar100", "tinyImageNet", "colored_emnist"], default="cifar10")
    arg_parser.add_argument("-label_noise_rate", "--label_noise_rate", type=float, default=0.0) 
    arg_parser.add_argument("-gray_scale", "--gray_scale", action='store_true', help="グレースケールに変換するかどうか")
    arg_parser.add_argument("-batch", "--batch", type=int, default=128, help="バッチサイズ")
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

def apply_transform(x, transform):
    transformed_x = []
    for img in x:
        img = transform(img)
        transformed_x.append(img)
    return torch.stack(transformed_x)

def load_datasets(dataset, target, gray_scale):
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

def add_label_noise(targets, label_noise_rate, num_classes):
    """
    Add label noise to the targets based on the specified label noise rate.
    
    Args:
        targets (torch.Tensor): The original target labels.
        label_noise_rate (float): The rate of label noise to add.
        num_classes (int): The number of classes in the dataset.
        
    Returns:
        torch.Tensor: The noisy target labels.
    """
    noisy_targets = targets.clone()
    # ラベルノイズを追加するインデックスをランダムに選ぶ
    num_noisy = int(label_noise_rate * len(targets))
    noisy_indices = torch.randperm(len(targets))[:num_noisy]

    # ランダムなラベルに変更
    for idx in noisy_indices:
        original_label = targets[idx].item()
        new_label = random.randint(0, num_classes - 1)
        # 元のラベルと同じ場合は変更し続ける
        while new_label == original_label:
            new_label = random.randint(0, num_classes - 1)
        noisy_targets[idx] = new_label

    return noisy_targets

def train_model(model, train_loader, optimizer, criterion, device):
    """
    Train the model on the training dataset.
    
    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        criterion (torch.nn.Module): The loss function to use.
        device (torch.device): The device to use for computation.
        
    Returns:
        tuple: A tuple containing the average loss and accuracy on the training dataset.
    """
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        
    accuracy = total_correct / total_samples
    error = 1 - accuracy
    return running_loss / len(train_loader), accuracy, error

def test_model(model, test_loader, criterion, device):
    """
    Evaluate the model on the test dataset.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (torch.utils.data.DataLoader): The test data loader.
        criterion (torch.nn.Module): The loss function to use.
        device (torch.device): The device to use for computation.
    
    Returns:
        tuple: A tuple containing the average loss, predicted labels, true labels, and accuracy on the test dataset.
    """
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (preds == labels).sum().item()
            
    accuracy = total_correct / total_samples
    error = 1 - accuracy
    return running_loss / len(test_loader), accuracy, error

def main():
    print("start training and testing")
    
    args = parse_args()
    # set seed
    set_seed(args.fix_seed)
    
    # set device
    device = set_device(args.gpu)
    
    # load datasets
    train_dataset, test_dataset, imagesize, num_classes, in_channels = load_datasets(args.dataset, args.target, args.gray_scale)

    # set label noise
    if args.label_noise_rate > 0.0:
        if args.dataset == "colored_emnist":
            train_dataset.tensors = (train_dataset.tensors[0], add_label_noise(train_dataset.tensors[1], args.label_noise_rate, num_classes))
        else:
            train_dataset.targets = add_label_noise(train_dataset.targets, args.label_noise_rate, num_classes)

    # set dataloader    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    # load model
    model = load_models(in_channels, args, imagesize, num_classes)
    model = model.to(device)    

    # set optimizer
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Invalid optimizer name.")
    
    # set loss function
    if args.loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "focal_loss":
        criterion = FocalLoss()
    else:
        raise ValueError("Invalid loss function name.")
    
    # set expericment name
    if args.dataset != "colored_emnist":
        experiment_name = f'{args.model}_{args.dataset}_lr{args.lr}_batch{args.batch}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}'
    else:
        experiment_name = f'{args.model}_{args.dataset}_{args.target}_lr{args.lr}_batch{args.batch}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}'
    print(f'Experiment name: {experiment_name}')
    
    # set wandb
    if args.wandb:
        wandb.init(project=args.wandb_project, name=experiment_name, entity=args.wandb_entity)
        wandb.config.update(args)
    
    #get initial values
    train_loss, train_accuracy, train_error = test_model(model, test_loader, criterion, device)
    test_loss, test_accuracy, test_error = test_model(model, test_loader, criterion, device)
    os.makedirs(f"./csv/32/{experiment_name}", exist_ok=True)
    with open(f"./csv/32/{experiment_name}/log.csv", 'a') as f:
        writer = csv.writer(f)
        f.write("epoch, train_loss, train_accuracy, train_error, test_loss, test_accuracy, test_error\n")
        f.write(f"0, {train_loss}, {train_accuracy}, {train_error}, {test_loss}, {test_accuracy}, {test_error}\n")
    
    # train model
    for epoch in range(args.epoch):
        epoch = epoch + 1 # start from 1
        train_loss, train_accuracy, train_error = train_model(model, train_loader, optimizer, criterion, device)
        test_loss, test_accuracy, test_error = test_model(model, test_loader, criterion, device)
        print("epoch: {}, train_loss: {:.4f}, train_accuracy: {:.4f}, train_error: {:.4f},test_loss: {:.4f}, test_accuracy: {:.4f}, test_error: {:.4f}" \
                .format(epoch, train_loss, train_accuracy, train_error, test_loss, test_accuracy, test_error))
        if args.wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy, "train_error": train_error, "test_loss": test_loss, "test_accuracy": test_accuracy, "test_error": test_error})
        
        #save logs to ./csv/
        with open(f"./csv/32/{experiment_name}/log.csv", 'a') as f:
            writer = csv.writer(f)
            f.write(f"{epoch}, {train_loss}, {train_accuracy}, {train_error}, {test_loss}, {test_accuracy}, {test_error}\n")
        # save model. make directry if not exist
        os.makedirs(f"./model_weights/32/{experiment_name}", exist_ok=True)
        torch.save(model.state_dict(), f"./model_weights/32/{experiment_name}/model_{epoch}.pth")

if __name__ == "__main__":
    main()