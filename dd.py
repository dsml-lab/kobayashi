import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms;
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import math
from model import resnet18k 
import torchvision.models as models
import numpy as np
import time
import wandb
import argparse
import random
import matplotlib.pyplot as plt
import csv 
import warnings

# settings
def parse_args():
    arg_parser = argparse.ArgumentParser(description="ResNet trained by CIFAR-10")
    
    arg_parser.add_argument("-k", "--model_width", type=int, default=1)
    arg_parser.add_argument("-e", "--epoch", type=int, default=4000)
    arg_parser.add_argument("-l", "--label_noise_rate", type=float, default=0.0)
    arg_parser.add_argument("--model", type=str, choices=["resnet18k", "resnet18", "resnet50", "wide_resnet50", "vgg16", "vgg19", "densenet121", ], help="モデルアーキテクチャの選択")
    arg_parser.add_argument("-pt", "--pretrained", action='store_true', help="事前学習モデルの利用,ResNet18のときのみ有効,引数をつけるとTrue")
    arg_parser.add_argument("-ds", "--dataset", type=str, choices=["cifar10", "cifar100", "tinyImageNet"], default="cifar10")
    arg_parser.add_argument("--num_classes", type=int, default=10, help="分類クラス数")
    arg_parser.add_argument("-seed", "--fix_seed", type=int, default=42)
    arg_parser.add_argument("-gray_scale", "--gray_scale", action='store_true', help="グレースケールに変換するかどうか")
    # GPUを指定
    arg_parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU device ID")

    return arg_parser.parse_args()

def fix_seed(seed=42):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    # Tensorflow
    # tf.random.set_seed(seed)

class WeightedGrayscaleTransform:
    def __call__(self, img):
        # img: (C, H, W) -> (H, W, C)
        img = img.permute(1, 2, 0)
        # 重みを適用してグレースケールに変換
        gray_img = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
        # (H, W) -> (1, H, W)
        gray_img = gray_img.unsqueeze(0)
        return gray_img

def get_model(args):
    """モデルの読み込み

    Arguments:
        args: 
    Returns:
        model, model_fullname
    """
    model_fullname = ""  # Initialize an empty string for the model_fullname

    if args.model == "resnet18k":
        assert args.model_width is not None, "please check k value"
        model = resnet18k.make_resnet18k(k=args.model_width, num_classes=args.num_classes)
        model_fullname = "SR_resnet18k-{}".format(args.model_width)
    elif args.model == "resnet18":
        if not args.pretrained:
            model = models.resnet18(pretrained=False, num_classes=args.num_classes)
            model_fullname = "SR_resnet18"
        else:
            model = models.resnet18(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features=in_features, out_features=args.num_classes, bias=True)
            model_fullname = "IN_resnet18"
    elif args.model == "resnet50":
        if not args.pretrained:
            model = models.resnet50(pretrained=False, num_classes=args.num_classes)
            model_fullname = "SR_resnet50"
        else:
            model = models.resnet50(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features=in_features, out_features=args.num_classes, bias=True)
            model_fullname = "IN_resnet50"
    elif args.model == "wide_resnet50":
        if not args.pretrained:
            model = models.wide_resnet50_2(pretrained=False, num_classes=args.num_classes)
            model_fullname = "SR_wide_resnet50"
        else:
            model = models.wide_resnet50_2(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features=in_features, out_features=args.num_classes, bias=True)
            model_fullname = "IN_wide_resnet50"
    elif args.model == "resnet101":
        if not args.pretrained:
            model = models.resnet101(pretrained=False, num_classes=args.num_classes)
            model_fullname = "SR_resnet101"
        else:
            model = models.resnet101(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features=in_features, out_features=args.num_classes, bias=True)
            model_fullname = "IN_resnet101"
    elif args.model == "wide_resnet101":
        if not args.pretrained:
            model = models.wide_resnet101_2(pretrained=False, num_classes=args.num_classes)
            model_fullname = "SR_wide_resnet101"
        else:
            model = models.wide_resnet101_2(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features=in_features, out_features=args.num_classes, bias=True)
            model_fullname = "IN_wide_resnet101"
    elif args.model == "vgg16":
        if not args.pretrained:
            model = models.vgg16(pretrained=False, num_classes=args.num_classes)
            model_fullname = "SR_vgg16"
        else:
            model = models.vgg16(pretrained=True)
            in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_features=in_features, out_features=args.num_classes, bias=True)
            model_fullname = "IN_vgg16"
    elif args.model == "vgg19":
        if not args.pretrained:
            model = models.vgg19(pretrained=False, num_classes=args.num_classes)
            model_fullname = "SR_vgg19"
        else:
            model = models.vgg19(pretrained=True)
            in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_features=in_features, out_features=args.num_classes, bias=True)
            model_fullname = "IN_vgg19"
    elif args.model == "densenet121":
        if not args.pretrained:
            model = models.densenet121(pretrained=False, num_classes=args.num_classes)
            model_fullname = "SR_densenet121"
        else:
            model = models.densenet121(pretrained=True)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features=in_features, out_features=args.num_classes, bias=True)
            model_fullname = "IN_densenet121"
    elif args.model == "densenet161":
        if not args.pretrained:
            model = models.densenet161(pretrained=False, num_classes=args.num_classes)
            model_fullname = "SR_densenet161"
        else:
            model = models.densenet161(pretrained=True)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features=in_features, out_features=args.num_classes, bias=True)
            model_fullname = "IN_densenet161"

    return model, model_fullname

def get_dataset_train(args, transform):
    if args.dataset == "cifar10":
        return torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif args.dataset == "cifar100":
        return torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    elif args.dataset == "tinyImageNet":
        return torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform)
        
def get_dataset_test(args, transform):
    if args.dataset == "cifar10":
        return torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif args.dataset == "cifar100":
        return torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    elif args.dataset == "tinyImageNet":
        return torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=transform)
    

def get_imagesize(args):
    if args.dataset == "cifar10":
        return 32
    elif args.dataset == "cifar100":
        return 32
    elif args.dataset == "tinyImageNet":
        return 64

def get_num_classes(args):
    if args.dataset == "cifar10":
        return 10
    elif args.dataset == "cifar100":
        return 100
    elif args.dataset == "tinyImageNet":
        return 200

def calc_score(true_list, predict_list, running_loss, data_loader_length):
    acc = accuracy_score(true_list, predict_list)
    error_rate = 1 - acc
    loss = running_loss / data_loader_length
    
    return acc, error_rate, loss

def train(model, device, train_loader, criterion, optimizer, epoch, model_fullname, dataset, label_noise_rate, args):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        output_list = []
        target_list = []
        running_loss = 0.0
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        output_list.extend([int(o.argmax()) for o in outputs])
        target_list.extend([int(t) for t in targets])

        train_acc, train_error, train_loss = calc_score(target_list, output_list, running_loss, len(train_loader))

    return train_acc, train_error, train_loss  # return average loss

def test(model, device, test_loader, criterion, epoch, model_fullname, dataset, label_noise_rate):
    model.eval()
    
    running_loss = 0.0
    output_list = []
    target_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            output_list.extend([int(o.argmax()) for o in outputs])
            target_list.extend([int(t) for t in targets])

    test_acc, test_error, test_loss = calc_score(target_list, output_list, running_loss, len(test_loader))
    
    return test_acc, test_error, test_loss

def main():
    #setting
    args = parse_args()
    epoch = args.epoch
    label_noise_rate = args.label_noise_rate
    args.num_classes = get_num_classes(args)
    fix_seed(args.fix_seed)
    imageSize = get_imagesize(args)
    fix_seed(42)
    
    # Set the specified GPU device
    device_id = args.gpu
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        device = f'cuda:{device_id}'
    else:
        print("CUDA is not available. Using CPU instead.")
        device = 'cpu'
    
    # Data preparation
    if not args.gray_scale:
        transform_train = transforms.Compose([
            transforms.Resize(imageSize),
            transforms.RandomCrop(imageSize, padding=imageSize // 8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
    train_set = get_dataset_train(args, transform_train)
    
    # put label noise
    for i in range(len(train_set.targets)):
        if(random.randint(0, 9999) < int(label_noise_rate * 10000)):
            train_set.targets[i] += random.randint(1, args.num_classes - 1)
            train_set.targets[i] %= args.num_classes
    
    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=512, shuffle=True, num_workers=4)
    test_set = get_dataset_test(args, transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=512, shuffle=False, num_workers=4)
    
    # Classes of CIFAR10
    class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Model preparation
    model, model_fullname = get_model(args)
    args.model_fullname = model_fullname
    model = model.to(device)
    print(args.model_fullname)

    print(torch.cuda.is_available())
    
    # Use DataParallel if more than one GPU is available
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs.")
    #     # model = nn.DataParallel(model)
    #     model = torch.nn.DataParallel(model)
    
    #wandb init
    wandb.init(project="iwase_dd", entity="dsml-kernel24", name="{}_{}_{}epoch_ln{}pc".format(args.model_fullname, args.dataset, args.epoch, int(label_noise_rate * 100)), config=args)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
    # Getting initial values before training begins
    initial_epoch = 0
    train_acc, train_error, train_loss = test(model, device, test_loader, criterion, initial_epoch, model_fullname, args.dataset, args.label_noise_rate)
    test_acc, test_error, test_loss = test(model, device, test_loader, criterion, initial_epoch, model_fullname, args.dataset, args.label_noise_rate)
    
    # logging initial epoch
    # wandb.log({
    #     "epoch": initial_epoch,
    #     "train_loss": train_loss,
    #     "train_acc": train_acc,
    #     "train_error": train_error,
    #     "test_loss": test_loss,
    #     "test_acc": test_acc,
    #     "test_error": test_error
    # })
    # ## csv writer
    # with open('./csv/{}_{}_{}epochs_ln{}pc_train.csv'.format(args.model_fullname, args.dataset, args.epoch, int(label_noise_rate * 100)),'w') as file:
    #     file.write("epoch,error,loss\n")
    #     file.write(f"0,{train_error},{train_loss}" + "\n")
    # # test
    # with open('./csv/{}_{}_{}epochs_ln{}pc_test.csv'.format(args.model_fullname, args.dataset, args.epoch, int(label_noise_rate * 100)),'w') as file:
    #     file.write("epoch,error,loss\n")
    #     file.write(f"0,{test_error},{test_loss}" + "\n")
    
    
    # Training loop
    for epoch in range(args.epoch):
        # Training and testing
        train_acc, train_error, train_loss = train(model, device, train_loader, criterion, optimizer, epoch, model_fullname, args.dataset, args.label_noise_rate, args)
        test_acc, test_error, test_loss = test(model, device, test_loader, criterion, epoch, model_fullname, args.dataset, args.label_noise_rate)
 
        wandb.log({
            "epoch": epoch+1, 
            "train_loss": train_loss, 
            "train_acc": train_acc, 
            "train_error": train_error,
            "test_loss": test_loss, 
            "test_acc": test_acc, 
            "test_error": test_error
        })

        print(f'Epoch: {epoch + 1}, Train Acc: {train_acc:<8.5f}, Train Loss: {train_loss:<8.5f}, Test Acc: {test_acc:<8.5f}, Test Loss: {test_loss:<8.5f}')
        # CSV writer
        with open('./csv/{}_{}_{}epochs_ln{}pc_seed{}_train.csv'.format(model_fullname, args.dataset, epoch + 1, int(label_noise_rate * 100), args.fix_seed),'a') as file:
            file.write(f"{epoch+1},{1.0 - train_acc},{train_loss}\n")
        with open('./csv/{}_{}_{}epochs_ln{}pc_seed{}_test.csv'.format(model_fullname, args.dataset, epoch + 1, int(label_noise_rate * 100), args.fix_seed),'a') as file:
            file.write(f"{epoch+1},{1.0 - test_acc},{test_loss}\n")

        # モデルの保存パス
        model_save_dir = f'./model_weights/{args.model_fullname}'
        model_dir = os.path.dirname(model_save_dir)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir, exist_ok=True)
        ## model save
        model_path = f'./model_weights/{args.model_fullname}/{args.model_fullname}_epoch{(epoch + 1):04}_{args.dataset}_ln{int(label_noise_rate * 100)}pc.pth'

        # モデルを保存
        torch.save(model.state_dict(), model_path)
        # wandb.save(model_path)

    # checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, 
    f'./checkpoint/{args.model_fullname}_{(epoch + 1)}epochs_{args.dataset}_ln{int(label_noise_rate * 100)}pc.tar')
    wandb.finish()

if __name__ =='__main__':
    warnings.filterwarnings('ignore')
    start = time.perf_counter()
    main()
    
    print(time.perf_counter() - start)
