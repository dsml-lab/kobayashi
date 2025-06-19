import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # 初段: CIFAR 用に変更（3x3 conv）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # MaxPool は省略

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # He初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, 32, 32]

        x = self.layer1(x)  # [B, 256, 32, 32]
        x = self.layer2(x)  # [B, 512, 16, 16]
        x = self.layer3(x)  # [B, 1024, 8, 8]
        x = self.layer4(x)  # [B, 2048, 4, 4]

        x = self.avgpool(x)  # [B, 2048, 1, 1]
        x = torch.flatten(x, 1)  # [B, 2048]
        x = self.fc(x)  # [B, num_classes]

        return x

def resnet50_cifar100():
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=100)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# === Load tr_mem and assign groups ===
infl = np.load('data/cifar100_infl_matrix.npz')
tr_mem = infl['tr_mem']

def assign_mem_group(mem):
    if mem < 0.25:
        return 0
    elif mem >= 0.9:
        return 4
    elif mem >= 0.8:
        return 3
    elif mem >= 0.5:
        return 2
    else:
        return 1

print("assigning memory groups...")
group_flags = np.array([assign_mem_group(m) for m in tr_mem])
for g in range(5):
    count = np.sum(group_flags == g)
    print(f"  Group {g}: {count} samples")
# === Dataset wrapper ===
class CIFAR100WithGroup(Dataset):
    def __init__(self, base_dataset, group_flags):
        self.base = base_dataset
        self.group_flags = group_flags

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data, label = self.base[idx]
        group_id = self.group_flags[idx]
        return data, label, group_id

# === Transforms ===
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
])

# === Data Loaders ===
base_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainset = CIFAR100WithGroup(base_trainset, group_flags)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

# === Model and optimizer ===
model = model = resnet50_cifar100().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# === Training function ===
def train_model(model, dataloader, optimizer, criterion):
    model.train()
    correct_by_group = defaultdict(int)
    total_by_group = defaultdict(int)
    loss_sum_by_group = defaultdict(float)
    count_by_group = defaultdict(int)

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    for inputs, targets, group_ids in dataloader:
        inputs, targets, group_ids = inputs.to(device), targets.to(device), group_ids.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct = preds.eq(targets)

        total_correct += correct.sum().item()
        total_samples += targets.size(0)
        total_loss += loss.item() * targets.size(0)

        for g in range(5):
            mask = group_ids == g
            mask_sum = mask.sum().item()
            if mask_sum > 0:
                group_loss = criterion(outputs[mask], targets[mask])
                correct_by_group[g] += correct[mask].sum().item()
                total_by_group[g] += mask_sum
                loss_sum_by_group[g] += group_loss.item() * mask_sum
                count_by_group[g] += mask_sum

    acc_by_group = {
        g: correct_by_group[g] / total_by_group[g] if total_by_group[g] > 0 else 0.0
        for g in range(5)
    }
    loss_by_group = {
        g: loss_sum_by_group[g] / count_by_group[g] if count_by_group[g] > 0 else 0.0
        for g in range(5)
    }

    return {
        "acc": acc_by_group,
        "loss": loss_by_group,
        "overall_acc": total_correct / total_samples,
        "overall_loss": total_loss / total_samples
    }

# === Test function ===
def test_model(model, dataloader, criterion):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            preds = outputs.argmax(dim=1)
            total_correct += preds.eq(targets).sum().item()
            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)

    return {
        "acc": total_correct / total_samples,
        "loss": total_loss / total_samples
    }

# === Training loop ===
num_epochs = 100
train_log = []
test_log = []

for epoch in range(num_epochs):
    train_metrics = train_model(model, trainloader, optimizer, criterion)
    test_metrics = test_model(model, testloader, criterion)

    train_log.append(train_metrics)
    test_log.append(test_metrics)

    print(f"\nEpoch {epoch+1}")
    for g in range(5):
        print(f"  Group {g}: Acc = {train_metrics['acc'][g]*100:.2f}%, Loss = {train_metrics['loss'][g]:.4f}")
    print(f"  Train Overall: Acc = {train_metrics['overall_acc']*100:.2f}%, Loss = {train_metrics['overall_loss']:.4f}")
    print(f"  Test Overall : Acc = {test_metrics['acc']*100:.2f}%, Loss = {test_metrics['loss']:.4f}")

# === Save results to CSV ===
base_dir = "result_mem"
csv_dir = os.path.join(base_dir, "csv")
fig_dir = os.path.join(base_dir, "fig")
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

epochs = list(range(1, num_epochs + 1))
train_acc_df = pd.DataFrame([{**{"epoch": e + 1}, **train_log[e]["acc"]} for e in range(num_epochs)])
train_loss_df = pd.DataFrame([{**{"epoch": e + 1}, **train_log[e]["loss"]} for e in range(num_epochs)])
train_overall_df = pd.DataFrame({
    "epoch": epochs,
    "overall_acc": [log["overall_acc"] for log in train_log],
    "overall_loss": [log["overall_loss"] for log in train_log],
})
test_df = pd.DataFrame({
    "epoch": epochs,
    "test_acc": [log["acc"] for log in test_log],
    "test_loss": [log["loss"] for log in test_log],
})

train_acc_df.to_csv(os.path.join(csv_dir, "train_group_accuracy.csv"), index=False)
train_loss_df.to_csv(os.path.join(csv_dir, "train_group_loss.csv"), index=False)
train_overall_df.to_csv(os.path.join(csv_dir, "train_overall.csv"), index=False)
test_df.to_csv(os.path.join(csv_dir, "test_overall.csv"), index=False)

# === Plot: train_group_accuracy.png ===
plt.figure(figsize=(10, 6))
for g in range(5):
    plt.plot(train_acc_df["epoch"], train_acc_df[g], label=f"Train Group {g}")
plt.plot(train_overall_df["epoch"], train_overall_df["overall_acc"], label="Train Overall", linestyle="--", linewidth=2)
plt.plot(test_df["epoch"], test_df["test_acc"], label="Test Overall", linestyle=":", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train Group Accuracy vs Overall Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "train_group_accuracy.png"))
plt.close()
