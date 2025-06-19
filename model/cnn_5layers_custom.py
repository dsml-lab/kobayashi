import torch
import torch.nn as nn

class CNN5Layer_cus(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, width_multiplier=1, img_size=(32, 32)):
        super(CNN5Layer_cus, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, int(1 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(int(1 * width_multiplier), int(2 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(int(2 * width_multiplier), int(4 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(int(4 * width_multiplier), int(8 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(int(8 * width_multiplier), int(16 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        
        # Fully connected層の入力サイズの計算
        self.fc_input_size = int(16 * width_multiplier) * (img_size[0] // 8) * (img_size[1] // 8)
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        x = self.conv5(x)
        x = self.relu5(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
