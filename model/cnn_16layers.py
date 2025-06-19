import torch
import torch.nn as nn


class CNN16Layer(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, width_multiplier=1, img_size=(32, 32)):
        super(CNN16Layer, self).__init__()
        
        # Block 1 (入力処理)
        self.conv1 = nn.Conv2d(in_channels, int(64 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(int(64 * width_multiplier))
        self.relu1 = nn.ReLU()
        
        # Block 2
        self.conv2 = nn.Conv2d(int(64 * width_multiplier), int(64 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(int(64 * width_multiplier))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv3 = nn.Conv2d(int(64 * width_multiplier), int(128 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(int(128 * width_multiplier))
        self.relu3 = nn.ReLU()
        
        # Block 4
        self.conv4 = nn.Conv2d(int(128 * width_multiplier), int(128 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(int(128 * width_multiplier))
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 5
        self.conv5 = nn.Conv2d(int(128 * width_multiplier), int(256 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(int(256 * width_multiplier))
        self.relu5 = nn.ReLU()
        
        # Block 6
        self.conv6 = nn.Conv2d(int(256 * width_multiplier), int(256 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(int(256 * width_multiplier))
        self.relu6 = nn.ReLU()
        
        # Block 7
        self.conv7 = nn.Conv2d(int(256 * width_multiplier), int(256 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(int(256 * width_multiplier))
        self.relu7 = nn.ReLU()
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 8
        self.conv8 = nn.Conv2d(int(256 * width_multiplier), int(512 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(int(512 * width_multiplier))
        self.relu8 = nn.ReLU()
        
        # Block 9
        self.conv9 = nn.Conv2d(int(512 * width_multiplier), int(512 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(int(512 * width_multiplier))
        self.relu9 = nn.ReLU()
        
        # Block 10
        self.conv10 = nn.Conv2d(int(512 * width_multiplier), int(512 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(int(512 * width_multiplier))
        self.relu10 = nn.ReLU()
        self.pool10 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 11
        self.conv11 = nn.Conv2d(int(512 * width_multiplier), int(512 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(int(512 * width_multiplier))
        self.relu11 = nn.ReLU()
        
        # Block 12
        self.conv12 = nn.Conv2d(int(512 * width_multiplier), int(512 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(int(512 * width_multiplier))
        self.relu12 = nn.ReLU()
        
        # Block 13
        self.conv13 = nn.Conv2d(int(512 * width_multiplier), int(512 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.bn13 = nn.BatchNorm2d(int(512 * width_multiplier))
        self.relu13 = nn.ReLU()
        self.pool13 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 14-16 (最終特徴抽出)
        self.conv14 = nn.Conv2d(int(512 * width_multiplier), int(512 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.bn14 = nn.BatchNorm2d(int(512 * width_multiplier))
        self.relu14 = nn.ReLU()
        
        self.conv15 = nn.Conv2d(int(512 * width_multiplier), int(512 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.bn15 = nn.BatchNorm2d(int(512 * width_multiplier))
        self.relu15 = nn.ReLU()
        
        self.conv16 = nn.Conv2d(int(512 * width_multiplier), int(512 * width_multiplier), kernel_size=3, stride=1, padding=1)
        self.bn16 = nn.BatchNorm2d(int(512 * width_multiplier))
        self.relu16 = nn.ReLU()
        
        # Pooling後のサイズを計算（5回のpoolingで32）
        self.fc_input_size = int(512 * width_multiplier) * (img_size[0] // 32) * (img_size[1] // 32)
        
        # 分類層
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        # Block 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        
        # Block 6
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        
        # Block 7
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.pool7(x)
        
        # Block 8
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)
        
        # Block 9
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu9(x)
        
        # Block 10
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu10(x)
        x = self.pool10(x)
        
        # Block 11
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu11(x)
        
        # Block 12
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu12(x)
        
        # Block 13
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.relu13(x)
        x = self.pool13(x)
        
        # Block 14-16
        x = self.conv14(x)
        x = self.bn14(x)
        x = self.relu14(x)
        
        x = self.conv15(x)
        x = self.bn15(x)
        x = self.relu15(x)
        
        x = self.conv16(x)
        x = self.bn16(x)
        x = self.relu16(x)
        
        # 分類層
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x