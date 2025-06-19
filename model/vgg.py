import torch
import torch.nn as nn

class ModifiedVGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, width_multiplier=1, img_size=(32, 32), model_type='VGG16'):
        super(ModifiedVGG, self).__init__()
        
        # Configure model architectures
        self.cfgs = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        }
        
        # Create features layers with width multiplier
        self.features = self._make_layers(self.cfgs[model_type], in_channels, width_multiplier)
        
        # Calculate output size after features
        feature_output_size = img_size[0]
        for layer in self.cfgs[model_type]:
            if layer == 'M':
                feature_output_size //= 2
        
        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Calculate classifier input size
        self.classifier_input_size = int(512 * width_multiplier) * 7 * 7
        
        # Classifier with width multiplier
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_size, int(4096 * width_multiplier)),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(int(4096 * width_multiplier), int(4096 * width_multiplier)),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(int(4096 * width_multiplier), num_classes),
        )
        
        self._initialize_weights()

    def _make_layers(self, cfg, in_channels, width_multiplier):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                adjusted_v = int(v * width_multiplier)
                conv2d = nn.Conv2d(in_channels, adjusted_v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = adjusted_v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# Helper functions to create specific VGG variants
def vgg11(in_channels=3, num_classes=10, width_multiplier=1, img_size=(32, 32)):
    return ModifiedVGG(in_channels, num_classes, width_multiplier, img_size, model_type='VGG11')

def vgg13(in_channels=3, num_classes=10, width_multiplier=1, img_size=(32, 32)):
    return ModifiedVGG(in_channels, num_classes, width_multiplier, img_size, model_type='VGG13')

def vgg16(in_channels=3, num_classes=10, width_multiplier=1, img_size=(32, 32)):
    return ModifiedVGG(in_channels, num_classes, width_multiplier, img_size, model_type='VGG16')

def vgg19(in_channels=3, num_classes=10, width_multiplier=1, img_size=(32, 32)):
    return ModifiedVGG(in_channels, num_classes, width_multiplier, img_size, model_type='VGG19')