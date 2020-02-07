from torchvision import models
from torch import nn
import torch
from utils import swish, mish


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=10, activation = 'relu'):
        super().__init__()
        
        if activation == 'relu':
            f_activation = nn.ReLU6(inplace=True)
            
        if activation == 'swish':
            f_activation = swish()
            
        if activation == 'mish':
            f_activation = mish()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            f_activation,

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            f_activation,

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x
        
        return residual
    
class MobileNetV2(nn.Module):

    def __init__(self, class_num=10, activation = 'relu'):
        super().__init__()
        
        if activation == 'relu':
            f_activation = nn.ReLU6(inplace=True)
            
        if activation == 'swish':
            f_activation = swish()
            
        if activation == 'mish':
            f_activation = mish()

        self.pre = nn.Sequential(
            nn.Conv2d(1, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            f_activation
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1, activation = activation)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6, activation = activation)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6, activation = activation)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6, activation = activation)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6, activation = activation)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6, activation = activation)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6, activation = activation)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            f_activation
        )

        self.conv2 = nn.Conv2d(1280, class_num, 1)
        
    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x
    
    def _make_stage(self, repeat, in_channels, out_channels, stride, t, activation = 'relu'):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t, activation = activation))
        
        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t, activation = activation))
            repeat -= 1
        
        return nn.Sequential(*layers)

def mobilenetv2(activation = 'relu'):
    return MobileNetV2(class_num = 1000, activation = activation)


class BengaliModel(nn.Module):
    def __init__(self, backbone_model):
        super(BengaliModel, self).__init__()
        #self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
        self.backbone_model = backbone_model
        self.fc1 = nn.Linear(in_features=1000, out_features=168) # grapheme_root
        self.fc2 = nn.Linear(in_features=1000, out_features=11) # vowel_diacritic
        self.fc3 = nn.Linear(in_features=1000, out_features=7) # consonant_diacritic
        
    def forward(self, x):
        # pass through the backbone model
        #y = self.conv(x)
        y = self.backbone_model(x)
        
        # multi-output
        grapheme_root = self.fc1(y)
        vowel_diacritic = self.fc2(y)
        consonant_diacritic = self.fc3(y)
        
        return grapheme_root, vowel_diacritic, consonant_diacritic