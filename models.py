"""
IBC-EDL Model: Three-expert architecture with EDL heads
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class IBC_EDL_Model(nn.Module):
    """
    Multi-expert model with EDL heads for long-tailed classification
    
    Architecture:
    - Shared backbone: PreActResNet18
    - Three classification heads: main, tail, medium
    """
    def __init__(self, num_classes=10):
        super(IBC_EDL_Model, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes
        
        # Shared backbone (PreActResNet18)
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(PreActBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(PreActBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(PreActBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(PreActBlock, 512, 2, stride=2)
        
        # Three classification heads
        self.fc_main = nn.Linear(512, num_classes)
        self.fc_tail = nn.Linear(512, num_classes)
        self.fc_medium = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x, return_features=False):
        # Backbone forward
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        features = out.view(out.size(0), -1)
        
        # Three expert heads
        logits_main = self.fc_main(features)
        logits_tail = self.fc_tail(features)
        logits_medium = self.fc_medium(features)
        
        if return_features:
            return features, logits_main, logits_tail, logits_medium
        return logits_main, logits_tail, logits_medium
