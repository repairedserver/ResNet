import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    mul = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()

        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleNeck(nn.Module):
    mul = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.conv3 = nn.Conv2d(out_planes, out_planes * self.mul, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes * self.mul)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != out_planes * self.mul:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes * self.mul, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes * self.mul)
            )
            