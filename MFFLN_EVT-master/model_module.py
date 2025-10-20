# model_module.py
# ------------------------------------------------------------
# Contains all network definitions: GRL, SEBlock, MultiScaleBlock,
# FeatureExtractor, DomainClassifier, and HealthStatusClassifier.
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x): return GradientReversalFunction.apply(x, self.alpha)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.relu, self.sigmoid = nn.ReLU(True), nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class MultiScaleBlock(nn.Module):
    """Depthwise separable convs + SEBlock fusion."""
    def __init__(self, in_c, out_c, reduction=16):
        super().__init__()
        self.dw1 = nn.Conv2d(in_c, in_c, 1, 1, 0, groups=in_c, bias=False)
        self.dw3 = nn.Conv2d(in_c, in_c, 3, 1, 1, groups=in_c, bias=False)
        self.dw5 = nn.Conv2d(in_c, in_c, 5, 1, 2, groups=in_c, bias=False)
        self.se = SEBlock(in_c, reduction)
        self.bn, self.relu = nn.BatchNorm2d(in_c * 5), nn.ReLU(True)
        self.pw = nn.Conv2d(in_c * 5, out_c, 1, 1, 0, bias=False)

    def forward(self, x):
        out = torch.cat([x, self.dw1(x), self.dw3(x), self.dw5(x), self.se(x)], dim=1)
        return self.pw(self.relu(self.bn(out)))


class FeatureExtractor(nn.Module):
    """Five stacked MultiScale blocks."""
    def __init__(self):
        super().__init__()
        self.init = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, 1),
        )
        self.blocks = nn.ModuleList([
            MultiScaleBlock(16, 32), MultiScaleBlock(32, 64),
            MultiScaleBlock(64, 128), MultiScaleBlock(128, 256),
            MultiScaleBlock(256, 512)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(c) for c in [32, 64, 128, 256, 512]])
        self.relu = nn.ReLU(True)
        self.pool, self.flat = nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()

    def forward(self, x):
        x = self.init(x)
        for blk, bn in zip(self.blocks, self.bns):
            x = self.relu(bn(blk(x)))
        return self.flat(self.pool(x))


class DomainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 2), nn.Softmax(dim=1)
        )

    def forward(self, x, alpha): return self.net(GradientReversalFunction.apply(x, alpha))


class HealthStatusClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(512, num_classes + 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.fc(x))
